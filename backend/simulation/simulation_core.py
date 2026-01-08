import time
import numpy as np
import logging
from typing import Dict, Any

# Import Environment logic
from .smart_home_env import SmartHomeEnv
from .device_config import DEVICE_CONFIG, ACTION_INDICES

logger = logging.getLogger('SimulationCore')
logger.setLevel(logging.INFO)

class SimulationEngine:
    def __init__(self):
        self.step_count = 0
        self.start_time = time.time()
        
        # --- 1. CONFIGURATION (Shared) ---
        # Cấu hình mới cho Device-Specific Environment
        self.config = {
            'time_step_hours': 1.0,
            'sim_start': '2025-01-01',
            'sim_steps': 24,
            'sim_freq': '1h',
            'battery': {
                'capacity_kwh': 10.0,
                'soc_init': 0.5,
                'soc_min': 0.1,
                'soc_max': 0.9,
                'p_charge_max_kw': 3.0,
                'p_discharge_max_kw': 3.0,
                'eta_ch': 0.95,
                'eta_dis': 0.95
            },
            'pv_config': {
                'latitude': 10.762622,
                'longitude': 106.660172,
                'tz': 'Asia/Ho_Chi_Minh',
                'surface_tilt': 10.0,
                'surface_azimuth': 180.0,
                'module_parameters': {'pdc0': 3.0}
            },
            'behavior': {
                'residents': [],
                'must_run_base': 0.15
            }
        }

        # --- 2. INIT DUAL ENVIRONMENTS ---
        # Khởi tạo 2 môi trường riêng biệt cho PPO và Hybrid
        self.env_ppo = SmartHomeEnv(price_profile=None, pv_profile=None, config=self.config)
        self.env_hybrid = SmartHomeEnv(price_profile=None, pv_profile=None, config=self.config)

        # Reset với CÙNG SEED để môi trường (thời tiết, PV) giống hệt nhau
        seed = 42
        np.random.seed(seed)
        self.obs_ppo, _ = self.env_ppo.reset(seed=seed)
        self.obs_hybrid, _ = self.env_hybrid.reset(seed=seed)

        # Lưu thông tin trả về từ step
        self.info_ppo = {}
        self.info_hybrid = {}
        
        # Lưu actions để trả về đúng trong get_data_packet
        self.last_action_ppo = None
        self.last_action_hybrid = None
        
        # Lưu SOC trước để xác định battery state
        self.prev_soc_ppo = self.env_ppo.soc
        self.prev_soc_hybrid = self.env_hybrid.soc
        
        # Trạng thái ban đầu
        self.done = False

    def _get_heuristic_action(self, env_instance, obs, agent_type='ppo'):
        """
        Giả lập hành động dựa trên logic đơn giản.
        TODO: Sau này thay bằng loading model thực tế:
        action, _ = model.predict(obs)
        """
        # Action space mới: [battery, ac_living, ac_master, ac_bed2, ev, wm, dw]
        # Shape: (7,), range: [-1, 1]
        
        hour = self.step_count % 24
        action = np.zeros(7, dtype=np.float32)
        
        # --- Baseline PPO Logic (Mock) ---
        # Battery: charge during day (solar), discharge at night
        if 6 <= hour <= 16:
            action[ACTION_INDICES['battery']] = 0.5  # Charge
        else:
            action[ACTION_INDICES['battery']] = -0.5  # Discharge
        
        # AC: Cool when people are home and it's hot
        temp_out = env_instance.temp_out if hasattr(env_instance, 'temp_out') else 30
        n_home = env_instance.load_schedules[min(self.step_count, 23)]['n_home'] if hasattr(env_instance, 'load_schedules') else 0
        
        if n_home > 0 and temp_out > 28:
            # Turn on ACs when hot and people are home
            action[ACTION_INDICES['ac_living']] = 0.6
            action[ACTION_INDICES['ac_master']] = 0.4 if hour >= 21 else -0.5
            action[ACTION_INDICES['ac_bed2']] = 0.3 if hour >= 21 else -0.5
        else:
            action[ACTION_INDICES['ac_living']] = -0.8
            action[ACTION_INDICES['ac_master']] = -0.8
            action[ACTION_INDICES['ac_bed2']] = -0.8
        
        # EV: Charge at night (off-peak)
        if 22 <= hour or hour < 6:
            action[ACTION_INDICES['ev']] = 0.8
        else:
            action[ACTION_INDICES['ev']] = -0.5
        
        # WM/DW: Random trigger in the evening
        if agent_type == 'ppo':
            action[ACTION_INDICES['wm']] = 0.5 if (18 <= hour <= 20 and np.random.random() > 0.5) else -0.5
            action[ACTION_INDICES['dw']] = 0.5 if (19 <= hour <= 21 and np.random.random() > 0.6) else -0.5
        else:
            # Hybrid: More intelligent scheduling
            wm_remaining = getattr(env_instance, 'wm_remaining', 0)
            dw_remaining = getattr(env_instance, 'dw_remaining', 0)
            
            # Force on if deadline approaching
            if wm_remaining > 0 and hour >= 20:
                action[ACTION_INDICES['wm']] = 1.0
            elif wm_remaining > 0 and 15 <= hour <= 18:
                action[ACTION_INDICES['wm']] = 0.7
            else:
                action[ACTION_INDICES['wm']] = -0.5
            
            if dw_remaining > 0 and hour >= 21:
                action[ACTION_INDICES['dw']] = 1.0
            elif dw_remaining > 0 and 19 <= hour <= 21:
                action[ACTION_INDICES['dw']] = 0.7
            else:
                action[ACTION_INDICES['dw']] = -0.5
        
        return action

    def update(self):
        """Hàm này được gọi liên tục bởi server để next step"""
        if self.done:
            # Nếu hết 24h, reset lại để chạy vòng lặp mới
            seed = np.random.randint(0, 1000)
            np.random.seed(seed)
            self.obs_ppo, _ = self.env_ppo.reset(seed=seed)
            self.obs_hybrid, _ = self.env_hybrid.reset(seed=seed)
            self.done = False
            self.step_count = 0
            self.prev_soc_ppo = self.env_ppo.soc
            self.prev_soc_hybrid = self.env_hybrid.soc
            logger.info(f"--- Simulation Reset (Seed {seed}) ---")
            return

        # Lưu SOC trước khi step
        self.prev_soc_ppo = self.env_ppo.soc
        self.prev_soc_hybrid = self.env_hybrid.soc

        # 1. Lấy Action (Mock hoặc Model)
        self.last_action_ppo = self._get_heuristic_action(self.env_ppo, self.obs_ppo, 'ppo')
        self.last_action_hybrid = self._get_heuristic_action(self.env_hybrid, self.obs_hybrid, 'hybrid')

        # 2. Step Environment
        self.obs_ppo, _, done_ppo, _, self.info_ppo = self.env_ppo.step(self.last_action_ppo)
        self.obs_hybrid, _, done_hybrid, _, self.info_hybrid = self.env_hybrid.step(self.last_action_hybrid)

        self.done = done_ppo or done_hybrid
        self.step_count += 1

    def _parse_actions(self, action_raw, info, agent_type):
        """Chuyển đổi raw action vector + env info sang dict dễ đọc cho Frontend"""
        
        if action_raw is None or info is None:
            return self._default_actions()
        
        try:
            action = np.array(action_raw).flatten()
            
            return {
                # ACs (from action vector, thresholded)
                "ac_living": int(info.get('ac_living', 0)),
                "ac_master": int(info.get('ac_master', 0)),
                "ac_bed2": int(info.get('ac_bed2', 0)),
                
                # Lights (from env_info - rule-based)
                "light_living": int(info.get('light_living', 0)),
                "light_master": int(info.get('light_master', 0)),
                "light_bed2": int(info.get('light_bed2', 0)),
                "light_kitchen": int(info.get('light_kitchen', 0)),
                "light_toilet": int(info.get('light_toilet', 0)),
                
                # Shiftable devices
                "wm": int(info.get('wm', 0)),
                "dw": int(info.get('dw', 0)),
                "ev": float(info.get('ev', 0)),
                
                # Battery state
                "battery": info.get('battery', 'idle')
            }
        except Exception as e:
            logger.warning(f"Action parsing error: {e}")
            return self._default_actions()
    
    def _default_actions(self):
        """Return default action state when data is unavailable"""
        return {
            "ac_living": 0,
            "ac_master": 0,
            "ac_bed2": 0,
            "light_living": 0,
            "light_master": 0,
            "light_bed2": 0,
            "light_kitchen": 0,
            "light_toilet": 0,
            "wm": 0,
            "dw": 0,
            "ev": 0,
            "battery": "idle"
        }

    def _determine_battery_state(self, old_soc, new_soc):
        """Xác định trạng thái pin dựa trên sự thay đổi SOC"""
        delta = new_soc - old_soc
        if delta > 0.001:
            return "charge"
        if delta < -0.001:
            return "discharge"
        return "idle"

    def _get_price_tier(self, cumulative_kwh: float) -> int:
        """Xác định tier hiện tại dựa trên lượng điện đã dùng"""
        if cumulative_kwh <= 50:
            return 1
        elif cumulative_kwh <= 100:
            return 2
        elif cumulative_kwh <= 200:
            return 3
        elif cumulative_kwh <= 300:
            return 4
        elif cumulative_kwh <= 400:
            return 5
        else:
            return 6

    def get_data_packet(self) -> Dict[str, Any]:
        """Đóng gói dữ liệu JSON theo Data Contract mới với device-specific states"""
        
        # --- MAPPING DATA ---
        
        # 1. Environment Info (dùng info_ppo làm chuẩn vì chung seed)
        env_data = {
            "weather": str(self.info_ppo.get('weather', 'sunny')),
            "temp": float(round(float(self.info_ppo.get('temp', 30.0)), 1)),
            "pv": float(round(float(self.info_ppo.get('pv', 0.0)), 2)),
            "price_tier": int(self._get_price_tier(self.env_ppo.cumulative_import_kwh))
        }

        # 2. PPO Agent Data
        ppo_actions = self._parse_actions(self.last_action_ppo, self.info_ppo, 'ppo')
        
        ppo_data = {
            "bill": int(self.env_ppo.total_cost),
            "soc": float(round(float(self.env_ppo.soc) * 100, 1)),
            "grid": float(round(float(self.env_ppo.cumulative_import_kwh - self.env_ppo.cumulative_export_kwh), 2)),
            "actions": ppo_actions,
            "comfort": 0.0  # TODO: Calculate from room temperature deviations
        }

        # 3. Hybrid Agent Data
        hybrid_actions = self._parse_actions(self.last_action_hybrid, self.info_hybrid, 'hybrid')
        
        hybrid_data = {
            "bill": int(self.env_hybrid.total_cost),
            "soc": float(round(float(self.env_hybrid.soc) * 100, 1)),
            "grid": float(round(float(self.env_hybrid.cumulative_import_kwh - self.env_hybrid.cumulative_export_kwh), 2)),
            "actions": hybrid_actions,
            "comfort": 0.0
        }

        # Format timestamp as HH:mm
        hour = self.step_count % 24
        timestamp = f"{hour:02d}:00"

        return {
            "timestamp": timestamp,
            "env": env_data,
            "ppo": ppo_data,
            "hybrid": hybrid_data
        }

# Singleton instance
sim_instance = SimulationEngine()