import time
import numpy as np
import logging
from typing import Dict, Any

# Import Environment logic
from smart_home_env import SmartHomeEnv, calculate_vietnam_tiered_bill

logger = logging.getLogger('SimulationCore')
logger.setLevel(logging.INFO)

class SimulationEngine:
    def __init__(self):
        self.step_count = 0
        self.start_time = time.time()
        
        # --- 1. CONFIGURATION (Shared) ---
        # Cấu hình giống main.py để đảm bảo logic Environment chuẩn
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
            'residents': [],  # Basic config
            'shiftable_su': [{'rate': 0.5, 'L': 2, 't_s': 0, 't_f': 23}],  # WM
            'shiftable_si': [],
            'adjustable': [{'P_min': 0.5, 'P_max': 2.0, 'P_com': 1.2, 'alpha': 0.06}],  # AC
            'pv_config': {
                'latitude': 10.762622,
                'longitude': 106.660172,
                'tz': 'Asia/Ho_Chi_Minh',
                'surface_tilt': 10.0,
                'surface_azimuth': 180.0,
                'module_parameters': {'pdc0': 3.0}
            },
            'price_tiers': []  # Will use default inside Env
        }

        # --- 2. INIT DUAL ENVIRONMENTS ---
        # Khởi tạo 2 môi trường riêng biệt cho PPO và Hybrid
        # Price và PV để None để Env tự sinh theo logic nội tại
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

    def _get_heuristic_action(self, env_instance, agent_type='ppo'):
        """
        Giả lập hành động. 
        TODO: Sau này thay bằng loading model thực tế:
        action, _ = model.predict(obs)
        """
        # Action space: [SU_1, ..., SI_1, ..., AD_1, ...]
        # Giả sử: 1 SU (Máy giặt), 0 SI, 1 AD (AC) -> Size = 2
        # Logic ngẫu nhiên có trọng số để tạo sự khác biệt
        
        action = env_instance.action_space.sample()
        
        # Mock logic: Hybrid thông minh hơn xíu (ví dụ)
        if agent_type == 'hybrid':
            # Ví dụ: Hybrid thích bật AC khi nóng (bit cuối)
            if len(action) > 0: 
                action[-1] = 1
            
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
        self.last_action_ppo = self._get_heuristic_action(self.env_ppo, 'ppo')
        self.last_action_hybrid = self._get_heuristic_action(self.env_hybrid, 'hybrid')

        # 2. Step Environment
        self.obs_ppo, _, done_ppo, _, self.info_ppo = self.env_ppo.step(self.last_action_ppo)
        self.obs_hybrid, _, done_hybrid, _, self.info_hybrid = self.env_hybrid.step(self.last_action_hybrid)

        self.done = done_ppo or done_hybrid
        self.step_count += 1

    def _parse_actions(self, action_raw, agent_type):
        """Chuyển đổi raw action vector sang dict dễ đọc cho Frontend"""
        # Giả định action vector: [WM, AC] (dựa trên config init ở trên)
        # Cần map chính xác với cấu hình env.su_devs, env.ad_devs
        
        # Fallback an toàn
        ac_status = 0
        wm_status = 0
        ev_status = 0
        
        if action_raw is None:
            return {
                "ac": ac_status,
                "wm": wm_status,
                "ev": ev_status,
                "battery": "idle"
            }
        
        try:
            flat_act = np.array(action_raw).flatten()
            if len(flat_act) >= 1: 
                wm_status = int(flat_act[0])  # SU device
            if len(flat_act) >= 2: 
                ac_status = int(flat_act[1])  # AD device
        except Exception as e:
            logger.warning(f"Action parsing error: {e}")
            
        return {
            "ac": ac_status,
            "wm": wm_status,
            "ev": ev_status,
            "battery": "idle"  # Will be updated in get_data_packet
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
        """Đóng gói dữ liệu JSON theo Data Contract mới"""
        
        # --- MAPPING DATA ---
        
        # 1. Environment Info (dùng info_ppo làm chuẩn vì chung seed)
        env_data = {
            "weather": self.info_ppo.get('weather', 'sunny'),
            "temp": round(float(self.info_ppo.get('temp', 30.0)), 1),
            "pv": round(float(self.info_ppo.get('pv', 0.0)), 2),
            "price_tier": self._get_price_tier(self.env_ppo.cumulative_import_kwh)
        }

        # 2. PPO Agent Data
        ppo_actions = self._parse_actions(self.last_action_ppo, 'ppo')
        ppo_actions['battery'] = self._determine_battery_state(self.prev_soc_ppo, self.env_ppo.soc)
        
        ppo_data = {
            "bill": int(self.env_ppo.total_cost),
            "soc": round(self.env_ppo.soc * 100, 1),
            "grid": round(self.env_ppo.cumulative_import_kwh - self.env_ppo.cumulative_export_kwh, 2),
            "actions": ppo_actions,
            "comfort": 0.0  # TODO: Extract from reward function if needed
        }

        # 3. Hybrid Agent Data
        hybrid_actions = self._parse_actions(self.last_action_hybrid, 'hybrid')
        hybrid_actions['battery'] = self._determine_battery_state(self.prev_soc_hybrid, self.env_hybrid.soc)
        
        hybrid_data = {
            "bill": int(self.env_hybrid.total_cost),
            "soc": round(self.env_hybrid.soc * 100, 1),
            "grid": round(self.env_hybrid.cumulative_import_kwh - self.env_hybrid.cumulative_export_kwh, 2),
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