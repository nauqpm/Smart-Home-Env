"""
WebSocket Server for Smart Home Simulation
Uses REAL trained PPO and Hybrid models from .zip files
Streams real-time comparison data to frontend
"""
import asyncio
import os
import logging
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import from new package structure
from simulation.smart_home_env import SmartHomeEnv
from simulation.device_config import DEVICE_CONFIG, ACTION_INDICES, ROOM_OCCUPANCY_HOURS
# from rl_ppo_hybrid_new import HybridAgentWrapper # REMOVED: Using pure PPO model

# Try to import stable_baselines3
try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("WARNING: stable_baselines3 not installed. Using fallback mode.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('WebSocketServer')

app = FastAPI(title="Smart Home HEMS WebSocket Server - Real Models")

# CORS for HTTP endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths
PPO_MODEL_PATH = "ppo_smart_home.zip"
HYBRID_MODEL_PATH = "ppo_hybrid_smart_home.zip"

# Demo Scenarios for presentation mode
DEMO_SCENARIOS = {
    "ideal": {
        "name": "Ideal Day",
        "description": "Sunny bell-curve PV, normal pricing",
        "pv_profile": [0, 0, 0, 0, 0, 0, 0.2, 0.8, 1.5, 2.5, 3.2, 3.8, 4.0, 3.8, 3.0, 2.0, 1.0, 0.3, 0, 0, 0, 0, 0, 0],
        "temp_profile": [24, 24, 23, 23, 23, 24, 25, 26, 27, 28, 29, 30, 31, 31, 30, 29, 28, 27, 26, 26, 25, 25, 24, 24],
        "price_tier": [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 4, 3, 2, 1, 1],
        "weather": "sunny"
    },
    "erratic": {
        "name": "Erratic Day",
        "description": "Intermittent clouds, PV drops at 10h and 14h",
        "pv_profile": [0, 0, 0, 0, 0, 0, 0.1, 0.5, 1.8, 2.2, 0.2, 0.3, 2.5, 1.0, 0.1, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0],
        "temp_profile": [25, 25, 25, 24, 24, 25, 26, 28, 30, 32, 31, 30, 29, 27, 26, 25, 25, 24, 24, 24, 23, 23, 23, 23],
        "price_tier": [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 4, 3, 2, 1, 1],
        "weather": "cloudy"
    },
    "heatwave": {
        "name": "Heatwave",
        "description": "Extreme heat (41°C peak), high PV, tier 6 pricing",
        "pv_profile": [0, 0, 0, 0, 0, 0.5, 1.2, 2.5, 3.5, 4.2, 4.5, 4.5, 4.5, 4.2, 3.8, 3.0, 2.0, 1.0, 0.2, 0, 0, 0, 0, 0],
        "temp_profile": [28, 28, 27, 27, 27, 28, 30, 33, 36, 38, 40, 41, 41, 40, 38, 36, 34, 32, 31, 30, 30, 29, 29, 28],
        "price_tier": [1, 1, 1, 1, 1, 2, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 5, 4, 3, 2, 1],
        "weather": "sunny"
    }
}


# Session Statistics for Demo Report
class SessionStats:
    """Track metrics during demo session for final report generation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_bill = 0.0
        self.total_grid_import = 0.0
        self.total_comfort_loss = 0.0
        self.temp_history = []  # [{outdoor, indoor}]
        self.energy_stack = []  # [{pv, grid, battery}]
        self.hourly_bills = []
        
        # Additional metrics
        self.total_solar_generated = 0.0
        self.total_solar_used = 0.0
        self.total_battery_discharged = 0.0
        self.peak_load = 0.0
        self.indoor_temps = []  # For avg/min/max calculation
    
    def update(self, step_bill, grid_kw, indoor_temp, outdoor_temp, pv_used, battery_discharged, 
               pv_generated=0, total_load=0):
        self.total_bill += float(step_bill)
        self.total_grid_import += max(0.0, float(grid_kw))
        self.hourly_bills.append(float(step_bill))
        
        # Comfort loss: deviation from 24-27°C comfort range
        indoor = float(indoor_temp)
        if indoor < 24:
            self.total_comfort_loss += (24 - indoor)
        elif indoor > 27:
            self.total_comfort_loss += (indoor - 27)
        
        # Track indoor temps for stats
        self.indoor_temps.append(indoor)
        
        # Solar tracking
        self.total_solar_generated += float(pv_generated)
        self.total_solar_used += float(pv_used)
        
        # Battery tracking
        self.total_battery_discharged += max(0.0, float(battery_discharged))
        
        # Peak load tracking
        if total_load > self.peak_load:
            self.peak_load = total_load
            
        self.temp_history.append({
            "outdoor": round(float(outdoor_temp), 1),
            "indoor": round(indoor, 1)
        })
        
        self.energy_stack.append({
            "pv": round(float(pv_used), 2),
            "grid": round(max(0.0, float(grid_kw)), 2),
            "battery": round(max(0.0, float(battery_discharged)), 2)
        })
    
    def get_avg_indoor_temp(self):
        return sum(self.indoor_temps) / len(self.indoor_temps) if self.indoor_temps else 25.0
    
    def get_min_indoor_temp(self):
        return min(self.indoor_temps) if self.indoor_temps else 25.0
    
    def get_max_indoor_temp(self):
        return max(self.indoor_temps) if self.indoor_temps else 25.0
    
    def get_solar_self_consumption_rate(self):
        if self.total_solar_generated > 0:
            return (self.total_solar_used / self.total_solar_generated) * 100
        return 0.0


class RealModelSimulation:
    """
    Simulation using REAL trained PPO and Hybrid models.
    Falls back to heuristic mode if models are not available.
    """
    
    def __init__(self):
        self.step_count = 0
        self.use_real_models = False
        self.ppo_model = None
        self.hybrid_model = None
        self.hybrid_wrapper = None
        
        # Demo mode state (OFF by default - uses normal simulation)
        self.is_demo_mode = False
        self.current_scenario = "ideal"
        self.paused = False  # Pause state for reports

        
        # Session statistics for demo report
        self.stats_ppo = SessionStats()
        self.stats_hybrid = SessionStats()
        
        # Environment config (Synced with main.py)
        # Keeping 24 hours simulation as standard
        T = 24
        # Scaling for VND roughly (mock) or just unitless cost
        self.price_profile = np.array([0.1] * 6 + [0.15] * 6 + [0.25] * 6 + [0.18] * 6) * 10000 
        
        self.config = {
            "critical": [0.33] * 24, # Tải nền
            "adjustable": [ # Điều hòa
                {"P_min": 0.5, "P_max": 2.0, "P_com": 1.5, "alpha": 0.06}, 
                {"P_min": 0.0, "P_max": 2.0, "P_com": 1.5, "alpha": 0.08}
            ],
            "shiftable_su": [ # Máy giặt, máy rửa bát
                {"name": "washing_machine", "rate": 0.5, "L": 1},
                {"name": "dishwasher", "rate": 0.8, "L": 1}
            ],
            "shiftable_si": [ # Xe điện
                {"name": "ev_charger", "rate": 3.3, "E": 7.0}
            ],
            "beta": 0.5,
            "battery": {
                "capacity_kwh": 10.0,
                "soc_init": 0.5,
                "soc_min": 0.1,
                "soc_max": 0.9,
                "p_charge_max_kw": 3.0,
                "p_discharge_max_kw": 3.0,
                "eta_ch": 0.95,
                "eta_dis": 0.95
            },
            "pv_config": {
                "latitude": 10.762622,
                "longitude": 106.660172,
                "module_parameters": {"pdc0": 3.0}
            },
            "sim_steps": 24
        }
        
        # Initialize environments
        self._init_environments()
        
        # Try to load models
        self._load_models()
    
    def _init_environments(self):
        """Initialize dual environments for PPO and Hybrid"""
        seed = 42
        np.random.seed(seed)
        
        # Pass Price Profile, config and dummy PV (will be generated by env)
        self.env_ppo = SmartHomeEnv(self.price_profile, np.zeros(24), self.config)
        self.env_hybrid = SmartHomeEnv(self.price_profile, np.zeros(24), self.config)
        
        self.obs_ppo, _ = self.env_ppo.reset(seed=seed)
        self.obs_hybrid, _ = self.env_hybrid.reset(seed=seed)
        
        self.info_ppo = {}
        self.info_hybrid = {}
        self.done = False
        
        logger.info("✅ Dual environments initialized")
    
    def _load_models(self):
        """Load trained PPO models from .zip files"""
        if not HAS_SB3:
            logger.warning("⚠️ stable_baselines3 not available. Using heuristic mode.")
            return
        
        try:
            # Construct absolute paths relative to this script
            base_path = os.path.dirname(os.path.abspath(__file__))
            # Updated paths to point to 'models' folder
            ppo_path = os.path.join(base_path, "models", "ppo_smart_home.zip")
            hybrid_path = os.path.join(base_path, "models", "ppo_hybrid_smart_home.zip")
            
            # Check if model files exist
            if os.path.exists(ppo_path):
                self.ppo_model = PPO.load(ppo_path)
                logger.info(f"✅ Loaded PPO model from {ppo_path}")
            else:
                logger.warning(f"⚠️ PPO model not found: {ppo_path}")
            
            if os.path.exists(hybrid_path):
                self.hybrid_model = PPO.load(hybrid_path)
                logger.info(f"✅ Loaded Hybrid model from {hybrid_path}")
            else:
                logger.warning(f"⚠️ Hybrid model not found: {hybrid_path}")
            
            # Enable real model mode if at least one model loaded
            # Check for observation space compatibility
            if self.ppo_model is not None:
                expected_shape = self.ppo_model.observation_space.shape
                actual_shape = self.obs_ppo.shape
                if expected_shape != actual_shape:
                    logger.warning(f"⚠️ OBSERVATION MISMATCH! Model expects {expected_shape}, Env provides {actual_shape}")
                    logger.warning("⚠️ Models were trained with different env version. Using HEURISTIC mode.")
                    self.use_real_models = False
                    return
            
            if self.ppo_model is not None or self.hybrid_model is not None:
                self.use_real_models = True
                logger.info("🚀 REAL MODEL MODE ENABLED")
            else:
                logger.warning("⚠️ No models loaded. Using heuristic fallback.")
                
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            logger.warning("⚠️ Falling back to heuristic mode")
    
    def _get_heuristic_action(self, hour, temp_out, n_home, agent_type='ppo'):
        """Fallback heuristic action when models are not available"""
        action = np.zeros(7, dtype=np.float32)
        
        # Battery logic
        if 6 <= hour <= 16:
            action[ACTION_INDICES['battery']] = 0.5
        else:
            action[ACTION_INDICES['battery']] = -0.5
        
        # AC logic
        if n_home > 0 and temp_out > 28:
            action[ACTION_INDICES['ac_living']] = 0.6
            action[ACTION_INDICES['ac_master']] = 0.4 if hour >= 21 else -0.5
            action[ACTION_INDICES['ac_bed2']] = 0.3 if hour >= 21 else -0.5
        else:
            action[ACTION_INDICES['ac_living']] = -0.8
            action[ACTION_INDICES['ac_master']] = -0.8
            action[ACTION_INDICES['ac_bed2']] = -0.8
        
        # EV charging
        if 22 <= hour or hour < 6:
            action[ACTION_INDICES['ev']] = 0.8
        else:
            action[ACTION_INDICES['ev']] = -0.5
        
        # WM/DW
        if agent_type == 'ppo':
            action[ACTION_INDICES['wm']] = 0.5 if 18 <= hour <= 20 else -0.5
            action[ACTION_INDICES['dw']] = 0.5 if 19 <= hour <= 21 else -0.5
        else:
            # Hybrid is smarter
            action[ACTION_INDICES['wm']] = 0.5 if 14 <= hour <= 16 else -0.5
            action[ACTION_INDICES['dw']] = 0.5 if 15 <= hour <= 17 else -0.5
        
        return action
    
    def _get_env_state(self, env):
        """Extract environment state for Hybrid rules"""
        hour = self.step_count % 24
        return {
            'hour': hour,
            'soc': env.soc,
            'ev_soc': getattr(env, 'ev_soc', 0.5),
            'wm_remaining': getattr(env, 'wm_remaining', 0),
            'wm_deadline': getattr(env, 'wm_deadline', 22),
            'dw_remaining': getattr(env, 'dw_remaining', 0),
            'dw_deadline': getattr(env, 'dw_deadline', 23),
            'n_home': env.load_schedules[min(self.step_count, 23)]['n_home'] if hasattr(env, 'load_schedules') else 0,
            'price_tier': self._get_price_tier(env.cumulative_import_kwh),
            'temp_out': getattr(env, 'temp_out', 30)
        }
    
    def _get_price_tier(self, cumulative_kwh):
        if cumulative_kwh <= 50: return 1
        elif cumulative_kwh <= 100: return 2
        elif cumulative_kwh <= 200: return 3
        elif cumulative_kwh <= 300: return 4
        elif cumulative_kwh <= 400: return 5
        else: return 6
    
    def _generate_final_report(self):
        """Generate FINAL_REPORT packet at end of demo day (hour 23)"""
        scenario = DEMO_SCENARIOS[self.current_scenario]
        
        # Build temperature chart data
        temps_chart = []
        for i in range(min(len(self.stats_ppo.temp_history), len(self.stats_hybrid.temp_history))):
            temps_chart.append({
                "hour": i,
                "outdoor": self.stats_ppo.temp_history[i]["outdoor"],
                "ppo": self.stats_ppo.temp_history[i]["indoor"],
                "hybrid": self.stats_hybrid.temp_history[i]["indoor"]
            })
        
        # Build energy stack chart data (for Hybrid)
        energy_chart = []
        for i, e in enumerate(self.stats_hybrid.energy_stack):
            energy_chart.append({
                "hour": i,
                "pv": e["pv"],
                "grid": e["grid"],
                "battery": e["battery"]
            })
        
        # Merge with regular data packet so UI stays updated, but override type
        base_packet = self.get_data_packet()
        report = {
            **base_packet,
            "type": "FINAL_REPORT",
            "data": {
                "scenario": scenario["name"],
                "metrics": {
                    # Cost metrics
                    "ppo_bill": round(self.stats_ppo.total_bill, 0),
                    "hybrid_bill": round(self.stats_hybrid.total_bill, 0),
                    
                    # Comfort metrics
                    "ppo_comfort": round(self.stats_ppo.total_comfort_loss, 2),
                    "hybrid_comfort": round(self.stats_hybrid.total_comfort_loss, 2),
                    
                    # Grid metrics
                    "ppo_grid": round(self.stats_ppo.total_grid_import, 2),
                    "hybrid_grid": round(self.stats_hybrid.total_grid_import, 2),
                    
                    # Solar metrics
                    "solar_generated": round(self.stats_ppo.total_solar_generated, 2),
                    "ppo_solar_used": round(self.stats_ppo.total_solar_used, 2),
                    "hybrid_solar_used": round(self.stats_hybrid.total_solar_used, 2),
                    "ppo_solar_self_consumption": round(self.stats_ppo.get_solar_self_consumption_rate(), 1),
                    "hybrid_solar_self_consumption": round(self.stats_hybrid.get_solar_self_consumption_rate(), 1),
                    
                    # Battery metrics
                    "ppo_battery_discharged": round(self.stats_ppo.total_battery_discharged, 2),
                    "hybrid_battery_discharged": round(self.stats_hybrid.total_battery_discharged, 2),
                    
                    # Temperature metrics
                    "ppo_avg_temp": round(self.stats_ppo.get_avg_indoor_temp(), 1),
                    "hybrid_avg_temp": round(self.stats_hybrid.get_avg_indoor_temp(), 1),
                    "ppo_min_temp": round(self.stats_ppo.get_min_indoor_temp(), 1),
                    "hybrid_min_temp": round(self.stats_hybrid.get_min_indoor_temp(), 1),
                    "ppo_max_temp": round(self.stats_ppo.get_max_indoor_temp(), 1),
                    "hybrid_max_temp": round(self.stats_hybrid.get_max_indoor_temp(), 1),
                    
                    # Peak load metrics
                    "ppo_peak_load": round(self.stats_ppo.peak_load, 2),
                    "hybrid_peak_load": round(self.stats_hybrid.peak_load, 2),
                },
                "charts": {
                    "temps": temps_chart,
                    "energy_stack": energy_chart
                }
            }
        }
        
        logger.info(f"📊 Generated FINAL_REPORT for scenario: {scenario['name']}")
        
        # NOTE: Don't reset stats here - keep cumulative values so dashboard 
        # continues showing the final totals. Stats only reset when user 
        # starts a NEW demo session via set_demo_mode().
        
        return report
    
    def update(self):
        """Run one simulation step"""
        try:
            if self.paused:
                return None

            if self.done:
                self.reset()
                return self.get_data_packet()
            
            hour = self.step_count % 24
            
            # Inject demo scenario data if demo mode is active
            # This happens BEFORE AI prediction so models see the demo data
            if self.is_demo_mode:
                self._inject_demo_data()
            
            # Get environment states
            ppo_state = self._get_env_state(self.env_ppo)
            hybrid_state = self._get_env_state(self.env_hybrid)
            
            # --- GET PPO ACTION ---
            if self.use_real_models and self.ppo_model is not None:
                # Use REAL trained PPO model
                action_ppo, _ = self.ppo_model.predict(self.obs_ppo, deterministic=True)
                action_ppo = np.array(action_ppo, dtype=np.float32).flatten()
                logger.debug(f"[PPO] Real model action: {action_ppo}")
            else:
                # Fallback to heuristic
                action_ppo = self._get_heuristic_action(
                    hour, ppo_state['temp_out'], ppo_state['n_home'], 'ppo'
                )
            
            # --- GET HYBRID ACTION ---
            # True Hybrid: PPO + Imitation Learning (Pure Model)
            if self.use_real_models and self.hybrid_model is not None:
                if self.is_demo_mode and self.obs_hybrid is not None:
                    # print(f"DEBUG OBS HYBRID: {self.obs_hybrid}")
                    action_hybrid, _ = self.hybrid_model.predict(self.obs_hybrid)
                else:
                    action_hybrid, _ = self.hybrid_model.predict(self.obs_hybrid)
                action_hybrid = np.array(action_hybrid, dtype=np.float32).flatten()
                # DEBUG: Log PV and Action to console
                try:
                    pv_obs = self.obs_hybrid[1] # Index 1 is PV
                    logger.info(f"[Step {hour}] Hybrid PV(obs): {pv_obs:.2f}, Act[0](Bat): {action_hybrid[0]:.2f}, SOC: {self.env_hybrid.soc:.2f}")
                except:
                    pass
                logger.debug(f"[Hybrid] Real model action: {action_hybrid}")
            else:
                # Fallback to heuristic
                action_hybrid = self._get_heuristic_action(
                    hour, hybrid_state['temp_out'], hybrid_state['n_home'], 'hybrid'
                )
            
            # Step environments
            self.obs_ppo, _, done_ppo, _, self.info_ppo = self.env_ppo.step(action_ppo)
            self.obs_hybrid, _, done_hybrid, _, self.info_hybrid = self.env_hybrid.step(action_hybrid)
            
            self.done = done_ppo or done_hybrid
            self.step_count += 1
            
            # --- ALWAYS Accumulate stats for cumulative bill tracking ---
            # Get data from environment info
            ppo_room_temps = self.info_ppo.get('room_temps', {})
            hybrid_room_temps = self.info_hybrid.get('room_temps', {})
            
            ppo_indoor = np.mean(list(ppo_room_temps.values())) if ppo_room_temps else 25.0
            hybrid_indoor = np.mean(list(hybrid_room_temps.values())) if hybrid_room_temps else 25.0
            
            # Step costs from environment
            ppo_step_cost = self.info_ppo.get('step_cost', 0)
            hybrid_step_cost = self.info_hybrid.get('step_cost', 0)
            
            # Grid import - per-step value
            ppo_grid_step = self.info_ppo.get('step_grid_import', 0)
            hybrid_grid_step = self.info_hybrid.get('step_grid_import', 0)
            
            # Battery discharge
            ppo_battery_discharge = max(0, -action_ppo[0]) * 3.0
            hybrid_battery_discharge = max(0, -action_hybrid[0]) * 3.0
            
            # Use demo scenario data if in demo mode, otherwise use environment data
            if self.is_demo_mode:
                scenario = DEMO_SCENARIOS[self.current_scenario]
                outdoor_temp = scenario["temp_profile"][hour]
                pv_available = scenario["pv_profile"][hour]
            else:
                outdoor_temp = getattr(self.env_ppo, 'temp_out', 30.0)
                pv_available = self.env_ppo.pv_profile[hour] if hasattr(self.env_ppo, 'pv_profile') else 0
            
            # Estimate total load
            ppo_total_load = ppo_grid_step + min(pv_available, 4.0)
            hybrid_total_load = hybrid_grid_step + min(pv_available, 4.0)
            
            # Update stats (ALWAYS, not just demo mode)
            self.stats_ppo.update(
                step_bill=ppo_step_cost,
                grid_kw=ppo_grid_step,
                indoor_temp=ppo_indoor,
                outdoor_temp=outdoor_temp,
                pv_used=min(pv_available, 4.0),
                battery_discharged=ppo_battery_discharge,
                pv_generated=pv_available,
                total_load=ppo_total_load
            )
            
            self.stats_hybrid.update(
                step_bill=hybrid_step_cost,
                grid_kw=hybrid_grid_step,
                indoor_temp=hybrid_indoor,
                outdoor_temp=outdoor_temp,
                pv_used=min(pv_available, 4.0),
                battery_discharged=hybrid_battery_discharge,
                pv_generated=pv_available,
                total_load=hybrid_total_load
            )
            
            # Demo mode: generate final report at end of day
            if self.is_demo_mode and hour == 23:
                try:
                    logger.info("📊 Generating FINAL REPORT for hour 23...")
                    report = self._generate_final_report()
                    self.paused = True  # PAUSE to show report
                    return report
                except Exception as e:
                    logger.error(f"❌ Report generation FAILED: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    self.paused = False
            
            return self.get_data_packet()
        except Exception as e:
            logger.error(f"❌ Error in update(): {e}")
            import traceback
            traceback.print_exc()
            # Return safe fallback data
            return self.get_data_packet()
    
    def get_data_packet(self):
        """Build JSON data packet for frontend"""
        hour = self.step_count % 24
        
        # Environment data (shared)
        # When demo mode is ON, use scenario data instead of environment data
        if self.is_demo_mode:
            scenario = DEMO_SCENARIOS[self.current_scenario]
            env_data = {
                "weather": scenario["weather"],
                "temp": float(scenario["temp_profile"][hour]),
                "pv": float(scenario["pv_profile"][hour]),
                "price_tier": int(scenario["price_tier"][hour]),
                "scenario_name": scenario["name"],
                "demo_mode": True
            }
        else:
            # Original logic - use actual environment data
            env_data = {
                "weather": str(self.info_ppo.get('weather', 'sunny')),
                "temp": float(round(float(self.info_ppo.get('temp', 30.0)), 1)),
                "pv": float(round(float(self.info_ppo.get('pv', 0.0)), 2)),
                "price_tier": int(self._get_price_tier(self.env_ppo.cumulative_import_kwh)),
                "demo_mode": False
            }
        
        # Get room temperatures from environments
        ppo_room_temps = self.info_ppo.get('room_temps', {})
        hybrid_room_temps = self.info_hybrid.get('room_temps', {})
        
        # Also try to get from env directly if not in info
        if not ppo_room_temps and hasattr(self.env_ppo, 'room_temps'):
            ppo_room_temps = self.env_ppo.room_temps
        if not hybrid_room_temps and hasattr(self.env_hybrid, 'room_temps'):
            hybrid_room_temps = self.env_hybrid.room_temps
        
        # PPO agent data
        ppo_actions = self._parse_device_states(self.info_ppo)
        ppo_comfort = self._calculate_comfort_score(ppo_room_temps)
        
        # ALWAYS use cumulative stats for bill tracking (works for both demo and normal mode)
        ppo_bill = int(self.stats_ppo.total_bill)
        hybrid_bill = int(self.stats_hybrid.total_bill)
        
        ppo_data = {
            "bill": ppo_bill,
            "soc": float(round(float(self.env_ppo.soc) * 100, 1)),
            "grid": float(round(float(self.env_ppo.cumulative_import_kwh - self.env_ppo.cumulative_export_kwh), 2)),
            "actions": ppo_actions,
            "comfort": ppo_comfort,
            "temp_living": float(round(float(ppo_room_temps.get('living', 25.0)), 1)),
            "temp_master": float(round(float(ppo_room_temps.get('master', 25.0)), 1)),
            "temp_bed2": float(round(float(ppo_room_temps.get('bed2', 25.0)), 1))
        }
        
        # Hybrid agent data
        hybrid_actions = self._parse_device_states(self.info_hybrid)
        hybrid_comfort = self._calculate_comfort_score(hybrid_room_temps)
        hybrid_data = {
            "bill": hybrid_bill,
            "soc": float(round(float(self.env_hybrid.soc) * 100, 1)),
            "grid": float(round(float(self.env_hybrid.cumulative_import_kwh - self.env_hybrid.cumulative_export_kwh), 2)),
            "actions": hybrid_actions,
            "comfort": hybrid_comfort,
            "temp_living": float(round(float(hybrid_room_temps.get('living', 25.0)), 1)),
            "temp_master": float(round(float(hybrid_room_temps.get('master', 25.0)), 1)),
            "temp_bed2": float(round(float(hybrid_room_temps.get('bed2', 25.0)), 1))
        }
        
        return {
            "timestamp": f"{hour:02d}:00",
            "env": env_data,
            "ppo": ppo_data,
            "hybrid": hybrid_data,
            "model_mode": "real" if self.use_real_models else "heuristic"
        }
    
    def _calculate_comfort_score(self, room_temps):
        """Calculate comfort score based on room temperatures (0-100)"""
        if not room_temps:
            return 100.0
        
        target_temp = 25.0  # Ideal temperature
        total_deviation = 0.0
        
        for room, temp in room_temps.items():
            deviation = abs(float(temp) - target_temp)
            total_deviation += deviation
        
        # Average deviation across rooms
        avg_deviation = total_deviation / max(len(room_temps), 1)
        
        # Score: 100 - (deviation * 10), clamped to 0-100
        score = max(0, min(100, 100 - avg_deviation * 10))
        return float(round(score, 1))
    
    def _parse_device_states(self, info):
        """Parse device states from environment info"""
        return {
            "ac_living": int(info.get('ac_living', 0)),
            "ac_master": int(info.get('ac_master', 0)),
            "ac_bed2": int(info.get('ac_bed2', 0)),
            "light_living": int(info.get('light_living', 0)),
            "light_master": int(info.get('light_master', 0)),
            "light_bed2": int(info.get('light_bed2', 0)),
            "light_kitchen": int(info.get('light_kitchen', 0)),
            "light_toilet": int(info.get('light_toilet', 0)),
            "wm": int(info.get('wm', 0)),
            "dw": int(info.get('dw', 0)),
            "ev": float(info.get('ev', 0)),
            "battery": str(info.get('battery', 'idle'))
        }
    
    def set_demo_mode(self, enabled: bool, scenario: str = None):
        """Toggle demo mode and optionally change scenario"""
        self.is_demo_mode = enabled
        self.paused = False
        if scenario and scenario in DEMO_SCENARIOS:
            self.current_scenario = scenario
        if enabled:
            # Reset stats for fresh demo session
            self.stats_ppo.reset()
            self.stats_hybrid.reset()
            self.reset()  # Reset to hour 0 when enabling demo
            print(f"DEBUG: Demo mode ENABLED {scenario}")
            logger.info(f"🎬 Demo mode ENABLED - Scenario: {DEMO_SCENARIOS[self.current_scenario]['name']}")
        else:
            logger.info("🔄 Demo mode DISABLED - Returning to normal simulation")
    
    def _inject_demo_data(self):
        """Inject demo scenario data into environments when demo mode is ON"""
        if not self.is_demo_mode:
            return
        
        scenario = DEMO_SCENARIOS[self.current_scenario]
        hour = self.step_count % 24
        
        # FIX: Replace ENTIRE profile to ensure next_obs (hour+1) sees correct data
        # not just the current hour.
        for env in [self.env_ppo, self.env_hybrid]:
            env.pv_profile = list(scenario["pv_profile"]) # Copy full list
            env.temp_profile = list(scenario["temp_profile"]) # Copy full list
            
            # Also update load_schedules if they exist
            if hasattr(env, 'load_schedules'):
                for h in range(24):
                     env.load_schedules[h]['temp_out'] = scenario["temp_profile"][h]
        
        # CRITICAL FIX: Update the *current* observation to match the injected data
        # The agent makes a decision based on self.obs_*, which was generated in the prev step
        # If we just changed the environment, the old obs is stale.
        # Index 1 is PV, Index 7 is Temp
        
        current_pv = scenario["pv_profile"][hour]
        current_temp = scenario["temp_profile"][hour]
        
        # Update PPO obs
        self.obs_ppo[1] = current_pv
        self.obs_ppo[7] = current_temp
        
        # Update Hybrid obs
        self.obs_hybrid[1] = current_pv
        self.obs_hybrid[7] = current_temp
    
    def reset(self):
        """Reset simulation for new day"""
        seed = np.random.randint(0, 10000)
        np.random.seed(seed)
        
        self.obs_ppo, _ = self.env_ppo.reset(seed=seed)
        self.obs_hybrid, _ = self.env_hybrid.reset(seed=seed)
        
        self.step_count = 0
        self.done = False
        self.paused = False
        self.info_ppo = {}
        self.info_hybrid = {}
        
        # Inject demo data if demo mode is active
        if self.is_demo_mode:
            self._inject_demo_data()
            logger.info(f"🔄 Demo reset - Scenario: {self.current_scenario}")
        else:
            logger.info(f"🔄 Simulation reset (seed={seed})")


# Global simulation instance
sim = RealModelSimulation()

# Track connected clients
connected_clients = []


@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Smart Home HEMS WebSocket Server - Real Models",
        "model_mode": "real" if sim.use_real_models else "heuristic",
        "websocket_endpoint": "/ws",
        "http_endpoint": "/data",
        "connected_clients": len(connected_clients)
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_mode": "real" if sim.use_real_models else "heuristic",
        "ppo_loaded": sim.ppo_model is not None,
        "hybrid_loaded": sim.hybrid_model is not None
    }


@app.get("/data")
async def get_data():
    """HTTP polling endpoint as WebSocket fallback"""
    return JSONResponse(content=sim.update())


@app.get("/reset")
async def reset_sim():
    """Reset simulation"""
    sim.reset()
    return {"status": "reset"}


@app.post("/set_mode")
async def set_mode(payload: dict):
    """Toggle demo mode and optionally change scenario"""
    demo_mode = payload.get("demo_mode", False)
    scenario = payload.get("scenario", None)
    
    sim.set_demo_mode(demo_mode, scenario)
    
    return {
        "status": "ok",
        "demo_mode": sim.is_demo_mode,
        "scenario": sim.current_scenario,
        "scenario_name": DEMO_SCENARIOS[sim.current_scenario]["name"] if sim.is_demo_mode else None
    }


@app.get("/scenarios")
async def get_scenarios():
    """List available demo scenarios"""
    return {
        "scenarios": [
            {"key": key, "name": val["name"], "description": val["description"]}
            for key, val in DEMO_SCENARIOS.items()
        ],
        "current_scenario": sim.current_scenario,
        "demo_mode": sim.is_demo_mode
    }


@app.get("/reload-models")
async def reload_models():
    """Reload models from .zip files"""
    sim._load_models()
    return {
        "status": "reloaded",
        "model_mode": "real" if sim.use_real_models else "heuristic",
        "ppo_loaded": sim.ppo_model is not None,
        "hybrid_loaded": sim.hybrid_model is not None
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming"""
    logger.info(f"🔌 Connection attempt from {websocket.client}")
    
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info(f"✅ Client connected. Total: {len(connected_clients)}")
    logger.info(f"📊 Model mode: {'REAL' if sim.use_real_models else 'HEURISTIC'}")
    
    try:
        # Send initial data IMMEDIATELY on connection
        initial_data = sim.get_data_packet()
        await websocket.send_json(initial_data)
        logger.info("📤 Sent initial data packet")
        
        while True:
            # Just keep connection alive and listen for any messages (e.g. manual control)
            # This prevents the handler from exiting and closing the socket
            data = await websocket.receive_text()
            # Optional: Handle manual control messages here if needed
            # For now just log heartbeat or ignore
            # logger.debug(f"Received from client: {data}")
            
    except WebSocketDisconnect:
        logger.info(f"❌ Client disconnected normally")
    except Exception as e:
        logger.error(f"⚠️ WebSocket error: {e}")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.info(f"Remaining clients: {len(connected_clients)}")


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("🚀 Smart Home HEMS WebSocket Server - REAL MODELS")
    logger.info("=" * 60)
    logger.info(f"📊 Model Mode: {'REAL AI' if sim.use_real_models else 'HEURISTIC FALLBACK'}")
    logger.info(f"   PPO Model: {'✅ Loaded' if sim.ppo_model else '❌ Not found'}")
    logger.info(f"   Hybrid Model: {'✅ Loaded' if sim.hybrid_model else '❌ Not found'}")
    logger.info("=" * 60)
    logger.info("📡 WebSocket: ws://localhost:8001/ws")
    logger.info("🌐 HTTP Poll: http://localhost:8001/data")
    logger.info("🔄 Reload Models: http://localhost:8001/reload-models")
    logger.info("=" * 60)
    
    # Start background simulation loop
    asyncio.create_task(simulation_background_loop())

async def simulation_background_loop():
    """Independent simulation loop that broadcasts updates to all clients"""
    logger.info("🕰️ Background simulation loop started")
    while True:
        try:
            await asyncio.sleep(1.0)
            
            # Skip if paused
            if sim.paused:
                continue
                
            # Update simulation
            data = sim.update()
            
            # Broadcast if we have data
            if data and connected_clients:
                # Broadcast to all connected clients
                disconnected = []
                for client in connected_clients:
                    try:
                        await client.send_json(data)
                    except Exception:
                        disconnected.append(client)
                
                # Cleanup disconnected
                for client in disconnected:
                    if client in connected_clients:
                        connected_clients.remove(client)
                        
        except Exception as e:
            logger.error(f"❌ Error in background loop: {e}")
            await asyncio.sleep(1.0)  # Prevent tight loop on error


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8012)
