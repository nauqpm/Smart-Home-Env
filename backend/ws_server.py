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

# Import environment and config
from smart_home_env import SmartHomeEnv
from device_config import DEVICE_CONFIG, ACTION_INDICES, ROOM_OCCUPANCY_HOURS
from rl_ppo_hybrid_new import HybridAgentWrapper

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
        
        # Environment config
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
        
        # Initialize environments
        self._init_environments()
        
        # Try to load models
        self._load_models()
    
    def _init_environments(self):
        """Initialize dual environments for PPO and Hybrid"""
        seed = 42
        np.random.seed(seed)
        
        self.env_ppo = SmartHomeEnv(None, None, self.config)
        self.env_hybrid = SmartHomeEnv(None, None, self.config)
        
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
            # Check if model files exist
            if os.path.exists(PPO_MODEL_PATH):
                self.ppo_model = PPO.load(PPO_MODEL_PATH)
                logger.info(f"✅ Loaded PPO model from {PPO_MODEL_PATH}")
            else:
                logger.warning(f"⚠️ PPO model not found: {PPO_MODEL_PATH}")
            
            if os.path.exists(HYBRID_MODEL_PATH):
                hybrid_base_model = PPO.load(HYBRID_MODEL_PATH)
                self.hybrid_wrapper = HybridAgentWrapper(hybrid_base_model)  # Only 1 arg!
                logger.info(f"✅ Loaded Hybrid model from {HYBRID_MODEL_PATH}")
            else:
                logger.warning(f"⚠️ Hybrid model not found: {HYBRID_MODEL_PATH}")
            
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
            
            if self.ppo_model is not None or self.hybrid_wrapper is not None:
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
    
    def update(self):
        """Run one simulation step"""
        try:
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
            # NOTE: Rules are now baked into Hybrid model during training
            # No need for HybridAgentWrapper rules - just use direct predict
            if self.use_real_models and self.hybrid_wrapper is not None:
                # Use REAL trained Hybrid model (rules already applied during training)
                action_hybrid, _ = self.hybrid_wrapper.model.predict(
                    self.obs_hybrid, 
                    deterministic=True
                )
                action_hybrid = np.array(action_hybrid, dtype=np.float32).flatten()
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
        ppo_data = {
            "bill": int(self.env_ppo.total_cost),
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
            "bill": int(self.env_hybrid.total_cost),
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
        if scenario and scenario in DEMO_SCENARIOS:
            self.current_scenario = scenario
        if enabled:
            self.reset()  # Reset to hour 0 when enabling demo
            logger.info(f"🎬 Demo mode ENABLED - Scenario: {DEMO_SCENARIOS[self.current_scenario]['name']}")
        else:
            logger.info("🔄 Demo mode DISABLED - Returning to normal simulation")
    
    def _inject_demo_data(self):
        """Inject demo scenario data into environments when demo mode is ON"""
        if not self.is_demo_mode:
            return
        
        scenario = DEMO_SCENARIOS[self.current_scenario]
        hour = self.step_count % 24
        
        demo_pv = scenario["pv_profile"][hour]
        demo_temp = scenario["temp_profile"][hour]
        
        # Override PV profile in environments
        for env in [self.env_ppo, self.env_hybrid]:
            if hasattr(env, 'pv_profile') and len(env.pv_profile) > hour:
                env.pv_profile[hour] = demo_pv
            # Override temperature in load_schedules if available
            if hasattr(env, 'load_schedules') and len(env.load_schedules) > hour:
                env.load_schedules[hour]['temp_out'] = demo_temp
    
    def reset(self):
        """Reset simulation for new day"""
        seed = np.random.randint(0, 10000)
        np.random.seed(seed)
        
        self.obs_ppo, _ = self.env_ppo.reset(seed=seed)
        self.obs_hybrid, _ = self.env_hybrid.reset(seed=seed)
        
        self.step_count = 0
        self.done = False
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
        "hybrid_loaded": sim.hybrid_wrapper is not None
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
        "hybrid_loaded": sim.hybrid_wrapper is not None
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
            # Wait before next update
            await asyncio.sleep(1.0)
            
            # Update and get data
            data = sim.update()
            
            # Send to client
            await websocket.send_json(data)
            
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
    logger.info(f"   Hybrid Model: {'✅ Loaded' if sim.hybrid_wrapper else '❌ Not found'}")
    logger.info("=" * 60)
    logger.info("📡 WebSocket: ws://localhost:8000/ws")
    logger.info("🌐 HTTP Poll: http://localhost:8000/data")
    logger.info("🔄 Reload Models: http://localhost:8000/reload-models")
    logger.info("=" * 60)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
