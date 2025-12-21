import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import os
from stable_baselines3 import PPO
from backend.smart_home_env import SmartHomeEnv

app = FastAPI()

# Input Models
class ResidentPars(BaseModel):
    name: str = "user"
    profile: str = "office_worker" # office_worker, remote_worker, etc

class SimulationRequest(BaseModel):
    num_people: int = 2
    weather_condition: str = "sunny" # sunny, mild, cloudy, rainy, stormy
    must_run_base: float = 0.2
    seed: int = 42

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Model Store
models = {}

# --- DEFAULT CONFIGURATION (Aligned with run_episode_plot.py) ---
# Keeping 24 hours simulation as standard
T = 24
PRICE_PROFILE = np.array([0.1] * 6 + [0.15] * 6 + [0.25] * 6 + [0.18] * 6) * 10000 # Scaling for VND roughly (mock) or just unitless cost

BASE_CONFIG = {
    "critical": [0.33] * 24,
    "adjustable": [
        {"P_min": 0.5, "P_max": 2.0, "P_com": 1.5, "alpha": 0.06}, # AC Living
        {"P_min": 0.0, "P_max": 2.0, "P_com": 1.5, "alpha": 0.08}  # AC Master
    ],
    "shiftable_su": [
        {"name": "washing_machine", "rate": 0.5, "L": 1}, # Using simpler config logic if env supports it, else mapping manually below
        {"name": "dishwasher", "rate": 0.8, "L": 1}
    ],
    "shiftable_si": [
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
        "tz": "Asia/Ho_Chi_Minh",
        "surface_tilt": 10.0,
        "surface_azimuth": 180.0,
        "module_parameters": {"pdc0": 3.0}
    },
    "sim_steps": T
}

# Explicit Device Mapping for Env Init
# The env expects lists of dicts. We need to match what's in run_episode_plot loop
# shiftable_su: washer (idx 0), dish (idx 1)
ENV_SU_DEVS = [
    {"rate": 0.5, "L": 1, "t_s": 0, "t_f": 23}, # Washer
    {"rate": 0.8, "L": 1, "t_s": 0, "t_f": 23}  # Dishwasher
]
ENV_SI_DEVS = [
    {"rate": 3.3, "E": 7.0, "t_s": 0, "t_f": 23} # EV
]


@app.on_event("startup")
def load_models():
    # Load PPO Models
    base_path = os.path.dirname(os.path.abspath(__file__))
    ppo_path = os.path.join(base_path, "ppo_smart_home.zip")
    hybrid_path = os.path.join(base_path, "ppo_hybrid_smart_home.zip")
    
    if os.path.exists(ppo_path):
        models['ppo'] = PPO.load(ppo_path)
        print("✅ Loaded PPO Model")
    else:
        print(f"❌ PPO Model not found at {ppo_path}")

    if os.path.exists(hybrid_path):
        models['hybrid'] = PPO.load(hybrid_path)
        print("✅ Loaded Hybrid Model")
    else:
        print(f"❌ Hybrid Model not found at {hybrid_path}")


def interpret_action(action, expected_len):
    # Helper to parse MultiBinary or Box action
    # env action space is: SU (2) + SI (1) + AD (2) = 5 dims
    if np.isscalar(action): action = [action]
    action = np.array(action, dtype=int).flatten()
    if action.size != expected_len:
         if action.size > expected_len: action = action[:expected_len]
         else: action = np.pad(action, (0, expected_len - action.size))
    return action

@app.post("/simulate")
def run_simulation(req: SimulationRequest):
    if 'ppo' not in models or 'hybrid' not in models:
        raise HTTPException(status_code=500, detail="Models not loaded")

    # 1. Setup Common Environment Context (weather, behavior)
    # We create a config with the requested Human Behavior params
    sim_config = BASE_CONFIG.copy()
    sim_config['behavior'] = {
        "residents": [{'name': f'user{i}'} for i in range(req.num_people)],
        "must_run_base": req.must_run_base,
        "weather": req.weather_condition 
    }
    # Update device lists
    sim_config['shiftable_su'] = ENV_SU_DEVS
    sim_config['shiftable_si'] = ENV_SI_DEVS
    
    # We will use the SAME seed for both envs to ensure weather/behavior is identical
    seed = req.seed
    
    # 2. Run PPO Simulation
    data_ppo = run_single_agent(models['ppo'], sim_config, seed, "PPO")
    
    # 3. Run Hybrid Simulation
    data_hybrid = run_single_agent(models['hybrid'], sim_config, seed, "Hybrid")
    
    return {
        "ppo": data_ppo,
        "hybrid": data_hybrid
    }

def run_single_agent(model, config, seed, label):
    # Re-instantiate env to guarantee clean state
    # Note: We pass Dummy PV profile as placeholder, Env calculates it if pv_profile_input is zeros but config exists
    env = SmartHomeEnv(PRICE_PROFILE, np.zeros(T), config)
    
    # Strict Seed Reset
    # We set np.random.seed for the environment's internal generators if they use global np.random
    np.random.seed(seed)
    
    # Override Env internal state if needed
    obs, info = env.reset(seed=seed)
    
    history = []
    done = False
    
    while not done:
        # Predict
        action, _ = model.predict(obs, deterministic=True)
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Format Data for Frontend ---
        # Action Decoding
        # 5 output dims: [Washer, Dishwasher, EV, AC_Living, AC_Master]
        raw_act = interpret_action(action, 5)
        
        # Map raw output to named devices
        # Note: AC actions in this env (Adjustable) are complex (0/1 or continuous). 
        # SmartHomeEnv line 353: if act == 1: p += P_com. 
        # So 1 = ON, 0 = OFF/Eco for AC.
        
        devices_status = {
            "washer": bool(raw_act[0]),
            "dishwasher": bool(raw_act[1]),
            "charger": bool(raw_act[2]), # EV
            "ac_living": bool(raw_act[3]),
            "ac_master": bool(raw_act[4]),
            # Base/Passive Devices from Info or assumption
            "tv": info.get('n_home', 0) > 0, # Simple heuristic from env logic
            "fridge": True,
            "lights": info.get('n_home', 0) > 0, # Simple heuristic
        }
        
        step_data = {
            "hour": int(len(history)),
            "soc": float(info.get('soc', 0.0)),
            "grid": float(info.get('cumulative_import', 0.0)) - float(info.get('cumulative_export', 0.0)), # Net Grid
             # Frontend expects simple Grid power (Watts) or similar. 
             # Let's provide 'grid_power' for the chart (Load - PV + Battery). 
             # Env 'grid' return is actually Step Grid Energy? 
             # SmartHomeEnv: grid_kwh = grid * time_step. 
             # We want Power (kW) or Watts. 
             # info['grid'] is likely not populated in reset, only implicitly in env.
             # Let's verify env code...
             # Env Code Line 417: info = { ... 'load': total_load ... }
             # We can calculate grid_power = load - pv + batt_power. 
             # Or just use Load and PV for now.
             
            "load": float(info.get('load', 0.0)) * 1000, # kW -> Watts
            "pv": float(info.get('pv', 0.0)) * 1000, # kW -> Watts
            "temp": float(info.get('temp', 25.0)),
            "reward": float(reward),
            "total_bill": float(info.get('total_cost', 0.0)),
            "n_home": int(info.get('n_home', 0)),
            "weather": info.get('weather', 'sunny'),
            "devices": devices_status
        }
        history.append(step_data)

    return history

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)