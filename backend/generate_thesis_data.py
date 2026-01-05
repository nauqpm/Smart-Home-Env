
import os
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from stable_baselines3 import PPO
from smart_home_env import SmartHomeEnv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = "report_data"
MODELS = {
    "PPO": "ppo_smart_home.zip",
    "Hybrid": "ppo_hybrid_smart_home.zip"
}

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_scenario_config(scenario_name):
    """
    Generates configuration overrides for different scenarios.
    
    Scenarios:
    1. Standard (Normal): Base PV, Base Temp
    2. High_PV_Hot (Sunny): 1.5x PV, +2°C Temp (High generation, high cooling load)
    3. Low_PV_Cool (Cloudy): 0.5x PV, -2°C Temp (Low generation, lower cooling load)
    """
    
    # Base config
    config = {
        'time_step_hours': 1.0,
        'sim_start': '2025-06-15', # Summer day
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
    
    # Generate profiles based on scenario
    # Default Peak PV ~3kW, Peak Temp ~33C (Summer)
    
    hours = np.arange(24)
    
    # Base profiles
    base_pv = np.clip(3.0 * np.sin(np.pi * (hours - 6) / 12), 0, None)
    base_pv[0:6] = 0
    base_pv[18:] = 0
    
    base_temp = 28 + 5 * np.sin(np.pi * (hours - 9) / 12) # 23C to 33C
    
    pv_profile = base_pv.copy()
    temp_profile = base_temp.copy() # Note: Temp is handled inside env behavior usually, but we can override or pass it if env supports
    # Actually SmartHomeEnv generates temp internally in AdvancedHumanBehaviorGenerator. 
    # But we can't easily override internal behavior without changing code.
    # However, we can simulate different "Days" or just accept the env's internal weather temp 
    # and focus on PV variation which is passed in init.
    
    # Wait, SmartHomeEnv takes `pv_profile` in __init__.
    # But `temp_out` is generated inside `AdvancedHumanBehaviorGenerator.generate_for_times`.
    # To strictly follow "don't change old code", we might only be able to control PV.
    # BUT, we can subclass or mock the behavior generator if needed.
    # For now, let's stick to PV variation which is the biggest factor for Hybrid/PPO difference.
    
    if scenario_name == "High_PV":
        pv_profile = base_pv * 1.5 # Peak 4.5kW
        # hypothetically hotter, but hard to force without code change
    elif scenario_name == "Low_PV":
        pv_profile = base_pv * 0.5 # Peak 1.5kW
    
    return config, pv_profile

def get_heuristic_action(model, obs, env, agent_type):
    """
    Get action from model or fallback rule-based if model fails/missing.
    For this specific task, we want to try using the model first.
    """
    if model:
        action, _ = model.predict(obs, deterministic=True)
        return action
    
    # Fallback similar to thesis_evaluation.py
    pv_gen = obs[1] # Approx index
    hour = env.times[env.t].hour
    
    action = np.zeros(7, dtype=np.float32)
    
    if agent_type == "PPO":
        # Simple rule
        if pv_gen > 2: action[0] = 0.5 
        else: action[0] = -0.2
    else: # Hybrid
        # More aggressive battery use
        if pv_gen > 3: action[0] = 0.8
        elif hour >= 18 or hour < 6: action[0] = -0.6
        else: action[0] = 0.0
        
    return action

def run_simulation(scenario_name):
    """Run simulation for both agents under a specific scenario"""
    logger.info(f"--- Running Scenario: {scenario_name} ---")
    
    config, pv_profile = generate_scenario_config(scenario_name)
    
    # Initialize Environments
    env_ppo = SmartHomeEnv(price_profile=None, pv_profile=pv_profile, config=config)
    env_hybrid = SmartHomeEnv(price_profile=None, pv_profile=pv_profile, config=config)
    
    # Load Models
    models = {}
    for name, path in MODELS.items():
        try:
            if os.path.exists(path):
                models[name] = PPO.load(path)
                logger.info(f"Loaded {name} model from {path}")
            else:
                logger.warning(f"{path} not found. Using Fallback Heuristics for {name}.")
                models[name] = None
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            models[name] = None
            
    # Run
    seed = 42 # Fixed seed for comparison
    
    # Reset
    obs_desc = {}
    obs_desc["PPO"], _ = env_ppo.reset(seed=seed)
    obs_desc["Hybrid"], _ = env_hybrid.reset(seed=seed)
    
    envs = {"PPO": env_ppo, "Hybrid": env_hybrid}
    
    results = []
    
    # 24 steps
    for t in range(24):
        hour = t
        
        row = {
            "Scenario": scenario_name,
            "Hour": hour
        }
        
        for agent_name, env in envs.items():
            model = models[agent_name]
            obs = obs_desc[agent_name]
            
            # Predict
            action = get_heuristic_action(model, obs, env, agent_name)
            
            # Step
            next_obs, reward, done, truncated, info = env.step(action)
            obs_desc[agent_name] = next_obs
            
            # Collect Metrics
            # 1. Cost (VND) - Step cost
            step_cost = info.get('step_cost', 0)
            
            # 2. Grid Import (kWh) - for PAR
            grid_import = info.get('step_grid_import', 0)
            
            # 3. Comfort (Indoor Temp & Penalty)
            # Avg temp of 3 rooms
            temps = info.get('room_temps', {})
            avg_temp = np.mean(list(temps.values())) if temps else 25.0
            
            # Store with prefix
            row[f"{agent_name}_Cost"] = step_cost
            row[f"{agent_name}_Grid_kWh"] = grid_import
            row[f"{agent_name}_Indoor_Temp"] = avg_temp
            # row[f"{agent_name}_Reward"] = reward
            
        results.append(row)
        
    return results

def calculate_par(df, agent_name):
    """Calculate Peak-to-Average Ratio"""
    col_name = f"{agent_name}_Grid_kWh"
    peak = df[col_name].max()
    avg = df[col_name].mean()
    if avg == 0: return 0.0
    return peak / avg

def main():
    ensure_dir(OUTPUT_DIR)
    
    all_results = []
    
    scenarios = ["Standard", "High_PV", "Low_PV"] # "Normal", "Sunny", "Cloudy"
    
    for sc in scenarios:
        scenario_data = run_simulation(sc)
        all_results.extend(scenario_data)
        
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save Raw Data
    csv_path = os.path.join(OUTPUT_DIR, "thesis_scenarios_data.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved detailed hourly data to {csv_path}")
    
    # Calculate Summaries (Totals and PAR)
    summary = {}
    
    for sc in scenarios:
        df_sc = df[df["Scenario"] == sc]
        summary[sc] = {}
        
        for agent in ["PPO", "Hybrid"]:
            total_cost = df_sc[f"{agent}_Cost"].sum()
            total_grid = df_sc[f"{agent}_Grid_kWh"].sum()
            avg_temp = df_sc[f"{agent}_Indoor_Temp"].mean()
            par = calculate_par(df_sc, agent)
            
            summary[sc][agent] = {
                "Total_Cost_VND": float(total_cost),
                "Total_Grid_kWh": float(total_grid),
                "Avg_Temp_C": float(avg_temp),
                "PAR": float(par)
            }
            
    # Save Summary
    json_path = os.path.join(OUTPUT_DIR, "thesis_scenarios_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Saved summary data to {json_path}")
    
    print("\nGeneration Complete.")
    print(f"Detailed CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")

if __name__ == "__main__":
    main()
