import os
import json
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

# Add current directory to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_home_env import SmartHomeEnv

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUTPUT_JSON_PATH = os.path.join(PROJECT_ROOT, "frontend", "public", "data", "agent_comparison.json")

# Model Paths
PPO_MODEL_PATH = os.path.join(BASE_DIR, "ppo_smart_home.zip")
HYBRID_MODEL_PATH = os.path.join(BASE_DIR, "ppo_hybrid_smart_home.zip")

def run_agent_episode(model, env_config, seed=42, agent_name="Agent"):
    """
    Runs a single episode with the given model and config.
    Returns a dictionary of time-series data.
    """
    # Initialize Env with fixed seed
    # For fair comparison, we essentially want the same external factors (Price, PV, Behavior)
    # The 'seed' in reset() handles stochastic generation if implemented, 
    # but we also pass seed to HumanBehavior via config if needed, or rely on env.reset(seed=...)
    
    # Create Env
    # Mock prices/pv if not provided (Env handles it)
    env = SmartHomeEnv(None, None, env_config)
    
    # Reset
    obs, info = env.reset(seed=seed)
    
    metrics = {
        "soc": [],
        "grid_import": [],
        "cost": [],
        "load": [],
        "reward": [],
        # [NEW] Environmental Context
        "weather": [],
        "temp": [],
        "n_home": []
    }
    
    terminated = False
    truncated = False
    
    cumulative_cost = 0.0
    
    while not (terminated or truncated):
        # Action
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Random fallback
            action = env.action_space.sample()
            
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record
        metrics["soc"].append(float(info["soc"]))
        
        # Safe Grid Access (Fallback if attribute missing)
        grid_val = getattr(env, 'last_grid_action', 0.0) 
        # Or deduce from info if possible, but let's stick to safe attr for now
        metrics["grid_import"].append(float(grid_val)) 
        
        metrics["cost"].append(float(info["total_cost"]))
        metrics["load"].append(float(info["load"]))
        metrics["reward"].append(float(reward))
        
        # [NEW] Collect Context
        metrics["weather"].append(str(info.get("weather", "unknown")))
        metrics["temp"].append(float(info.get("temp", 25.0)))
        metrics["n_home"].append(int(info.get("n_home", 0)))
        
    return metrics

def main():
    print("--- Starting Agent Comparison Export ---")
    
    # Common Config
    config = {
        'residents': ['office_worker', 'student', 'father', 'mother'],
        'must_run_base': 0.4,
        'battery': {'capacity_kwh': 10.0, 'soc_init': 0.5, 'soc_min': 0.1, 'soc_max': 0.9},
        'sim_steps': 24, 
        'sim_freq': '1h',
        # Dummy devices for action space sizing
        'shiftable_su': [{'L': 2, 'rate': 1.0}], 
        'shiftable_si': [{'E': 5.0, 'rate': 2.0}],
        'adjustable': [{'P_com': 1.5}] 
    }
    
    # 1. Load Models
    print(f"Loading Pure PPO from: {PPO_MODEL_PATH}")
    try:
        ppo_model = PPO.load(PPO_MODEL_PATH)
        print(">> Pure PPO Loaded.")
    except Exception as e:
        print(f">> Failed to load Pure PPO ({e}). Using Random Agent.")
        ppo_model = None

    print(f"Loading Hybrid Agent from: {HYBRID_MODEL_PATH}")
    try:
        hybrid_model = PPO.load(HYBRID_MODEL_PATH)
        print(">> Hybrid Agent Loaded.")
    except Exception as e:
        print(f">> Failed to load Hybrid Agent ({e}). Using Random Agent (Seed+1).")
        hybrid_model = None

    # 3. Continuous Multi-Day Simulation (30 Days)
    days = 30
    
    # Storage for continuous time series
    ppo_series = {"soc": [], "grid": [], "bill": [], "load": [], "weather": [], "temp": [], "n_home": []}
    hybrid_series = {"soc": [], "grid": [], "bill": [], "load": []} # Hybrid doesn't need env context again if identical
    
    # State Persistence
    ppo_soc = config['battery']['soc_init']
    hybrid_soc = config['battery']['soc_init']
    
    ppo_bill_offset = 0.0
    hybrid_bill_offset = 0.0
    
    print(f"Starting {days}-Day Simulation...")
    
    for day in range(days):
        day_seed = 42 + day
        
        # --- Update Configs with carried-over SOC ---
        config_ppo = config.copy()
        config_ppo['battery'] = config['battery'].copy()
        config_ppo['battery']['soc_init'] = ppo_soc
        
        config_hybrid = config.copy()
        config_hybrid['battery'] = config['battery'].copy()
        config_hybrid['battery']['soc_init'] = hybrid_soc
        
        # --- Run Daily Episodes ---
        # Note: run_agent_episode creates a new Env each time, so it's fresh.
        # We assume the models (PPO) handle the state (SOC) via observation.
        
        ppo_day = run_agent_episode(ppo_model, config_ppo, seed=day_seed, agent_name=f"PPO-Day{day}")
        hybrid_day = run_agent_episode(hybrid_model, config_hybrid, seed=day_seed, agent_name=f"Hybrid-Day{day}")
        
        # --- Accumulate Data ---
        # SOC: Direct append
        ppo_series["soc"].extend(ppo_day["soc"])
        hybrid_series["soc"].extend(hybrid_day["soc"])
        
        # Grid/Load/Env: Direct append
        ppo_series["grid"].extend(ppo_day["grid_import"])
        ppo_series["load"].extend(ppo_day["load"])
        ppo_series["weather"].extend(ppo_day["weather"])
        ppo_series["temp"].extend(ppo_day["temp"])
        ppo_series["n_home"].extend(ppo_day["n_home"])
        
        hybrid_series["grid"].extend(hybrid_day["grid_import"])
        hybrid_series["load"].extend(hybrid_day["load"])
        
        # Bill: Needs Offset (Env starts at 0 each day)
        # We take the day's bill curve and add the offset
        for b in ppo_day["cost"]:
            ppo_series["bill"].append(b + ppo_bill_offset)
        for b in hybrid_day["cost"]:
            hybrid_series["bill"].append(b + hybrid_bill_offset)
            
        # --- Persist State for Next Day ---
        if ppo_day["soc"]: ppo_soc = ppo_day["soc"][-1]
        if hybrid_day["soc"]: hybrid_soc = hybrid_day["soc"][-1]
        
        if ppo_day["cost"]: ppo_bill_offset += ppo_day["cost"][-1]
        if hybrid_day["cost"]: hybrid_bill_offset += hybrid_day["cost"][-1]
        
        print(f">> Day {day+1}/{days} Complete. PPO Bill: {ppo_bill_offset:,.0f} | Hybrid Bill: {hybrid_bill_offset:,.0f}")

    # 4. Structure Data for JSON
    series_data = []
    total_steps = len(ppo_series["soc"])
    
    for t in range(total_steps):
        item = {
            "index": t,
            "day": (t // 24) + 1,
            "hour": t % 24,
            
            "ppo_soc": ppo_series["soc"][t],
            "hybrid_soc": hybrid_series["soc"][t],
            
            "ppo_total_bill": ppo_series["bill"][t],
            "hybrid_total_bill": hybrid_series["bill"][t],
            
            "ppo_grid": ppo_series["grid"][t],
            "hybrid_grid": hybrid_series["grid"][t],
            
            "load": ppo_series["load"][t],
            "weather": ppo_series["weather"][t],
            "temp": ppo_series["temp"][t],
            "n_home": ppo_series["n_home"][t]
        }
        series_data.append(item)
        
    # Summary
    ppo_final = ppo_bill_offset
    hybrid_final = hybrid_bill_offset
    savings = 0.0
    if ppo_final > 0:
        savings = ((ppo_final - hybrid_final) / ppo_final) * 100.0
        
    output = {
        "summary": {
            "days": days,
            "pure_ppo_total_cost": int(ppo_final),
            "hybrid_total_cost": int(hybrid_final),
            "savings_percent": round(savings, 2)
        },
        "series": series_data
    }
    
    # 5. Save
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(output, f, indent=2)
        
    print(f"Export completed. Data saved to: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
