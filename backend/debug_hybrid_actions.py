import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from smart_home_env import SmartHomeEnv

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HYBRID_MODEL_PATH = os.path.join(BASE_DIR, "ppo_hybrid_smart_home.zip")

def main():
    print(f"Loading Hybrid Model from: {HYBRID_MODEL_PATH}")
    try:
        model = PPO.load(HYBRID_MODEL_PATH)
        print(">> Model Loaded Successfully.")
    except Exception as e:
        print(f"!! Failed to load model: {e}")
        return

    # Create dummy config
    config = {
        'residents': ['office_worker', 'student'],
        'sim_steps': 24, 
        'sim_freq': '1h'
    }

    env = SmartHomeEnv(None, None, config)
    obs, info = env.reset(seed=42)
    
    print("\n--- Starting Debug Episode ---")
    print(f"{'Step':<5} | {'SOC':<6} | {'Price':<6} | {'Action (Index 0 - Battery)':<25} | {'Reward':<8}")
    
    for _ in range(24):
        action, _states = model.predict(obs, deterministic=True)
        
        # Capture pre-step state
        soc_before = obs[0] # Index 0 is SOC
        living_temp = obs[8]
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Action[0]: Batt, [1-3]: AC
        ac_actions = action[1:4]
        
        print(f"{env.t:<4} | {soc_before:.2f} | {living_temp:.1f}C | AC: {[f'{x:.2f}' for x in ac_actions]} | R: {reward:.2f}")

        if terminated or truncated:
            break

if __name__ == "__main__":
    main()
