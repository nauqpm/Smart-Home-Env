
import numpy as np
from stable_baselines3 import PPO

def debug():
    print("Loading model...")
    try:
        model = PPO.load("ppo_hybrid_smart_home.zip")
        print("Model loaded.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Mock Obs with PV=3.2
    # SmartHomeEnv obs: [SOC, PV, ...]
    obs = np.zeros(13)
    obs[0] = 0.5 # SOC
    obs[1] = 3.2 # PV
    obs[7] = 25.0 # Temp
    
    print(f"Predicting for PV={obs[1]}...")
    action, _ = model.predict(obs, deterministic=True)
    
    print(f"Action: {action}")
    print(f"Battery Action (Idx 0): {action[0]}")
    
    if action[0] > 0.5:
        print("PASS: Model charges.")
    else:
        print("FAIL: Model discharges/idles.")

if __name__ == "__main__":
    debug()
