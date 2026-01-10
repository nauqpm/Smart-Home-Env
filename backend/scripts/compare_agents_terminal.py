import os
import sys
import numpy as np
from stable_baselines3 import PPO

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.smart_home_env import SmartHomeEnv
from algorithms.hybrid_wrapper import HybridAgentWrapper
from simulation.device_config import DEVICE_CONFIG

def compare_agents():
    # Paths
    ppo_path = "ppo_smart_home.zip"
    hybrid_path = "ppo_hybrid_smart_home.zip"
    
    if not os.path.exists(ppo_path) or not os.path.exists(hybrid_path):
        print("❌ Model files not found! Please run 'python training/retrain_models.py' first.")
        return

    # Load Models
    print(f"Loading PPO from {ppo_path}...")
    ppo_model = PPO.load(ppo_path)
    
    print(f"Loading Hybrid from {hybrid_path}...")
    hybrid_base = PPO.load(hybrid_path)
    hybrid_model = HybridAgentWrapper(hybrid_base)
    
    # Init Env
    env = SmartHomeEnv()
    obs, _ = env.reset(seed=42)
    
    print("\n" + "="*80)
    print(f"{'Hour':<6} | {'PPO Action (Bat/EV/AC)':<25} | {'Hybrid Action (Bat/EV/AC)':<25} | {'Diff?':<5}")
    print("="*80)
    
    diff_count = 0
    
    # Run 24 steps
    for t in range(24):
        # 1. Get PPO Action
        act_ppo, _ = ppo_model.predict(obs, deterministic=True)
        
        # 2. Get Hybrid Action (needs env state for wrapper)
        hybrid_state = {
             "hour": env.times[env.t].hour,
             "soc": env.soc,
             "ev_soc": env.ev_soc,
             "n_home": env.load_schedules[env.t]["n_home"]
        }
        act_hybrid, _ = hybrid_model.predict(obs, env_state=hybrid_state, deterministic=True)
        
        # 3. Compare
        # Focus on Battery[0], PPO AC[1], EV[4]
        ppo_summary = f"[{act_ppo[0]:.2f}, {act_ppo[4]:.2f}, {act_ppo[1]:.2f}]"
        hyb_summary = f"[{act_hybrid[0]:.2f}, {act_hybrid[4]:.2f}, {act_hybrid[1]:.2f}]"
        
        is_diff = not np.allclose(act_ppo, act_hybrid, atol=0.01)
        diff_mark = "❌" if is_diff else "=="
        if is_diff: diff_count += 1
        
        print(f"{t:<6} | {ppo_summary:<25} | {hyb_summary:<25} | {diff_mark:<5}")
        
        # Determine strict action to step env (using Hybrid mainly or just PPO)
        # We just want to see decisions on same obs, so we step with one to evolve state
        obs, _, done, _, _ = env.step(act_hybrid)
        
    print("="*80)
    if diff_count == 0:
        print("⚠️ WARNING: Agents are IDENTICAL. Retraining might have failed or Models are copies.")
    else:
        print(f"✅ Agents behaved differently in {diff_count}/24 steps.")
        print("Wrapper is working correctly.")

if __name__ == "__main__":
    compare_agents()
