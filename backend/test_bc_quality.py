import torch
import numpy as np
import sys
import os
from stable_baselines3 import PPO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.smart_home_env import SmartHomeEnv
from training.train_il_bc import BCPolicyPPOCompat, get_env_inputs

def evaluate_bc_policy(model_path="models/bc_policy.pt", n_episodes=50):
    print(f"Evaluating BC Policy from {model_path}...")
    
    if not os.path.exists(model_path):
        print("Model file not found!")
        return

    # Load BC Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Determine dimensions (assuming default 13 obs, 7 act)
    bc_model = BCPolicyPPOCompat(13, 7).to(device)
    try:
        ckpt = torch.load(model_path, map_location=device)
        bc_model.load_state_dict(ckpt["model_state_dict"])
        bc_model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Setup Env
    price, pv = get_env_inputs()
    costs = []
    
    for ep in range(n_episodes):
        config = {"sim_steps": 24}
        env = SmartHomeEnv(price, pv, config)
        obs, _ = env.reset()
        total_cost = 0
        
        for t in range(24):
            # Predict
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = bc_model(obs_t).cpu().numpy().flatten()
            
            # Clip action to valid range [-1, 1] as PPO would
            action = np.clip(action, -1, 1)
            
            obs, reward, done, _, info = env.step(action)
            
            # SmartHomeEnv reward is usually -cost. 
            # We can check specific metrics if available in info, but reward is a good proxy.
            # Assuming reward = -cost/scale. 
            # Let's try to track actual electricity cost if possible.
            # Usually info contains details?
            
            # Simple approximation: Accumulate negative reward as proxy for "goodness"
            # Higher reward = Lower cost.
            total_cost += -reward 

        costs.append(total_cost)

    avg_cost = np.mean(costs)
    print(f"Average Negative Reward (Proxy for Cost): {avg_cost:.4f} (over {n_episodes} episodes)")
    return avg_cost

def evaluate_random_policy(n_episodes=50):
    print("Evaluating Random Policy...")
    price, pv = get_env_inputs()
    costs = []
    
    for ep in range(n_episodes):
        config = {"sim_steps": 24}
        env = SmartHomeEnv(price, pv, config)
        obs, _ = env.reset()
        total_cost = 0
        
        for t in range(24):
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            total_cost += -reward

        costs.append(total_cost)

    avg_cost = np.mean(costs)
    print(f"Random Policy Average Negative Reward: {avg_cost:.4f}")
    return avg_cost

if __name__ == "__main__":
    bc_score = evaluate_bc_policy()
    rand_score = evaluate_random_policy()
    
    print("\n--- Benchmark Results ---")
    print(f"Random: {rand_score:.4f}")
    print(f"BC (LBWO): {bc_score:.4f}")
    
    if bc_score < rand_score:
        print("✅ BC is BETTER than Random (Lower 'Cost').")
    else:
        print("❌ BC is WORSE than Random.")
