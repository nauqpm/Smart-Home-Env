"""
Agent Diagnosis Script
Compares 4 agents to identify root cause of Hybrid underperformance:
1. PPO (Baseline)
2. Hybrid (Trained)
3. BC Only (Untrained PPO with BC weights)
4. Heuristic (Expert)
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from smart_home_env import SmartHomeEnv
from expert_utils import expert_heuristic_action
from device_config import THERMAL_CONSTANTS

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PPO_PATH = os.path.join(BASE_DIR, "ppo_smart_home.zip")
HYBRID_PATH = os.path.join(BASE_DIR, "ppo_hybrid_smart_home.zip")
BC_PATH = os.path.join(BASE_DIR, "bc_policy.pt")


# ==========================================
# 1. Define Heuristic Agent Wrapper
# ==========================================
class HeuristicAgent:
    def __init__(self):
        self.env = None
        self.price_profile = np.array([0.1] * 6 + [0.15] * 6 + [0.25] * 6 + [0.18] * 6)

    def set_env(self, env):
        self.env = env

    def predict(self, obs, deterministic=True):
        hour = self.env.t % 24 if self.env else 0
        action = expert_heuristic_action(obs, hour, self.price_profile)
        return action, {}


# ==========================================
# 2. Define BC Agent Loader
# ==========================================
def load_bc_agent(env):
    """Load BC weights into an untrained PPO model (with 256x256 net)"""
    # MUST match the network size used in BC training
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    model = PPO("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs)
    try:
        import torch
        ckpt = torch.load(BC_PATH, map_location="cpu")
        bc_state = ckpt.get("model_state_dict", ckpt)
        ppo_state = model.policy.state_dict()
        
        pretrained_dict = {k: v for k, v in bc_state.items() if k in ppo_state and v.shape == ppo_state[k].shape}
        ppo_state.update(pretrained_dict)
        model.policy.load_state_dict(ppo_state)
        print(f"Loaded BC Weights [256x256]: {len(pretrained_dict)} layers matched.")
    except Exception as e:
        print(f"Error loading BC: {e}")
        return None
    return model


# ==========================================
# 3. Evaluation Loop
# ==========================================
def evaluate_agent(name, agent, n_episodes=30):
    print(f"--- Evaluating {name} ---")
    
    config = {
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
        'behavior': {'must_run_base': 0.4}
    }
    
    costs = []
    avg_temps = []
    comfort_violations = 0
    
    for i in range(n_episodes):
        seed = 42 + i
        env = SmartHomeEnv(None, None, config)
        obs, _ = env.reset(seed=seed)
        
        # Set env reference for Heuristic agent
        if isinstance(agent, HeuristicAgent):
            agent.set_env(env)

        done = False
        temp_sum = 0
        steps = 0
        
        while not done:
            if agent:
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            obs, reward, done, _, info = env.step(action)
            
            temps = info.get('room_temps', {})
            t_avg = sum(temps.values()) / len(temps) if temps else 25.0
            temp_sum += t_avg
            
            if t_avg > 28.0 or t_avg < 22.0:
                comfort_violations += 1
            
            steps += 1
            
            if done:
                costs.append(info['total_cost'])
        
        avg_temps.append(temp_sum / steps)
    
    mean_cost = np.mean(costs)
    mean_temp = np.mean(avg_temps)
    print(f"  > Mean Cost: {mean_cost:,.0f} VND")
    print(f"  > Mean Temp: {mean_temp:.2f} °C")
    print(f"  > Comfort Violations (Steps): {comfort_violations}")
    return mean_cost, mean_temp, comfort_violations


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("AGENT DIAGNOSIS: Garbage In, Garbage Out Analysis")
    print("="*60)
    
    # Create dummy env for initialization
    dummy_config = {'sim_steps': 24}
    dummy_env = SmartHomeEnv(None, None, dummy_config)
    dummy_env.reset()
    
    # 1. Heuristic Agent
    heuristic_agent = HeuristicAgent()
    
    # 2. BC Agent (Untrained PPO initialized with BC)
    bc_agent = load_bc_agent(dummy_env)
    
    # 3. Load Trained Agents
    try:
        ppo_agent = PPO.load(PPO_PATH)
        print("PPO Model loaded.")
    except Exception as e:
        ppo_agent = None
        print(f"PPO Model not found: {e}")
        
    try:
        hybrid_agent = PPO.load(HYBRID_PATH)
        print("Hybrid Model loaded.")
    except Exception as e:
        hybrid_agent = None
        print(f"Hybrid Model not found: {e}")

    # 4. Evaluate All
    results = {}
    
    results["Heuristic (Expert)"] = evaluate_agent("Heuristic (Expert)", heuristic_agent)
    
    if bc_agent:
        results["BC Only (Start)"] = evaluate_agent("BC Policy (Untrained)", bc_agent)
    
    if ppo_agent:
        results["Pure PPO"] = evaluate_agent("Pure PPO", ppo_agent)
    
    if hybrid_agent:
        results["Hybrid (Trained)"] = evaluate_agent("Hybrid (Trained)", hybrid_agent)
    
    # 5. Summary Report
    print("\n" + "="*70)
    print("DIAGNOSIS REPORT")
    print("="*70)
    print(f"{'Agent':<22} | {'Cost (VND)':<14} | {'Temp (°C)':<10} | {'Violations'}")
    print("-" * 70)
    
    for name, (cost, temp, viol) in results.items():
        print(f"{name:<22} | {cost:>12,.0f} | {temp:>10.2f} | {viol}")
        
    print("-" * 70)
    print("\nINTERPRETATION GUIDE:")
    print("1. If Heuristic Cost > PPO Cost: Expert logic is expensive. Fix expert_utils.py.")
    print("2. If Heuristic Cost LOW but BC Cost HIGH: BC training failed. Need more epochs/data.")
    print("3. If BC Cost LOW but Hybrid Cost HIGH: PPO destroyed BC knowledge. Need regularization.")
    print("="*70)
