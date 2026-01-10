# train_il_bc.py
"""
Behavior Cloning (IL) for Smart Home PPO Warm-Start
- Outputs 7 actions to match PPO action space
- Uses PPO-compatible key names for warm-start
- Expert: Heuristic logic centralized in expert_utils.py
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta

import sys
import os

# Add parent directory (backend) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.smart_home_env import SmartHomeEnv
from simulation.device_config import ROOM_OCCUPANCY_HOURS, EV_CONFIG, THERMAL_CONSTANTS
from algorithms.expert_utils import expert_heuristic_action  # <--- Centralized Import

try:
    from algorithms.milp_expert import build_and_solve_milp
    HAS_MILP = True
except Exception:
    HAS_MILP = False



SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Action Space: 7 dimensions
ACTION_DIM = 7
OBS_DIM = 13

DEFAULT_CFG = {
    "critical": [0.33] * 24,
    "adjustable": [],
    "shiftable_su": [
        {"rate": 0.5, "L": 2, "t_s": 7, "t_f": 22},
        {"rate": 0.8, "L": 1, "t_s": 19, "t_f": 23},
    ],
    "shiftable_si": [
        {"rate": 3.3, "E": 7.0, "t_s": 0, "t_f": 23}
    ],
    "sim_steps": 24,
}


# ------------------------------
# ------------------------------
def get_env_inputs():
    # Use FLAT price to force Self-Consumption strategy (Optimal for Tiered Bill)
    # RTP arbitrage causes losses due to efficiency < 100% when bill is Tiered/Flat.
    price = np.array([0.15] * 24) 
    # Match "Ideal Day" profile from ws_server.py to ensures expert sees solar!
    pv = np.array([0, 0, 0, 0, 0, 0, 0.2, 0.8, 1.5, 2.5, 3.2, 3.8, 4.0, 3.8, 3.0, 2.0, 1.0, 0.3, 0, 0, 0, 0, 0, 0])
    return price, pv


# ------------------------------
class BCPolicyPPOCompat(nn.Module):
    """
    BC Policy with LARGER Network (Matches PPO net_arch=[256, 256])
    """
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM):
        super().__init__()
        
        # Upgraded: 64 -> 256 neurons for better capacity
        self.mlp_extractor = nn.ModuleDict({
            'policy_net': nn.Sequential(
                nn.Linear(obs_dim, 256),   # UPGRADE
                nn.Tanh(),
                nn.Linear(256, 256),       # UPGRADE
                nn.Tanh(),
            )
        })
        
        self.action_net = nn.Linear(256, action_dim)  # UPGRADE
        
    def forward(self, x):
        features = self.mlp_extractor['policy_net'](x)
        return self.action_net(features)


# ------------------------------
from algorithms.lbwo_solver import LBWOOptimizer 

def collect_expert_data(cfg, n_episodes=10):
    """
    Collect expert demonstrations using LBWO for Battery Optimization 
    and Rule-based Heuristics for other devices.
    """
    obs_list, act_list = [], []
    price, pv = get_env_inputs()

    print(f"Starting LBWO Expert Data Collection ({n_episodes} episodes)...")

    # LBWO Config (Match Environment)
    lbwo = LBWOOptimizer(
        n_whales=30, 
        max_iter=50, # Sufficient for convergence 
        n_vars=24,
        lb=-3.0, ub=3.0, # Battery Power Limits
        soc_min=1.0, soc_max=9.0, initial_soc=5.0, ess_capacity=10.0
    )

    for ep in range(n_episodes):
        # Random start day
        day = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 365))
        cfg_ep = cfg.copy()
        cfg_ep["sim_start"] = day.strftime("%Y-%m-%d")

        env = SmartHomeEnv(price, pv, cfg_ep)
        obs, _ = env.reset()
        
        # 1. Get 24h Forecast
        day_data = env.get_current_day_forecast()
        
        # 2. Setup LBWO
        lbwo.initial_soc = obs[0] * 10.0 # Convert normalized SOC (0-1) to kWh (0-10)
        lbwo.set_environment_data(
            day_data['price_buy'],
            day_data['price_sell'],
            day_data['load'],
            day_data['pv'],
            day_data['wind']
        )
        
        # 3. Optimize Battery Schedule (Global Optimization)
        # print(f"  > Optimize Day {ep+1}...", end="\r")
        best_schedule_24h = lbwo.optimize() # Returns array of 24 float values (Battery kW)
        
        # 4. Step Environment
        for t in range(env.sim_steps):
            hour = t % 24
            
            # Action 1: Get Battery Action from LBWO
            # LBWO output is Power (kW) [-3, 3]. Env expects Normalized Action [-1, 1].
            # Map [-3, 3] -> [-1, 1]
            bat_kw = best_schedule_24h[t]
            bat_action_norm = np.clip(bat_kw / 3.0, -1.0, 1.0)
            
            # Action 2: Get Other Devices Actions from Rules (Heuristic)
            # We use expert_heuristic_action but override the battery component
            rule_action = expert_heuristic_action(obs, hour, price)
            
            # Combine
            final_action = rule_action.copy()
            final_action[0] = bat_action_norm # OVERRIDE Battery with Optimal Plan
            
            # Store
            obs_list.append(obs.astype(np.float32))
            act_list.append(final_action)

            obs, _, done, _, _ = env.step(final_action)
            if done:
                break
                
        if (ep + 1) % 5 == 0:
            print(f"Collected episode {ep + 1}/{n_episodes}")

    return np.stack(obs_list), np.stack(act_list)


# ------------------------------
def train_bc(X, Y, epochs=100):
    """Train BC policy with PPO-compatible architecture"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCPolicyPPOCompat(obs_dim=X.shape[1], action_dim=Y.shape[1]).to(device)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X, device=device)
    Y_t = torch.tensor(Y, device=device)

    print(f"\n{'='*50}")
    print(f"Training BC Policy (PPO-Compatible)")
    print(f"  Dataset size: {X.shape[0]}")
    print(f"{'='*50}\n")

    for e in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, Y_t)
        loss.backward()
        opt.step()

        if (e + 1) % 20 == 0:
            print(f"Epoch {e+1}/{epochs} | Loss {loss.item():.4f}")

    save_dict = {"model_state_dict": model.state_dict()}
    os.makedirs("models", exist_ok=True)
    torch.save(save_dict, "models/bc_policy.pt")
    
    print(f"\n✅ Saved bc_policy.pt to models/")

# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50) # Fewer episodes needed due to high quality? Keep 50 for speed testing.
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    X, Y = collect_expert_data(DEFAULT_CFG, args.episodes)
    train_bc(X, Y, args.epochs)
