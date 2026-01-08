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

from backend.simulation.smart_home_env import SmartHomeEnv
from backend.simulation.device_config import ROOM_OCCUPANCY_HOURS, EV_CONFIG, THERMAL_CONSTANTS
from backend.algorithms.expert_utils import expert_heuristic_action  # <--- Centralized Import

try:
    from backend.algorithms.milp_expert import build_and_solve_milp
    HAS_MILP = True
except Exception:
    HAS_MILP = False

from backend.algorithms.lbwo_solver import LBWOSolver

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
def get_env_inputs():
    price = np.array([0.1] * 6 + [0.15] * 6 + [0.25] * 6 + [0.18] * 6)
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
def collect_expert_data(cfg, n_episodes=10):
    """Collect expert demonstrations for all 7 actions"""
    obs_list, act_list = [], []
    
    price, pv = get_env_inputs()

    for ep in range(n_episodes):
        day = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 365))
        cfg_ep = cfg.copy()
        cfg_ep["sim_start"] = day.strftime("%Y-%m-%d")

        env = SmartHomeEnv(price, pv, cfg_ep)
        obs, _ = env.reset()

        for t in range(env.sim_steps):
            hour = t % 24
            
            # Expert action using centralized utility
            expert_action = expert_heuristic_action(obs, hour, price)
            
            obs_list.append(obs.astype(np.float32))
            act_list.append(expert_action)

            obs, _, done, _, _ = env.step(expert_action)
            if done:
                break

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
    torch.save(save_dict, "bc_policy.pt")
    
    print(f"\n✅ Saved bc_policy.pt")


# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)  # Increased for 256x256 net
    parser.add_argument("--epochs", type=int, default=500)      # Increased 2.5x
    args = parser.parse_args()

    X, Y = collect_expert_data(DEFAULT_CFG, args.episodes)
    train_bc(X, Y, args.epochs)
