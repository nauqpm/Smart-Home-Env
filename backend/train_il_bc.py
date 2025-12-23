# train_il_bc.py
"""
Behavior Cloning (IL) for Smart Home PPO Warm-Start
- Outputs 7 actions to match PPO action space
- Uses PPO-compatible key names for warm-start
- Expert: LBWO/MILP for shiftable + heuristic for AC/Battery
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta

from smart_home_env import SmartHomeEnv
from device_config import ROOM_OCCUPANCY_HOURS, EV_CONFIG, THERMAL_CONSTANTS

try:
    from milp_expert import build_and_solve_milp
    HAS_MILP = True
except Exception:
    HAS_MILP = False

from lbwo_solver import LBWOSolver

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Action Space: 7 dimensions
# [0] battery, [1] ac_living, [2] ac_master, [3] ac_bed2, [4] ev, [5] wm, [6] dw
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
    pv = np.zeros(24)
    return price, pv


# ------------------------------
def is_room_occupied(room, hour):
    """Check if room is occupied at given hour"""
    for start, end in ROOM_OCCUPANCY_HOURS.get(room, []):
        if start <= end and start <= hour < end:
            return True
        if start > end and (hour >= start or hour < end):
            return True
    return False


# ------------------------------
def expert_heuristic_action(obs, hour, price):
    """
    Generate expert action for all 7 dimensions.
    Uses heuristics for battery and AC, LBWO result for shiftable.
    
    Returns: array of 7 floats in [-1, 1]
    """
    action = np.zeros(7, dtype=np.float32)
    
    # Extract from obs
    soc = obs[0]
    pv_now = obs[1]
    n_home = obs[6]
    temp_out = obs[7]
    room_temps = obs[8:11]  # living, master, bed2
    ev_soc = obs[12]
    
    # -------- [0] Battery: Charge when cheap, discharge when expensive --------
    if price[hour] < 0.12:  # Off-peak (0-5h)
        action[0] = 0.8 if soc < 0.8 else 0.0  # Charge
    elif price[hour] > 0.20:  # Peak (12-17h)
        action[0] = -0.8 if soc > 0.3 else 0.0  # Discharge
    else:
        action[0] = 0.0
    
    # -------- [1-3] ACs: Turn on when occupied and hot --------
    comfort_temp = THERMAL_CONSTANTS["comfort_temp"]
    for idx, room in enumerate(["living", "master", "bed2"]):
        if is_room_occupied(room, hour) and n_home > 0:
            temp_diff = room_temps[idx] - comfort_temp
            if temp_diff > 2:  # Too hot
                action[1 + idx] = min(1.0, temp_diff / 5)  # Proportional cooling
            elif temp_diff < -2:  # Too cold
                action[1 + idx] = -0.5
            else:
                action[1 + idx] = 0.3  # Maintain
        else:
            action[1 + idx] = -1.0  # Off
    
    # -------- [4] EV: Charge during off-peak or when deadline approaching --------
    deadline = EV_CONFIG["deadline_hour"]
    target_soc = EV_CONFIG["min_target_soc"]
    hours_left = (deadline - hour) % 24
    hours_left = max(1, hours_left)
    
    if (hour >= 22 or hour < 4) and ev_soc < 0.9:
        action[4] = 1.0  # Off-peak charging
    elif ev_soc < target_soc and hours_left < 6:
        action[4] = 1.0  # Urgent charging
    elif ev_soc < 0.5 and price[hour] < 0.15:
        action[4] = 0.7  # Opportunistic charging
    else:
        action[4] = -1.0  # Don't charge
    
    # -------- [5-6] WM/DW: Run during off-peak hours --------
    if 0 <= hour < 6 or 22 <= hour < 24:  # Off-peak
        action[5] = 1.0  # WM on
        action[6] = 1.0  # DW on
    else:
        action[5] = -1.0
        action[6] = -1.0
    
    return np.clip(action, -1, 1)


# ------------------------------
class BCPolicyPPOCompat(nn.Module):
    """
    BC Policy with PPO-compatible layer names.
    Uses EXACT same key names as SB3's MlpPolicy for warm-start.
    
    PPO MlpPolicy structure:
    - mlp_extractor.policy_net.0: Linear(obs_dim, 64)
    - mlp_extractor.policy_net.2: Linear(64, 64)
    - action_net: Linear(64, action_dim)
    """
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM):
        super().__init__()
        
        # Create mlp_extractor with policy_net using Sequential
        # Keys will be: mlp_extractor.policy_net.0.weight, .0.bias, .2.weight, .2.bias
        self.mlp_extractor = nn.ModuleDict({
            'policy_net': nn.Sequential(
                nn.Linear(obs_dim, 64),   # index 0
                nn.Tanh(),                 # index 1 (no weights)
                nn.Linear(64, 64),         # index 2
                nn.Tanh(),                 # index 3 (no weights)
            )
        })
        
        # action_net matches PPO exactly
        self.action_net = nn.Linear(64, action_dim)
        
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
            
            # Expert action using heuristics
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
    loss_fn = nn.MSELoss()  # Use MSE for continuous actions

    X_t = torch.tensor(X, device=device)
    Y_t = torch.tensor(Y, device=device)

    print(f"\n{'='*50}")
    print(f"Training BC Policy (PPO-Compatible)")
    print(f"  Observation dim: {X.shape[1]}")
    print(f"  Action dim: {Y.shape[1]}")
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

    # Save with model_state_dict key for compatibility with load_il_weights_into_ppo
    save_dict = {"model_state_dict": model.state_dict()}
    torch.save(save_dict, "bc_policy.pt")
    
    print(f"\n✅ Saved bc_policy.pt")
    print("Layer shapes:")
    for k, v in model.state_dict().items():
        print(f"  {k}: {list(v.shape)}")


# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)  # More expert demos
    parser.add_argument("--epochs", type=int, default=200)    # More training
    args = parser.parse_args()

    X, Y = collect_expert_data(DEFAULT_CFG, args.episodes)
    print(f"\nDataset: X={X.shape}, Y={Y.shape}")

    train_bc(X, Y, args.epochs)
