# train_il_bc.py
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta

from smart_home_env import SmartHomeEnv

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
def adapt_expert_action(device_action):
    battery = 0.0
    adjustable = [0.0, 0.0, 0.0]
    return [battery] + adjustable + device_action.tolist()


# ------------------------------
class BCPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------
def collect_expert_data(cfg, n_episodes=10):
    obs_list, act_list = [], []

    n_devices = (
        len(cfg["shiftable_su"])
        + len(cfg["shiftable_si"])
        + len(cfg["adjustable"])
    )

    lbwo = LBWOSolver(
        horizon=24,
        num_devices=n_devices,
        population_size=20,
        max_iter=30,
    )

    for ep in range(n_episodes):
        day = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 365))
        cfg_ep = cfg.copy()
        cfg_ep["sim_start"] = day.strftime("%Y-%m-%d")

        price, pv = get_env_inputs()

        # -------- expert ----------
        if HAS_MILP:
            try:
                sched = build_and_solve_milp(price, pv, cfg_ep)
                expert_schedule = np.array(sched["z_su"] + sched["z_si"]).T
            except Exception:
                expert_schedule = lbwo.solve(cfg_ep, price, pv)
        else:
            expert_schedule = lbwo.solve(cfg_ep, price, pv)

        env = SmartHomeEnv(price, pv, cfg_ep)
        obs, _ = env.reset()

        for t in range(env.sim_steps):
            device_action = expert_schedule[t]
            env_action = adapt_expert_action(device_action)

            obs_list.append(obs.astype(np.float32))
            act_list.append(device_action.astype(np.float32))

            obs, _, done, _, _ = env.step(env_action)
            if done:
                break

        print(f"Collected episode {ep + 1}/{n_episodes}")

    return np.stack(obs_list), np.stack(act_list)


# ------------------------------
def train_bc(X, Y, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCPolicy(X.shape[1], Y.shape[1]).to(device)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X, device=device)
    Y_t = torch.tensor(Y, device=device)

    for e in range(epochs):
        opt.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, Y_t)
        loss.backward()
        opt.step()

        if (e + 1) % 10 == 0:
            print(f"Epoch {e+1}/{epochs} | Loss {loss.item():.4f}")

    torch.save(model.state_dict(), "bc_policy.pt")
    print("Saved bc_policy.pt")


# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    X, Y = collect_expert_data(DEFAULT_CFG, args.episodes)
    print("Dataset:", X.shape, Y.shape)

    train_bc(X, Y, args.epochs)
