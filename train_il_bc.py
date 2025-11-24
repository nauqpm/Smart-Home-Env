# train_il_bc.py
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import timedelta, datetime

from smart_home_env import SmartHomeEnv
from lbwo_solver import LBWOSolver

# Config Mặc định
DEFAULT_CFG = {
    "critical": [0.33] * 24,
    "adjustable": [],
    "shiftable_su": [
        {"rate": 0.5, "L": 2, "t_s": 7, "t_f": 22},
        {"rate": 0.8, "L": 1, "t_s": 19, "t_f": 23}
    ],
    "shiftable_si": [
        {"rate": 3.3, "E": 7.0, "t_s": 0, "t_f": 23}
    ],
    "time_step_hours": 1.0,
    "battery": {
        "capacity_kwh": 6.0, "soc_init": 0.5, "soc_min": 0.1, "soc_max": 0.9,
        "eta_ch": 0.95, "eta_dis": 0.95, "p_charge_max_kw": 2.0, "p_discharge_max_kw": 2.0
    },
    "pv_config": {
        "latitude": 10.762622, "longitude": 106.660172, "tz": "Asia/Ho_Chi_Minh",
        "surface_tilt": 10.0, "surface_azimuth": 180.0,
        "module_parameters": {"pdc0": 3.0, "gamma_pdc": -0.0045, "inv_eff": 0.96}
    },
    "behavior": {
        "mode": "deterministic",
        "tz": "Asia/Ho_Chi_Minh"
    },
    "sim_start": "2025-05-01",
    "sim_steps": 24,
    "sim_freq": "1h",  # Đã sửa thành 'h' thường
    "price_tiers": [(0, 1500), (7, 3000), (17, 5000)]
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_env_inputs():
    price_profile = np.array([0.1] * 6 + [0.15] * 6 + [0.25] * 6 + [0.18] * 6)
    pv_profile = np.zeros(24)
    return price_profile, pv_profile


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=[128, 64]):
        super().__init__()
        layers = []
        inp = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(inp, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.1))
            inp = h
        layers.append(nn.Linear(inp, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def collect_expert_data(base_cfg: dict, n_episodes: int = 50):
    obs_list = []
    act_list = []

    # Đếm thiết bị chuẩn từ config
    num_devices = len(base_cfg.get("shiftable_su", [])) + len(base_cfg.get("shiftable_si", []))
    dim = num_devices * 24

    # Init Solver
    solver = LBWOSolver(dim=dim, population_size=20, max_iter=40)

    print(f"--- Bắt đầu thu thập dữ liệu ({n_episodes} episodes) ---")
    print(f"Số thiết bị cần điều khiển: {num_devices}")

    for ep in range(n_episodes):
        day_offset = np.random.randint(0, 365)
        start_date = datetime.strptime("2025-01-01", "%Y-%m-%d") + timedelta(days=day_offset)

        current_cfg = base_cfg.copy()
        current_cfg['sim_start'] = start_date.strftime("%Y-%m-%d")

        # 1. Giải bài toán tối ưu (Solver sẽ tự xử lý PV bên trong)
        best_schedule = solver.solve(current_cfg)

        # 2. Khởi tạo Env để replay lại kết quả
        price, pv = get_env_inputs()
        env = SmartHomeEnv(price, pv, current_cfg)
        obs, _ = env.reset()

        # 3. Ghi dữ liệu
        for t in range(24):
            expert_action = best_schedule[t]

            obs_list.append(obs.astype(np.float32))
            act_list.append(np.array(expert_action, dtype=np.int64))

            # Step env (nhận 5 giá trị)
            obs, _, done, _, _ = env.step(expert_action)
            if done: break

        if (ep + 1) % 5 == 0:
            print(f"Collected Episode {ep + 1}/{n_episodes}...")

    X = np.stack(obs_list)
    Y = np.stack(act_list)
    return X, Y


def train_bc_weighted(X, Y, obs_dim, action_dim, epochs=50, batch_size=64, lr=1e-3, save_path='bc_policy.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BCPolicy(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_samples = Y.shape[0]
    num_pos = np.sum(Y, axis=0)
    num_pos = np.clip(num_pos, 1, num_samples)
    num_neg = num_samples - num_pos

    pos_weights_tensor = torch.tensor(num_neg / num_pos, dtype=torch.float32).to(device)
    print(f"Class Balancing Weights: {pos_weights_tensor.cpu().numpy()}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)

    dataset_size = X.shape[0]
    indices = np.arange(dataset_size)

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        model.train()

        for start in range(0, dataset_size, batch_size):
            batch_idx = indices[start:start + batch_size]
            xb = torch.tensor(X[batch_idx], dtype=torch.float32, device=device)
            yb = torch.tensor(Y[batch_idx], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / (dataset_size / batch_size)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")

    torch.save({'model_state_dict': model.state_dict(), 'obs_dim': obs_dim, 'action_dim': action_dim}, save_path)
    print(f"Saved model to {save_path}")
    return model


def evaluate_agent(model_path, base_cfg, n_test_episodes=5):
    ckpt = torch.load(model_path, map_location='cpu')
    model = BCPolicy(ckpt['obs_dim'], ckpt['action_dim'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    total_rewards = []

    print("\n--- Evaluation Results ---")
    for i in range(n_test_episodes):
        day_offset = np.random.randint(0, 365)
        test_cfg = base_cfg.copy()
        test_cfg['sim_start'] = (datetime.strptime("2025-01-01", "%Y-%m-%d") + timedelta(days=day_offset)).strftime(
            "%Y-%m-%d")

        price, pv = get_env_inputs()
        env = SmartHomeEnv(price, pv, test_cfg)
        obs, _ = env.reset()

        done = False
        ep_reward = 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(obs_tensor)
                probs = torch.sigmoid(logits).numpy().squeeze()

            action = (probs > 0.5).astype(int).tolist()
            if isinstance(action, int): action = [action]

            obs, r, done, _, _ = env.step(action)
            ep_reward += r

        total_rewards.append(ep_reward)
        print(f"Episode {i + 1}: Reward = {ep_reward:.2f}")

    print(f"Average Reward: {np.mean(total_rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    # Chạy lại
    cfg = DEFAULT_CFG.copy()
    X, Y = collect_expert_data(cfg, n_episodes=args.n_episodes)
    print(f"Dataset Shape: X={X.shape}, Y={Y.shape}")
    model = train_bc_weighted(X, Y, X.shape[1], Y.shape[1], epochs=args.epochs)
    evaluate_agent('bc_policy.pt', cfg)