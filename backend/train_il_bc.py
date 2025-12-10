# train_il_bc.py (sửa lại để tương thích MILP hoặc LBWO)
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import timedelta, datetime

from smart_home_env import SmartHomeEnv

# Cố gắng import MILP expert (nếu có), nếu không sẽ dùng LBWO (nếu có)
try:
    from milp_expert import build_and_solve_milp
    HAS_MILP = True
except Exception:
    HAS_MILP = False

try:
    from lbwo_solver import LBWOSolver
    HAS_LBWO = True
except Exception:
    HAS_LBWO = False

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
    "sim_freq": "1h",
    "price_tiers": [(0, 1500), (7, 3000), (17, 5000)]
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_env_inputs():
    # Bạn có thể làm phong phú price/pv tùy ngày; hiện tĩnh để đơn giản
    price_profile = np.array([0.1] * 6 + [0.15] * 6 + [0.25] * 6 + [0.18] * 6)
    pv_profile = np.zeros(24)
    return price_profile, pv_profile


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=[64, 64]):
        super().__init__()
        layers = []
        inp = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(inp, h))
            layers.append(nn.ReLU())
            inp = h
        layers.append(nn.Linear(inp, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _build_schedule_from_milp(schedule_dict, cfg):
    """
    Convert schedule returned by build_and_solve_milp into array shape (T, num_devices)
    Expect schedule_dict contains keys 'z_su' (n_su x T) and 'z_si' (n_si x T).
    Order required by SmartHomeEnv: [SU..., SI..., AD...]
    """
    su_cfgs = cfg.get("shiftable_su", [])
    si_cfgs = cfg.get("shiftable_si", [])
    ad_cfgs = cfg.get("adjustable", [])

    n_su = len(su_cfgs)
    n_si = len(si_cfgs)
    n_ad = len(ad_cfgs)

    T = len(schedule_dict.get('P_b', [])) if 'P_b' in schedule_dict else 24

    z_su = np.zeros((n_su, T), dtype=int)
    z_si = np.zeros((n_si, T), dtype=int)

    if 'z_su' in schedule_dict:
        arr = np.array(schedule_dict['z_su'])
        # ensure shape (n_su, T)
        if arr.ndim == 2 and arr.shape[0] == n_su and arr.shape[1] == T:
            z_su = arr.astype(int)
        else:
            # try transpose or pad/crop
            try:
                z_su = arr.reshape((n_su, T)).astype(int)
            except Exception:
                z_su = np.zeros((n_su, T), dtype=int)

    if 'z_si' in schedule_dict:
        arr = np.array(schedule_dict['z_si'])
        if arr.ndim == 2 and arr.shape[0] == n_si and arr.shape[1] == T:
            z_si = arr.astype(int)
        else:
            try:
                z_si = arr.reshape((n_si, T)).astype(int)
            except Exception:
                z_si = np.zeros((n_si, T), dtype=int)

    # Build schedule (T x num_devices)
    num_devices = n_su + n_si + n_ad
    schedule = np.zeros((T, num_devices), dtype=int)
    for t in range(T):
        row = []
        # SU
        for i in range(n_su):
            row.append(int(z_su[i, t]))
        # SI
        for j in range(n_si):
            row.append(int(z_si[j, t]))
        # AD: for now mark 0 (MILP may produce P_ad separately)
        for a in range(n_ad):
            row.append(0)
        schedule[t, :] = np.array(row, dtype=int)
    return schedule


def collect_expert_data(base_cfg: dict, n_episodes: int = 50, use_milp_if_available=True):
    """
    Thu thập dữ liệu expert. Thử dùng MILP nếu có, nếu không thì LBWO (nếu có).
    Trả về X (obs) shape (N_samples, obs_dim) và Y (actions) shape (N_samples, action_dim)
    """
    obs_list = []
    act_list = []

    # Số thiết bị từ config
    su_cfgs = base_cfg.get("shiftable_su", [])
    si_cfgs = base_cfg.get("shiftable_si", [])
    ad_cfgs = base_cfg.get("adjustable", [])
    num_devices = len(su_cfgs) + len(si_cfgs) + len(ad_cfgs)

    # init LBWO nếu MILP không có
    lbwo_solver = None
    if not HAS_MILP and HAS_LBWO:
        dim = num_devices * 24
        lbwo_solver = LBWOSolver(dim=dim, population_size=20, max_iter=30, verbose=False)

    print(f"--- Bắt đầu thu thập dữ liệu ({n_episodes} episodes) ---")
    print(f"Số thiết bị cần điều khiển: {num_devices}")

    for ep in range(n_episodes):
        # random ngày trong năm để thay sim_start
        day_offset = np.random.randint(0, 365)
        start_date = datetime.strptime("2025-01-01", "%Y-%m-%d") + timedelta(days=day_offset)
        current_cfg = base_cfg.copy()
        current_cfg['sim_start'] = start_date.strftime("%Y-%m-%d")

        # get price/pv for this episode
        price, pv = get_env_inputs()

        # 1) tạo expert schedule: ưu tiên MILP nếu có
        best_schedule = None
        if HAS_MILP and use_milp_if_available:
            try:
                sched_dict = build_and_solve_milp(price, pv, current_cfg)
                best_schedule = _build_schedule_from_milp(sched_dict, current_cfg)
            except Exception as e:
                print(f"[collect_expert_data] MILP failed: {e} — trying LBWO (if available).")
                best_schedule = None

        if best_schedule is None and lbwo_solver is not None:
            # LBWO expects env_config and optionally prices/pv (our solver supports prices,pv)
            try:
                best_schedule = lbwo_solver.solve(current_cfg, prices=price, pv_profile=pv)
                # ensure shape (T, num_devices)
                best_schedule = np.array(best_schedule, dtype=int)
                if best_schedule.ndim == 1:
                    best_schedule = best_schedule.reshape((24, num_devices))
            except Exception as e:
                print(f"[collect_expert_data] LBWO failed: {e}")
                best_schedule = None

        if best_schedule is None:
            # fallback trivial schedule: all zeros
            print("[collect_expert_data] No expert available for this episode — using zero schedule.")
            best_schedule = np.zeros((24, num_devices), dtype=int)

        # 2) replay the schedule on env to collect (obs, action)
        env = SmartHomeEnv(price, pv, current_cfg)
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out

        T = env.sim_steps

        for t in range(T):
            # get action at t
            action_t = best_schedule[t].tolist() if t < best_schedule.shape[0] else [0] * num_devices

            # store obs/action
            obs_list.append(np.array(obs, dtype=np.float32))
            act_list.append(np.array(action_t, dtype=np.int64))

            out = env.step(action_t)
            # env.step returns 5-tuple (obs, reward, done, False, info) in your env; handle both shapes
            if len(out) == 5:
                obs, r, done, truncated, info = out
                done = done or truncated
            elif len(out) == 4:
                obs, r, done, info = out
            else:
                raise RuntimeError("Unexpected env.step() output length")
            if done:
                break

        if (ep + 1) % 5 == 0:
            print(f"Collected Episode {ep + 1}/{n_episodes}...")

    X = np.stack(obs_list)
    Y = np.stack(act_list)
    return X, Y


def train_bc_weighted(X, Y, obs_dim, action_dim, epochs=50, batch_size=64, lr=1e-3, save_path='bc_policy.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BCPolicy(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset_size = X.shape[0]

    # compute balancing weights per action dimension (pos_weight for BCEWithLogitsLoss)
    num_pos = np.sum(Y, axis=0).astype(float)  # how many positives per action
    # avoid division by zero
    num_pos = np.clip(num_pos, 1.0, dataset_size)
    num_neg = dataset_size - num_pos
    pos_weights = (num_neg / num_pos)
    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)
    print(f"Class Balancing Weights: {pos_weights}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)

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

            epoch_loss += float(loss.item()) * xb.size(0)

        avg_loss = epoch_loss / dataset_size
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.6f}")

    # save checkpoint with dims
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
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out

        done = False
        ep_reward = 0.0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(obs_tensor)
                probs = torch.sigmoid(logits).numpy().squeeze()

            action = (probs > 0.5).astype(int).tolist()
            if isinstance(action, int):
                action = [action]

            out = env.step(action)
            if len(out) == 5:
                obs, r, done, truncated, info = out
                done = done or truncated
            elif len(out) == 4:
                obs, r, done, info = out
            else:
                raise RuntimeError("Unexpected env.step() output length")

            ep_reward += float(r)

        total_rewards.append(ep_reward)
        print(f"Episode {i + 1}: Reward = {ep_reward:.2f}")

    print(f"Average Reward: {np.mean(total_rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--use_milp', action='store_true', help='Prefer MILP expert if available')
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()
    X, Y = collect_expert_data(cfg, n_episodes=args.n_episodes, use_milp_if_available=args.use_milp)
    print(f"Dataset Shape: X={X.shape}, Y={Y.shape}")
    # obs_dim and action_dim
    obs_dim = X.shape[1]
    action_dim = Y.shape[1]
    model = train_bc_weighted(X, Y, obs_dim, action_dim, epochs=args.epochs)
    evaluate_agent('bc_policy.pt', cfg)
