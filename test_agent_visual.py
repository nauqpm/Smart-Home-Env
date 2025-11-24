# test_agent_visual.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from smart_home_env import SmartHomeEnv
from train_il_bc import BCPolicy, get_env_inputs, DEFAULT_CFG


def visualize_agent_behavior():
    # 1. Load Model
    model_path = 'bc_policy.pt'
    try:
        ckpt = torch.load(model_path, map_location='cpu')
    except FileNotFoundError:
        print("Chưa thấy file model. Hãy chạy train trước!")
        return

    model = BCPolicy(ckpt['obs_dim'], ckpt['action_dim'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 2. Setup Env (Chọn 1 ngày cố định để test)
    cfg = DEFAULT_CFG.copy()
    cfg['sim_start'] = "2025-06-15"  # Một ngày mùa hè
    price, pv = get_env_inputs()

    # Quan trọng: Reset để Env tự tính toán PV
    env = SmartHomeEnv(price, pv, cfg)
    obs, _ = env.reset()

    # 3. Chạy mô phỏng
    prices = []
    pvs = []
    actions = []
    socs = []

    done = False
    print("\n--- Bắt đầu chạy thử Agent ---")
    print("Time | Price | PV   | Action (Dev 1, 2, 3...)")

    while not done:
        # Agent suy nghĩ
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(obs_tensor)
            probs = torch.sigmoid(logits).numpy().squeeze()

        # Quyết định (Ngưỡng 0.5)
        action = (probs > 0.5).astype(int).tolist()
        if isinstance(action, int): action = [action]

        # Ghi log
        idx = min(env.t, 23)
        p_now = env.price_profile[idx] if env.price_profile is not None else 0
        pv_now = env.pv_profile[idx] if env.pv_profile is not None else 0

        prices.append(p_now)
        pvs.append(pv_now)
        actions.append(action)
        socs.append(env.soc)

        print(f"{idx:02d}:00| {p_now:.3f} | {pv_now:.2f} | {action}")

        obs, r, done, _, _ = env.step(action)

    # 4. Vẽ biểu đồ
    actions = np.array(actions)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Vẽ Giá điện (Nền)
    ax1.set_xlabel('Giờ trong ngày')
    ax1.set_ylabel('Giá điện ($)', color='red')
    ax1.plot(prices, color='red', linestyle='--', label='Price')
    ax1.tick_params(axis='y', labelcolor='red')

    # Vẽ Hành động (Cột)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Trạng thái Bật/Tắt', color='blue')

    # Vẽ từng thiết bị lệch nhau một chút để dễ nhìn
    num_devs = actions.shape[1]
    for i in range(num_devs):
        offset = i * 0.05
        ax2.step(np.arange(24), actions[:, i] + offset, label=f'Dev {i + 1}', where='post', alpha=0.7)

    ax2.set_ylim(-0.1, num_devs + 0.5)
    ax2.legend(loc='upper left')

    plt.title("Hành vi của Agent: Bật thiết bị lúc nào?")
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    visualize_agent_behavior()