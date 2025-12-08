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
        print("⚠️ Chưa thấy file model 'bc_policy.pt'.")
        return

    # Lấy kích thước từ checkpoint để init model đúng
    obs_dim = ckpt['obs_dim']
    action_dim = ckpt['action_dim']

    model = BCPolicy(obs_dim, action_dim)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 2. Setup Env
    # Reset Seed để đảm bảo test ngẫu nhiên
    import random
    import time
    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)

    cfg = DEFAULT_CFG.copy()
    cfg['sim_start'] = "2025-06-20"  # Ngày hè nóng nực
    price, pv = get_env_inputs()

    env = SmartHomeEnv(price, pv, cfg)
    obs, _ = env.reset()

    # 3. Chạy mô phỏng & Ghi log chi tiết
    history = {
        "price": [], "pv": [], "temp": [], "n_home": [],
        "action_su": [], "action_si": [], "action_ac": [],
        "reward": [], "soc": []
    }

    done = False
    print(f"\n--- CHẠY THỬ NGHIỆM (Seed: {seed}) ---")

    while not done:
        # Agent suy nghĩ
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(obs_tensor)
            probs = torch.sigmoid(logits).numpy().squeeze()

        action = (probs > 0.5).astype(int).tolist()
        if isinstance(action, int): action = [action]

        # Lấy thông tin môi trường hiện tại để log
        idx = min(env.t, 23)
        t_info = env.load_schedules[idx]  # {'temp_out': ..., 'n_home': ...}

        history["price"].append(env.price_profile[idx])
        history["pv"].append(env.pv_profile[idx])
        history["temp"].append(t_info['temp_out'])
        history["n_home"].append(t_info['n_home'])
        history["soc"].append(env.soc)

        # Phân tách Action (Giả sử: 2 SU, 1 SI, 1 AC = 4 actions)
        # Cần map đúng với config của bạn
        # Config mặc định: 2 SU (Giặt, Rửa), 1 SI (Xe), 0 AC (Cũ) -> Check lại logic đếm
        # Nếu config mới: 2 SU, 1 SI, 1 AC

        su_len = env.N_su
        si_len = env.N_si

        # Lưu action từng loại để vẽ
        if len(action) >= su_len + si_len + 1:
            history["action_su"].append(max(action[:su_len]))  # Max để xem có bật cái nào ko
            history["action_si"].append(action[su_len])
            history["action_ac"].append(action[su_len + si_len])
        else:
            history["action_su"].append(0)
            history["action_si"].append(0)
            history["action_ac"].append(0)

        obs, r, done, _, info = env.step(action)
        history["reward"].append(r)

    # 4. In kết quả check lỗi
    print(f"Tổng Reward: {sum(history['reward']):.2f}")
    print(f"Trạng thái cuối: SU Status={env.su_status}, SI Status={env.si_status}")

    # Check Failures
    if env.su_status[0] < env.su_devs[0]['L']:
        print("❌ LỖI: Máy giặt chưa chạy xong!")
    if env.si_status[0] < env.si_devs[0]['E']:
        print("❌ LỖI: Xe điện chưa sạc đầy!")

    # 5. Vẽ biểu đồ
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Biểu đồ 1: Nhiệt độ & Người & AC
    ax1 = axs[0]
    ax1.set_title("Tiện nghi Nhiệt độ (AC)")
    ax1.plot(history["temp"], color='orange', label='Nhiệt độ ngoài trời')
    ax1.set_ylabel("Độ C")
    ax1.axhline(y=28, color='red', linestyle='--', alpha=0.5, label='Ngưỡng nóng (28C)')

    ax1b = ax1.twinx()
    ax1b.fill_between(range(24), 0, history["n_home"], color='gray', alpha=0.2, label='Người ở nhà')
    ax1b.step(range(24), history["action_ac"], color='blue', where='post', label='Bật AC')
    ax1b.set_ylabel("Trạng thái / Số người")
    ax1b.legend(loc='upper left')
    ax1.legend(loc='upper right')

    # Biểu đồ 2: Giá điện & Máy giặt/Xe điện
    ax2 = axs[1]
    ax2.set_title("Thiết bị vs Giá điện")
    ax2.plot(history["price"], color='red', linestyle='--', label='Giá điện')
    ax2.set_ylabel("Giá ($)")

    ax2b = ax2.twinx()
    ax2b.step(range(24), history["action_su"], color='purple', where='post', label='Máy giặt (SU)')
    ax2b.step(range(24), history["action_si"], color='green', where='post', label='Sạc xe (SI)')
    ax2b.set_ylim(-0.1, 1.5)
    ax2b.legend(loc='upper left')

    # Biểu đồ 3: Reward từng bước
    ax3 = axs[2]
    ax3.set_title("Reward từng giờ (Âm nhiều = Bị phạt)")
    ax3.bar(range(24), history["reward"], color='brown')
    ax3.set_xlabel("Giờ trong ngày")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_agent_behavior()