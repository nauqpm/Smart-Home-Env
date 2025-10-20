import numpy as np
import matplotlib.pyplot as plt
from smart_home_env import SmartHomeEnv

# ===== MÔ PHỎNG THỜI TIẾT =====
T = 24
weather_states = ["sunny", "cloudy", "rainy", "mixed"]

# Base PV profile (trước khi thêm nhiễu thời tiết)
base_pv = np.clip(
    1.5 * np.sin(np.linspace(0, 3.14, T)) + 0.2 * np.random.randn(T),
    0,
    None
)

# Vì mô hình thời tiết đã xử lý trong SmartHomeEnv,
# ta chỉ cần truyền PV cơ bản để mô phỏng.
pv = base_pv

# Giá điện theo giờ tại Việt Nam (mô phỏng)
price = np.array([0.1]*6 + [0.15]*6 + [0.25]*6 + [0.18]*6)  # VN electricity rate (day-night pattern)

# ===== CẤU HÌNH HỆ THỐNG =====
cfg = {
    "critical": [  # baseline 0.33, tăng vào buổi tối
        0.33,0.33,0.33,0.33,0.33,0.33,  # 0–5h
        0.33,0.33,0.33,0.33,0.33,0.33,  # 6–11h
        0.33,0.33,0.33,0.33,0.33,0.53,  # 12–17h
        0.53,0.53,0.53,0.53,0.53,0.33   # 18–23h
    ],
    "adjustable": [
        {"P_min": 0.5, "P_max": 2.0, "P_com": 1.5, "alpha": 0.06},  # AC
        {"P_min": 0.0, "P_max": 2.0, "P_com": 1.5, "alpha": 0.08}   # Water heater
    ],
    "shiftable_su": [
        {"rate": 0.5, "L": 1, "t_s": 7,  "t_f": 22},  # Washing machine
        {"rate": 0.8, "L": 1, "t_s": 19, "t_f": 23}   # Dishwasher
    ],
    "shiftable_si": [
        {"rate": 3.3, "E": 7.0, "t_s": 0, "t_f": 23}  # EV charger
    ],
    "beta": 0.5,
    "battery": {"soc0": 0.5, "soc_min": 0.1, "soc_max": 0.9, "eta_ch": 0.95, "eta_dis": 0.95}
}

print("🚀 Mô phỏng SmartHomeEnv đang khởi tạo...")

# ===== KHỞI TẠO MÔI TRƯỜNG =====
env = SmartHomeEnv(price, pv, cfg)
obs = env.reset()
done = False
rewards, soc_hist, pv_hist, load_hist, grid_hist, weather_hist = [], [], [], [], [], []

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    soc_hist.append(info["SOC"])
    pv_hist.append(info["P_pv"])
    load_hist.append(info["P_load"])
    grid_hist.append(info["P_grid"])
    weather_hist.append(info.get("weather", "N/A"))

# ===== BIỂU ĐỒ =====
fig, axs = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
axs[0].plot(pv_hist, label="PV (kW)")
axs[0].plot(load_hist, label="Load (kW)")
axs[0].set_ylabel("Power (kW)")
axs[0].set_title("PV vs Load")
axs[0].legend()

axs[1].plot(soc_hist, color="orange")
axs[1].set_ylabel("SOC")
axs[1].set_title("Battery SOC")

axs[2].plot(grid_hist, color="red")
axs[2].set_ylabel("Grid Power (kW)")
axs[2].set_title("Power Bought from Grid")

axs[3].bar(range(T), rewards, color="green")
axs[3].set_ylabel("Reward")
axs[3].set_xlabel("Hour")
axs[3].set_title("Reward per Hour")

# Biểu đồ thời tiết
weather_numeric = [weather_states.index(w) if w in weather_states else -1 for w in weather_hist]
axs[4].plot(weather_numeric, marker="o", color="blue")
axs[4].set_yticks(range(len(weather_states)))
axs[4].set_yticklabels(weather_states)
axs[4].set_ylabel("Weather")
axs[4].set_title("Weather Pattern (Simulated)")

plt.tight_layout()
plt.show()
