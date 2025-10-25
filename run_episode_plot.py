import numpy as np
import json
import http.server
import socketserver
import threading
import matplotlib.pyplot as plt
from smart_home_env import SmartHomeEnv
from human_behavior import HumanBehavior  # 🧩 Mô phỏng hành vi con người
import webbrowser  # 🚀 Thêm thư viện này
import os  # 🚀 Thêm thư viện này

# ===== MÔ PHỎNG THỜI TIẾT =====
T = 24
weather_states = ["sunny", "mild", "cloudy", "rainy", "stormy"]
weather_condition = np.random.choice(weather_states)

base_pv = np.clip(
    1.5 * np.sin(np.linspace(0, 3.14, T)) + 0.2 * np.random.randn(T),
    0, None
)

weather_factors = {
    "sunny": np.clip(np.random.normal(1.0, 0.05, T), 0.9, 1.1),
    "mild": np.clip(np.random.normal(0.85, 0.08, T), 0.7, 1.0),
    "cloudy": np.clip(np.random.normal(0.6, 0.1, T), 0.4, 0.8),
    "rainy": np.clip(np.random.normal(0.4, 0.1, T), 0.2, 0.6),
    "stormy": np.clip(np.random.normal(0.2, 0.1, T), 0.05, 0.4),
}
pv = np.clip(base_pv * weather_factors[weather_condition], 0, None)
price = np.array([0.1] * 6 + [0.15] * 6 + [0.25] * 6 + [0.18] * 6)

# ===== CẤU HÌNH HỆ THỐNG =====
cfg = {
    "critical": [
        0.33, 0.33, 0.33, 0.33, 0.33, 0.33,
        0.33, 0.33, 0.33, 0.33, 0.33, 0.33,
        0.33, 0.33, 0.33, 0.33, 0.33, 0.53,
        0.53, 0.53, 0.53, 0.53, 0.53, 0.33
    ],
    "adjustable": [
        {"P_min": 0.5, "P_max": 2.0, "P_com": 1.5, "alpha": 0.06},  # AC
        {"P_min": 0.0, "P_max": 2.0, "P_com": 1.5, "alpha": 0.08}  # Water heater
    ],
    "shiftable_su": [
        {"rate": 0.5, "L": 1, "t_s": 7, "t_f": 22},  # Washing machine
        {"rate": 0.8, "L": 1, "t_s": 19, "t_f": 23}  # Dishwasher
    ],
    "shiftable_si": [
        {"rate": 3.3, "E": 7.0, "t_s": 0, "t_f": 23}  # EV charger
    ],
    "beta": 0.5,
    "battery": {"soc0": 0.5, "soc_min": 0.1, "soc_max": 0.9, "eta_ch": 0.95, "eta_dis": 0.95}
}

print(f"🌦️ Thời tiết khởi đầu: {weather_condition}")

# ===== KHỞI TẠO MÔI TRƯỜNG =====
env = SmartHomeEnv(price, pv, cfg)
human = HumanBehavior(num_people=4, T=T)
occupancy_profile, activity_profile = human.generate_daily_behavior()

obs = env.reset()
done = False
rewards, soc_hist, pv_hist, load_hist, grid_hist, weather_hist, occ_hist = [], [], [], [], [], [], []

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    # Ghi dữ liệu
    rewards.append(reward)
    soc_hist.append(info["SOC"])
    # SỬA LỖI JSON: Chuyển numpy.float64 sang float
    pv_hist.append(float(info["P_pv"]))
    load_hist.append(info["P_load"])
    grid_hist.append(info["P_grid"])
    weather_hist.append(info.get("weather", weather_condition))
    # SỬA LỖI JSON: Chuyển numpy.float64 sang float
    occ_hist.append(float(occupancy_profile[env.t % T]))

# ===== BIỂU ĐỒ =====
fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

axs[0].plot(pv_hist, label="PV (kW)")
axs[0].plot(load_hist, label="Load (kW)")
axs[0].set_ylabel("Power (kW)")
axs[0].set_title("PV vs Load theo thời tiết thực tế")
axs[0].legend()

axs[1].plot(soc_hist, color="orange")
axs[1].set_ylabel("SOC")
axs[1].set_title("Battery SOC")

axs[2].plot(grid_hist, color="red")
axs[2].set_ylabel("Grid Power (kW)")
axs[2].set_title("Điện mua từ lưới")

axs[3].bar(range(T), rewards, color="green")
axs[3].set_ylabel("Reward")
axs[3].set_title("Reward theo thời điểm")

# Biểu đồ thời tiết
weather_numeric = [weather_states.index(w) if w in weather_states else -1 for w in weather_hist]
axs[4].plot(weather_numeric, marker="o", color="blue")
axs[4].set_yticks(range(len(weather_states)))
axs[4].set_yticklabels(weather_states)
axs[4].set_ylabel("Weather")
axs[4].set_title("Mô phỏng chuỗi thời tiết trong ngày")

# Biểu đồ Occupancy
axs[5].plot(occ_hist, color="purple")
axs[5].set_ylabel("Occupancy (người ở nhà)")
axs[5].set_xlabel("Giờ")
axs[5].set_title("Mức độ hiện diện trong nhà (theo hành vi con người)")

plt.tight_layout()

# ===== LƯU BIỂU ĐỒ RA FILE (Tránh lag) =====
plot_filename = "simulation_plot.png"
plot_filepath = os.path.abspath(plot_filename)
try:
    plt.savefig(plot_filename)
    print(f"✅ Đã lưu biểu đồ vào file: {plot_filename}")

    # Tự động mở file ảnh biểu đồ
    webbrowser.open_new_tab(f'file://{plot_filepath}')
    print(f"🚀 Đã mở file biểu đồ {plot_filename} trong tab mới.")
except Exception as e:
    print(f"Lỗi khi lưu/mở file biểu đồ: {e}")

plt.close(fig)  # Đóng đối tượng plot để giải phóng bộ nhớ

simulation_data = {
    "timesteps": list(range(T)),
    "weather": weather_hist,
    "occupancy": occ_hist,
    "soc": soc_hist,
    "pv": pv_hist,
    "load": load_hist,
    "grid": grid_hist,
    "rewards": rewards,
    "devices": []
}

# Lấy trạng thái thiết bị từ human behavior
for t in range(T):
    # SỬA LỖI JSON: Chuyển numpy.bool_ sang bool
    devices_t = {
        "lights": bool(human.device_usage["lights"][t] > 0.5),
        "ac": bool(human.device_usage["ac_prob"][t] > 0.5),
        "heater": bool(human.device_usage["heater_prob"][t] > 0.5),
        "tv": bool(human.device_usage["tv_prob"][t] > 0.5),
        "washing_machine": bool(human.device_usage["washing_machine_prob"][t] > 0.3),
        "ev_charger": bool(human.device_usage["ev_charger_prob"][t] > 0.5)
    }
    simulation_data["devices"].append(devices_t)

with open("simulation_data.json", "w") as f:
    json.dump(simulation_data, f)
print("✅ Đã lưu simulation_data.json")

PORT = 8000
FILE_TO_OPEN = 'visualizer.html'
URL = f"http://localhost:{PORT}/{FILE_TO_OPEN}"

# 1. Định nghĩa một máy chủ đơn giản
Handler = http.server.SimpleHTTPRequestHandler
httpd = None

# 2. Hàm để khởi động máy chủ
def start_server():
    global httpd
    try:
        # Chạy máy chủ
        httpd = socketserver.TCPServer(("", PORT), Handler)
        print(f"✅ Máy chủ đang chạy tại: http://localhost:{PORT}")
        print("Nhấn Ctrl+C trong terminal này để dừng máy chủ.")
        httpd.serve_forever()
    except OSError:
        print(f"❗️ Lỗi: Cổng {PORT} đã được sử dụng. Vui lòng thử cổng khác (vd: 8001).")
        print(f"Vui lòng tự mở file '{FILE_TO_OPEN}' bằng tay sau khi chạy máy chủ thủ công.")
    except KeyboardInterrupt:
        print("\nTắt máy chủ...")
        if httpd:
            httpd.shutdown()

# 3. Chạy máy chủ trong một luồng (thread) riêng để không làm treo script
# (Mặc dù script đã xong, chạy ở luồng chính vẫn tốt hơn)
print("\n" + "="*30)
print(f"Khởi động máy chủ web để xem {FILE_TO_OPEN}...")

# 4. Mở trình duyệt trỏ đến localhost
try:
    webbrowser.open_new_tab(URL)
    print(f"🚀 Đã tự động mở {URL} trong trình duyệt.")
except Exception as e:
    print(f"Lỗi khi mở trình duyệt: {e}. Vui lòng tự mở link: {URL}")

# 5. Bắt đầu chạy máy chủ (đoạn này sẽ giữ cho script chạy)
start_server()