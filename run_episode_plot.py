import numpy as np
import json
import http.server
import socketserver
import threading
import matplotlib.pyplot as plt
from smart_home_env import SmartHomeEnv  # Đảm bảo file này tồn tại
from human_behavior import HumanBehavior  # Đảm bảo file này tồn tại
import webbrowser
import os

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
devices_hist = []
device_power_hist = []

# Định nghĩa công suất (kW) cho từng thiết bị
DEVICE_POWER_MAP = {
    "lights": 0.1,
    "fridge": 0.2,
    "tv": 0.15,
    "ac": 1.5,
    "heater": 1.0,
    "washing_machine": 0.5,
    "dishwasher": 0.8,
    "laptop": 0.08,
    "ev_charger": 3.3
}

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    # Ghi dữ liệu cơ bản
    rewards.append(reward)
    soc_hist.append(info["SOC"])
    pv_hist.append(float(info["P_pv"]))
    load_hist.append(info["P_load"])
    grid_hist.append(info["P_grid"])
    weather_hist.append(info.get("weather", weather_condition))
    current_occupancy = float(occupancy_profile[env.t % T]) # Lưu lại để dùng bên dưới
    occ_hist.append(current_occupancy)

    # Lấy trạng thái thiết bị VÀ tính toán điện năng chi tiết
    t_index = (env.t - 1) % T # Chỉ số thời gian (0-23)
    behavior = human.device_usage
    default_prob = [0] * T

    devices_t = {
        "lights": bool(behavior.get("lights", default_prob)[t_index] > 0.5),
        "ac": bool(behavior.get("ac_prob", default_prob)[t_index] > 0.5),
        "heater": bool(behavior.get("heater_prob", default_prob)[t_index] > 0.5),
        "tv": bool(behavior.get("tv_prob", default_prob)[t_index] > 0.5),
        "washing_machine": bool(behavior.get("washing_machine_prob", default_prob)[t_index] > 0.3),
        "ev_charger": bool(behavior.get("ev_charger_prob", default_prob)[t_index] > 0.5),
        "fridge": True,
        "laptop": bool(current_occupancy > 0.1 and 8 <= t_index <= 23), # Sửa điều kiện occupancy
        "dishwasher": bool(behavior.get("washing_machine_prob", default_prob)[t_index] > 0.3 and 19 <= t_index <= 23)
    }
    devices_hist.append(devices_t)

    power_t = {}
    for device_name, is_on in devices_t.items():
        if is_on:
            # Sửa lỗi logic: AC và Heater có thể có công suất thay đổi từ env.step
            if device_name == "ac" and "P_ac" in info:
                 power_t[device_name] = info["P_ac"]
            elif device_name == "heater" and "P_heater" in info: # Giả sử env trả về P_heater
                 power_t[device_name] = info["P_heater"]
            # Các thiết bị shiftable có thể có công suất khác nhau
            elif device_name == "washing_machine" and "P_washing_machine" in info: # Giả sử
                 power_t[device_name] = info["P_washing_machine"]
            elif device_name == "dishwasher" and "P_dishwasher" in info: # Giả sử
                 power_t[device_name] = info["P_dishwasher"]
            elif device_name == "ev_charger" and "P_ev_charger" in info: # Giả sử
                 power_t[device_name] = info["P_ev_charger"]
            else:
                # Dùng công suất mặc định nếu không có thông tin từ env
                power_t[device_name] = DEVICE_POWER_MAP.get(device_name, 0.0)
        else:
            power_t[device_name] = 0.0

    power_t["pv"] = float(info["P_pv"]) # Thêm PV
    device_power_hist.append(power_t)

# ===== BIỂU ĐỒ =====
fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
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
axs[2].set_title("Grid Power")
axs[3].bar(range(T), rewards, color="green")
axs[3].set_ylabel("Reward")
axs[3].set_title("Reward")
weather_numeric = [weather_states.index(w) if w in weather_states else -1 for w in weather_hist]
axs[4].plot(weather_numeric, marker="o", color="blue")
axs[4].set_yticks(range(len(weather_states)))
axs[4].set_yticklabels(weather_states)
axs[4].set_ylabel("Weather")
axs[4].set_title("Weather Simulation")
axs[5].plot(occ_hist, color="purple")
axs[5].set_ylabel("Occupancy")
axs[5].set_xlabel("Hour")
axs[5].set_title("Occupancy")
plt.tight_layout()
plot_filename = "simulation_plot.png"
try:
    plt.savefig(plot_filename)
    print(f"✅ Đã lưu biểu đồ vào file: {plot_filename}")
except Exception as e:
    print(f"Lỗi khi lưu/mở file biểu đồ: {e}")
plt.close(fig)

# ===== XUẤT JSON =====
simulation_data = {
    "timesteps": list(range(T)),
    "weather": weather_hist,
    "occupancy": occ_hist,
    "soc": soc_hist,
    "pv": pv_hist,
    "load": load_hist,
    "grid": grid_hist,
    "rewards": rewards,
    "devices": devices_hist,
    "device_power": device_power_hist
}

with open("simulation_data.json", "w") as f:
    json.dump(simulation_data, f, indent=2) # Thêm indent=2 để dễ đọc file JSON
print("✅ Đã lưu simulation_data.json (v2.1 - Sửa lỗi công suất)")

# ===== MÁY CHỦ =====
PORT = 8000
FILE_TO_OPEN = 'visualizer.html'
URL = f"http://localhost:{PORT}/{FILE_TO_OPEN}"
Handler = http.server.SimpleHTTPRequestHandler
httpd = None

def start_server():
    global httpd
    # Thay đổi thư mục làm việc để đảm bảo server phục vụ đúng file
    web_dir = os.path.dirname(os.path.abspath(__file__)) # Lấy thư mục chứa file python
    os.chdir(web_dir)
    print(f"Thư mục phục vụ web: {web_dir}")

    try:
        httpd = socketserver.TCPServer(("", PORT), Handler)
        print(f"✅ Máy chủ đang chạy tại: http://localhost:{PORT}")
        print(f"Đang phục vụ tệp: {FILE_TO_OPEN}")
        print("Nhấn Ctrl+C trong terminal này để dừng máy chủ.")
        httpd.serve_forever()
    except OSError as e:
        print(f"❗️ Lỗi: Cổng {PORT} đã được sử dụng hoặc lỗi khác: {e}")
        print(f"Vui lòng tự mở file '{FILE_TO_OPEN}' bằng tay hoặc thử cổng khác.")
    except KeyboardInterrupt:
        print("\nTắt máy chủ...")
        if httpd:
            httpd.shutdown()

print("\n" + "="*30)
print(f"Khởi động máy chủ web để xem {FILE_TO_OPEN}...")

try:
    webbrowser.open_new_tab(URL)
    print(f"🚀 Đã tự động mở {URL} trong trình duyệt.")
except Exception as e:
    print(f"Lỗi khi mở trình duyệt: {e}. Vui lòng tự mở link: {URL}")

start_server()