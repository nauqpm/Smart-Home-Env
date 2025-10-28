import numpy as np
import json
import http.server
import socketserver
import threading
import matplotlib.pyplot as plt
from smart_home_env import SmartHomeEnv  # Ensure this exists
from human_behavior import HumanBehavior  # Ensure this exists (upgraded API)
import webbrowser
import os

# ===== CONFIG =====
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

# ===== SYSTEM CONFIG =====
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

# ===== INIT ENV & HUMAN BEHAVIOR =====
env = SmartHomeEnv(price, pv, cfg)

human = HumanBehavior(num_people=4, T=T, seed=42, month=None)
multi_day_mode = True

if multi_day_mode:
    print("\n🧭 Bắt đầu mô phỏng nhiều ngày (30 ngày) với lịch sự kiện thực tế...")
    # sinh hành vi cho cả tháng
    month_behavior = human.generate_month_behavior_with_schedule(start_day="monday", days=30)

    # === THÊM DÒNG NÀY ===
    # Nạp hành vi của tháng vào môi trường
    env.set_month_behavior(month_behavior)
    # ======================

    # đếm thống kê loại ngày
    event_stats = {}
    for d, data in month_behavior.items():
        event_type = data.get("event_type", "unknown")
        event_stats[event_type] = event_stats.get(event_type, 0) + 1
    print("📊 Thống kê loại ngày:")
    for ev, cnt in event_stats.items():
        print(f" - {ev}: {cnt} ngày")

    # chọn ngày để mô phỏng
    selected_day = 0
    behavior = month_behavior[selected_day]
    print(f"▶️ Mô phỏng ngày {selected_day + 1}: {behavior['event_type']}")
else:
    # dùng mô phỏng 1 ngày như cũ
    behavior = human.generate_daily_behavior(sample_device_states=True)


    # --- PHẦN BỔ TRỢ: ĐẢM BẢO TƯƠNG THÍCH NGƯỢC VỚI ENV ---
    # Bọc dữ liệu behavior mới để env.step() có thể dùng như cũ (occupancy, device_usage)
    class BehaviorWrapper:
        def __init__(self, b):
            # Lưu ý: env cũ (File 1) đang tìm .occupancy và .device_usage
            self.occupancy = b.get("occupancy_ratio", [1.0] * T)
            self.device_usage = b.get("device_probs", {})  # Ánh xạ device_probs -> device_usage


    env.behavior = BehaviorWrapper(behavior)

# unpack behavior

presence_counts = behavior.get("presence_counts")
occupancy_profile = behavior.get("occupancy_ratio")
activity_profile = behavior.get("activity_level")
device_probs = behavior.get("device_probs")
device_states = behavior.get("device_states")  # dict: device -> list[bool]

# start environment
obs = env.reset()
done = False
rewards, soc_hist, pv_hist, load_hist, grid_hist, weather_hist, occ_hist = [], [], [], [], [], [], []
devices_hist = []
device_power_hist = []

# device nominal powers (kW)
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

# run episode
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    # record
    rewards.append(reward)
    soc_hist.append(info.get("SOC", 0.0))
    pv_hist.append(float(info.get("P_pv", 0.0)))
    load_hist.append(info.get("P_load", 0.0))
    grid_hist.append(info.get("P_grid", 0.0))
    weather_hist.append(info.get("weather", weather_condition))

    # ensure valid timestep index
    t_index = max(0, (env.t - 1) % T)

    # occupancy from new API
    current_occupancy = float(occupancy_profile[t_index])
    occ_hist.append(current_occupancy)

    # get device on/off from sampled device_states
    devices_t = {}
    for d in DEVICE_POWER_MAP.keys():
        # device_states keys follow device names; fridge is always True in behavior generation
        ds = device_states.get(d)
        if ds is not None:
            devices_t[d] = bool(ds[t_index])
        else:
            # fallback to probability-based threshold
            p = device_probs.get(d, [0]*T)[t_index]
            devices_t[d] = (p > 0.5)

    devices_hist.append(devices_t)

    # compute power per device (use info if environment provides device-level power)
    power_t = {}
    for device_name, is_on in devices_t.items():
        if is_on:
            # check if env provided a device-specific power in info
            key_map = {
                "ac": "P_ac",
                "heater": "P_heater",
                "washing_machine": "P_washing_machine",
                "dishwasher": "P_dishwasher",
                "ev_charger": "P_ev_charger"
            }
            if device_name in key_map and key_map[device_name] in info:
                power_t[device_name] = info[key_map[device_name]]
            else:
                power_t[device_name] = DEVICE_POWER_MAP.get(device_name, 0.0)
        else:
            power_t[device_name] = 0.0

    power_t["pv"] = float(info.get("P_pv", 0.0))
    device_power_hist.append(power_t)

# ===== PLOTTING =====
fig, axs = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
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
axs[3].bar(range(len(rewards)), rewards, color="green")
axs[3].set_ylabel("Reward")
axs[3].set_title("Reward")
weather_numeric = [weather_states.index(w) if w in weather_states else -1 for w in weather_hist]
axs[4].plot(weather_numeric, marker="o")
axs[4].set_yticks(range(len(weather_states)))
axs[4].set_yticklabels(weather_states)
axs[4].set_ylabel("Weather")
axs[4].set_title("Weather Simulation")
axs[5].plot(occ_hist, color="purple")
axs[5].set_ylabel("Occupancy")
axs[5].set_xlabel("Hour")
axs[5].set_title("Occupancy")
# device power summary (stacked or total)
total_device_power = [sum(d.values()) for d in device_power_hist]
axs[6].plot(total_device_power, color="brown")
axs[6].set_ylabel("Total Device Power (kW)")
axs[6].set_title("Total Device Power (incl. PV)")
plt.tight_layout()
plot_filename = "simulation_plot.png"
try:
    plt.savefig(plot_filename)
    print(f"✅ Đã lưu biểu đồ vào file: {plot_filename}")
except Exception as e:
    print(f"Lỗi khi lưu/mở file biểu đồ: {e}")
plt.close(fig)

# ===== EXPORT JSON =====
simulation_data = {
    "timesteps": list(range(T)),
    "weather": weather_hist,
    "occupancy": occ_hist,
    "presence_counts": presence_counts,
    "activity_level": activity_profile,
    "soc": soc_hist,
    "pv": pv_hist,
    "load": load_hist,
    "grid": grid_hist,
    "rewards": rewards,
    "devices": devices_hist,
    "device_power": device_power_hist,
    "device_probs": device_probs
}

with open("simulation_data.json", "w") as f:
    json.dump(simulation_data, f, indent=2)
print("✅ Đã lưu simulation_data.json (tương thích API mới)")

if multi_day_mode:
    summary = {ev: cnt for ev, cnt in event_stats.items()}
    with open("month_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("✅ Đã lưu month_summary.json (thống kê theo loại ngày)")

# ===== SIMPLE WEB SERVER TO SERVE VISUALIZER =====
PORT = 8000
FILE_TO_OPEN = 'visualizer.html'
URL = f"http://localhost:{PORT}/{FILE_TO_OPEN}"
Handler = http.server.SimpleHTTPRequestHandler
httpd = None

def start_server():
    global httpd
    web_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(web_dir)
    print(f"Thư mục phục vụ web: {web_dir}")
    try:
        httpd = socketserver.TCPServer(("", PORT), Handler)
        print(f"✅ Máy chủ đang chạy tại: http://localhost:{PORT}")
        print(f"Đang phục vụ tệp: {FILE_TO_OPEN}")
        httpd.serve_forever()
    except OSError as e:
        print(f"❗️ Lỗi: Cổng {PORT} đã được sử dụng hoặc lỗi khác: {e}")
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
