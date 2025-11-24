import numpy as np
import json
import http.server
import socketserver
import threading
import matplotlib.pyplot as plt
from smart_home_env import SmartHomeEnv  # File mới của bạn
from human_behavior import HumanBehavior  # File behavior của bạn
import webbrowser
import os

# ===== 1. CẤU HÌNH CƠ BẢN =====
T = 24  # Độ dài 1 ngày (giờ)

# Giá điện (Time of Use - TOU)
price = np.array([0.1] * 6 + [0.15] * 6 + [0.25] * 6 + [0.18] * 6)

# [THAY ĐỔI QUAN TRỌNG]: PV Profile đầu vào giờ chỉ là "placeholder" (giữ chỗ).
# Environment sẽ tự tính toán lại dựa trên Vật lý (Ineichen/Zenith) bên trong.
# Ta để mảng 0 để tránh nhầm lẫn.
dummy_pv_profile = np.zeros(T)

# ===== 2. CẤU HÌNH THIẾT BỊ (CONFIG) =====
cfg = {
    "critical": [0.33] * 24,  # Tải nền
    # Tải có thể điều chỉnh công suất (AC, Bình nóng lạnh)
    "adjustable": [
        {"P_min": 0.5, "P_max": 2.0, "P_com": 1.5, "alpha": 0.06},  # AC
        {"P_min": 0.0, "P_max": 2.0, "P_com": 1.5, "alpha": 0.08}  # Heater
    ],
    # Tải có thể dời lịch (Máy giặt, Máy rửa bát) - Shiftable Uninterruptible
    "shiftable_su": [
        {"rate": 0.5, "L": 1, "t_s": 7, "t_f": 22},  # Washing machine
        {"rate": 0.8, "L": 1, "t_s": 19, "t_f": 23}  # Dishwasher
    ],
    # Tải có thể ngắt quãng (Sạc xe điện) - Shiftable Interruptible
    "shiftable_si": [
        {"rate": 3.3, "E": 7.0, "t_s": 0, "t_f": 23}  # EV charger
    ],
    "beta": 0.5,  # Trọng số ưu tiên bán điện (nếu có logic bán)
    "battery": {
        "soc0": 0.5, "soc_min": 0.1, "soc_max": 0.9,
        "eta_ch": 0.95, "eta_dis": 0.95
    }
}

# ===== 3. KHỞI TẠO MÔI TRƯỜNG & HUMAN BEHAVIOR =====
print("⚙️ Đang khởi tạo Môi trường Smart Home (Physics-based)...")
# Lưu ý: dummy_pv_profile được truyền vào nhưng sẽ bị class AdvancedPV ghi đè logic
env = SmartHomeEnv(price, dummy_pv_profile, cfg)

# Khởi tạo hành vi con người
human = HumanBehavior(num_people=4, T=T, seed=42, month=None)
multi_day_mode = True  # Chạy mô phỏng 30 ngày để thấy sự thay đổi thời tiết

if multi_day_mode:
    print("\n🗓️ Đang sinh lịch trình sinh hoạt cho 30 ngày...")
    month_behavior = human.generate_month_behavior_with_schedule(start_day="monday", days=30)

    # Nạp hành vi vào môi trường
    env.set_month_behavior(month_behavior)

    # Thống kê sơ bộ
    event_stats = {}
    for d, data in month_behavior.items():
        event_type = data.get("event_type", "unknown")
        event_stats[event_type] = event_stats.get(event_type, 0) + 1
    print(f"📊 Thống kê: {event_stats}")
else:
    # Chế độ 1 ngày đơn giản
    print("\n🗓️ Chạy mô phỏng 1 ngày đơn lẻ...")
    # Env mới đã tự có logic fallback nếu không set behavior,
    # nhưng ta set thủ công để kiểm soát tốt hơn.
    behavior = human.generate_daily_behavior(sample_device_states=True)
    # Env mới hỗ trợ nhận dict behavior trực tiếp (qua logic fallback trong reset),
    # hoặc ta có thể gán vào biến tạm nếu cần (tuy nhiên logic multi-day tốt hơn).
    env.behavior = behavior

# ===== 4. VÒNG LẶP MÔ PHỎNG (RUN EPISODE) =====
obs = env.reset()
done = False

# Các danh sách để lưu lịch sử chạy
history = {
    "rewards": [], "soc": [], "pv": [], "load": [],
    "grid": [], "weather": [], "occupancy": [],
    "devices": [], "device_power": []
}

# Mapping công suất danh định để vẽ biểu đồ
DEVICE_POWER_MAP = {
    "lights": 0.1, "fridge": 0.2, "tv": 0.15, "ac": 1.5, "heater": 1.0,
    "washing_machine": 0.5, "dishwasher": 0.8, "laptop": 0.08, "ev_charger": 3.3
}

print("\n▶️ Bắt đầu chạy mô phỏng...")
while not done:
    # 1. Chọn hành động ngẫu nhiên (hoặc thay bằng Agent RL của bạn ở đây)
    action = env.action_space.sample()

    # 2. Bước chạy môi trường
    obs, reward, done, info = env.step(action)

    # 3. Ghi lại dữ liệu từ INFO (Quan trọng: Lấy dữ liệu thực tế từ Env)
    history["rewards"].append(reward)
    history["soc"].append(info.get("SOC", 0.0))

    # [QUAN TRỌNG] Lấy PV từ info (được tính bằng pvlib) chứ không phải mảng đầu vào
    history["pv"].append(float(info.get("P_pv", 0.0)))

    history["load"].append(info.get("P_load", 0.0))
    history["grid"].append(info.get("P_grid", 0.0))
    history["weather"].append(info.get("weather", "unknown"))

    # Lấy thông tin thiết bị từ info (nếu Env trả về) hoặc behavior
    # Logic lấy occupancy cho biểu đồ
    if hasattr(env, 'current_behavior') and env.current_behavior:
        occ = env.current_behavior.get("occupancy_ratio", [0] * T)
        t_idx = (env.t - 1) % T
        history["occupancy"].append(occ[t_idx])
    else:
        history["occupancy"].append(0)

    # Lưu trạng thái thiết bị (On/Off) từ info
    history["devices"].append(info.get("device_states", {}))

    # Tính công suất từng thiết bị để vẽ biểu đồ stacked
    # (Kết hợp trạng thái On/Off với công suất danh định)
    dev_states = info.get("device_states", {})
    p_t = {}
    for d_name, is_on in dev_states.items():
        if is_on:
            p_t[d_name] = DEVICE_POWER_MAP.get(d_name, 0.0)
        else:
            p_t[d_name] = 0.0
    p_t["pv"] = float(info.get("P_pv", 0.0))  # Lưu cả PV để tham chiếu
    history["device_power"].append(p_t)

print(f"✅ Hoàn thành mô phỏng. Tổng reward: {sum(history['rewards']):.2f}")

# ===== 5. VẼ BIỂU ĐỒ (PLOTTING) =====
# Chỉ vẽ 24 giờ đầu tiên hoặc ngày cuối cùng để dễ nhìn,
# hoặc vẽ toàn bộ nếu thích. Ở đây vẽ toàn bộ chuỗi thời gian.
fig, axs = plt.subplots(7, 1, figsize=(12, 16), sharex=True)

# Plot 1: PV vs Load
axs[0].plot(history["pv"], label="PV (Physics-based)", color="orange")
axs[0].plot(history["load"], label="Total Load", color="blue", alpha=0.7)
axs[0].set_ylabel("Power (kW)")
axs[0].set_title("PV Generation (Ineichen Model) vs House Load")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Plot 2: Battery SOC
axs[1].plot(history["soc"], color="green")
axs[1].set_ylabel("SOC (0-1)")
axs[1].set_title("Battery State of Charge")
axs[1].grid(True, alpha=0.3)

# Plot 3: Grid Interaction
axs[2].plot(history["grid"], color="red")
axs[2].set_ylabel("Grid Import (kW)")
axs[2].set_title("Grid Energy Bought")
axs[2].grid(True, alpha=0.3)

# Plot 4: Rewards
axs[3].bar(range(len(history["rewards"])), history["rewards"], color="purple", alpha=0.6)
axs[3].set_ylabel("Reward")
axs[3].set_title("Agent Reward per Step")

# Plot 5: Weather (Categorical to Numeric)
weather_states_list = ["sunny", "mild", "cloudy", "rainy", "stormy"]
w_numeric = [weather_states_list.index(w) if w in weather_states_list else -1 for w in history["weather"]]
axs[4].plot(w_numeric, marker=".", linestyle="none", color="cyan")
axs[4].set_yticks(range(len(weather_states_list)))
axs[4].set_yticklabels(weather_states_list)
axs[4].set_ylabel("Condition")
axs[4].set_title("Simulated Weather")
axs[4].grid(True, axis='y')

# Plot 6: Occupancy
axs[5].plot(history["occupancy"], color="brown")
axs[5].set_ylabel("Occupancy Ratio")
axs[5].set_title("Human Occupancy")

# Plot 7: Total Device Power Consumption
total_dev_p = [sum([v for k, v in d.items() if k != 'pv']) for d in history["device_power"]]
axs[6].plot(total_dev_p, color="black", linestyle="--")
axs[6].set_ylabel("kW")
axs[6].set_xlabel("Time Step (Hour)")
axs[6].set_title("Total Appliance Power")

plt.tight_layout()
plot_filename = "simulation_physics_plot.png"
plt.savefig(plot_filename)
print(f"📊 Đã lưu biểu đồ vào: {plot_filename}")
plt.close(fig)

# ===== 6. XUẤT JSON & WEB SERVER =====
# Chuẩn bị dữ liệu JSON (Cần convert numpy types sang python types)
sim_data_export = {
    "timesteps": list(range(len(history["pv"]))),
    "weather": history["weather"],
    "occupancy": history["occupancy"],
    "soc": history["soc"],
    "pv": history["pv"],
    "load": history["load"],
    "grid": history["grid"],
    "rewards": history["rewards"],
    "devices": history["devices"],  # List of dicts
    "device_power": history["device_power"]  # List of dicts
}

with open("simulation_data.json", "w") as f:
    json.dump(sim_data_export, f, indent=2)
print("💾 Đã xuất file simulation_data.json")

# Server Code (Giữ nguyên như cũ)
PORT = 8000
FILE_TO_OPEN = 'visualizer.html'
URL = f"http://localhost:{PORT}/{FILE_TO_OPEN}"
Handler = http.server.SimpleHTTPRequestHandler


def start_server():
    web_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(web_dir)
    try:
        httpd = socketserver.TCPServer(("", PORT), Handler)
        print(f"🚀 Server running at: {URL}")
        httpd.serve_forever()
    except OSError:
        print(f"⚠️ Port {PORT} busy. Check: {URL}")
    except KeyboardInterrupt:
        pass


# Tự động mở web
try:
    # Tạo file html giả nếu chưa có để test (Optional)
    if not os.path.exists(FILE_TO_OPEN):
        with open(FILE_TO_OPEN, "w") as f:
            f.write("<h1>Simulation Data Generated. Check console.</h1>")

    threading.Thread(target=start_server, daemon=True).start()
    webbrowser.open_new_tab(URL)
    input("\n🔴 Nhấn Enter để dừng server và thoát chương trình...\n")
except Exception as e:
    print(f"Error: {e}")