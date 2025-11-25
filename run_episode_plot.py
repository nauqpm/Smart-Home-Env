# run_episode_plot.py - Fixed Config
import numpy as np
import json
import http.server
import socketserver
import threading
import matplotlib.pyplot as plt
from smart_home_env import SmartHomeEnv
from human_behavior import HumanBehavior
import webbrowser
import os

T = 24

price = np.array([0.1] * 6 + [0.15] * 6 + [0.25] * 6 + [0.18] * 6)

dummy_pv_profile = np.zeros(T)

cfg = {
    "critical": [0.33] * 24,
    "adjustable": [
        {"P_min": 0.5, "P_max": 2.0, "P_com": 1.5, "alpha": 0.06},
        {"P_min": 0.0, "P_max": 2.0, "P_com": 1.5, "alpha": 0.08}
    ],
    "shiftable_su": [
        {"rate": 0.5, "L": 1, "t_s": 7, "t_f": 22},
        {"rate": 0.8, "L": 1, "t_s": 19, "t_f": 23}
    ],
    "shiftable_si": [
        {"rate": 3.3, "E": 7.0, "t_s": 0, "t_f": 23}
    ],
    "beta": 0.5,
    "battery": {
        "capacity_kwh": 10.0,
        "soc_init": 0.5,
        "soc_min": 0.1,
        "soc_max": 0.9,
        "p_charge_max_kw": 3.0,
        "p_discharge_max_kw": 3.0,
        "eta_ch": 0.95,
        "eta_dis": 0.95
    },

    "behavior": {
        "residents": [{'name': 'user1'}, {'name': 'user2'}],
        "must_run_base": 0.2
    },
    "pv_config": {
        "latitude": 10.762622,
        "longitude": 106.660172,
        "tz": "Asia/Ho_Chi_Minh",
        "surface_tilt": 10.0,
        "surface_azimuth": 180.0,
        "module_parameters": {"pdc0": 3.0}
    },
    "sim_steps": T
}

print("⚙️ Đang khởi tạo Môi trường Smart Home (Physics-based)...")
env = SmartHomeEnv(price, dummy_pv_profile, cfg)

human = HumanBehavior(num_people=4, T=T, seed=42, month=None)
multi_day_mode = True

if multi_day_mode:
    print("\n🗓️ Đang sinh lịch trình sinh hoạt cho 30 ngày...")
    month_behavior = human.generate_month_behavior_with_schedule(start_day="monday", days=30)
    # Env mới hỗ trợ set month behavior (nếu bạn đã implement hàm set_month_behavior trong env)
    if hasattr(env, 'set_month_behavior'):
        env.set_month_behavior(month_behavior)
else:
    print("\n🗓️ Chạy mô phỏng 1 ngày đơn lẻ...")
    behavior = human.generate_daily_behavior(sample_device_states=True)
    pass

obs, info = env.reset()
done = False

history = {
    "rewards": [], "soc": [], "pv": [], "load": [],
    "grid": [], "weather": [], "occupancy": [],
    "devices": [], "device_power": []
}

DEVICE_POWER_MAP = {
    "lights": 0.1, "fridge": 0.2, "tv": 0.15, "ac": 1.5, "heater": 1.0,
    "washing_machine": 0.5, "dishwasher": 0.8, "laptop": 0.08, "ev_charger": 3.3
}

print("\n▶️ Bắt đầu chạy mô phỏng...")
while not done:
    action = env.action_space.sample()

    step_res = env.step(action)
    if len(step_res) == 5:
        obs, reward, terminated, truncated, info = step_res
        done = terminated or truncated
    else:
        obs, reward, done, info = step_res

    history["rewards"].append(reward)
    history["soc"].append(info.get("soc", 0.0))
    history["pv"].append(info.get("pv", 0.0))
    history["load"].append(info.get("load", 0.0))
    history["grid"].append(info.get("grid", 0.0))
    history["weather"].append(info.get("weather", "unknown"))
    history["occupancy"].append(info.get("n_home", 0))

    dev_states = {
        "lights": True, "fridge": True,
        "tv": info.get("n_home", 0) > 0,
        "ac": info.get("temp", 25) > 28,
    }
    history["devices"].append(dev_states)

    # Power breakdown
    p_t = {}
    for d, is_on in dev_states.items():
        p_t[d] = DEVICE_POWER_MAP.get(d, 0.0) if is_on else 0.0
    p_t["pv"] = info.get("pv", 0.0)
    history["device_power"].append(p_t)

print(f"✅ Hoàn thành mô phỏng. Tổng reward: {sum(history['rewards']):.2f}")


sim_data_export = {
    "timesteps": list(range(len(history["pv"]))),
    "weather": history["weather"],
    "occupancy": history["occupancy"],
    "soc": history["soc"],
    "pv": history["pv"],
    "load": history["load"],
    "grid": history["grid"],
    "rewards": history["rewards"],
    "devices": history["devices"],
    "device_power": history["device_power"]
}

with open("simulation_data.json", "w") as f:
    json.dump(sim_data_export, f, indent=2)
print("💾 Đã xuất file simulation_data.json")

# Server Code
#PORT = 8000
"""
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

try:
    if not os.path.exists(FILE_TO_OPEN):
        with open(FILE_TO_OPEN, "w", encoding='utf-8') as f:
            f.write("<h1>Simulation Data Generated. Please copy visualizer.html here.</h1>")

    threading.Thread(target=start_server, daemon=True).start()
    webbrowser.open_new_tab(URL)
    input("\n🔴 Nhấn Enter để dừng server và thoát chương trình...\n")
except Exception as e:
    print(f"Error: {e}")
"""