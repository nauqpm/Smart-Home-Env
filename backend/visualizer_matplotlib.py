import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import json

DATA_FILE = 'simulation_data.json'

try:
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    print("✅ Đã load dữ liệu thành công!")
except FileNotFoundError:
    print(f"❌ Không tìm thấy file {DATA_FILE}. Hãy chạy run_episode_plot.py trước.")
    exit()

timesteps = data['timesteps']
T = len(timesteps)

all_devices = list(data['device_power'][0].keys())
if 'pv' in all_devices: all_devices.remove('pv')

DEVICE_COLORS = {
    'ac': '#1f77b4', 'ev_charger': '#d62728',
    'washing_machine': '#9467bd', 'heater': '#ff7f0e',
    'dishwasher': '#8c564b', 'lights': '#bcbd22',
    'laptop': '#7f7f7f', 'fridge': '#17becf', 'tv': '#e377c2'
}

plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(14, 10))
fig.canvas.manager.set_window_title('Smart Home Analytics Dashboard')

spec = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], hspace=0.4, bottom=0.1)

ax_overview = fig.add_subplot(spec[0, :])
ax_overview.set_title("TONG QUAN: Nang luong Mat troi (PV) vs Tong Tai (Load)", fontsize=11, fontweight='bold')
ax_overview.set_ylabel("Cong suat (kW)")
ax_overview.set_xlim(0, T - 1)
max_val = max(max(data['load']), max(data['pv']))
ax_overview.set_ylim(0, max_val * 1.2 if max_val > 0 else 5)

line_pv, = ax_overview.plot([], [], color='#ffa500', label='PV Gen', linewidth=2, linestyle='--')
line_load, = ax_overview.plot([], [], color='#000080', label='Total Load', linewidth=2)
fill_pv = ax_overview.fill_between([], 0, 0, color='#ffa500', alpha=0.2)
ax_overview.legend(loc='upper right', frameon=True)

ax_devices = fig.add_subplot(spec[1, :])
ax_devices.set_title("CHI TIET: Cong suat tung thiet bi", fontsize=11, fontweight='bold')
ax_devices.set_ylabel("Cong suat (kW)")
ax_devices.set_xlim(0, T - 1)

max_dev_p = 0.5
for d in all_devices:
    p_list = [step[d] for step in data['device_power']]
    if p_list: max_dev_p = max(max_dev_p, max(p_list))
ax_devices.set_ylim(0, max_dev_p * 1.2)

dev_lines = {}
for dev in all_devices:
    color = DEVICE_COLORS.get(dev, '#333333')
    # Format tên thiết bị cho đẹp
    label = dev.upper().replace("_", " ")
    line, = ax_devices.plot([], [], label=label, color=color, linewidth=1.5)
    dev_lines[dev] = line
ax_devices.legend(loc='upper left', ncol=4, fontsize=8, frameon=True)

ax_soc = fig.add_subplot(spec[2, 0])
ax_soc.set_title("PIN: Trang thai sac (SOC)", fontsize=11, fontweight='bold')
ax_soc.set_ylabel("% SOC")
ax_soc.set_ylim(0, 100)
ax_soc.set_xlim(0, T - 1)
ax_soc.axhspan(0, 10, color='red', alpha=0.1)  # Vùng nguy hiểm

line_soc, = ax_soc.plot([], [], color='#2ca02c', linewidth=3)
fill_soc = ax_soc.fill_between([], 0, 0, color='#2ca02c', alpha=0.3)

ax_cost = fig.add_subplot(spec[2, 1])
ax_cost.set_title("TAI CHINH: Chi phi tich luy", fontsize=11, fontweight='bold')
ax_cost.set_ylabel("VND (Nghin dong)")
ax_cost.set_xlim(0, T - 1)

cumulative_cost = np.cumsum([-r for r in data['rewards']])
max_cost = max(cumulative_cost) if len(cumulative_cost) > 0 else 10
ax_cost.set_ylim(0, max_cost * 1.1 + 1)

line_cost, = ax_cost.plot([], [], color='#d62728', linewidth=2)
ax_cost.fill_between([], 0, 0, color='#d62728', alpha=0.1)

vertical_lines = []
for ax in [ax_overview, ax_devices, ax_soc, ax_cost]:
    vl = ax.axvline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    vertical_lines.append(vl)

info_text = fig.text(0.5, 0.02, "", fontsize=12, fontweight='bold',
                     ha='center', va='bottom',
                     bbox=dict(facecolor='#f0f0f0', edgecolor='gray', boxstyle='round,pad=0.5'))



def init():
    return line_pv, line_load, line_soc, line_cost


def update(frame):
    x_data = range(frame + 1)

    pv_data = data['pv'][:frame + 1]
    load_data = data['load'][:frame + 1]
    line_pv.set_data(x_data, pv_data)
    line_load.set_data(x_data, load_data)

    global fill_pv, fill_soc
    fill_pv.remove()
    fill_pv = ax_overview.fill_between(x_data, 0, pv_data, color='#ffa500', alpha=0.2)

    for dev, line in dev_lines.items():
        p_series = [step[dev] for step in data['device_power'][:frame + 1]]
        line.set_data(x_data, p_series)

    soc_data = [s * 100 for s in data['soc'][:frame + 1]]
    line_soc.set_data(x_data, soc_data)

    fill_soc.remove()
    fill_soc = ax_soc.fill_between(x_data, 0, soc_data, color='#2ca02c', alpha=0.3)

    cost_data = cumulative_cost[:frame + 1]
    line_cost.set_data(x_data, cost_data)

    for vl in vertical_lines:
        vl.set_xdata([frame, frame])

    weather = data['weather'][frame].upper()

    occupancy = int(data['occupancy'][frame])

    current_cost = cost_data[-1]

    status_str = (f"GIO: {frame:02d}:00  |  "
                  f"THOI TIET: {weather}  |  "
                  f"NGUOI: {occupancy}/4  |  "
                  f"CHI PHI: {current_cost:.2f}k VND")

    bg_color = '#fffacd' if 6 <= frame <= 18 else '#e6e6fa'
    info_text.set_bbox(dict(facecolor=bg_color, edgecolor='gray', boxstyle='round,pad=0.5'))
    info_text.set_text(status_str)

    return [line_pv, line_load, line_soc, line_cost] + list(dev_lines.values()) + vertical_lines


ani = FuncAnimation(fig, update, frames=T, init_func=init, blit=False, interval=300, repeat=True)
plt.show()