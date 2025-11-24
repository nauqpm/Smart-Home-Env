# Tên file: deterministic_simulator.py
# Mô tả: Cung cấp hàm mục tiêu (fitness function)
# cho LBWO dựa trên mô phỏng xác định (deterministic).

import numpy as np
from human_behavior import HumanBehavior
# Lấy cấu hình và bản đồ công suất từ file gốc của bạn
from run_episode_plot import cfg, DEVICE_POWER_MAP, T

# Sao chép lại cấu hình pin từ file của bạn để dễ truy cập
BAT_CFG = cfg.get('battery', {})
BAT_SOC0 = BAT_CFG.get('soc0', 0.5)
BAT_SOC_MIN = BAT_CFG.get('soc_min', 0.1)
BAT_SOC_MAX = BAT_CFG.get('soc_max', 0.9)
BAT_ETA_CH = BAT_CFG.get('eta_ch', 0.95)
BAT_ETA_DIS = BAT_CFG.get('eta_dis', 0.95)

# Đếm số lượng thiết bị có thể điều khiển
N_SU = len(cfg.get('shiftable_su', []))
N_SI = len(cfg.get('shiftable_si', []))
N_AD = len(cfg.get('adjustable', []))  # Tải điều chỉnh


def _generate_forecasts():
    """
    Tạo một bộ dữ liệu dự báo 24h (giá, PV, và tải người dùng).
    Đây là phiên bản "xác định" của môi trường.
    """
    # 1. Dự báo giá (lấy từ run_episode_plot.py)
    price_forecast = np.array([0.1] * 6 + [0.15] * 6 + [0.25] * 6 + [0.18] * 6)

    # 2. Dự báo PV (lấy từ run_episode_plot.py)
    weather_condition = np.random.choice(["sunny", "mild", "cloudy", "rainy", "stormy"])
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
    pv_forecast = np.clip(base_pv * weather_factors[weather_condition], 0, None)

    # 3. Dự báo Tải người dùng (Tải không thể điều khiển)
    human = HumanBehavior(num_people=4, T=T, weather=weather_condition)
    behavior_data = human.generate_daily_behavior()

    _occ = behavior_data.get("occupancy_ratio")  # Hoặc behavior_data.get("presence_counts")
    activity_profile = behavior_data.get("device_probs")

    human_load_forecast = np.zeros(T)

    # 3a. Tải 'critical' từ config
    human_load_forecast += np.array(cfg.get("critical", [0.0] * T))

    # 3b. Tải từ hành vi (ví dụ: TV, Lights, Laptop)
    # Chúng ta giả định chúng là 'không thể điều khiển' trong kịch bản này
    # và chỉ lấy giá trị trung bình (xác suất > 0.5)

    # Các thiết bị này LÀ MỘT PHẦN của human_behavior.py
    # nhưng KHÔNG có trong cfg['shiftable']
    human_devices = ["lights", "fridge", "tv", "laptop"]
    # Map tên từ activity_profile sang DEVICE_POWER_MAP
    device_map_keys = {
        "lights": "lights",
        "fridge": "fridge",
        "tv": "tv",
        "laptop": "laptop"
    }

    for t in range(T):
        for key, profile_key in device_map_keys.items():
            if activity_profile[key][t] > 0.5:  # Giả định đơn giản
                human_load_forecast[t] += DEVICE_POWER_MAP.get(profile_key, 0.0)

    # 3c. Tải điều chỉnh (Adjustable) - Giả định chúng chạy ở mức 'P_com'
    # Đây là một sự đơn giản hóa.
    for ad_cfg in cfg.get('adjustable', []):
        human_load_forecast[t] += ad_cfg.get('P_com', 0.0)

    return price_forecast, pv_forecast, human_load_forecast


def calculate_total_cost(X, price_forecast, pv_forecast, human_load_forecast):
    """
    Hàm Mục tiêu (Fitness Function) cho LBWO.

    Args:
        X (np.array): Vector giải pháp (lịch trình 24h cho các thiết bị).
                      Là một vector nhị phân dài (N_su + N_si) * 24.
        price_forecast (np.array): Dự báo giá 24h.
        pv_forecast (np.array): Dự báo PV 24h.
        human_load_forecast (np.array): Dự báo tải nền (critical + human) 24h.

    Returns:
        float: Tổng chi phí + Phạt. Càng thấp càng tốt.
    """

    # 1. Giải mã (Decode) vector giải pháp X
    X = np.round(X).astype(int)  # Đảm bảo X là nhị phân

    schedules = {}
    start_idx = 0
    for i in range(N_SU):
        schedules[f'su_{i}'] = X[start_idx: start_idx + T]
        start_idx += T
    for i in range(N_SI):
        schedules[f'si_{i}'] = X[start_idx: start_idx + T]
        start_idx += T

    # 2. Khởi tạo các biến mô phỏng
    total_cost = 0.0
    total_penalty = 0.0
    SOC = BAT_SOC0

    # Biến theo dõi ràng buộc
    su_remaining = [cfg['shiftable_su'][i]['L'] for i in range(N_SU)]
    si_energy_total = [0.0] * N_SI

    # 3. Chạy mô phỏng 24 giờ
    for t in range(T):

        # --- 3a. Tính toán P_load từ lịch trình X ---
        P_su_t = 0.0
        for i in range(N_SU):
            su_cfg = cfg['shiftable_su'][i]
            is_on = schedules[f'su_{i}'][t]

            if is_on == 1 and su_cfg['t_s'] <= t < su_cfg['t_f'] and su_remaining[i] > 0:
                P_su_t += su_cfg['rate']
                su_remaining[i] -= 1  # Giả sử 1 giờ chạy = 1 đơn vị L
            elif is_on == 1:
                total_penalty += 50  # Phạt vì cố chạy ngoài giờ hoặc chạy thừa

        P_si_t = 0.0
        for i in range(N_SI):
            si_cfg = cfg['shiftable_si'][i]
            is_on = schedules[f'si_{i}'][t]

            if is_on == 1 and si_cfg['t_s'] <= t < si_cfg['t_f']:
                P_si_t += si_cfg['rate']
                si_energy_total[i] += si_cfg['rate'] * 1.0  # 1.0 = time_step
            elif is_on == 1:
                total_penalty += 50  # Phạt vì cố chạy ngoài giờ

        # Lấy tải nền từ dự báo
        P_uncontrollable = human_load_forecast[t]

        P_load = P_uncontrollable + P_su_t + P_si_t
        P_pv = pv_forecast[t]

        # --- 3b. Sao chép logic Pin và Lưới từ smart_home_env.py ---
        # (Tham chiếu: smart_home_env.py, dòng 142-160)
        P_ch, P_dis = 0.0, 0.0
        if P_pv >= P_load:
            P_surplus = P_pv - P_load
            if SOC < BAT_SOC_MAX:
                P_ch = P_surplus
                # Giả sử T=24, time_step=1.0
                SOC = min(BAT_SOC_MAX, SOC + BAT_ETA_CH * P_ch / 24.0)
        else:
            P_deficit = P_load - P_pv
            if SOC > BAT_SOC_MIN:
                # Giả sử T=24, time_step=1.0
                P_dis = min(P_deficit, (SOC - BAT_SOC_MIN) * 24.0 / BAT_ETA_DIS)
                SOC = max(BAT_SOC_MIN, SOC - P_dis * BAT_ETA_DIS / 24.0)

        supply = P_pv + P_dis
        demand = P_load + P_ch
        P_grid = max(0, demand - supply)  # Chỉ mua điện

        # --- 3c. Tính chi phí ---
        cost_t = P_grid * price_forecast[t]
        total_cost += cost_t

    # 4. Xử lý Phạt (Penalties) sau khi kết thúc 24h

    # Phạt 1: Tải SU không chạy đủ thời gian L
    for i in range(N_SU):
        if su_remaining[i] > 0:
            total_penalty += su_remaining[i] * 1000.0  # Phạt nặng

    # Phạt 2: Tải SI không chạy đủ năng lượng E
    for i in range(N_SI):
        si_cfg = cfg['shiftable_si'][i]
        if si_energy_total[i] < si_cfg['E']:
            total_penalty += (si_cfg['E'] - si_energy_total[i]) * 1000.0  # Phạt nặng

    return total_cost + total_penalty