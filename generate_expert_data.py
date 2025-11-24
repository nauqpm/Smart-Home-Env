# Tên file: generate_expert_data.py
# Mô tả: Chạy Giai đoạn 2 - Tạo dữ liệu chuyên gia
#
# Yêu cầu:
# 1. lbwo_optimizer.py (Từ Bước 1)
# 2. deterministic_simulator.py (File ở trên)
# 3. Các file gốc: smart_home_env.py, human_behavior.py, run_episode_plot.py

import numpy as np
import json
import time

# Từ Bước 1
from lbwo_optimizer import LBWO
# Từ File 1 (ở trên)
from deterministic_simulator import _generate_forecasts, calculate_total_cost, N_SU, N_SI, T, cfg, BAT_SOC_MAX, BAT_SOC_MIN, BAT_ETA_CH, BAT_ETA_DIS

# Lấy hàm _get_obs từ smart_home_env.py (nếu nó độc lập)
# Tái định nghĩa một hàm _get_obs đơn giản hóa ở đây
# Vì hàm _get_obs trong env phụ thuộc vào trạng thái env

def _get_obs_at_t(t, soc, price_fc, pv_fc, ot_si, su_started, forecast_horizon=3):
    """
    Tái tạo logic của _get_obs() từ SmartHomeEnv
    nhưng với các giá trị xác định.
    (Tham chiếu: smart_home_env.py, dòng 72-87)
    """
    t_norm = t / max(1, T - 1)
    rho = price_fc[t]
    pv_now = pv_fc[t]

    # Lấy dự báo ngắn hạn
    forecast_prices = price_fc[t: min(t + forecast_horizon, T)]
    forecast_pv = pv_fc[t: min(t + forecast_horizon, T)]

    # Thêm padding nếu dự báo ngắn hơn horizon
    if len(forecast_prices) < forecast_horizon:
        forecast_prices = np.pad(forecast_prices, (0, forecast_horizon - len(forecast_prices)), 'edge')
    if len(forecast_pv) < forecast_horizon:
        forecast_pv = np.pad(forecast_pv, (0, forecast_horizon - len(forecast_pv)), 'edge')

    obs = [t_norm, rho, pv_now, soc]
    obs += forecast_prices.tolist()
    obs += forecast_pv.tolist()
    obs += ot_si  # Trạng thái năng lượng SI
    obs += [1.0 if s else 0.0 for s in su_started]  # Trạng thái bắt đầu SU

    # Cần đảm bảo độ dài obs khớp với observation_space
    # obs_len = 4 + 2*forecast_horizon + N_si + N_su
    # Hãy kiểm tra smart_home_env.py
    # Dòng 69: obs_len = 4 + self.N_si + self.N_su (SAI!)
    # Dòng 86: obs += self.Ot_si + [1.0 if s else 0.0 for s in self.su_started]
    # Dòng 85: obs += forecast_prices.tolist() + forecast_pv.tolist()
    # -> obs_len = 4 + 2*forecast_horizon + N_si + N_su

    # Mã gốc của bạn (smart_home_env.py) có thể có lỗi ở dòng 69
    # (obs_len không cộng 2*forecast_horizon)
    # Chúng ta sẽ trả về obs dựa trên logic ở dòng 84-87

    return np.array(obs, dtype=np.float32)


def run_expert_generation(num_episodes, pop_size, max_iter):
    """
    Hàm chính để tạo dữ liệu chuyên gia.
    """
    print(f"Bắt đầu tạo {num_episodes} kịch bản chuyên gia...")
    print(f"Cấu hình: {N_SU} SU, {N_SI} SI. Vector dim = {(N_SU + N_SI) * T}")
    print(f"LBWO: pop_size={pop_size}, max_iter={max_iter}\n")

    expert_trajectories = []

    for ep in range(num_episodes):
        start_time = time.time()
        print(f"--- Kịch bản {ep + 1}/{num_episodes} ---")

        # 1. Tạo kịch bản dự báo mới
        price_fc, pv_fc, human_load_fc = _generate_forecasts()

        # 2. Tạo hàm fitness (wrapper) cho kịch bản này
        def fitness_wrapper(X):
            return calculate_total_cost(X, price_fc, pv_fc, human_load_fc)

        # 3. Định nghĩa biên (bounds)
        dim = (N_SU + N_SI) * T
        lb = np.zeros(dim)
        ub = np.ones(dim)

        # 4. Chạy LBWO để tìm lịch trình X_optimal
        print("  Đang chạy LBWO để tìm lịch trình tối ưu...")
        lbwo_solver = LBWO(
            fitness_func=fitness_wrapper,
            dim=dim,
            lb=lb,
            ub=ub,
            pop_size=pop_size,
            max_iter=max_iter
        )
        X_optimal, best_cost = lbwo_solver.run()
        X_optimal = np.round(X_optimal).astype(int)  # Làm tròn kết quả
        print(f"  LBWO hoàn tất. Chi phí dự kiến tốt nhất: {best_cost:.2f}")

        # 5. Giải mã X_optimal thành các lịch trình
        schedules = {}
        start_idx = 0
        for i in range(N_SU):
            schedules[f'su_{i}'] = X_optimal[start_idx: start_idx + T]
            start_idx += T
        for i in range(N_SI):
            schedules[f'si_{i}'] = X_optimal[start_idx: start_idx + T]
            start_idx += T

        # 6. Chạy lại mô phỏng để ghi (state, action)
        print("  Đang trích xuất (state, action) ...")
        ep_data = []

        # Khởi tạo trạng thái mô phỏng (giống env.reset())
        SOC = cfg['battery']['soc0']
        Ot_si = [0.0] * N_SI
        su_started = [False] * N_SU
        su_remaining_track = [cfg['shiftable_su'][i]['L'] for i in range(N_SU)]

        for t in range(T):
            # 6a. Lấy STATE (obs) tại thời điểm t
            obs_t = _get_obs_at_t(t, SOC, price_fc, pv_fc, Ot_si, su_started)

            # 6b. Lấy ACTION (expert_action) tại thời điểm t
            action_t = []
            for i in range(N_SU):
                action_t.append(schedules[f'su_{i}'][t])
            for i in range(N_SI):
                action_t.append(schedules[f'si_{i}'][t])

            action_t = np.array(action_t, dtype=int)

            # 6c. Lưu trữ cặp (state, action)
            ep_data.append({
                "state": obs_t.tolist(),
                "action": action_t.tolist()
            })

            # 6d. Cập nhật trạng thái (SOC, Ot_si, su_started) cho t+1
            # (Chạy lại logic mô phỏng *tối giản* từ fitness function)

            P_su_t = 0.0
            for i in range(N_SU):
                if action_t[i] == 1 and su_remaining_track[i] > 0:
                    P_su_t += cfg['shiftable_su'][i]['rate']
                    su_remaining_track[i] -= 1
                    su_started[i] = True

            P_si_t = 0.0
            si_idx_offset = N_SU  # index của SI trong 'action_t'
            for i in range(N_SI):
                if action_t[si_idx_offset + i] == 1:
                    rate = cfg['shiftable_si'][i]['rate']
                    P_si_t += rate
                    Ot_si[i] += rate * 1.0  # Cập nhật năng lượng đã sạc

            P_load = human_load_fc[t] + P_su_t + P_si_t
            P_pv = pv_fc[t]

            # Cập nhật SOC (logic sao chép)
            P_ch, P_dis = 0.0, 0.0
            if P_pv >= P_load:
                P_surplus = P_pv - P_load
                if SOC < BAT_SOC_MAX:
                    P_ch = P_surplus
                    SOC = min(BAT_SOC_MAX, SOC + BAT_ETA_CH * P_ch / 24.0)
            else:
                P_deficit = P_load - P_pv
                if SOC > BAT_SOC_MIN:
                    P_dis = min(P_deficit, (SOC - BAT_SOC_MIN) * 24.0 / BAT_ETA_DIS)
                    SOC = max(BAT_SOC_MIN, SOC - P_dis * BAT_ETA_DIS / 24.0)

        expert_trajectories.append(ep_data)
        end_time = time.time()
        print(f"  Hoàn tất kịch bản. Thời gian: {end_time - start_time:.2f}s\n")

    # 7. Lưu toàn bộ dữ liệu
    output_filename = "expert_data.json"
    with open(output_filename, 'w') as f:
        json.dump(expert_trajectories, f, indent=2)

    print(f"=== TẤT CẢ HOÀN TẤT ===")
    print(f"Đã lưu {len(expert_trajectories)} kịch bản chuyên gia vào file: {output_filename}")


if __name__ == "__main__":
    # === CẤU HÌNH CHẠY ===

    # Số lượng kịch bản (ngày) bạn muốn tạo
    NUM_EPISODES = 5

    # Cấu hình LBWO (Rất quan trọng!)
    # Cần giữ ở mức thấp để chạy thử nghiệm
    # (pop_size=30, max_iter=50) có thể mất vài phút MỖI kịch bản.
    POP_SIZE = 20  # Số lượng 'cá voi'
    MAX_ITER = 40  # Số thế hệ

    # ---------------------

    run_expert_generation(
        num_episodes=NUM_EPISODES,
        pop_size=POP_SIZE,
        max_iter=MAX_ITER
    )