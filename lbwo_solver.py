# lbwo_solver.py
import numpy as np
import random
from smart_home_env import SmartHomeEnv


class LBWOSolver:
    """
    Leader Beluga Whale Optimization (LBWO) - Solver Mode.
    Sử dụng để tìm lịch trình tối ưu (Expert Action) cho một cấu hình môi trường cụ thể.
    """

    def __init__(self,
                 dim,
                 population_size=30,  # Tăng nhẹ để tìm kiếm tốt hơn
                 max_iter=50,
                 lb=-3.0,  # Logit bounds
                 ub=3.0):

        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.population_size = population_size
        self.max_iter = max_iter

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def evaluate_fitness(self, whale_vector, env_config):
        """
        Chạy mô phỏng nhanh (Fast Simulation) để tính Reward cho lịch trình này.
        """
        # Quan trọng: Tạo env mới với cùng config để đảm bảo tính nhất quán
        env = SmartHomeEnv(env_config)
        env.reset()  # Đảm bảo reset về trạng thái đầu ngày

        num_devices = len(env_config["behavior"]["shiftable_devices"])
        horizon = env.sim_steps

        total_reward = 0.0

        # Giải mã vector cá voi thành chuỗi hành động 24h
        actions_matrix = []
        for t in range(horizon):
            actions_t = []
            for dev_id in range(num_devices):
                idx = dev_id * horizon + t
                prob = self._sigmoid(whale_vector[idx])
                actions_t.append(1 if prob > 0.5 else 0)
            actions_matrix.append(actions_t)

        # Chạy step-by-step
        for t in range(horizon):
            obs, r, done, truncated, info = env.step(actions_matrix[t])

            # Phạt nặng nếu vi phạm ràng buộc (nếu SmartHomeEnv chưa phạt đủ)
            # Ví dụ: Nếu cần thiết có thể cộng thêm logic phạt ở đây
            total_reward += r

            if done: break

        return total_reward

    def solve(self, env_config):
        """
        Hàm chính: Tìm input tối ưu cho env_config đã cho.
        """
        # 1. Khởi tạo quần thể ngẫu nhiên
        whales = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.full(self.population_size, -np.inf)  # Reward càng cao càng tốt

        best_whale = None
        best_fitness = -np.inf

        # 2. Vòng lặp tối ưu
        for iter in range(self.max_iter):
            # Đánh giá fitness
            for i in range(self.population_size):
                fit = self.evaluate_fitness(whales[i], env_config)
                fitness[i] = fit

                if fit > best_fitness:
                    best_fitness = fit
                    best_whale = whales[i].copy()

            # --- LBWO Mechanism (Update positions) ---
            # Sắp xếp để tìm các Leader (trong LBWO gốc dùng 3 leader, ở đây dùng phiên bản WOA tiêu chuẩn + lai ghép đơn giản)

            a = 2 - 2 * (iter / self.max_iter)  # Giảm dần từ 2 xuống 0

            for i in range(self.population_size):
                r1 = random.random()
                r2 = random.random()
                A = 2 * a * r1 - a
                C = 2 * r2

                p = random.random()

                if p < 0.5:
                    if abs(A) < 1:
                        # Bao vây con mồi (Best Whale)
                        D = abs(C * best_whale - whales[i])
                        whales[i] = best_whale - A * D
                    else:
                        # Tìm kiếm ngẫu nhiên (Exploration)
                        rand_idx = random.randint(0, self.population_size - 1)
                        X_rand = whales[rand_idx]
                        D = abs(C * X_rand - whales[i])
                        whales[i] = X_rand - A * D
                else:
                    # Xoắn ốc (Spiral update)
                    distance_to_best = abs(best_whale - whales[i])
                    whales[i] = distance_to_best * np.exp(1 * 2 * np.pi * r1) * np.cos(2 * np.pi * r1) + best_whale

                # Kẹp giá trị trong biên
                whales[i] = np.clip(whales[i], self.lb, self.ub)

        # 3. Trả về lịch trình đã giải mã (Binary Schedule)
        # Convert best_whale (logits) -> Binary Schedule [Steps, Devices]
        final_schedule = []
        num_devices = len(env_config["behavior"]["shiftable_devices"])
        horizon = env_config["sim_steps"]

        for t in range(horizon):
            act_t = []
            for dev_id in range(num_devices):
                idx = dev_id * horizon + t
                prob = self._sigmoid(best_whale[idx])
                act_t.append(1 if prob > 0.5 else 0)
            final_schedule.append(act_t)

        return np.array(final_schedule)  # Shape: (24, Num_Devices)