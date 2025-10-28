# Tên file: lbwo_optimizer.py
# Mô tả: Triển khai thuật toán Beluga Whale Optimization (BWO)
# và Leader BWO (LBWO) dựa trên bài báo PDF.

import numpy as np

try:
    # Cần thư viện scipy để tính hàm Gamma cho Levy flight
    from scipy.special import gamma
except ImportError:
    print("Cảnh báo: Thư viện 'scipy' không được cài đặt.")
    print("Hàm Levy flight (sử dụng trong BWO/LBWO) sẽ không hoạt động.")
    print("Vui lòng cài đặt: pip install scipy")
    gamma = None


class BWO:
    """
    Triển khai thuật toán Beluga Whale Optimization (BWO) cơ bản.

    Dựa trên Phần IV của bài báo.
    """

    def __init__(self, fitness_func, dim, lb, ub, pop_size, max_iter):
        """
        Khởi tạo trình tối ưu hóa BWO.

        Args:
            fitness_func (function): Hàm mục tiêu (fitness) để tối ưu hóa.
                                     Phải nhận 1 vector giải pháp và trả về 1 số (cost).
            dim (int): Số chiều của vector giải pháp.
            lb (float or array): Giới hạn dưới của các biến.
            ub (float or array): Giới hạn trên của các biến.
            pop_size (int): Kích thước quần thể (số lượng 'cá voi').
            max_iter (int): Số lần lặp (thế hệ) tối đa.
        """
        self.fitness_func = fitness_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter

        # Đảm bảo lb và ub là các mảng numpy
        self.lb = np.array(lb) * np.ones(dim)
        self.ub = np.array(ub) * np.ones(dim)

        # Khởi tạo quần thể
        self.pop = np.zeros((self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)

        # Lưu trữ giải pháp tốt nhất toàn cục
        self.g_best_sol = np.zeros(self.dim)
        self.g_best_fit = np.inf

        # Beta cho Levy flight
        self.beta = 1.5  # Giá trị hằng số được định nghĩa trong bài báo [cite: 505]
        self._sigma = self._calculate_sigma(self.beta)

    def _calculate_sigma(self, beta):
        """Tính hằng số sigma cho Levy flight, dựa trên Equation (40)[cite: 502]."""
        if gamma is None:
            return 0.0  # Trả về giá trị mặc định nếu scipy không tồn tại

        num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
        den = gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        return (num / den) ** (1 / beta)

    def _calculate_levy(self):
        """Tính một bước Levy flight, dựa trên Equation (39)[cite: 501]."""
        u = np.random.normal(0, self._sigma)
        v = np.random.normal(0, 1)
        step = u / (np.abs(v) ** (1 / self.beta))
        return 0.05 * step  # [cite: 501]

    def _clip_solution(self, solution):
        """Đảm bảo giải pháp nằm trong giới hạn [lb, ub]."""
        return np.clip(solution, self.lb, self.ub)

    def _exploration_phase(self, i, r_idx, r1, r2):
        """
        Tính toán vị trí mới dựa trên Giai đoạn Khám phá (Exploration).

        LƯU Ý: Bài báo có sự không nhất quán.
        - Text [cite: 480] đề cập đến 'sin(2*pi*r2)' và 'cos(2*pi*r2)'.
        - Text [cite: 481] đề cập đến "even and odd numbers" (số chẵn và lẻ).
        - Tuy nhiên, Equation (37) [cite: 479] được viết 2 lần *giống hệt nhau*
          và *không* chứa sin/cos.

        Do đó, chúng tôi sẽ triển khai một phiên bản 'hòa giải' (reconciled)
        sử dụng sin/cos dựa trên gợi ý từ văn bản[cite: 480, 481].
        """
        X_new = self.pop[i].copy()

        # Chọn các chiều ngẫu nhiên như trong text [cite: 480]
        p_j = np.random.randint(self.dim)
        p_1 = np.random.randint(self.dim)

        for j in range(self.dim):
            # Dựa trên gợi ý "even and odd" [cite: 481]
            if j % 2 == 1:  # Số lẻ
                X_new[j] = self.pop[i, p_j] + (self.pop[r_idx, p_1] - self.pop[i, p_j]) \
                           * (1 + r1) * np.sin(2 * np.pi * r2)
            else:  # Số chẵn
                X_new[j] = self.pop[i, p_j] + (self.pop[r_idx, p_1] - self.pop[i, p_j]) \
                           * (1 + r1) * np.cos(2 * np.pi * r2)
        return X_new

    def _exploitation_phase(self, i, r_idx, T, x_best):
        """
        Tính toán vị trí mới dựa trên Giai đoạn Khai thác (Exploitation).
        Dựa trên Equation (38)[cite: 495].
        """
        r3, r4 = np.random.rand(), np.random.rand()  # [cite: 496]

        C1 = 2 * r4 * (1 - (T + 1) / self.max_iter)  # [cite: 496]
        LF = self._calculate_levy()  # [cite: 501]

        X_new = r3 * x_best - r4 * self.pop[i] + C1 * LF * (self.pop[r_idx] - self.pop[i])  # [cite: 495]
        return X_new

    def _whale_fall_phase(self, i, r_idx, T, C2):
        """
        Tính toán vị trí mới dựa trên Giai đoạn Whale Fall.
        Dựa trên Equation (41) và (42)[cite: 517, 519].
        """
        r5, r6, r7 = np.random.rand(3)  # [cite: 518, 521]

        # Eq (42): Tính toán step size [cite: 519]
        step_size = (self.ub - self.lb) * np.exp(-C2 * (T + 1) / self.max_iter)

        # Eq (41): Cập nhật vị trí [cite: 517]
        X_new = r5 * self.pop[i] - r6 * self.pop[r_idx] + r7 * step_size
        return X_new

    def run(self):
        """Chạy thuật toán tối ưu hóa BWO."""

        # 1. Khởi tạo quần thể (Step 1) [cite: 534, 536]
        for i in range(self.pop_size):
            self.pop[i] = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
            self.fitness[i] = self.fitness_func(self.pop[i])

        # Tìm giải pháp tốt nhất ban đầu [cite: 562]
        best_idx = np.argmin(self.fitness)
        self.g_best_fit = self.fitness[best_idx]
        self.g_best_sol = self.pop[best_idx].copy()

        # 2. Vòng lặp chính (Main loop)
        for T in range(self.max_iter):
            # Tính toán các yếu tố kiểm soát
            # Wf: Whale fall probability, Eq (43) [cite: 525]
            Wf = 0.1 - 0.05 * (T + 1) / self.max_iter

            # B0: Yếu tố cân bằng ngẫu nhiên, Eq (36) [cite: 491]
            B0 = np.random.rand()

            # C2: Yếu tố bước Whale fall, Eq (42) [cite: 523]
            C2 = 2 * Wf * self.pop_size

            # Lấy giải pháp tốt nhất của thế hệ hiện tại
            x_best_iter = self.g_best_sol.copy()

            for i in range(self.pop_size):
                # 3. Cập nhật pha (Step 2) [cite: 537]

                # Bf: Yếu tố cân bằng, Eq (36) [cite: 490]
                Bf = B0 * (1 - (T + 1) / (2 * self.max_iter))

                # Chọn một cá voi ngẫu nhiên [cite: 480, 495, 517]
                r_idx = np.random.randint(self.pop_size)
                while r_idx == i:
                    r_idx = np.random.randint(self.pop_size)

                X_new = self.pop[i].copy()

                # --- Các pha BWO ---
                if Bf > 0.5:  # 3.1 Giai đoạn Khám phá [cite: 539]
                    r1, r2 = np.random.rand(), np.random.rand()
                    X_new = self._exploration_phase(i, r_idx, r1, r2)
                else:  # 3.2 Giai đoạn Khai thác [cite: 540]
                    X_new = self._exploitation_phase(i, r_idx, T, x_best_iter)

                # 4. Giai đoạn Whale Fall (Step 3) [cite: 542, 572]
                if Bf <= Wf:  # Logic từ Flowchart (Fig. 6) [cite: 572]
                    X_new = self._whale_fall_phase(i, r_idx, T, C2)

                # --- Đánh giá và Lựa chọn ---
                X_new = self._clip_solution(X_new)
                fit_new = self.fitness_func(X_new)

                if fit_new < self.fitness[i]:
                    self.pop[i] = X_new
                    self.fitness[i] = fit_new

                    if fit_new < self.g_best_fit:
                        self.g_best_fit = fit_new
                        self.g_best_sol = X_new.copy()

            # print(f"Iter {T+1}/{self.max_iter}, Best Cost: {self.g_best_fit}")

        # 5. Trả về kết quả (Step 4) [cite: 545]
        return self.g_best_sol, self.g_best_fit


class LBWO(BWO):
    """
    Triển khai thuật toán Leader Beluga Whale Optimization (LBWO).

    Kế thừa từ BWO và thêm cơ chế "Leader-based mutation-selection".
    Dựa trên Phần V của bài báo.
    """

    def __init__(self, fitness_func, dim, lb, ub, pop_size, max_iter):
        super().__init__(fitness_func, dim, lb, ub, pop_size, max_iter)

    def run(self):
        """Chạy thuật toán tối ưu hóa LBWO."""

        # 1. Khởi tạo quần thể [cite: 534, 536]
        for i in range(self.pop_size):
            self.pop[i] = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
            self.fitness[i] = self.fitness_func(self.pop[i])

        # Tìm giải pháp tốt nhất ban đầu [cite: 562]
        best_idx = np.argmin(self.fitness)
        self.g_best_fit = self.fitness[best_idx]
        self.g_best_sol = self.pop[best_idx].copy()

        # 2. Vòng lặp chính (Main loop)
        for T in range(self.max_iter):
            # Tính toán các yếu tố kiểm soát (như BWO)
            Wf = 0.1 - 0.05 * (T + 1) / self.max_iter  # [cite: 525]
            B0 = np.random.rand()  # [cite: 491]
            C2 = 2 * Wf * self.pop_size  # [cite: 523]

            # === PHẦN BỔ SUNG CỦA LBWO ===
            # Tìm 3 "leader" tốt nhất (best, best-1, best-2) [cite: 597]
            sorted_indices = np.argsort(self.fitness)
            x_best = self.pop[sorted_indices[0]]
            x_best_1 = self.pop[sorted_indices[1]]
            x_best_2 = self.pop[sorted_indices[2]]
            # ===============================

            # Tạo quần thể mới để lưu trữ kết quả
            new_pop = self.pop.copy()
            new_fitness = self.fitness.copy()

            for i in range(self.pop_size):
                # 3. Cập nhật pha BWO (Step 2) [cite: 537]
                Bf = B0 * (1 - (T + 1) / (2 * self.max_iter))  # [cite: 490]

                r_idx = np.random.randint(self.pop_size)
                while r_idx == i:
                    r_idx = np.random.randint(self.pop_size)

                # Áp dụng các pha BWO để có giải pháp x_i(new)
                if Bf > 0.5:  # 3.1 Khám phá [cite: 539]
                    r1, r2 = np.random.rand(), np.random.rand()
                    X_bwo_step = self._exploration_phase(i, r_idx, r1, r2)
                else:  # 3.2 Khai thác [cite: 540]
                    # Sử dụng x_best tìm được ở đầu vòng lặp
                    X_bwo_step = self._exploitation_phase(i, r_idx, T, x_best)

                    # 4. Giai đoạn Whale Fall (Step 3) [cite: 542, 572]
                if Bf <= Wf:  # [cite: 572]
                    X_bwo_step = self._whale_fall_phase(i, r_idx, T, C2)

                X_bwo_step = self._clip_solution(X_bwo_step)
                # Lưu ý: fit_bwo_step được tính sau (nếu cần)

                # === 5. PHA ĐỘT BIẾN LBWO ===
                # Dựa trên Flowchart (Fig. 7) [cite: 668] và Equation (44) [cite: 600]

                rand1, rand2 = np.random.rand(), np.random.rand()

                # Hệ số giảm dần
                factor1 = 2 * (1 - (T + 1) / self.max_iter)

                # Thành phần thứ nhất của đột biến
                term1 = factor1 * (2 * rand1 - 1) * (2 * x_best - (x_best_1 + x_best_2))

                # Thành phần thứ hai của đột biến
                term2 = (2 * rand2 - 1) * (x_best - X_bwo_step)

                # Equation (44): Tính giải pháp đột biến x_i(mut) [cite: 600]
                X_mut = X_bwo_step + term1 + term2

                X_mut = self._clip_solution(X_mut)

                # === 6. CHỌN LỌC LBWO ===
                # Dựa trên Equation (45) [cite: 604] và Fig. 7 [cite: 664]

                # Đánh giá fitness của cả hai ứng cử viên
                fit_bwo_step = self.fitness_func(X_bwo_step)
                fit_mut = self.fitness_func(X_mut)

                # Chọn giải pháp tốt hơn
                if fit_mut < fit_bwo_step:
                    new_pop[i] = X_mut
                    new_fitness[i] = fit_mut
                else:
                    new_pop[i] = X_bwo_step
                    new_fitness[i] = fit_bwo_step

            # Cập nhật toàn bộ quần thể cho thế hệ tiếp theo
            self.pop = new_pop
            self.fitness = new_fitness

            # Cập nhật giải pháp tốt nhất toàn cục (g_best)
            best_idx_iter = np.argmin(self.fitness)
            if self.fitness[best_idx_iter] < self.g_best_fit:
                self.g_best_fit = self.fitness[best_idx_iter]
                self.g_best_sol = self.pop[best_idx_iter].copy()

            # print(f"Iter {T+1}/{self.max_iter}, Best Cost: {self.g_best_fit}")

        # 7. Trả về kết quả (Step 4) [cite: 545]
        return self.g_best_sol, self.g_best_fit


# --- VÍ DỤ SỬ DỤNG ---
if __name__ == "__main__":
    # 1. Định nghĩa một hàm mục tiêu (fitness function) đơn giản
    # Ví dụ: Hàm Sphere (mục tiêu là tìm X=[0,0,...,0])
    def sphere_function(X):
        return np.sum(X ** 2)


    # 2. Định nghĩa các tham số của bài toán
    DIMENSIONS = 10  # Số chiều (ví dụ: số biến cần tối ưu)
    LOWER_BOUND = -10  # Giới hạn dưới
    UPPER_BOUND = 10  # Giới hạn trên
    POP_SIZE = 30  # Kích thước quần thể
    MAX_ITER = 100  # Số thế hệ

    print("Đang chạy BWO (Beluga Whale Optimization)...")

    # 3. Khởi tạo và chạy BWO
    bwo_solver = BWO(
        fitness_func=sphere_function,
        dim=DIMENSIONS,
        lb=LOWER_BOUND,
        ub=UPPER_BOUND,
        pop_size=POP_SIZE,
        max_iter=MAX_ITER
    )

    best_solution_bwo, best_fitness_bwo = bwo_solver.run()

    print("\n--- Kết quả BWO ---")
    print(f"Fitness (Cost) tốt nhất: {best_fitness_bwo}")
    # print(f"Giải pháp tốt nhất: {best_solution_bwo}")

    print("\n" + "=" * 30 + "\n")

    print("Đang chạy LBWO (Leader Beluga Whale Optimization)...")

    # 4. Khởi tạo và chạy LBWO
    lbwo_solver = LBWO(
        fitness_func=sphere_function,
        dim=DIMENSIONS,
        lb=LOWER_BOUND,
        ub=UPPER_BOUND,
        pop_size=POP_SIZE,
        max_iter=MAX_ITER
    )

    best_solution_lbwo, best_fitness_lbwo = lbwo_solver.run()

    print("\n--- Kết quả LBWO ---")
    print(f"Fitness (Cost) tốt nhất: {best_fitness_lbwo}")
    # print(f"Giải pháp tốt nhất: {best_solution_lbwo}")