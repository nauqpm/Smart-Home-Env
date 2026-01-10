import numpy as np
import math
from copy import deepcopy

class LBWOOptimizer:
    def __init__(self, n_whales=30, max_iter=100, n_vars=24, 
                 lb=-3.0, ub=3.0, 
                 soc_min=1.0, soc_max=9.0, initial_soc=5.0, ess_capacity=10.0,
                 eta_ch=0.95, eta_dis=0.95):
        """
        Tham số:
        - n_whales: Số lượng cá voi (Population).
        - max_iter: Số vòng lặp tối đa.
        - n_vars: Số biến (24 giờ).
        - lb, ub: Giới hạn công suất sạc/xả (kW). (+) là sạc, (-) là xả.
        - soc_min, soc_max, ess_capacity: Thông số pin (kWh).
        - eta_ch, eta_dis: Hiệu suất sạc/xả.
        """
        self.n_whales = n_whales
        self.max_iter = max_iter
        self.n_vars = n_vars
        self.lb = lb
        self.ub = ub
        
        # Thông số môi trường để tính Fitness
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.initial_soc = initial_soc
        self.ess_capacity = ess_capacity
        self.eta_ch = eta_ch
        self.eta_dis = eta_dis
        
        # Dữ liệu môi trường (sẽ được set mỗi ngày)
        self.prices_buy = None
        self.prices_sell = None
        self.load = None
        self.pv = None
        self.wind = None

        # Khởi tạo quần thể
        self.positions = np.random.uniform(self.lb, self.ub, (n_whales, n_vars))
        self.fitness = np.zeros(n_whales)
        
        # Global Best
        self.best_whale = np.zeros(n_vars)
        self.best_fitness = float('inf')

    def set_environment_data(self, prices_buy, prices_sell, load, pv, wind):
        """Cập nhật dữ liệu dự báo cho ngày hiện tại (24h)"""
        self.prices_buy = np.array(prices_buy)
        self.prices_sell = np.array(prices_sell)
        self.load = np.array(load)
        self.pv = np.array(pv)
        self.wind = np.array(wind)
        
        # Tính lại fitness cho quần thể hiện tại với data mới
        for i in range(self.n_whales):
            self.fitness[i] = self.calculate_fitness(self.positions[i])
        
        self.sort_population()
        self.best_whale = deepcopy(self.positions[0])
        self.best_fitness = self.fitness[0]

    def calculate_fitness(self, schedule):
        """
        Hàm mục tiêu: Tính tổng chi phí điện (Cost) + Phạt vi phạm (Penalty).
        schedule: mảng 24 phần tử (Action sạc/xả).
        """
        total_cost = 0
        current_soc = self.initial_soc
        penalty = 0
        
        for t in range(self.n_vars):
            action = schedule[t] # kW (+ sạc, - xả)
            
            # 1. Cập nhật SOC (Eq. 30)
            if action >= 0: # Sạc
                energy_change = action * self.eta_ch
            else: # Xả
                energy_change = action * self.eta_dis # Fixed: Discharging reduces SOC. If action is negative, energy_change should be negative effectively.
                # Wait, standard logic:
                # If action > 0 (Charge): SOC increases by action * eta_ch * dt
                # If action < 0 (Discharge): SOC decreases by |action| / eta_dis * dt
                pass

            # CORRECT SOC LOGIC based on physics
            # action is Power (kW). dt = 1h.
            dt = 1.0
            
            if action >= 0: 
                # Charging
                soc_change = (action * dt * self.eta_ch)  # kWh added
            else:
                # Discharging (action is negative)
                # Energy removed from battery = Output Energy / eta_dis
                # e.g. Output -1kW. Removed = 1 / 0.95 = 1.05 kWh
                soc_change = (action * dt) / self.eta_dis # Negative value
            
            next_soc = current_soc + soc_change
            
            # 2. Kiểm tra ràng buộc SOC (Eq. 31-33)
            # Nếu vi phạm, cộng phạt cực lớn để cá voi tránh xa
            if next_soc < self.soc_min:
                penalty += 1000 * (self.soc_min - next_soc)
                next_soc = self.soc_min # Clamping ảo để tính tiếp
            elif next_soc > self.soc_max:
                penalty += 1000 * (next_soc - self.soc_max)
                next_soc = self.soc_max
            
            current_soc = next_soc

            # 3. Tính cân bằng năng lượng lưới (Net Grid)
            # P_grid = Load - PV - Wind + P_ess
            # P_ess = action (dương là sạc = tải thêm, âm là xả = nguồn phát)
            p_grid = self.load[t] - self.pv[t] - self.wind[t] + action
            
            # 4. Tính chi phí (Eq. 11, 13)
            if p_grid >= 0: # Mua điện
                cost = p_grid * self.prices_buy[t]
            else: # Bán điện
                cost = p_grid * self.prices_sell[t]
            
            total_cost += cost

        return total_cost + penalty

    def sort_population(self):
        indices = np.argsort(self.fitness)
        self.positions = self.positions[indices]
        self.fitness = self.fitness[indices]

    def levy_flight(self, beta=1.5):
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(self.n_vars) * sigma
        v = np.random.randn(self.n_vars)
        step = u / (np.abs(v) ** (1 / beta))
        return 0.05 * step

    def optimize(self):
        """Chạy thuật toán LBWO để tìm lịch trình tối ưu cho 24h"""
        T = 0
        while T < self.max_iter:
            # Tham số động
            B0 = np.random.rand()
            Bf = B0 * (1 - T / (2 * self.max_iter)) # Balance factor (Eq. 36)
            Wf = 0.1 - 0.05 * (T / self.max_iter)   # Whale fall prob (Eq. 43)
            
            new_positions = deepcopy(self.positions)
            
            # Lấy 3 Leader tốt nhất cho LBWO Mutation
            self.sort_population()
            x_best = self.positions[0]
            x_best_1 = self.positions[1]
            x_best_2 = self.positions[2]

            for i in range(self.n_whales):
                r1, r2, r3, r4, r5, r6, r7 = np.random.rand(7)
                r_idx = np.random.randint(0, self.n_whales)
                
                # --- Phase 1: Exploration vs Exploitation ---
                if Bf > 0.5: # Exploration (Swimming) - Eq. 37
                    if i % 2 == 0:
                         new_positions[i] = self.positions[i] + (self.positions[r_idx] - self.positions[i]) * (1 + r1)
                    else:
                         new_positions[i] = self.positions[i] + (self.positions[r_idx] - self.positions[i]) * (1 + r1) * math.sin(2 * math.pi * r2)
                else: # Exploitation (Hunting) - Eq. 38
                    C1 = 2 * r4 * (1 - T / self.max_iter)
                    LF = self.levy_flight()
                    new_positions[i] = r3 * x_best - r4 * self.positions[i] + C1 * LF * (self.positions[r_idx] - self.positions[i])
                
                # --- Phase 2: Whale Fall --- (Eq. 41)
                if Bf <= Wf:
                    C2 = 2 * self.n_whales * Wf
                    X_step = (self.ub - self.lb) * math.exp(-C2 * T / self.max_iter)
                    new_positions[i] = r5 * new_positions[i] - r6 * self.positions[r_idx] + r7 * X_step

                # Clip giá trị
                new_positions[i] = np.clip(new_positions[i], self.lb, self.ub)

                # --- Phase 3: LBWO Leader-based Mutation --- (Eq. 44)
                # Đây là phần cải tiến quan trọng của LBWO so với BWO
                term2 = 2 * (1 - T/self.max_iter) * (2 * np.random.rand() - 1)
                term3 = (2 * x_best - (x_best_1 + x_best_2))
                term4 = (2 * np.random.rand() - 1) * (x_best - new_positions[i])
                
                x_mut = new_positions[i] + term2 * term3 + term4
                x_mut = np.clip(x_mut, self.lb, self.ub)
                
                # Greedy Selection (Eq. 45)
                fit_new = self.calculate_fitness(new_positions[i])
                fit_mut = self.calculate_fitness(x_mut)
                
                if fit_mut < fit_new:
                    new_positions[i] = x_mut
                    current_fit = fit_mut
                else:
                    current_fit = fit_new

                # Cập nhật cá thể
                if current_fit < self.fitness[i]:
                    self.positions[i] = new_positions[i]
                    self.fitness[i] = current_fit

            # Cập nhật Global Best
            min_idx = np.argmin(self.fitness)
            if self.fitness[min_idx] < self.best_fitness:
                self.best_fitness = self.fitness[min_idx]
                self.best_whale = deepcopy(self.positions[min_idx])
            
            T += 1
            
        return self.best_whale
