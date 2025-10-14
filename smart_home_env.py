# smart_home_env.py
"""
Gym environment for Smart Home HEMS with hybrid scheme:
- agent controls shiftable loads (SU and SI)
- adjustable loads + buy/sell are optimized online using one-step MILP (pulp) or heuristic fallback
Usage:
    from smart_home_env import SmartHomeEnv
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pulp
except Exception:
    pulp = None
    print("WARNING: pulp not installed. _solve_one_step will use heuristic fallback.")

class SmartHomeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, price_profile, pv_profile, config, forecast_horizon=3):
        super().__init__()
        self.price = np.array(price_profile)
        self.pv = np.array(pv_profile)
        self.T = len(self.price)
        self.cfg = config
        self.forecast_horizon = forecast_horizon
        self.P_grid = 0.0
        self.time_step = 1.0
        self.total_energy_bought = 0.0
        self.N_ad = len(config.get('adjustable', []))
        self.N_su = len(config.get('shiftable_su', []))
        self.N_si = len(config.get('shiftable_si', []))

        bat = self.cfg.get('battery', {})
        self.C_bat = bat.get('C_bat', 10.0)

        obs_len = 4 + self.N_si + self.N_su + (2 * self.forecast_horizon)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)
        self.action_space = spaces.MultiBinary(self.N_su + self.N_si)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        bat = self.cfg.get('battery', {})
        self.SOC = bat.get('soc0', 0.5)
        self.Ot_si = [0] * self.N_si
        self.su_started = [False] * self.N_su
        self.su_remaining = [0] * self.N_su
        self.total_cost = 0.0
        self.total_energy_bought = 0.0

        return self._get_obs(), {}

    def _get_obs(self):
        current_t = min(self.t, self.T - 1)

        t_norm = current_t / max(1, self.T - 1)
        rho = self.price[current_t]
        pv = self.pv[current_t]

        obs = [t_norm, rho, pv, self.SOC]
        obs += self.Ot_si
        obs += [1.0 if s > 0 else 0.0 for s in self.su_remaining]

        # Lấy lát cắt dự báo
        future_prices = self.price[current_t + 1: current_t + 1 + self.forecast_horizon]
        future_pvs = self.pv[current_t + 1: current_t + 1 + self.forecast_horizon]

        pad_width_prices = self.forecast_horizon - len(future_prices)
        pad_width_pvs = self.forecast_horizon - len(future_pvs)

        # Nếu cần đệm, sử dụng mode='constant' với giá trị cuối cùng được biết
        if pad_width_prices > 0:
            future_prices = np.pad(future_prices, (0, pad_width_prices), mode='constant', constant_values=rho)

        if pad_width_pvs > 0:
            future_pvs = np.pad(future_pvs, (0, pad_width_pvs), mode='constant', constant_values=pv)


        obs.extend(future_prices)
        obs.extend(future_pvs)

        return np.array(obs, dtype=np.float32)

    def _get_tiered_price(self, total_kwh: float) -> float:
        """
        Tính giá điện trung bình theo kWh dựa trên biểu giá bậc thang.
        Hàm này được tách ra để tăng tính rõ ràng và hiệu quả.
        """
        tiers = [
            (50, 1984), (100, 2050), (200, 2380),
            (300, 2998), (400, 3350), (float('inf'), 3460)
        ]

        # Nếu không sử dụng điện, giá là bậc thấp nhất để tránh chia cho 0
        if total_kwh <= 0:
            return tiers[0][1] / 1000.0

        remaining_kwh, cost, last_limit = total_kwh, 0, 0
        for limit, price_vnd in tiers:
            usage_in_tier = min(remaining_kwh, limit - last_limit)
            cost += usage_in_tier * price_vnd
            remaining_kwh -= usage_in_tier
            last_limit = limit
            if remaining_kwh <= 0:
                break

        # Trả về giá trung bình, quy đổi từ VND sang đơn vị tiền tệ của môi trường (nếu cần)
        return (cost / total_kwh) / 1000.0

    def step(self, action: np.ndarray):
        """
        Thực hiện một bước trong môi trường, tính toán trạng thái tiếp theo và phần thưởng.
        """
        # ===== 1. XỬ LÝ HÀNH ĐỘNG VÀ TÍNH TẢI CƠ BẢN =====
        action = np.array(action).astype(int)
        act_su = action[:self.N_su]
        act_si = action[self.N_su:]

        # Tải SU (không thể ngắt quãng)
        P_su_t = 0.0
        for i in range(self.N_su):
            su = self.cfg["shiftable_su"][i]
            if self.su_remaining[i] > 0:
                P_su_t += su["rate"]
                self.su_remaining[i] -= 1
            elif not self.su_started[i] and act_su[i] == 1 and su["t_s"] <= self.t <= su["t_f"]:
                P_su_t += su["rate"]
                self.su_started[i] = True
                self.su_remaining[i] = su["L"] - 1

        # Tải SI (có thể ngắt quãng)
        P_si_t = 0.0
        for i in range(self.N_si):
            si = self.cfg["shiftable_si"][i]
            if act_si[i] == 1 and si["t_s"] <= self.t <= si["t_f"]:
                P_si_t += si["rate"]
                self.Ot_si[i] += 1

        # Tổng tải cơ bản (chưa bao gồm tải điều chỉnh được)
        P_cr_t = self.cfg.get("critical", [0.0] * self.T)[self.t]
        P_base_load = P_cr_t + P_su_t + P_si_t

        # ===== 2. CÂN BẰNG NĂNG LƯỢNG SƠ BỘ (PV & PIN) =====
        P_pv = self.pv[self.t]
        P_net_after_pv = P_base_load - P_pv

        # Lấy thông số pin
        bat_cfg = self.cfg.get("battery", {})
        soc_min, soc_max = bat_cfg.get("soc_min", 0.1), bat_cfg.get("soc_max", 0.9)
        eta_ch, eta_dis = bat_cfg.get("eta_ch", 0.95), bat_cfg.get("eta_dis", 0.95)
        P_ch, P_dis = 0.0, 0.0

        if P_net_after_pv < 0:  # TH1: Thừa năng lượng mặt trời -> Sạc pin
            P_to_charge = -P_net_after_pv
            max_charge_power = (soc_max - self.SOC) * self.C_bat / (eta_ch * self.time_step)
            P_ch = min(P_to_charge, max_charge_power)
            self.SOC += P_ch * eta_ch * self.time_step / self.C_bat if self.C_bat > 0 else 0
        else:  # TH2: Thiếu năng lượng mặt trời -> Xả pin
            P_to_discharge = P_net_after_pv
            max_discharge_power = (self.SOC - soc_min) * self.C_bat * eta_dis / self.time_step
            P_dis = min(P_to_discharge, max_discharge_power)
            self.SOC -= P_dis / eta_dis * self.time_step / self.C_bat if self.C_bat > 0 and eta_dis > 0 else 0

        # Công suất ròng còn lại sau khi đã dùng PV và pin
        P_net_after_battery = P_net_after_pv + P_ch - P_dis

        # ===== 3. TỐI ƯU HÓA TẢI ĐIỀU CHỈNH & LƯỚI ĐIỆN =====
        # Gọi bộ giải để quyết định công suất cho tải điều chỉnh và lượng điện mua từ lưới
        P_ad_list, P_b, _ = self._solve_one_step(P_net_after_battery)
        self.P_grid = P_b if P_b is not None else 0.0
        P_ad_t = sum(P_ad_list)
        P_total_load = P_base_load + P_ad_t

        # ===== 4. TÍNH TOÁN CHI PHÍ VÀ PHẦN THƯỞNG =====
        energy_bought = self.P_grid * self.time_step
        self.total_energy_bought += energy_bought
        price_per_kwh = self._get_tiered_price(self.total_energy_bought)
        cost = energy_bought * price_per_kwh
        self.total_cost += cost

        # Phạt nếu không đáp ứng đủ tải
        unmet_load = max(0, P_net_after_battery + P_ad_t - self.P_grid)

        # Tính toán phần thưởng dựa trên chế độ đã chọn
        mode = self.cfg.get("reward_mode", "balanced")
        penalty_battery = -0.05 * (abs(P_ch) + abs(P_dis))
        penalty_unmet = -10.0 * unmet_load
        reward = -cost + penalty_battery + penalty_unmet

        if mode == "advanced":
            pv_used = min(P_pv, P_total_load + P_ch)
            reward += 0.03 * pv_used  # Thưởng tận dụng PV
            if 17 <= self.t % 24 <= 21:
                reward -= 0.5 * cost  # Phạt thêm giờ cao điểm
            if 0.4 <= self.SOC <= 0.8:
                reward += 0.01  # Thưởng giữ SOC ổn định

        # ===== 5. CẬP NHẬT TRẠNG THÁI VÀ TRẢ VỀ =====
        self.t += 1
        done = (self.t >= self.T)
        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "P_pv": P_pv, "P_load": P_total_load, "P_grid": self.P_grid,
            "P_ch": P_ch, "P_dis": P_dis, "SOC": self.SOC, "cost": cost,
            "unmet_load": unmet_load, "price_per_kwh": price_per_kwh
        }

        return obs, reward, done, False, info

    def _solve_one_step(self, P_net_without_ad):
        # Heuristic fallback if pulp missing
        rho_b = self.price[self.t]
        if pulp is None or self.N_ad == 0:
            P_ad = []
            for ad in self.cfg.get('adjustable', []):
                P_ad.append(ad['P_com'] if rho_b < np.mean(self.price) else ad['P_min'])
            P_ad = np.array(P_ad)
            P_net = P_net_without_ad + P_ad.sum()
            if P_net >= 0:
                return P_ad.tolist(), float(P_net), 0.0
            else:
                return P_ad.tolist(), 0.0, float(-P_net)

        # build small LP with pulp
        prob = pulp.LpProblem('OneStep', pulp.LpMinimize)
        P_ad_vars = [pulp.LpVariable(f'P_ad_{i}', lowBound=ad['P_min'], upBound=ad['P_max']) for i,ad in enumerate(self.cfg.get('adjustable', []))]
        P_b_var = pulp.LpVariable('P_b', lowBound=0.0)
        P_s_var = pulp.LpVariable('P_s', lowBound=0.0)
        z_b = pulp.LpVariable('z_b', cat='Binary')
        z_s = pulp.LpVariable('z_s', cat='Binary')
        bigM = 1e5
        prob += (P_b_var - P_s_var == P_net_without_ad + pulp.lpSum(P_ad_vars))
        prob += P_b_var <= z_b * bigM
        prob += P_s_var <= z_s * bigM
        prob += z_b + z_s <= 1
        obj = rho_b * P_b_var - (self.cfg.get('beta',0.5)*rho_b) * P_s_var
        for i,ad in enumerate(self.cfg.get('adjustable', [])):
            u = pulp.LpVariable(f'u_{i}', lowBound=0.0)
            prob += u >= ad['P_com'] - P_ad_vars[i]
            prob += u >= P_ad_vars[i] - ad['P_com']
            obj += ad['alpha'] * u
            obj += rho_b * P_ad_vars[i]
        prob += obj
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=5)
        prob.solve(solver)
        P_ad = [float(pulp.value(v)) if pulp.value(v) is not None else 0.0 for v in P_ad_vars]
        P_b = float(pulp.value(P_b_var)) if pulp.value(P_b_var) is not None else 0.0
        P_s = float(pulp.value(P_s_var)) if pulp.value(P_s_var) is not None else 0.0
        return P_ad, P_b, P_s

    def get_tiered_price(total_consumption_kwh):
        # Giá điện theo bậc (đồng/kWh)
        tiers = [
            (50, 1984),
            (100, 2050),
            (200, 2380),
            (300, 2998),
            (400, 3350),
            (float('inf'), 3460),
        ]

        # Tính toán giá trung bình (theo mức dùng tích lũy)
        remaining = total_consumption_kwh
        cost = 0
        last_limit = 0

        for limit, price in tiers:
            usage = min(remaining, limit - last_limit)
            cost += usage * price
            remaining -= usage
            last_limit = limit
            if remaining <= 0:
                break

        avg_price = cost / total_consumption_kwh if total_consumption_kwh > 0 else tiers[0][1]
        return avg_price / 1000  # đổi sang kWh → đồng/kWh (nếu cần scale)

    def render(self, mode='human'):
        print(f"t={self.t}, total_cost={self.total_cost:.3f}")


if __name__ == "__main__":
    # quick demo
    T = 24
    price = 0.1 + 0.2 * np.random.rand(T)
    pv = np.clip(1.5 * np.sin(np.linspace(0, 3.14, T)) + 0.2*np.random.randn(T), 0, None)
    config = {
        'critical': [0.3]*T,
        'adjustable': [
            {'P_min':0.1, 'P_max':1.5, 'P_com':1.2, 'alpha':0.06},
            {'P_min':0.0, 'P_max':1.2, 'P_com':1.0, 'alpha':0.12}
        ],
        'shiftable_su': [ {'rate':0.5, 'L':2, 't_s':6, 't_f':20}, {'rate':0.6, 'L':1, 't_s':8, 't_f':22} ],
        'shiftable_si': [ {'rate':1.0, 'E':4.0, 't_s':0, 't_f':23} ],
        'beta': 0.5,
        'battery': {'soc0':0.5, 'soc_min':0.1, 'soc_max':0.9},
        "reward_mode": "advanced"
    }
    env = SmartHomeEnv(price, pv, config)
    obs = env.reset()
    done = False
    while not done:
        action = np.random.randint(0,2, size=env.N_su + env.N_si)
        obs, rew, done, info = env.step(action)
    print("Episode finished, total cost", env.total_cost)
