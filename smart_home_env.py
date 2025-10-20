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
        self.cfg = config
        self.T = len(self.price)
        self.forecast_horizon = forecast_horizon
        # === MÔ PHỎNG THỜI TIẾT ===
        self.weather_states = ["sunny", "mild", "cloudy", "rainy", "stormy"]
        self.weather_factors = {
            "sunny": 1.0,
            "mild": 0.8,
            "cloudy": 0.5,
            "rainy": 0.3,
            "stormy": 0.1
        }
        # Ma trận chuyển Markov cho thời tiết
        self.weather_transition = {
            "sunny": [0.6, 0.25, 0.1, 0.04, 0.01],
            "mild": [0.2, 0.5, 0.2, 0.08, 0.02],
            "cloudy": [0.1, 0.2, 0.4, 0.2, 0.1],
            "rainy": [0.05, 0.1, 0.25, 0.4, 0.2],
            "stormy": [0.02, 0.08, 0.2, 0.3, 0.4]
        }

        # Thông số chung
        self.time_step = 1.0  # 1h mỗi bước
        self.total_cost = 0.0
        self.total_energy_bought = 0.0

        # Cấu hình tải
        self.N_ad = len(config.get('adjustable', []))
        self.N_su = len(config.get('shiftable_su', []))
        self.N_si = len(config.get('shiftable_si', []))

        # Không gian quan sát và hành động
        obs_len = 4 + self.N_si + self.N_su
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(obs_len,), dtype=np.float32)
        self.action_space = spaces.MultiBinary(self.N_su + self.N_si)

        self.reset()

    def reset(self):
        self.t = 0
        bat = self.cfg.get('battery', {})
        self.SOC = bat.get('soc0', 0.5)
        self.Ot_si = [0] * self.N_si
        self.su_started = [False] * self.N_su
        self.su_remaining = [self.cfg['shiftable_su'][i]['L'] for i in range(self.N_su)]
        self.total_cost = 0.0
        self.total_energy_bought = 0.0
        # === KHỞI TẠO DỮ LIỆU THỜI TIẾT ===
        self.current_weather = "mild"
        self.weather_series = []
        for t in range(self.T):
            probs = self.weather_transition[self.current_weather]
            self.current_weather = np.random.choice(self.weather_states, p=probs)
            self.weather_series.append(self.current_weather)
        return self._get_obs()

    def _get_obs(self):
        # Observation bao gồm dự báo ngắn hạn
        t_norm = self.t / max(1, self.T - 1)
        rho = self.price[self.t]
        pv_now = self.pv[self.t]

        # dự báo PV và giá (theo horizon)
        forecast_prices = self.price[self.t:min(self.t + self.forecast_horizon, self.T)]
        forecast_pv = self.pv[self.t:min(self.t + self.forecast_horizon, self.T)]

        obs = [t_norm, rho, pv_now, self.SOC]
        obs += forecast_prices.tolist() + forecast_pv.tolist()
        obs += self.Ot_si
        obs += [1.0 if s else 0.0 for s in self.su_started]
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        action = np.array(action).astype(int)
        act_su = action[:self.N_su].tolist() if self.N_su > 0 else []
        act_si = action[self.N_su:].tolist() if self.N_si > 0 else []

        # ===== 1. TÍNH TOÁN TẢI =====
        P_su_t = sum(su["rate"] for i, su in enumerate(self.cfg["shiftable_su"])
                     if self.t >= su["t_s"] and self.t <= su["t_f"] and act_su[i] == 1)
        P_si_t = sum(si["rate"] for i, si in enumerate(self.cfg["shiftable_si"])
                     if self.t >= si["t_s"] and self.t <= si["t_f"] and act_si[i] == 1)

        P_cr_t = self.cfg.get("critical", [0.0] * self.T)[self.t]
        P_ad_t = sum(ad["P_com"] for ad in self.cfg.get("adjustable", []))
        P_load = P_cr_t + P_ad_t + P_su_t + P_si_t
        P_pv = self.pv[self.t]

        # === TÁC ĐỘNG CỦA THỜI TIẾT LÊN PV ===
        weather = self.weather_series[self.t]
        weather_factor = self.weather_factors[weather]
        P_pv = self.pv[self.t] * weather_factor

        # ===== 2. XỬ LÝ PIN =====
        bat = self.cfg.get("battery", {})
        soc_min = bat.get("soc_min", 0.1)
        soc_max = bat.get("soc_max", 0.9)
        eta_ch = bat.get("eta_ch", 0.95)
        eta_dis = bat.get("eta_dis", 0.95)

        P_ch, P_dis = 0.0, 0.0
        if P_pv >= P_load:
            P_surplus = P_pv - P_load
            if self.SOC < soc_max:
                P_ch = P_surplus
                self.SOC = min(soc_max, self.SOC + eta_ch * P_ch / self.T)
        else:
            P_deficit = P_load - P_pv
            if self.SOC > soc_min:
                P_dis = min(P_deficit, (self.SOC - soc_min) * self.T / eta_dis)
                self.SOC = max(soc_min, self.SOC - P_dis * eta_dis / self.T)

        # ===== 3. CÂN BẰNG LƯỚI =====
        supply = P_pv + P_dis
        demand = P_load + P_ch
        self.P_grid = max(0, demand - supply)  # chỉ mua điện

        # ===== 4. REWARD ADVANCED =====
        price = self.price[self.t]
        cost = self.P_grid * price
        self.total_cost += cost
        self.total_energy_bought += self.P_grid * self.time_step

        hour = self.t % 24
        is_night = (hour < 6 or hour >= 18)
        penalty_unmet = -10.0 * max(0, demand - supply - self.P_grid)
        penalty_battery = -0.05 * (abs(P_ch) + abs(P_dis))

        # Reward khởi tạo
        reward = -cost + penalty_unmet + penalty_battery

        # Logic ban đêm
        if is_night:
            if self.P_grid > 0:
                reward -= 0.2 * cost  # phạt thêm nếu dùng điện lưới
            elif P_dis > 0:
                reward += 0.05 * P_dis  # thưởng nhẹ nếu dùng pin

        # Logic nâng cao: SOC, giờ cao điểm, tận dụng PV
        if 17 <= hour <= 21:
            reward -= 0.5 * cost  # phạt giờ cao điểm
        if 0.4 <= self.SOC <= 0.8:
            reward += 0.02  # thưởng SOC ổn định
        reward += 0.03 * min(P_pv, P_load)  # thưởng tận dụng PV

        info = {
            "P_pv": P_pv,
            "P_load": P_load,
            "P_ch": P_ch,
            "P_dis": P_dis,
            "P_grid": self.P_grid,
            "SOC": self.SOC,
            "cost": cost,
            "is_night": is_night,
            "weather": weather,
            "weather_factor": weather_factor
        }

        self.t += 1
        done = (self.t >= self.T)
        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, info

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
