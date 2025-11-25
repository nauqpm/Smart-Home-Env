"""
Fixed SmartHomeEnv compatible with Gymnasium / Stable-Baselines3.
- reset(self, seed=None, options=None) follows Gymnasium API
- step(...) returns (obs, reward, terminated, truncated, info)
- get_tiered_price is a staticmethod
- small bugfixes: info definition in reset, obs shape checks

Lưu: file này chỉ sửa API và vài lỗi nhỏ; logic reward/battery/behavior giữ nguyên như bản gốc.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from human_behavior import HumanBehavior  # đảm bảo file human_behavior.py có mặt

try:
    import pulp
except Exception:
    pulp = None
    print("WARNING: pulp not installed. _solve_one_step will use heuristic fallback.")

DEVICE_POWER_MAP = {
    "lights": 0.1, "fridge": 0.2, "tv": 0.15, "ac": 1.5, "heater": 1.0,
    "washing_machine": 0.5, "dishwasher": 0.8, "laptop": 0.08, "ev_charger": 3.3
}


class SmartHomeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, price_profile, pv_profile, config, forecast_horizon=3):
        super().__init__()
        self.price = np.array(price_profile)
        self.pv = np.array(pv_profile)
        self.cfg = config
        self.T = len(self.price)
        self.forecast_horizon = forecast_horizon

        # behavior có thể được set từ bên ngoài
        self.behavior = None

        # weather model
        self.weather_states = ["sunny", "mild", "cloudy", "rainy", "stormy"]
        self.weather_factors = {
            "sunny": 1.0, "mild": 0.8, "cloudy": 0.5,
            "rainy": 0.3, "stormy": 0.1
        }
        self.weather_transition = {
            "sunny": [0.6, 0.25, 0.1, 0.04, 0.01],
            "mild": [0.2, 0.5, 0.2, 0.08, 0.02],
            "cloudy": [0.1, 0.2, 0.4, 0.2, 0.1],
            "rainy": [0.05, 0.1, 0.25, 0.4, 0.2],
            "stormy": [0.02, 0.08, 0.2, 0.3, 0.4]
        }

        # parameters
        self.time_step = 1.0
        self.total_cost = 0.0
        self.total_energy_bought = 0.0

        # load configuration
        self.N_ad = len(config.get('adjustable', []))
        self.N_su = len(config.get('shiftable_su', []))
        self.N_si = len(config.get('shiftable_si', []))

        # observation and action spaces
        obs_len = 4 + 2 * self.forecast_horizon + self.N_si + self.N_su
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(obs_len,), dtype=np.float32)
        self.action_space = spaces.MultiBinary(self.N_su + self.N_si)

        # internal state placeholders
        self.t = 0
        self.SOC = 0.0
        self.Ot_si = [0] * self.N_si
        self.su_started = [False] * self.N_su
        self.su_remaining = [0] * self.N_su
        self.weather_series = []

    def set_month_behavior(self, month_behavior):
        self.month_behavior = month_behavior
        self.current_day = 0
        self.current_behavior = month_behavior[self.current_day]
        print(f"📅 Mô phỏng bắt đầu: Ngày {self.current_day}, loại ngày = {self.current_behavior.get('event_type', 'unknown')}")

    def _update_behavior_for_new_day(self):
        if hasattr(self, "month_behavior"):
            self.current_day = (self.current_day + 1) % len(self.month_behavior)
            self.current_behavior = self.month_behavior[self.current_day]
            print(f"📅 Chuyển sang ngày {self.current_day}, loại ngày = {self.current_behavior.get('event_type', 'unknown')}")

    def reset(self, *, seed=None, options=None):
        # Gymnasium-compatible reset
        super().reset(seed=seed)
        if seed is not None:
            # seed numpy RNG to make environment deterministic when requested
            np.random.seed(seed)

        self.t = 0
        bat = self.cfg.get('battery', {})
        self.SOC = bat.get('soc0', 0.5)
        self.Ot_si = [0] * self.N_si
        self.su_started = [False] * self.N_su
        self.su_remaining = [self.cfg['shiftable_su'][i]['L'] for i in range(self.N_su)] if self.N_su > 0 else []
        self.total_cost = 0.0
        self.total_energy_bought = 0.0
        self.current_weather = "mild"
        self.weather_series = []

        # Do not overwrite externally set behavior
        if hasattr(self, "current_behavior"):
            self.behavior = self.current_behavior
        elif not hasattr(self, 'behavior') or self.behavior is None:
            # fallback single-day behavior
            hb_single = HumanBehavior(T=self.T, weather=self.current_weather)
            behavior_data = hb_single.generate_daily_behavior(sample_device_states=True)
            self.behavior = behavior_data

        # generate weather series
        cur = self.current_weather
        for _ in range(self.T):
            probs = self.weather_transition.get(cur, self.weather_transition['mild'])
            cur = np.random.choice(self.weather_states, p=probs)
            self.weather_series.append(cur)
        self.current_weather = self.weather_series[0] if len(self.weather_series) > 0 else 'mild'

        obs = self._get_obs()
        if obs.shape != self.observation_space.shape:
            raise ValueError(f"Lỗi Shape: observation_space shape {self.observation_space.shape} "
                             f"nhưng _get_obs() trả về shape {obs.shape}")
        info = {}
        return obs, info

    def _get_obs(self):
        # build observation vector
        t_norm = float(self.t) / max(1, self.T - 1)
        rho = float(self.price[self.t])
        pv_now = float(self.pv[self.t])

        # forecasts
        end = min(self.t + self.forecast_horizon, self.T)
        forecast_prices = self.price[self.t:end]
        forecast_pv = self.pv[self.t:end]
        if len(forecast_prices) < self.forecast_horizon:
            forecast_prices = np.pad(forecast_prices, (0, self.forecast_horizon - len(forecast_prices)), 'edge')
        if len(forecast_pv) < self.forecast_horizon:
            forecast_pv = np.pad(forecast_pv, (0, self.forecast_horizon - len(forecast_pv)), 'edge')

        obs = [t_norm, rho, pv_now, self.SOC]
        obs += forecast_prices.tolist() + forecast_pv.tolist()
        # Ot_si and su_started lengths
        obs += list(self.Ot_si) if self.N_si > 0 else []
        obs += [1.0 if s else 0.0 for s in self.su_started] if self.N_su > 0 else []
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        # Ensure action is array-like and length matches
        action = np.array(action, dtype=int).flatten()
        if action.size != self.N_su + self.N_si:
            raise ValueError(f"Action length {action.size} does not match expected {self.N_su + self.N_si}")

        act_su = action[:self.N_su].tolist() if self.N_su > 0 else []
        act_si = action[self.N_su:].tolist() if self.N_si > 0 else []

        # 1. compute loads
        P_su_t = 0.0
        for i, su in enumerate(self.cfg.get('shiftable_su', [])):
            if self.t >= su['t_s'] and self.t <= su['t_f'] and i < len(act_su) and act_su[i] == 1:
                P_su_t += su.get('rate', 0.0)

        P_si_t = 0.0
        for i, si in enumerate(self.cfg.get('shiftable_si', [])):
            if self.t >= si['t_s'] and self.t <= si['t_f'] and i < len(act_si) and act_si[i] == 1:
                P_si_t += si.get('rate', 0.0)

        P_cr_t = float(self.cfg.get('critical', [0.0] * self.T)[self.t])
        P_ad_t = sum(float(ad.get('P_com', 0.0)) for ad in self.cfg.get('adjustable', []))

        # human behavior loads
        P_human_t = 0.0
        device_states_t = {}
        if isinstance(self.behavior, dict):
            device_states = self.behavior.get('device_states', {})
            for device_name, power in DEVICE_POWER_MAP.items():
                states = device_states.get(device_name, [False]*self.T)
                is_on = bool(states[self.t]) if len(states) > self.t else False
                device_states_t[device_name] = is_on
                if is_on:
                    is_agent_controlled = (device_name in ['washing_machine', 'dishwasher', 'ev_charger'])
                    if not is_agent_controlled:
                        P_human_t += power
        elif hasattr(self.behavior, 'device_usage'):
            occ_factor = float(self.behavior.occupancy[self.t])
            device_profile = getattr(self.behavior, 'device_usage', {})
            if occ_factor > 0.7 and isinstance(device_profile, dict):
                if device_profile.get('tv', [0])[self.t] > 0.5:
                    P_human_t += DEVICE_POWER_MAP['tv']
                if device_profile.get('ac', [0])[self.t] > 0.5:
                    P_human_t += DEVICE_POWER_MAP['ac']
                if device_profile.get('laptop', [0])[self.t] > 0.5:
                    P_human_t += DEVICE_POWER_MAP['laptop']
                if device_profile.get('heater', [0])[self.t] > 0.5:
                    P_human_t += DEVICE_POWER_MAP['heater']

        P_load = P_cr_t + P_ad_t + P_su_t + P_si_t + P_human_t

        # weather effect on PV
        weather = self.weather_series[self.t] if len(self.weather_series) > self.t else 'mild'
        weather_factor = self.weather_factors.get(weather, 0.8)
        P_pv = float(self.pv[self.t]) * weather_factor

        # battery
        bat = self.cfg.get('battery', {})
        soc_min = bat.get('soc_min', 0.1)
        soc_max = bat.get('soc_max', 0.9)
        eta_ch = bat.get('eta_ch', 0.95)
        eta_dis = bat.get('eta_dis', 0.95)

        P_ch, P_dis = 0.0, 0.0
        if P_pv >= P_load:
            P_surplus = P_pv - P_load
            if self.SOC < soc_max:
                P_ch = P_surplus
                # simple SOC update (user should set realistic capacity)
                C_bat = bat.get('capacity_kwh', 10.0)
                self.SOC = min(soc_max, self.SOC + (eta_ch * P_ch * self.time_step) / C_bat)
        else:
            P_deficit = P_load - P_pv
            if self.SOC > soc_min:
                C_bat = bat.get('capacity_kwh', 10.0)
                P_dis = min(P_deficit, (self.SOC - soc_min) * C_bat / eta_dis)
                self.SOC = max(soc_min, self.SOC - (P_dis * eta_dis) / C_bat)

        # grid balancing (only buying)
        supply = P_pv + P_dis
        demand = P_load + P_ch
        self.P_grid = max(0.0, demand - supply)

        # reward
        price = float(self.price[self.t])
        cost = self.P_grid * price
        self.total_cost += cost
        self.total_energy_bought += self.P_grid * self.time_step

        hour = self.t % 24
        is_night = (hour < 6 or hour >= 18)
        penalty_unmet = -10.0 * max(0.0, demand - supply - self.P_grid)
        penalty_battery = -0.05 * (abs(P_ch) + abs(P_dis))

        reward = -cost + penalty_unmet + penalty_battery

        if is_night:
            if self.P_grid > 0:
                reward -= 0.2 * cost
            elif P_dis > 0:
                reward += 0.05 * P_dis

        if 17 <= hour <= 21:
            reward -= 0.5 * cost
        if 0.4 <= self.SOC <= 0.8:
            reward += 0.02
        reward += 0.03 * min(P_pv, P_load)

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
            "weather_factor": weather_factor,
            "P_human": P_human_t,
            "P_agent_su": P_su_t,
            "P_agent_si": P_si_t,
            "device_states": device_states_t
        }

        # increment time and done
        self.t += 1
        terminated = (self.t >= self.T)
        truncated = False
        if terminated:
            self._update_behavior_for_new_day()

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        if obs.shape != self.observation_space.shape:
            if terminated:
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            else:
                raise ValueError(f"Lỗi Shape sau khi step: observation_space shape {self.observation_space.shape} "
                                 f"nhưng _get_obs() trả về shape {obs.shape}")

        return obs, float(reward), bool(terminated), bool(truncated), info

    def _solve_one_step(self, P_net_without_ad):
        rho_b = float(self.price[self.t])
        if pulp is None or self.N_ad == 0:
            P_ad = []
            for ad in self.cfg.get('adjustable', []):
                P_ad.append(ad.get('P_com', 0.0) if rho_b < np.mean(self.price) else ad.get('P_min', 0.0))
            P_ad = np.array(P_ad)
            P_net = P_net_without_ad + P_ad.sum()
            if P_net >= 0:
                return P_ad.tolist(), float(P_net), 0.0
            else:
                return P_ad.tolist(), 0.0, float(-P_net)

        prob = pulp.LpProblem('OneStep', pulp.LpMinimize)
        P_ad_vars = [pulp.LpVariable(f'P_ad_{i}', lowBound=ad['P_min'], upBound=ad['P_max']) for i, ad in enumerate(self.cfg.get('adjustable', []))]
        P_b_var = pulp.LpVariable('P_b', lowBound=0.0)
        P_s_var = pulp.LpVariable('P_s', lowBound=0.0)
        z_b = pulp.LpVariable('z_b', cat='Binary')
        z_s = pulp.LpVariable('z_s', cat='Binary')
        bigM = 1e5
        prob += (P_b_var - P_s_var == P_net_without_ad + pulp.lpSum(P_ad_vars))
        prob += P_b_var <= z_b * bigM
        prob += P_s_var <= z_s * bigM
        prob += z_b + z_s <= 1
        obj = rho_b * P_b_var - (self.cfg.get('beta', 0.5) * rho_b) * P_s_var
        for i, ad in enumerate(self.cfg.get('adjustable', [])):
            u = pulp.LpVariable(f'u_{i}', lowBound=0.0)
            prob += u >= ad['P_com'] - P_ad_vars[i]
            prob += u >= P_ad_vars[i] - ad['P_com']
            obj += ad.get('alpha', 0.0) * u
            obj += rho_b * P_ad_vars[i]
        prob += obj
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=5)
        prob.solve(solver)
        P_ad = [float(pulp.value(v)) if pulp.value(v) is not None else 0.0 for v in P_ad_vars]
        P_b = float(pulp.value(P_b_var)) if pulp.value(P_b_var) is not None else 0.0
        P_s = float(pulp.value(P_s_var)) if pulp.value(P_s_var) is not None else 0.0
        return P_ad, P_b, P_s

    @staticmethod
    def get_tiered_price(total_consumption_kwh):
        tiers = [
            (50, 1984),
            (100, 2050),
            (200, 2380),
            (300, 2998),
            (400, 3350),
            (float('inf'), 3460),
        ]
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
        return avg_price / 1000

    def render(self, mode='human'):
        print(f"t={self.t}, total_cost={self.total_cost:.3f}, SOC={self.SOC:.3f}")


if __name__ == "__main__":
    # quick demo
    T = 24
    price = 0.1 + 0.2 * np.random.rand(T)
    pv = np.clip(1.5 * np.sin(np.linspace(0, 3.14, T)) + 0.2 * np.random.randn(T), 0, None)
    config = {
        'critical': [0.3] * T,
        'adjustable': [
            {'P_min': 0.1, 'P_max': 1.5, 'P_com': 1.2, 'alpha': 0.06},
            {'P_min': 0.0, 'P_max': 1.2, 'P_com': 1.0, 'alpha': 0.12}
        ],
        'shiftable_su': [{'rate': 0.5, 'L': 2, 't_s': 6, 't_f': 20}, {'rate': 0.6, 'L': 1, 't_s': 8, 't_f': 22}],
        'shiftable_si': [{'rate': 1.0, 'E': 4.0, 't_s': 0, 't_f': 23}],
        'beta': 0.5,
        'battery': {'soc0': 0.5, 'soc_min': 0.1, 'soc_max': 0.9, 'capacity_kwh': 10.0},
        "reward_mode": "advanced"
    }
    env = SmartHomeEnv(price, pv, config)
    print("--- Chạy Demo 1 ngày (Single-day) ---")
    obs, info = env.reset()
    done = False
    total_cost = 0.0
    while not done:
        action = np.random.randint(0, 2, size=env.N_su + env.N_si)
        obs, rew, terminated, truncated, info = env.step(action)
        total_cost = env.total_cost
        done = terminated or truncated
    print("Episode finished, total cost", total_cost)
