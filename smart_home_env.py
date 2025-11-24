import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from human_behavior import HumanBehavior

try:
    import pulp
except ImportError:
    pulp = None

try:
    import pvlib
    from pvlib.location import Location
    from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
    HAS_PVLIB = True
except ImportError:
    HAS_PVLIB = False

DEVICE_POWER_MAP = {
    "lights": 0.1, "fridge": 0.2, "tv": 0.15, "ac": 1.5, "heater": 1.0,
    "washing_machine": 0.5, "dishwasher": 0.8, "laptop": 0.08, "ev_charger": 3.3
}

class AdvancedPV:
    def __init__(self, latitude=10.8, longitude=106.6, timezone='Asia/Ho_Chi_Minh',
                 panel_capacity_kw=5.0, temp_coeff=-0.004):
        self.location = Location(latitude, longitude, tz=timezone)
        self.capacity = panel_capacity_kw
        self.temp_coeff = temp_coeff
        self.temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    def _simulate_weather_params(self, status):
        mapping = {
            "sunny":  {"temp": 32, "wind": 2.0, "cloud": 0.0},
            "mild":   {"temp": 28, "wind": 3.0, "cloud": 0.2},
            "cloudy": {"temp": 26, "wind": 4.0, "cloud": 0.6},
            "rainy":  {"temp": 24, "wind": 5.0, "cloud": 0.9},
            "stormy": {"temp": 22, "wind": 8.0, "cloud": 1.0}
        }
        base = mapping.get(status, mapping["mild"])
        return {
            'temp_air': base["temp"] + np.random.uniform(-2, 2),
            'wind_speed': max(0, base["wind"] + np.random.uniform(-1, 1)),
            'cloud_opacity': base["cloud"]
        }

    def calculate_generation(self, current_time, weather_status):
        if not HAS_PVLIB:
            return 0.0

        env_params = self._simulate_weather_params(weather_status)
        temp_air = env_params['temp_air']
        wind_speed = env_params['wind_speed']
        cloud_opacity = env_params['cloud_opacity']

        times = pd.DatetimeIndex([current_time])
        solpos = self.location.get_solarposition(times)
        zenith = solpos['apparent_zenith'].values[0]

        if zenith > 90:
            return 0.0

        linke_turbidity = 3.0 + (cloud_opacity * 2)
        clearsky = self.location.get_clearsky(times, model='ineichen', linke_turbidity=linke_turbidity)
        ghi_clearsky = clearsky['ghi'].values[0]

        ghi_real = ghi_clearsky * (1 - cloud_opacity * 0.8)

        cell_temp = pvlib.temperature.sapm_cell(
            poa_global=ghi_real,
            temp_air=temp_air,
            wind_speed=wind_speed,
            **self.temp_params
        )

        temp_loss = 1 + self.temp_coeff * (cell_temp - 25)
        power_output = self.capacity * (ghi_real / 1000.0) * temp_loss

        return max(0.0, power_output)

class SmartHomeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, price_profile, pv_profile, config, forecast_horizon=3):
        super().__init__()
        self.price = np.array(price_profile)
        self.pv = np.array(pv_profile)
        self.cfg = config
        self.T = len(self.price)
        self.forecast_horizon = forecast_horizon
        self.behavior = None

        self.weather_states = ["sunny", "mild", "cloudy", "rainy", "stormy"]
        self.weather_transition = {
            "sunny": [0.6, 0.25, 0.1, 0.04, 0.01],
            "mild": [0.2, 0.5, 0.2, 0.08, 0.02],
            "cloudy": [0.1, 0.2, 0.4, 0.2, 0.1],
            "rainy": [0.05, 0.1, 0.25, 0.4, 0.2],
            "stormy": [0.02, 0.08, 0.2, 0.3, 0.4]
        }

        self.pv_physics = AdvancedPV(latitude=10.8, longitude=106.6, panel_capacity_kw=5.0)
        self.start_date = pd.Timestamp('2025-01-01 00:00')

        self.time_step = 1.0
        self.total_cost = 0.0
        self.total_energy_bought = 0.0

        self.N_ad = len(config.get('adjustable', []))
        self.N_su = len(config.get('shiftable_su', []))
        self.N_si = len(config.get('shiftable_si', []))

        obs_len = 4 + 2 * self.forecast_horizon + self.N_si + self.N_su
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(obs_len,), dtype=np.float32)
        self.action_space = spaces.MultiBinary(self.N_su + self.N_si)

    def set_month_behavior(self, month_behavior):
        self.month_behavior = month_behavior
        self.current_day = 0
        self.current_behavior = month_behavior[self.current_day]

    def _update_behavior_for_new_day(self):
        if hasattr(self, "month_behavior"):
            self.current_day = (self.current_day + 1) % len(self.month_behavior)
            self.current_behavior = self.month_behavior[self.current_day]

    def reset(self):
        self.t = 0
        bat = self.cfg.get('battery', {})
        self.SOC = bat.get('soc0', 0.5)
        self.Ot_si = [0] * self.N_si
        self.su_started = [False] * self.N_su
        self.su_remaining = [self.cfg['shiftable_su'][i]['L'] for i in range(self.N_su)]
        self.total_cost = 0.0
        self.total_energy_bought = 0.0
        self.current_weather = "mild"
        self.weather_series = []

        if hasattr(self, "current_behavior"):
            self.behavior = self.current_behavior
        elif not hasattr(self, 'behavior') or self.behavior is None:
            hb_single = HumanBehavior(T=self.T, weather=self.current_weather)
            behavior_data = hb_single.generate_daily_behavior(sample_device_states=True)
            self.behavior = behavior_data

        for t in range(self.T):
            probs = self.weather_transition[self.current_weather]
            self.current_weather = np.random.choice(self.weather_states, p=probs)
            self.weather_series.append(self.current_weather)

        return self._get_obs()

    def _get_obs(self):
        t_norm = self.t / max(1, self.T - 1)
        rho = self.price[self.t]

        current_sim_time = self.start_date + pd.Timedelta(hours=self.t)
        pv_now = self.pv_physics.calculate_generation(current_sim_time, self.weather_series[self.t])

        forecast_prices = self.price[self.t:min(self.t + self.forecast_horizon, self.T)]
        forecast_pv = []

        for k in range(self.forecast_horizon):
            idx = min(self.t + k, self.T - 1)
            f_time = self.start_date + pd.Timedelta(hours=idx)
            f_weather = self.weather_series[idx]
            forecast_pv.append(self.pv_physics.calculate_generation(f_time, f_weather))
        forecast_pv = np.array(forecast_pv)

        if len(forecast_prices) < self.forecast_horizon:
            forecast_prices = np.pad(forecast_prices, (0, self.forecast_horizon - len(forecast_prices)), 'edge')
        if len(forecast_pv) < self.forecast_horizon:
            forecast_pv = np.pad(forecast_pv, (0, self.forecast_horizon - len(forecast_pv)), 'edge')

        obs = [t_norm, rho, pv_now, self.SOC]
        obs += forecast_prices.tolist() + forecast_pv.tolist()
        obs += self.Ot_si
        obs += [1.0 if s else 0.0 for s in self.su_started]
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        action = np.array(action).astype(int)
        act_su = action[:self.N_su].tolist() if self.N_su > 0 else []
        act_si = action[self.N_su:].tolist() if self.N_si > 0 else []

        P_su_t = sum(su["rate"] for i, su in enumerate(self.cfg["shiftable_su"])
                     if self.t >= su["t_s"] and self.t <= su["t_f"] and act_su[i] == 1)
        P_si_t = sum(si["rate"] for i, si in enumerate(self.cfg["shiftable_si"])
                     if self.t >= si["t_s"] and self.t <= si["t_f"] and act_si[i] == 1)

        P_cr_t = self.cfg.get("critical", [0.0] * self.T)[self.t]
        P_ad_t = sum(ad["P_com"] for ad in self.cfg.get("adjustable", []))

        P_human_t = 0.0
        device_states_t = {}

        if isinstance(self.behavior, dict):
            device_states = self.behavior.get("device_states")
            if device_states:
                for device_name, power in DEVICE_POWER_MAP.items():
                    is_on = device_states.get(device_name, [False]*self.T)[self.t]
                    device_states_t[device_name] = is_on
                    if is_on:
                        is_agent_controlled = False
                        if device_name == "washing_machine": is_agent_controlled = True
                        if device_name == "dishwasher": is_agent_controlled = True
                        if device_name == "ev_charger": is_agent_controlled = True

                        if not is_agent_controlled:
                            P_human_t += power

        P_load = P_cr_t + P_ad_t + P_su_t + P_si_t + P_human_t

        current_sim_time = self.start_date + pd.Timedelta(hours=self.t)
        P_pv = self.pv_physics.calculate_generation(current_sim_time, self.weather_series[self.t])

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

        supply = P_pv + P_dis
        demand = P_load + P_ch
        self.P_grid = max(0, demand - supply)

        price = self.price[self.t]
        cost = self.P_grid * price
        self.total_cost += cost
        self.total_energy_bought += self.P_grid * self.time_step

        hour = self.t % 24
        is_night = (hour < 6 or hour >= 18)
        penalty_unmet = -10.0 * max(0, demand - supply - self.P_grid)
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
            "weather": self.weather_series[self.t],
            "P_human": P_human_t,
            "P_agent_su": P_su_t,
            "P_agent_si": P_si_t,
            "device_states": device_states_t
        }

        self.t += 1
        done = (self.t >= self.T)
        if done:
            self._update_behavior_for_new_day()

        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, info

    def _solve_one_step(self, P_net_without_ad):
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

    def render(self, mode='human'):
        print(f"t={self.t}, total_cost={self.total_cost:.3f}")


if __name__ == "__main__":
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

    print("--- Chạy Demo 1 ngày (Single-day) ---")
    obs = env.reset() # reset() sẽ tự tạo behavior fallback
    done = False
    while not done:
        action = np.random.randint(0,2, size=env.N_su + env.N_si)
        obs, rew, done, info = env.step(action)
    print("Episode finished, total cost", env.total_cost)