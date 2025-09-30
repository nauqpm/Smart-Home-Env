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
    def __init__(self, price_profile, pv_profile, config):
        super().__init__()
        self.price = np.array(price_profile)
        self.pv = np.array(pv_profile)
        self.T = len(self.price)
        self.cfg = config
        self.N_ad = len(config.get('adjustable', []))
        self.N_su = len(config.get('shiftable_su', []))
        self.N_si = len(config.get('shiftable_si', []))
        obs_len = 4 + self.N_si + self.N_su
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(obs_len,), dtype=np.float32)
        self.action_space = spaces.MultiBinary(self.N_su + self.N_si)
        self.reset()

    def reset(self):
        self.t = 0
        bat = self.cfg.get('battery', {})
        self.SOC = bat.get('soc0', 0.5)
        self.Ot_si = [0]*self.N_si
        self.su_started = [False]*self.N_su
        self.su_remaining = [self.cfg['shiftable_su'][i]['L'] if self.N_su>0 else 0 for i in range(self.N_su)]
        self.total_cost = 0.0
        return self._get_obs()

    def _get_obs(self):
        t_norm = self.t / max(1, self.T-1)
        rho = self.price[self.t]
        pv = self.pv[self.t]
        obs = [t_norm, rho, pv, self.SOC]
        obs += self.Ot_si
        obs += [1.0 if s else 0.0 for s in self.su_started]
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        action = np.array(action).astype(int)
        act_su = action[:self.N_su].tolist() if self.N_su > 0 else []
        act_si = action[self.N_su:].tolist() if self.N_si > 0 else []

        # ===== TÍNH TOÁN TẢI =====
        # SU
        P_su_t = 0.0
        for i in range(self.N_su):
            su = self.cfg["shiftable_su"][i]
            if self.t >= su["t_s"] and self.t <= su["t_f"]:
                if act_su[i] == 1:  # bật
                    P_su_t += su["rate"]

        # SI
        P_si_t = 0.0
        for i in range(self.N_si):
            si = self.cfg["shiftable_si"][i]
            if self.t >= si["t_s"] and self.t <= si["t_f"]:
                if act_si[i] == 1:
                    P_si_t += si["rate"]
                    self.Ot_si[i] += 1

        P_cr_t = self.cfg.get("critical", [0.0] * self.T)[self.t]
        P_ad_t = sum(ad["P_com"] for ad in self.cfg.get("adjustable", []))  # đơn giản: luôn chạy ở P_com

        P_load = P_cr_t + P_ad_t + P_su_t + P_si_t
        P_pv = self.pv[self.t]

        # ===== XỬ LÝ PIN =====
        bat = self.cfg.get("battery", {})
        soc_min = bat.get("soc_min", 0.1)
        soc_max = bat.get("soc_max", 0.9)
        eta_ch = bat.get("eta_ch", 0.95)
        eta_dis = bat.get("eta_dis", 0.95)

        P_ch = 0.0
        P_dis = 0.0

        if P_pv >= P_load:
            # dư điện, ưu tiên sạc pin
            P_surplus = P_pv - P_load
            if self.SOC < soc_max:
                P_ch = P_surplus
                self.SOC = min(soc_max, self.SOC + eta_ch * P_ch / self.T)
        else:
            # thiếu điện, ưu tiên xả pin
            P_deficit = P_load - P_pv
            if self.SOC > soc_min:
                P_dis = min(P_deficit, (self.SOC - soc_min) * self.T / eta_dis)
                self.SOC = max(soc_min, self.SOC - P_dis * eta_dis / self.T)

        # ===== REWARD =====
        # reward phạt nếu còn thiếu điện (PV + pin < load)
        P_supplied = P_pv + P_dis
        unmet = max(0, P_load - P_supplied)
        reward = - (unmet * 10.0)  # phạt nặng nếu không đủ điện

        self.total_cost += reward
        self.t += 1
        done = self.t >= self.T
        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "P_pv": P_pv,
            "P_load": P_load,
            "P_ch": P_ch,
            "P_dis": P_dis,
            "SOC": self.SOC,
            "unmet": unmet
        }
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
        'battery': {'soc0':0.5, 'soc_min':0.1, 'soc_max':0.9}
    }
    env = SmartHomeEnv(price, pv, config)
    obs = env.reset()
    done = False
    while not done:
        action = np.random.randint(0,2, size=env.N_su + env.N_si)
        obs, rew, done, info = env.step(action)
    print("Episode finished, total cost", env.total_cost)
