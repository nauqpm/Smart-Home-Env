# lbwo_solver.py
import numpy as np
import random
from smart_home_env import SmartHomeEnv


class LBWOSolver:
    def __init__(self, dim, population_size=20, max_iter=30, lb=-3.0, ub=3.0):
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.population_size = population_size
        self.max_iter = max_iter

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def evaluate_fitness(self, whale_vector, env_ref):
        env_ref.reset()

        horizon = 24
        total_reward = 0.0
        penalty = 0.0

        su_configs = env_ref.config.get('shiftable_su', [])
        si_configs = env_ref.config.get('shiftable_si', [])

        su_track = np.zeros(len(su_configs))
        si_track = np.zeros(len(si_configs))

        num_devices = len(su_configs) + len(si_configs)

        actions_matrix = []
        for t in range(horizon):
            actions_t = []
            for dev_id in range(num_devices):
                idx = dev_id * horizon + t
                prob = self._sigmoid(whale_vector[idx])
                is_on = 1 if prob > 0.5 else 0
                actions_t.append(is_on)

                if is_on == 1:
                    if dev_id < len(su_configs):
                        su_track[dev_id] += 1
                    else:
                        si_idx = dev_id - len(su_configs)
                        rate = si_configs[si_idx]['rate']
                        si_track[si_idx] += rate

                        if si_track[si_idx] > si_configs[si_idx]['E'] + 0.1:
                            penalty += 5.0

            actions_matrix.append(actions_t)

        for t in range(horizon):
            obs, r, done, truncated, info = env_ref.step(actions_matrix[t])
            total_reward += r
            if done: break

        for i, cfg in enumerate(su_configs):
            required_hours = cfg['L']
            if su_track[i] < required_hours:
                diff = required_hours - su_track[i]
            elif su_track[i] > required_hours:
                diff = su_track[i] - required_hours
                penalty += diff * 5.0

        for i, cfg in enumerate(si_configs):
            required_energy = cfg['E']
            if si_track[i] < required_energy:
                diff = required_energy - si_track[i]
                penalty += diff * 10.0

        final_fitness = total_reward - penalty

        return final_fitness

    def solve(self, env_config):
        price_dummy = np.array([0.1] * 24)
        pv_dummy = np.zeros(24)

        temp_env = SmartHomeEnv(price_dummy, pv_dummy, env_config)
        temp_env.reset()

        real_pv_profile = temp_env.pv_profile.copy()

        fast_env = SmartHomeEnv(price_dummy, real_pv_profile, env_config)

        whales = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.full(self.population_size, -np.inf)

        best_whale = None
        best_fitness = -np.inf

        for iter in range(self.max_iter):
            for i in range(self.population_size):
                # Truyền Fast Env vào
                fit = self.evaluate_fitness(whales[i], fast_env)
                fitness[i] = fit

                if fit > best_fitness:
                    best_fitness = fit
                    best_whale = whales[i].copy()

            a = 2 - 2 * (iter / self.max_iter)
            for i in range(self.population_size):
                r1 = random.random()
                r2 = random.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                p = random.random()

                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C * best_whale - whales[i])
                        whales[i] = best_whale - A * D
                    else:
                        rand_idx = random.randint(0, self.population_size - 1)
                        X_rand = whales[rand_idx]
                        D = abs(C * X_rand - whales[i])
                        whales[i] = X_rand - A * D
                else:
                    dist = abs(best_whale - whales[i])
                    whales[i] = dist * np.exp(1 * 2 * np.pi * r1) * np.cos(2 * np.pi * r1) + best_whale

                whales[i] = np.clip(whales[i], self.lb, self.ub)

        final_schedule = []
        num_devices = len(env_config.get("shiftable_su", [])) + len(env_config.get("shiftable_si", []))

        for t in range(24):
            act_t = []
            for dev_id in range(num_devices):
                idx = dev_id * 24 + t
                prob = self._sigmoid(best_whale[idx])
                act_t.append(1 if prob > 0.5 else 0)
            final_schedule.append(act_t)

        return np.array(final_schedule)