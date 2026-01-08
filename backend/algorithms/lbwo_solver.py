# lbwo_solver.py
import numpy as np
import random
from smart_home_env import SmartHomeEnv


class LBWOSolver:
    """
    LBWO Expert Solver
    Chỉ tối ưu phần THIẾT BỊ RỜI RẠC (SU + SI + AD on/off)
    Battery + adjustable continuous -> để ENV tự xử lý
    """

    def __init__(
        self,
        horizon=24,
        num_devices=3,
        population_size=20,
        max_iter=30,
        lb=-1.0,
        ub=1.0,
    ):
        self.horizon = horizon
        self.num_devices = num_devices
        self.action_dim = horizon * num_devices

        self.population_size = population_size
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub

    # ------------------------------
    def _to_env_action(self, device_action):
        """
        Map expert discrete action -> full SmartHomeEnv action
        Env expects:
        [battery, adj1, adj2, adj3, su1, su2, si1, ...]
        """
        battery = 0.0                     # trung tính
        adjustable = [0.0, 0.0, 0.0]      # 50%
        shiftable = device_action.tolist()
        return [battery] + adjustable + shiftable

    # ------------------------------
    def evaluate_fitness(self, whale_vector, env: SmartHomeEnv):
        obs, _ = env.reset()
        total_reward = 0.0

        for t in range(self.horizon):
            device_action = whale_vector[
                t * self.num_devices:(t + 1) * self.num_devices
            ]

            env_action = self._to_env_action(device_action)
            obs, reward, done, _, _ = env.step(env_action)

            total_reward += reward
            if done:
                break

        return total_reward

    # ------------------------------
    def solve(self, env_config, price_profile, pv_profile):
        env = SmartHomeEnv(price_profile, pv_profile, env_config)

        whales = np.random.uniform(
            self.lb,
            self.ub,
            size=(self.population_size, self.action_dim)
        )

        best_whale = None
        best_fitness = -np.inf

        for it in range(self.max_iter):
            for i in range(self.population_size):
                fit = self.evaluate_fitness(whales[i], env)

                if fit > best_fitness:
                    best_fitness = fit
                    best_whale = whales[i].copy()

            a = 2 - 2 * (it / self.max_iter)

            for i in range(self.population_size):
                r1, r2 = random.random(), random.random()
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
                    D = abs(best_whale - whales[i])
                    whales[i] = (
                        D * np.exp(1 * r1) * np.cos(2 * np.pi * r1)
                        + best_whale
                    )

                whales[i] = np.clip(whales[i], self.lb, self.ub)

        # reshape -> (T, num_devices) & binarize
        schedule = best_whale.reshape(self.horizon, self.num_devices)
        schedule = (schedule > 0).astype(int)
        return schedule
