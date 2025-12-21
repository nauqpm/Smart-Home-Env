import numpy as np
import pandas as pd
import math
import logging
from typing import Dict
import gymnasium as gym
from gymnasium import spaces

from device_config import (
    DEVICE_CONFIG, ACTION_INDICES, ROOM_OCCUPANCY_HOURS,
    THERMAL_CONSTANTS, EV_CONFIG
)

logger = logging.getLogger("SmartHomeEnv")
logger.setLevel(logging.INFO)


# =====================================================
# Utility
# =====================================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def calculate_vietnam_tiered_bill(kwh: float) -> float:
    tiers = [
        (50, 1984),
        (50, 2050),
        (100, 2380),
        (100, 2998),
        (100, 3350),
        (float("inf"), 3460),
    ]
    bill = 0.0
    remaining = kwh
    for limit, price in tiers:
        if remaining <= 0:
            break
        amount = min(remaining, limit)
        bill += amount * price
        remaining -= amount
    return bill


def is_room_occupied(room: str, hour: int) -> bool:
    ranges = ROOM_OCCUPANCY_HOURS.get(room, [])
    for start, end in ranges:
        if start <= end:
            if start <= hour < end:
                return True
        else:
            if hour >= start or hour < end:
                return True
    return False


# =====================================================
# Behavior generator
# =====================================================
class AdvancedHumanBehaviorGenerator:
    def __init__(self, config: Dict):
        self.config = config

    def generate_for_times(self, times: pd.DatetimeIndex):
        schedules = []
        for t in times:
            hour = t.hour
            n_home = 2 if (18 <= hour <= 23 or 0 <= hour <= 7) else 0
            temp_out = 28 + 5 * math.sin(math.pi * (hour - 9) / 12)
            schedules.append(
                {
                    "must_run": DEVICE_CONFIG["fixed"]["fridge"]["power"],
                    "n_home": n_home,
                    "temp_out": temp_out,
                }
            )
        return schedules


# =====================================================
# ENVIRONMENT
# =====================================================
class SmartHomeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, price_profile=None, pv_profile=None, config=None):
        super().__init__()

        self.config = config or {}
        self.price_profile_input = price_profile
        self.pv_profile_input = pv_profile

        # Time
        self.sim_steps = int(self.config.get("sim_steps", 24))
        self.sim_start = pd.to_datetime(self.config.get("sim_start", "2025-01-01"))
        self.sim_freq = self.config.get("sim_freq", "1h")
        self.time_step_h = self.config.get("time_step_hours", 1.0)

        # Battery
        self.battery = self.config.get(
            "battery",
            {
                "capacity_kwh": 10.0,
                "soc_min": 0.1,
                "soc_max": 0.9,
                "p_charge_max_kw": 3.0,
                "p_discharge_max_kw": 3.0,
                "eta_ch": 0.95,
                "eta_dis": 0.95,
                "soc_init": 0.5,
            },
        )

        # Action & Observation
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float32)

        self.behavior_gen = AdvancedHumanBehaviorGenerator(
            self.config.get("behavior", {})
        )

        self._reset_internal()

    # -------------------------------------------------
    def _reset_internal(self):
        self.times = pd.date_range(
            self.sim_start, periods=self.sim_steps, freq=self.sim_freq
        )

        # PV
        self.pv_profile = (
            np.array(self.pv_profile_input[: self.sim_steps])
            if self.pv_profile_input is not None
            else np.zeros(self.sim_steps)
        )

        # Load
        self.load_schedules = self.behavior_gen.generate_for_times(self.times)

        # States
        self.t = 0
        self.soc = self.battery["soc_init"]
        self.cumulative_import_kwh = 0.0
        self.cumulative_export_kwh = 0.0
        self.total_cost = 0.0

        self.room_temps = {
            "living": THERMAL_CONSTANTS["comfort_temp"],
            "master": THERMAL_CONSTANTS["comfort_temp"],
            "bed2": THERMAL_CONSTANTS["comfort_temp"],
        }

        self.wm_remaining = np.random.choice([0, 2], p=[0.3, 0.7])
        self.dw_remaining = np.random.choice([0, 1], p=[0.4, 0.6])
        self.wm_deadline = 22
        self.dw_deadline = 23

        self.ev_soc = np.random.uniform(0.2, 0.5)
        self.ev_deadline = EV_CONFIG["deadline_hour"]

    # -------------------------------------------------
    def _get_obs(self):
        idx = self.t
        hour = self.times[idx].hour
        future_pv = np.sum(self.pv_profile[idx + 1 : idx + 7]) if idx + 1 < self.sim_steps else 0.0

        return np.array(
            [
                self.soc,
                self.pv_profile[idx],
                self.load_schedules[idx]["must_run"],
                future_pv,
                math.sin(2 * math.pi * hour / 24),
                math.cos(2 * math.pi * hour / 24),
                self.load_schedules[idx]["n_home"],
                self.load_schedules[idx]["temp_out"],
                self.room_temps["living"],
                self.room_temps["master"],
                self.room_temps["bed2"],
                self.wm_remaining,
                self.ev_soc,
            ],
            dtype=np.float32,
        )

    # -------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internal()
        return self._get_obs(), {}

    # -------------------------------------------------
    def _update_room_temp(self, room, ac_norm):
        k1 = THERMAL_CONSTANTS["k1"]
        k2 = THERMAL_CONSTANTS["k2"]
        t = self.room_temps[room]
        tout = self.load_schedules[self.t]["temp_out"]
        new_t = t + k1 * (tout - t) - k2 * ac_norm
        return clamp(new_t, 18.0, 38.0)

    def _comfort_penalty(self, room, hour):
        if not is_room_occupied(room, hour):
            return 0.0
        dev = abs(self.room_temps[room] - THERMAL_CONSTANTS["comfort_temp"])
        tol = THERMAL_CONSTANTS["comfort_tolerance"]
        return max(0.0, dev - tol) ** 2

    # -------------------------------------------------
    def step(self, action):
        hour = self.times[self.t].hour
        must_run = self.load_schedules[self.t]["must_run"]
        a = np.array(action).clip(-1, 1)

        # -------- Actions --------
        act_bat = a[0]              # [-1,1]
        ac_vals = [(a[i] + 1) / 2 for i in [1, 2, 3]]
        act_ev = (a[4] + 1) / 2
        act_wm = a[5] > 0
        act_dw = a[6] > 0

        total_load = 0.0
        comfort_penalty = 0.0

        # -------- AC --------
        for (room, ac), key in zip(
            zip(["living", "master", "bed2"], ac_vals),
            ["ac_living", "ac_master", "ac_bed2"],
        ):
            p = ac * DEVICE_CONFIG["adjustable"][key]["power_max"]
            total_load += p
            self.room_temps[room] = self._update_room_temp(room, ac)
            comfort_penalty += self._comfort_penalty(room, hour)

        # -------- WM --------
        if self.wm_remaining > 0 and act_wm:
            total_load += DEVICE_CONFIG["shiftable"]["wm"]["power"]
            self.wm_remaining -= 1

        # -------- DW --------
        if self.dw_remaining > 0 and act_dw:
            total_load += DEVICE_CONFIG["shiftable"]["dw"]["power"]
            self.dw_remaining -= 1

        # -------- EV --------
        ev_p = act_ev * DEVICE_CONFIG["shiftable"]["ev"]["power_max"]
        total_load += ev_p
        self.ev_soc = clamp(
            self.ev_soc + ev_p * self.time_step_h / DEVICE_CONFIG["shiftable"]["ev"]["capacity"],
            0.0,
            1.0,
        )

        # -------- Fixed --------
        total_load += must_run

        # -------- Battery --------
        bat_p = 0.0
        if act_bat > 0:  # charge
            bat_p = act_bat * self.battery["p_charge_max_kw"]
            delta = bat_p * self.time_step_h * self.battery["eta_ch"] / self.battery["capacity_kwh"]
            self.soc = clamp(self.soc + delta, self.battery["soc_min"], self.battery["soc_max"])
            total_load += bat_p
        else:  # discharge
            bat_p = -act_bat * self.battery["p_discharge_max_kw"]
            delta = bat_p * self.time_step_h / self.battery["eta_dis"] / self.battery["capacity_kwh"]
            self.soc = clamp(self.soc - delta, self.battery["soc_min"], self.battery["soc_max"])
            total_load -= bat_p

        # -------- Grid --------
        pv = self.pv_profile[self.t]
        grid_import = max(0.0, total_load - pv)
        grid_export = max(0.0, pv - total_load)

        self.cumulative_import_kwh += grid_import * self.time_step_h
        self.cumulative_export_kwh += grid_export * self.time_step_h

        prev_cost = self.total_cost
        import_bill = calculate_vietnam_tiered_bill(self.cumulative_import_kwh)
        export_revenue = self.cumulative_export_kwh * 2000
        self.total_cost = import_bill - export_revenue
        step_cost = self.total_cost - prev_cost

        # -------- Reward --------
        reward = -step_cost / 2000.0 - comfort_penalty

        # EV shaping
        hours_left = max(1, self.ev_deadline - hour)
        ev_deficit = max(0.0, EV_CONFIG["min_target_soc"] - self.ev_soc)
        reward -= ev_deficit * (5.0 / hours_left)

        # -------- Step --------
        self.t += 1
        done = self.t >= self.sim_steps

        if done:
            if self.wm_remaining > 0:
                reward -= 20
            if self.dw_remaining > 0:
                reward -= 15
            if self.ev_soc < EV_CONFIG["min_target_soc"]:
                reward -= 30

        return (
            self._get_obs() if not done else np.zeros(13, dtype=np.float32),
            float(reward),
            done,
            False,
            {"step_cost": step_cost, "total_cost": self.total_cost},
        )
