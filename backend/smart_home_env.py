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
        # Capture current step properties BEFORE incrementing t
        hour = self.times[self.t].hour
        must_run = self.load_schedules[self.t]["must_run"]
        n_home = self.load_schedules[self.t]["n_home"]
        temp_out = self.load_schedules[self.t]["temp_out"]
        pv_val = self.pv_profile[self.t]
        
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

        # -------- Battery (Physics Fix) --------
        bat_action_p = 0.0
        if act_bat > 0:  # Requested Charge
            max_charge_p = self.battery["p_charge_max_kw"]
            # Energy needed to reach 100% (or soc_max)
            max_energy_space = (self.battery["soc_max"] - self.soc) * self.battery["capacity_kwh"]
            # Max power limited by available space over this time step
            # P_lim = E_space / (dt * eta)
            max_p_by_soc = max_energy_space / (self.time_step_h * self.battery["eta_ch"])
            
            # Realizable Power
            bat_p = min(max_charge_p, max_p_by_soc)
            
            # Physics Update
            energy_in = bat_p * self.time_step_h * self.battery["eta_ch"]
            delta_soc = energy_in / self.battery["capacity_kwh"]
            self.soc = clamp(self.soc + delta_soc, self.battery["soc_min"], self.battery["soc_max"])
            
            total_load += bat_p # Load increases by input power
            
        else:  # Requested Discharge
            max_discharge_p = self.battery["p_discharge_max_kw"]
            # Energy available above soc_min
            available_energy = (self.soc - self.battery["soc_min"]) * self.battery["capacity_kwh"]
            # Max power limited by available energy
            # P_lim = E_avail * eta / dt   <-- Wait, discharge: E_out = P * dt / eta_dis? No.
            # Standard model: 
            #   Charge:   dSOC = P * dt * eta / Cap
            #   Dischg:   dSOC = P * dt / (eta * Cap) -> Energy taken from batt = P * dt / eta
            #   Here, 'bat_p' is usually the DC power or AC power? 
            #   Typically P is AC side.
            #   Discharge (AC side P): Energy removed from internal = P * dt / eta_dis
            
            max_p_by_soc = (available_energy * self.battery["eta_dis"]) / self.time_step_h
            
            # Realizable Power (magnitude)
            bat_p_mag = min(max_discharge_p * abs(act_bat), max_p_by_soc)
            
            # Physics Update
            energy_out_internal = (bat_p_mag * self.time_step_h) / self.battery["eta_dis"]
            delta_soc = energy_out_internal / self.battery["capacity_kwh"]
            self.soc = clamp(self.soc - delta_soc, self.battery["soc_min"], self.battery["soc_max"])
            
            total_load -= bat_p_mag # Load decreases by output power

        # -------- Grid --------
        pv = self.pv_profile[self.t]
        grid_import = max(0.0, total_load - pv)
        grid_export = max(0.0, pv - total_load)

        self.cumulative_import_kwh += grid_import * self.time_step_h
        self.cumulative_export_kwh += grid_export * self.time_step_h

        prev_cost = self.total_cost
        import_bill = calculate_vietnam_tiered_bill(self.cumulative_import_kwh)
        # NOTE: Export revenue disabled - no feed-in tariff (không bán điện về lưới)
        # export_revenue = self.cumulative_export_kwh * 2000
        self.total_cost = import_bill  # Only import cost, no export credit
        step_cost = self.total_cost - prev_cost

        # ======== IMPROVED REWARD FUNCTION ========
        # Base reward: negative of step cost (normalized)
        reward = -step_cost / 2000.0
        
        # --- 1. Comfort Penalty (Stronger) ---
        # If occupied and temp too high, penalize more
        for room, temp in self.room_temps.items():
            if is_room_occupied(room, hour):
                # Base comfort penalty
                reward -= self._comfort_penalty(room, hour)
                
                # Mandatory AC penalty: If temp > 28°C and AC essentially off
                if temp > 28.0:
                    room_idx = ["living", "master", "bed2"].index(room) if room in ["living", "master", "bed2"] else -1
                    if room_idx >= 0:
                        ac_usage = ac_vals[room_idx]  # 0-1 normalized
                        if ac_usage < 0.2:  # AC essentially off when needed
                            reward -= 2.0  # Strong penalty for ignoring comfort
                
                # --- NEW: Over-Cooling Penalty (Fair Competition) ---
                # Penalty: Cooling when already cold (wasting energy)
                comfort_temp = THERMAL_CONSTANTS["comfort_temp"]
                if temp < comfort_temp - 1.0:  # Room is already below 24°C
                    room_idx = ["living", "master", "bed2"].index(room) if room in ["living", "master", "bed2"] else -1
                    if room_idx >= 0:
                        ac_usage = ac_vals[room_idx]
                        if ac_usage > 0.2:  # AC is running when not needed
                            reward -= 3.0  # Strong penalty for over-cooling

        # --- 2. Per-Step Task Urgency Penalties (MUCH STRONGER) ---
        # WM: Must complete by hour 22 (deadline)
        if self.wm_remaining > 0:
            hours_until_wm_deadline = max(1, self.wm_deadline - hour)
            wm_urgency = self.wm_remaining / hours_until_wm_deadline
            reward -= wm_urgency * 5.0  # Increased from 1.0 to 5.0
        
        # DW: Must complete by hour 23
        if self.dw_remaining > 0:
            hours_until_dw_deadline = max(1, self.dw_deadline - hour)
            dw_urgency = self.dw_remaining / hours_until_dw_deadline
            reward -= dw_urgency * 4.0  # Increased from 0.8 to 4.0
        
        # --- 3. EV Charging Urgency (MUCH STRONGER) ---
        hours_left = max(1, self.ev_deadline - hour) if hour < self.ev_deadline else max(1, 24 - hour + self.ev_deadline)
        ev_deficit = max(0.0, EV_CONFIG["min_target_soc"] - self.ev_soc)
        # Quadratic penalty that grows faster as deadline approaches
        reward -= (ev_deficit ** 2) * (50.0 / hours_left)  # Increased from 10.0 to 50.0

        # --- 4. Task Completion Bonuses ---
        # Encourage actually completing tasks
        # Note: These are implicitly handled since completing tasks removes penalties

        # -------- Step --------
        self.t += 1
        done = self.t >= self.sim_steps

        if done:
            # --- 5. MASSIVE End-of-Episode Penalties (10x original) ---
            # These must be MUCH larger than any cost savings from not running devices
            if self.wm_remaining > 0:
                reward -= 200  # Was 60, now 200
            if self.dw_remaining > 0:
                reward -= 150  # Was 45, now 150
            if self.ev_soc < EV_CONFIG["min_target_soc"]:
                ev_shortfall = EV_CONFIG["min_target_soc"] - self.ev_soc
                reward -= 300 + ev_shortfall * 200  # Was 100+50, now 300+200

        # -------- Info for Visualization --------
        # Heuristic for lights based on occupancy AND time (avoid sleeping hours)
        # Use locally captured variables (hour, n_home)
        # Active hours: Morning (6-8) or Evening (17-23). 
        # Note: 23 is usually bedtime, so we can cut off AT 23 (lights off from 23:00 onwards).
        is_active_time = (6 <= hour < 9) or (17 <= hour < 23)
        lights_on = 1 if (n_home > 0 and is_active_time) else 0
        
        return (
            self._get_obs() if not done else np.zeros(13, dtype=np.float32),
            float(reward),
            done,
            False,
            {
                "step_cost": step_cost, 
                "total_cost": self.total_cost,
                "step_grid_import": grid_import * self.time_step_h,
                "room_temps": dict(self.room_temps),
                
                # Device States for Visualization
                "ac_living": 1 if ac_vals[0] > 0.4 else 0,
                "ac_master": 1 if ac_vals[1] > 0.4 else 0,
                "ac_bed2": 1 if ac_vals[2] > 0.4 else 0,
                "wm": 1 if act_wm else 0,
                "dw": 1 if act_dw else 0,
                "ev": 1 if act_ev > 0.1 else 0,
                "battery": "charge" if act_bat > 0.01 else ("discharge" if act_bat < -0.01 else "idle"),
                
                # Heuristic Lights
                "light_living": lights_on,
                "light_master": lights_on,
                "light_bed2": lights_on,
                "light_kitchen": lights_on,
                "light_toilet": lights_on,
                
                # Env context
                "weather": self.config.get("board_config", {}).get("weather", "sunny"), # Fallback
                "temp": temp_out, # Use captured variable
                "pv": pv_val,      # Use captured variable
                "load": total_load # Total home consumption
            },
        )
