# smart_home_env.py
import numpy as np
import pandas as pd
import math
import logging
from typing import Dict, List, Optional, Tuple

try:
    from gymnasium import spaces
except Exception:
    import gym.spaces as spaces

# PVLIB Imports (Optional)
try:
    import pvlib
    from pvlib.location import Location
    from pvlib.irradiance import get_total_irradiance
    from pvlib.temperature import sapm_cell
    from pvlib.pvsystem import pvwatts_dc, pvwatts_ac

    HAS_PVLIB = True
except Exception:
    HAS_PVLIB = False

logger = logging.getLogger('SmartHomeEnv')
logger.setLevel(logging.INFO)


# --- HELPER FUNCTIONS ---
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def get_tiered_price(hour: int, tiers: List[Tuple[int, float]]) -> float:
    for i in range(len(tiers)):
        start, price = tiers[i]
        next_start = tiers[i + 1][0] if i + 1 < len(tiers) else 24
        if start <= hour < next_start:
            return price
    return tiers[-1][1]


def compute_pv_power_with_weather(times, latitude, longitude, tz, surface_tilt, surface_azimuth,
                                  module_parameters, inverter_parameters=None):
    if HAS_PVLIB:
        try:
            loc = Location(latitude, longitude, tz=tz)
            # Localize timestamps
            times_tz = times.tz_localize(tz) if times.tz is None else times.tz_convert(tz)

            cs = loc.get_clearsky(times_tz, model='ineichen')
            ghi_used = cs['ghi']
            dni, dhi = cs['dni'], cs['dhi']

            solpos = loc.get_solarposition(times_tz)
            poa = get_total_irradiance(surface_tilt, surface_azimuth, solpos['zenith'], solpos['azimuth'],
                                       dni=dni, ghi=ghi_used, dhi=dhi, model='haydavies')
            poa_global = poa['poa_global']

            temp_air = 25.0
            wind_speed = 2.0
            celltemp = sapm_cell(poa_global, temp_air, wind_speed, u0=25.0, u1=6.0)
            temp_cell = celltemp['temp_cell']

            pdc0 = float(module_parameters.get('pdc0', 1.0))
            gamma_pdc = float(module_parameters.get('gamma_pdc', -0.004))
            p_dc = pvwatts_dc(poa_global, temp_cell, pdc0, gamma_pdc=gamma_pdc, temp_ref=25.0)

            inv_eff = float(module_parameters.get('inv_eff', 0.96))
            p_ac = p_dc * inv_eff

            return pd.DataFrame({'p_ac': p_ac.clip(lower=0.0).values}, index=times)
        except Exception as e:
            print(f"PVLib calculation error: {e}. Falling back to simple model.")
            # Fallback will happen below

    # Fallback simple logic
    pdc0 = float(module_parameters.get('pdc0', 1.0))
    inv_eff = float(module_parameters.get('inv_eff', 0.96))
    vals = []
    for t in times:
        hour = t.hour + t.minute / 60.0
        if 6 <= hour <= 18:
            vals.append(pdc0 * math.sin(math.pi * (hour - 6) / 12) * inv_eff)
        else:
            vals.append(0.0)
    return pd.DataFrame({'p_ac': vals}, index=times)


class AdvancedHumanBehaviorGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.people = config.get('residents', [])
        self.shiftable_devices = config.get('shiftable_devices', {'washing_machine': 1.0, 'dishwasher': 1.0})
        self.must_run_base = config.get('must_run_base', 0.2)

    def generate_for_times(self, times: pd.DatetimeIndex) -> List[Dict]:
        schedules = []
        for t in times:
            hour = int(t.hour)
            n_home = 0  # Simplified
            if 18 <= hour <= 23 or 6 <= hour <= 8:
                n_home = 2

            base = self.must_run_base + 0.15 * n_home
            if 18 <= hour <= 22: base += 0.6

            shiftable = {}
            for name, p in self.shiftable_devices.items():
                shiftable[name] = p if np.random.rand() < 0.1 else 0.0

            schedules.append({'must_run': base, 'shiftable': shiftable, 'n_home': n_home})
        return schedules


class SmartHomeEnv:
    def __init__(self, price_profile, pv_profile, config: Dict):
        self.config = config

        # Inputs from external source
        self.price_profile_input = price_profile
        self.pv_profile_input = pv_profile

        self.time_step_h = config.get('time_step_hours', 1.0)
        self.battery = config.get('battery', {'capacity_kwh': 10.0})
        self.C_bat = float(self.battery['capacity_kwh'])
        self.soc = float(self.battery.get('soc_init', 0.5))

        self.pv_config = config.get('pv_config', {})
        self.price_tiers = config.get('price_tiers', [])

        self.behavior = config.get('behavior', {})
        self.behavior_gen = AdvancedHumanBehaviorGenerator(self.behavior)

        self.sim_start = pd.to_datetime(config.get('sim_start', '2025-01-01'))
        # Fix Attribute Error by aliasing
        self.start_time = self.sim_start

        self.sim_steps = int(config.get('sim_steps', 24))
        # Fix Pandas Warning: Use 'h' instead of 'H'
        self.sim_freq = config.get('sim_freq', '1h')

        # Action Space
        self.N_su = len(config.get('shiftable_su', []))
        self.N_si = len(config.get('shiftable_si', []))

        if self.N_su + self.N_si == 0:
            num_shiftable = len(self.behavior.get('shiftable_devices', {}))
        else:
            num_shiftable = self.N_su + self.N_si

        if num_shiftable > 0:
            self.action_space = spaces.MultiBinary(num_shiftable)
        else:
            self.action_space = spaces.Discrete(1)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        # Internal placeholders
        self.times = None
        self.pv_profile = None
        self.price_profile = None
        self.load_schedules = None

        # Trigger init logic
        self._reset_internal()

    def _reset_internal(self):
        self.times = pd.date_range(start=self.sim_start, periods=self.sim_steps, freq=self.sim_freq)

        # 1. PV Logic
        if self.pv_profile_input is not None and np.sum(self.pv_profile_input) > 0.1:
            # Use provided input (Fast mode)
            req_len = len(self.times)
            if len(self.pv_profile_input) >= req_len:
                self.pv_profile = np.array(self.pv_profile_input[:req_len])
            else:
                self.pv_profile = np.pad(self.pv_profile_input, (0, req_len - len(self.pv_profile_input)))
        else:
            # Calculate Physics
            pvdf = compute_pv_power_with_weather(
                self.times,
                latitude=self.pv_config.get('latitude', 10.7),
                longitude=self.pv_config.get('longitude', 106.6),
                tz=self.pv_config.get('tz', 'Asia/Ho_Chi_Minh'),
                surface_tilt=self.pv_config.get('surface_tilt', 10),
                surface_azimuth=self.pv_config.get('surface_azimuth', 180),
                module_parameters=self.pv_config.get('module_parameters', {})
            )
            self.pv_profile = pvdf['p_ac'].values

        # 2. Price Logic
        if self.price_profile_input is not None and len(self.price_profile_input) > 0:
            self.price_profile = np.array(self.price_profile_input[:self.sim_steps])
        else:
            prices = [get_tiered_price(t.hour, self.price_tiers) for t in self.times]
            self.price_profile = np.array(prices)

        # 3. Load Logic
        self.load_schedules = self.behavior_gen.generate_for_times(self.times)

        self.t = 0
        self.soc = float(self.battery.get('soc_init', 0.5))

    def _get_obs(self):
        idx = min(self.t, self.sim_steps - 1)
        hour = int(self.times[idx].hour)

        must_run = float(self.load_schedules[idx]['must_run'])
        n_home = self.load_schedules[idx]['n_home']

        # Safe access to PV
        pv_now = float(self.pv_profile[idx]) if self.pv_profile is not None else 0.0

        horizon = 6
        end_idx = min(self.sim_steps, idx + 1 + horizon)
        if idx + 1 < self.sim_steps and self.pv_profile is not None:
            future_pv_sum = float(np.sum(self.pv_profile[idx + 1: end_idx]))
        else:
            future_pv_sum = 0.0

        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)

        return np.array([self.soc, pv_now, must_run, future_pv_sum, hour_sin, hour_cos, n_home], dtype=np.float32)

    def reset(self):
        self._reset_internal()
        return self._get_obs(), {}

    def step(self, action: Optional[List[int]] = None):
        idx = self.t
        must = float(self.load_schedules[idx]['must_run'])

        # Simplified Load Calculation
        p_shiftable_active = 0.0
        su_list = self.config.get('shiftable_su', [])
        si_list = self.config.get('shiftable_si', [])
        all_devs = su_list + si_list

        if action is not None and len(action) > 0:
            for i, act in enumerate(action):
                if i < len(all_devs) and act == 1:
                    p_shiftable_active += all_devs[i]['rate']

        total_load = must + p_shiftable_active
        pv_gen = float(self.pv_profile[idx])
        surplus = pv_gen - total_load

        # Battery Logic
        if surplus >= 0:
            p_ch = min(surplus, float(self.battery.get('p_charge_max_kw', 3.0)))
            eta = float(self.battery.get('eta_ch', 0.95))
            self.soc = clamp(self.soc + (p_ch * eta) / self.C_bat, self.battery['soc_min'], self.battery['soc_max'])
            grid = -(surplus - p_ch)  # Export rest
        else:
            deficit = -surplus
            p_dis = min(deficit, float(self.battery.get('p_discharge_max_kw', 3.0)))
            eta = float(self.battery.get('eta_dis', 0.95))
            kwh_needed = p_dis / eta
            self.soc = clamp(self.soc - kwh_needed / self.C_bat, self.battery['soc_min'], self.battery['soc_max'])
            grid = deficit - p_dis  # Import rest

        price = self.price_profile[idx]
        cost = max(0.0, grid) * price
        reward = -cost / 1000.0
        if self.soc <= self.battery['soc_min'] + 0.05:
            reward -= 0.5

        self.t += 1
        done = self.t >= self.sim_steps

        info = {'cost': cost, 'soc': self.soc}
        return self._get_obs(), reward, done, False, info