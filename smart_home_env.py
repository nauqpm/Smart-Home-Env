import numpy as np
import pandas as pd
import math
import logging
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces

try:
    import pvlib
    from pvlib.location import Location
    from pvlib.irradiance import get_total_irradiance
    from pvlib.temperature import sapm_cell
    from pvlib.pvsystem import pvwatts_dc, pvwatts_ac
    HAS_PVLIB = True
except ImportError:
    HAS_PVLIB = False
    print("WARNING: pvlib not installed. Simple PV model will be used.")

try:
    from human_behavior import HumanBehavior
    HAS_HUMAN_BEHAVIOR = True
except ImportError:
    HAS_HUMAN_BEHAVIOR = False
    print("WARNING: human_behavior.py not found. Using built-in generator.")

logger = logging.getLogger('SmartHomeEnv')
logger.setLevel(logging.INFO)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def get_tiered_price(hour: int, tiers: List[Tuple[int, float]]) -> float:
    for i in range(len(tiers)):
        start, price = tiers[i]
        next_start = tiers[i+1][0] if i+1 < len(tiers) else 24
        if start <= hour < next_start:
            return price
    return tiers[-1][1]

def compute_pv_power_with_weather(times, latitude, longitude, tz, surface_tilt, 
                                  surface_azimuth, module_parameters, weather_factor=1.0):
    """
    Compute PV power. Falls back to simple SIN curve if pvlib missing or error.
    """
    if HAS_PVLIB:
        try:
            loc = Location(latitude, longitude, tz=tz)
            # Localize timestamps
            times_tz = times.tz_localize(tz) if times.tz is None else times.tz_convert(tz)
            
            # Weather modification (Simplistic)
            cs = loc.get_clearsky(times_tz, model='ineichen')
            ghi = cs['ghi'] * weather_factor
            dni = cs['dni'] * weather_factor
            dhi = cs['dhi'] * weather_factor
            
            solpos = loc.get_solarposition(times_tz)
            poa = get_total_irradiance(surface_tilt, surface_azimuth, 
                                       solpos['zenith'], solpos['azimuth'],
                                       dni=dni, ghi=ghi, dhi=dhi, model='haydavies')
            
            # Cell Temp
            temp_air = 25.0
            wind_speed = 2.0
            celltemp = sapm_cell(poa['poa_global'], temp_air, wind_speed, u0=25.0, u1=6.0)
            
            # DC Power
            pdc0 = float(module_parameters.get('pdc0', 1.0))
            gamma_pdc = float(module_parameters.get('gamma_pdc', -0.004))
            p_dc = pvwatts_dc(poa['poa_global'], celltemp['temp_cell'], pdc0, gamma_pdc, 25.0)
            
            # AC Power
            inv_eff = float(module_parameters.get('inv_eff', 0.96))
            p_ac = pvwatts_ac(p_dc, inv_eff)
            
            # Return DataFrame (clip negative values)
            if isinstance(p_ac, (pd.Series, pd.DataFrame)):
                vals = p_ac.clip(lower=0.0).values.flatten() # Ensure 1D array
            else:
                vals = np.clip(p_ac, 0, None)
                
            return pd.DataFrame({'p_ac': vals}, index=times)
            
        except Exception as e:
            print(f"PVLib calculation error: {e}. Using fallback.")
            
    # Fallback simple model
    pdc0 = float(module_parameters.get('pdc0', 1.0))
    inv_eff = float(module_parameters.get('inv_eff', 0.96))
    vals = []
    for t in times:
        hour = t.hour + t.minute / 60.0
        if 6 <= hour <= 18:
            # Simple SIN curve
            val = pdc0 * math.sin(math.pi * (hour - 6) / 12) * inv_eff * weather_factor
            vals.append(max(0.0, val))
        else:
            vals.append(0.0)
    return pd.DataFrame({'p_ac': vals}, index=times)


# --- BEHAVIOR GENERATOR ---
class AdvancedHumanBehaviorGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.people = config.get('residents', [])
        self.must_run_base = config.get('must_run_base', 0.2)
        self.shiftable_devices = config.get('shiftable_devices', {'wm': 0.5})

    def generate_for_times(self, times: pd.DatetimeIndex, weather='mild') -> List[Dict]:
        schedules = []
        for t in times:
            hour = int(t.hour)
            # Occupancy Logic
            n_home = 2 if (17 <= hour <= 23 or 0 <= hour <= 7) else 0
            
            # Base Load Logic
            base = self.must_run_base + 0.15 * n_home
            if 18 <= hour <= 22: base += 0.4
            
            # Temperature Logic (Simulated)
            temp_base = {'sunny': 32, 'mild': 28, 'cloudy': 26, 'rainy': 24, 'stormy': 22}
            base_temp = temp_base.get(weather, 28)
            temp_out = base_temp + 5.0 * math.sin(math.pi * (hour - 9) / 12)
            
            schedules.append({
                'must_run': base, 
                'n_home': n_home, 
                'temp_out': temp_out
            })
        return schedules


# --- MAIN ENV CLASS ---
class SmartHomeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, price_profile, pv_profile, config):
        super().__init__()
        self.config = config
        
        # Inputs
        self.price_profile_input = price_profile
        self.pv_profile_input = pv_profile
        
        # Time
        self.time_step_h = config.get('time_step_hours', 1.0)
        self.sim_start = pd.to_datetime(config.get('sim_start', '2025-01-01'))
        self.sim_steps = int(config.get('sim_steps', 24))
        self.sim_freq = config.get('sim_freq', '1h')
        self.T = self.sim_steps
        
        # Battery
        self.battery = config.get('battery', {'capacity_kwh': 10.0})
        self.C_bat = float(self.battery['capacity_kwh'])
        
        # Configs
        self.pv_config = config.get('pv_config', {})
        self.price_tiers = config.get('price_tiers', [])
        
        # Weather States
        self.weather_states = ["sunny", "mild", "cloudy", "rainy", "stormy"]
        self.weather_factors = {"sunny":1.0, "mild":0.8, "cloudy":0.5, "rainy":0.3, "stormy":0.1}
        
        # Devices
        self.su_devs = config.get('shiftable_su', [])
        self.si_devs = config.get('shiftable_si', [])
        self.ad_devs = config.get('adjustable', [])
        
        self.N_su = len(self.su_devs)
        self.N_si = len(self.si_devs)
        self.N_ad = len(self.ad_devs)
        
        # Spaces
        total_action = self.N_su + self.N_si + self.N_ad
        if total_action > 0:
            self.action_space = spaces.MultiBinary(total_action)
        else:
            self.action_space = spaces.Discrete(1)
            
        # Obs: [SOC, PV, MustRun, FuturePV, Sin, Cos, Occupancy, TempOut, SU_Prog, SI_Prog]
        # Tăng lên 10 chiều để hỗ trợ tracking
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        # Behavior
        self.behavior_gen = AdvancedHumanBehaviorGenerator(config.get('behavior', {}))
        
        # Initialize
        self._reset_internal()

    def _reset_internal(self):
        # Time Index
        self.times = pd.date_range(start=self.sim_start, periods=self.sim_steps, freq=self.sim_freq)
        
        # Weather Series (Random)
        self.weather_series = np.random.choice(self.weather_states, size=self.sim_steps)
        
        # PV Logic
        if self.pv_profile_input is not None and np.sum(self.pv_profile_input) > 0.1:
            # Use Input
            req = len(self.times)
            if len(self.pv_profile_input) >= req:
                self.pv_profile = np.array(self.pv_profile_input[:req])
            else:
                self.pv_profile = np.pad(self.pv_profile_input, (0, req - len(self.pv_profile_input)))
        else:
            # Calculate Physics (FIXED CALL)
            avg_weather = np.mean([self.weather_factors[w] for w in self.weather_series])
            
            pvdf = compute_pv_power_with_weather(
                self.times,
                latitude=self.pv_config.get('latitude', 10.7),
                longitude=self.pv_config.get('longitude', 106.6),
                tz=self.pv_config.get('tz', 'Asia/Ho_Chi_Minh'),
                surface_tilt=self.pv_config.get('surface_tilt', 10),
                surface_azimuth=self.pv_config.get('surface_azimuth', 180),
                module_parameters=self.pv_config.get('module_parameters', {'pdc0': 1.0}),
                weather_factor=avg_weather
            )
            self.pv_profile = pvdf['p_ac'].values

        # Price Logic
        if self.price_profile_input is not None and len(self.price_profile_input) > 0:
            self.price_profile = np.array(self.price_profile_input[:self.sim_steps])
        else:
            self.price_profile = np.array([get_tiered_price(t.hour, self.price_tiers) for t in self.times])
            
        # Load Logic
        self.load_schedules = self.behavior_gen.generate_for_times(self.times, weather=self.weather_series[0])
        
        # Reset States
        self.t = 0
        self.soc = float(self.battery.get('soc_init', 0.5))
        self.su_status = np.zeros(self.N_su)
        self.si_status = np.zeros(self.N_si)
        self.total_cost = 0.0

    def _get_obs(self):
        idx = min(self.t, self.sim_steps - 1)
        hour = int(self.times[idx].hour)
        
        must_run = float(self.load_schedules[idx]['must_run'])
        n_home = self.load_schedules[idx]['n_home']
        temp_out = self.load_schedules[idx]['temp_out']
        
        pv_now = float(self.pv_profile[idx]) if self.pv_profile is not None else 0.0
        
        horizon = 6
        end_idx = min(self.sim_steps, idx + 1 + horizon)
        future_pv = float(np.sum(self.pv_profile[idx+1 : end_idx])) if idx+1 < self.sim_steps else 0.0
        
        # Progress Tracking (Normalize 0-1)
        if self.N_su > 0:
            su_prog = np.mean([self.su_status[i]/max(1, self.su_devs[i]['L']) for i in range(self.N_su)])
        else: su_prog = 1.0
            
        if self.N_si > 0:
            si_prog = np.mean([self.si_status[i]/max(1, self.si_devs[i]['E']) for i in range(self.N_si)])
        else: si_prog = 1.0
            
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        
        return np.array([self.soc, pv_now, must_run, future_pv, 
                         hour_sin, hour_cos, n_home, temp_out, 
                         su_prog, si_prog], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internal()
        return self._get_obs(), {}

    def step(self, action):
        idx = self.t
        must = float(self.load_schedules[idx]['must_run'])
        n_home = self.load_schedules[idx]['n_home']
        temp_out = self.load_schedules[idx]['temp_out']
        
        # Ensure action format
        if np.isscalar(action): action = [action]
        action = np.array(action, dtype=int).flatten()
        
        # [FIX]: Kiểm tra độ dài action đúng với khai báo Space
        expected_len = self.N_su + self.N_si + self.N_ad
        if action.size != expected_len:
            # Fallback tự động điều chỉnh nếu lỗi
            if action.size > expected_len: action = action[:expected_len]
            else: action = np.pad(action, (0, expected_len - action.size))
        
        p_dev_load = 0.0
        comfort_penalty = 0.0
        
        # 1. SU Devices
        for i in range(self.N_su):
            if i < len(action) and action[i] == 1:
                if self.su_status[i] < self.su_devs[i]['L']:
                    p_dev_load += self.su_devs[i]['rate']
                    self.su_status[i] += 1
                else:
                    comfort_penalty += 0.5 # Run thừa
                    
        # 2. SI Devices
        offset_si = self.N_su
        for i in range(self.N_si):
            act_idx = offset_si + i
            if act_idx < len(action) and action[act_idx] == 1:
                if self.si_status[i] < self.si_devs[i]['E']:
                    p_dev_load += self.si_devs[i]['rate']
                    self.si_status[i] += self.si_devs[i]['rate'] * self.time_step_h
                else:
                    comfort_penalty += 0.5 # Sạc thừa
                    
        # 3. AC (Adjustable)
        offset_ad = self.N_su + self.N_si
        for i in range(self.N_ad):
            act_idx = offset_ad + i
            if act_idx < len(action):
                act = action[act_idx]
                if act == 1:
                    p_dev_load += self.ad_devs[i]['P_com']
                
                # Comfort Check
                if n_home > 0:
                    if temp_out > 28.0 and act == 0: comfort_penalty += 5.0
                    if temp_out < 25.0 and act == 1: comfort_penalty += 1.0

        total_load = must + p_dev_load
        
        # Battery Logic
        pv_gen = float(self.pv_profile[idx])
        surplus = pv_gen - total_load
        
        # [FIX]: Rule-based Battery Control (Automatic)
        if surplus >= 0:
            p_ch = min(surplus, float(self.battery.get('p_charge_max_kw', 3.0)))
            eta = float(self.battery.get('eta_ch', 0.95))
            self.soc = clamp(self.soc + (p_ch * eta)/self.C_bat, self.battery['soc_min'], self.battery['soc_max'])
            grid = -(surplus - p_ch)
        else:
            deficit = -surplus
            p_dis_max = float(self.battery.get('p_discharge_max_kw', 3.0))
            
            # Check available energy in battery
            kwh_avail = (self.soc - self.battery['soc_min']) * self.C_bat
            # Power available to discharge
            p_dis_avail = (kwh_avail * float(self.battery.get('eta_dis', 0.95))) / self.time_step_h
            
            p_dis = min(deficit, p_dis_max, p_dis_avail)
            
            eta = float(self.battery.get('eta_dis', 0.95))
            self.soc = clamp(self.soc - (p_dis/eta)/self.C_bat, self.battery['soc_min'], self.battery['soc_max'])
            grid = deficit - p_dis

        current_weather = self.weather_series[idx] if idx < len(self.weather_series) else "unknown"
        price = self.price_profile[idx]
        cost = max(0.0, grid) * price
        self.total_cost += cost
        
        reward = -cost / 1000.0 - comfort_penalty
        if self.soc <= self.battery['soc_min'] + 0.01: reward -= 0.5
        
        self.t += 1
        done = self.t >= self.sim_steps
        
        if done:
            # Completion Penalty
            for i in range(self.N_su):
                if self.su_status[i] < self.su_devs[i]['L']: reward -= 20.0
            for i in range(self.N_si):
                if self.si_status[i] < self.si_devs[i]['E'] * 0.9: reward -= 20.0
                
        info = {
            'cost': cost, 'soc': self.soc, 'temp': temp_out, 
            'n_home': n_home, 'pv': pv_gen, 'load': total_load,
            'weather': current_weather
        }
        
        return self._get_obs() if not done else np.zeros(10, dtype=np.float32), float(reward), done, False, info