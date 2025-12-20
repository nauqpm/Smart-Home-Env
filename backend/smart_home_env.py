import numpy as np
import pandas as pd
import math
import logging
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces

# Import device configuration
from device_config import (
    DEVICE_CONFIG, ACTION_INDICES, ROOM_OCCUPANCY_HOURS,
    THERMAL_CONSTANTS, EV_CONFIG
)

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

def calculate_vietnam_tiered_bill(kwh: float) -> float:
    """
    Tier 1 (0 - 50 kWh): 1,984 VND
    Tier 2 (51 - 100 kWh): 2,050 VND
    Tier 3 (101 - 200 kWh): 2,380 VND
    Tier 4 (201 - 300 kWh): 2,998 VND
    Tier 5 (301 - 400 kWh): 3,350 VND
    Tier 6 (401+ kWh): 3,460 VND
    """
    tiers = [
        (50, 1984),
        (50, 2050), # Next 50 (Total 100)
        (100, 2380), # Next 100 (Total 200)
        (100, 2998), # Next 100 (Total 300)
        (100, 3350), # Next 100 (Total 400)
        (float('inf'), 3460)
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

def get_tiered_price(hour: int, tiers: List[Tuple[int, float]]) -> float:
    # Deprecated for billing, but kept if needed for momentary pricing (optional)
    # Returning average for backwards compatibility or safe fallback
    return 2500.0 

def compute_pv_power_with_weather(times, latitude, longitude, tz, surface_tilt, 
                                  surface_azimuth, module_parameters, weather_factor=1.0):
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
            
            # Base Load Logic (Fridge only now, devices are handled separately)
            base = DEVICE_CONFIG['fixed']['fridge']['power']
            
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


# --- HELPER FUNCTIONS ---
def is_room_occupied(room: str, hour: int) -> bool:
    """Check if a room is occupied at the given hour based on ROOM_OCCUPANCY_HOURS"""
    ranges = ROOM_OCCUPANCY_HOURS.get(room, [])
    for (start, end) in ranges:
        # Handle overnight ranges (e.g., 22-24, 0-6)
        if start <= end:
            if start <= hour < end:
                return True
        else:  # Overnight range like (22, 6) would be split
            if hour >= start or hour < end:
                return True
    return False


# --- MAIN ENV CLASS ---
class SmartHomeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def get_frontend_state(self):
        return {
            'total_bill': int(self.total_cost),
            'import_kwh': round(self.cumulative_import_kwh, 2),
            'export_kwh': round(self.cumulative_export_kwh, 2),
            'soc': round(self.soc * 100, 1)
        }

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
        
        # --- NEW: Action Space (7 dimensions) ---
        # [battery, ac_living, ac_master, ac_bed2, ev, wm, dw]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        
        # --- NEW: Observation Space (13 dimensions) ---
        # [SOC, PV, MustRun, FuturePV, Sin, Cos, Occupancy, TempOut, 
        #  TempLiving, TempMaster, TempBed2, WM_remaining, EV_soc]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        
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
        self.cumulative_import_kwh = 0.0
        self.cumulative_export_kwh = 0.0
        self.total_cost = 0.0
        
        # --- NEW: Room temperature states ---
        self.room_temps = {
            'living': THERMAL_CONSTANTS['comfort_temp'],
            'master': THERMAL_CONSTANTS['comfort_temp'],
            'bed2': THERMAL_CONSTANTS['comfort_temp']
        }
        
        # --- NEW: Shiftable device states ---
        # Washing Machine: randomly needs to run today
        self.wm_remaining = np.random.choice([0, 2], p=[0.3, 0.7])  # 70% chance needs washing
        self.wm_deadline = 22  # Must complete by 10 PM
        
        # Dishwasher: runs after dinner
        self.dw_remaining = np.random.choice([0, 1], p=[0.4, 0.6])  # 60% chance
        self.dw_deadline = 23
        
        # EV: needs to be charged by morning
        self.ev_soc = np.random.uniform(0.2, 0.5)  # Start with 20-50% charge
        self.ev_deadline = EV_CONFIG['deadline_hour']
        
        # Store current outdoor temperature
        self.temp_out = self.load_schedules[0]['temp_out']
        
        # Device states for info output
        self.device_states = {}

    def _get_obs(self):
        idx = min(self.t, self.sim_steps - 1)
        hour = int(self.times[idx].hour)
        
        must_run = float(self.load_schedules[idx]['must_run'])
        n_home = self.load_schedules[idx]['n_home']
        self.temp_out = self.load_schedules[idx]['temp_out']
        
        pv_now = float(self.pv_profile[idx]) if self.pv_profile is not None else 0.0
        
        horizon = 6
        end_idx = min(self.sim_steps, idx + 1 + horizon)
        future_pv = float(np.sum(self.pv_profile[idx+1 : end_idx])) if idx+1 < self.sim_steps else 0.0
        
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        
        # NEW: Extended observation with room temperatures and device states
        return np.array([
            self.soc,                           # 0: Battery SOC
            pv_now,                             # 1: Current PV generation
            must_run,                           # 2: Fixed load (fridge)
            future_pv,                          # 3: Future PV prediction
            hour_sin,                           # 4: Hour encoding (sin)
            hour_cos,                           # 5: Hour encoding (cos)
            n_home,                             # 6: Occupancy count
            self.temp_out,                      # 7: Outdoor temperature
            self.room_temps['living'],          # 8: Living room temperature
            self.room_temps['master'],          # 9: Master bedroom temperature
            self.room_temps['bed2'],            # 10: Bedroom 2 temperature
            self.wm_remaining,                  # 11: WM hours remaining
            self.ev_soc                         # 12: EV SOC
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internal()
        return self._get_obs(), {}

    def _update_room_temp(self, room: str, ac_power_normalized: float) -> float:
        """
        Update room temperature based on AC power and outdoor temperature.
        Formula: T_next = T_curr + k1*(T_out - T_curr) - k2*Power_AC
        """
        k1 = THERMAL_CONSTANTS['k1']
        k2 = THERMAL_CONSTANTS['k2']
        current_temp = self.room_temps[room]
        
        # Temperature tends towards outdoor temp, AC cools it down
        delta_temp = k1 * (self.temp_out - current_temp) - k2 * ac_power_normalized
        new_temp = current_temp + delta_temp
        
        # Clamp to reasonable range
        return clamp(new_temp, 18.0, 38.0)
    
    def _calculate_comfort_penalty(self, room: str, has_occupant: bool) -> float:
        """Calculate comfort penalty based on room temperature deviation"""
        if not has_occupant:
            return 0.0
        
        target = THERMAL_CONSTANTS['comfort_temp']
        tolerance = THERMAL_CONSTANTS['comfort_tolerance']
        current = self.room_temps[room]
        
        deviation = abs(current - target)
        if deviation <= tolerance:
            return 0.0
        else:
            # Exponential penalty for temperature deviation beyond tolerance
            return (deviation - tolerance) ** 2 * 2.0

    def step(self, action):
        idx = self.t
        hour = int(self.times[idx].hour)
        
        must_run = float(self.load_schedules[idx]['must_run'])  # Fridge power
        n_home = self.load_schedules[idx]['n_home']
        self.temp_out = self.load_schedules[idx]['temp_out']
        
        # Ensure action is numpy array
        action = np.array(action, dtype=np.float32).flatten()
        if len(action) != 7:
            # Fallback: pad or truncate
            if len(action) > 7:
                action = action[:7]
            else:
                action = np.pad(action, (0, 7 - len(action)))
        
        # --- 1. UNPACK ACTIONS ---
        act_battery = action[ACTION_INDICES['battery']]  # -1 to 1
        
        # Normalize AC actions from [-1, 1] to [0, 1]
        act_ac_living = (action[ACTION_INDICES['ac_living']] + 1) / 2
        act_ac_master = (action[ACTION_INDICES['ac_master']] + 1) / 2
        act_ac_bed2 = (action[ACTION_INDICES['ac_bed2']] + 1) / 2
        
        # EV: normalize to [0, 1]
        act_ev = (action[ACTION_INDICES['ev']] + 1) / 2
        
        # Shiftable: threshold at 0
        act_wm = 1 if action[ACTION_INDICES['wm']] > 0 else 0
        act_dw = 1 if action[ACTION_INDICES['dw']] > 0 else 0
        
        # --- 2. PROCESS DEVICES ---
        total_load = 0.0
        comfort_penalty = 0.0
        self.device_states = {}
        
        # --- A. AC Processing with Temperature Dynamics ---
        ac_rooms = [
            ('living', act_ac_living, 'ac_living'),
            ('master', act_ac_master, 'ac_master'),
            ('bed2', act_ac_bed2, 'ac_bed2')
        ]
        
        for room, act_val, device_key in ac_rooms:
            cfg = DEVICE_CONFIG['adjustable'][device_key]
            power_kw = act_val * cfg['power_max']
            
            # Update room temperature
            self.room_temps[room] = self._update_room_temp(room, act_val)
            
            # Add power consumption
            total_load += power_kw
            
            # Calculate comfort penalty if room is occupied
            is_occupied = is_room_occupied(room, hour)
            comfort_penalty += self._calculate_comfort_penalty(room, is_occupied)
            
            # Store device state (ON if power > 0.1 kW)
            self.device_states[device_key] = 1 if power_kw > 0.1 else 0
        
        # --- B. Shiftable Devices (WM, DW) with Deadline Logic ---
        
        # Washing Machine
        if self.wm_remaining > 0:
            if act_wm == 1:
                total_load += DEVICE_CONFIG['shiftable']['wm']['power']
                self.wm_remaining -= 1
                self.device_states['wm'] = 1
            else:
                self.device_states['wm'] = 0
                # Deadline penalty: if approaching deadline without completing
                hours_to_deadline = self.wm_deadline - hour
                if hours_to_deadline <= self.wm_remaining and hours_to_deadline > 0:
                    comfort_penalty += 5.0  # Penalize for risking deadline
        else:
            self.device_states['wm'] = 0
        
        # Dishwasher
        if self.dw_remaining > 0:
            if act_dw == 1:
                total_load += DEVICE_CONFIG['shiftable']['dw']['power']
                self.dw_remaining -= 1
                self.device_states['dw'] = 1
            else:
                self.device_states['dw'] = 0
                hours_to_dw_deadline = self.dw_deadline - hour
                if hours_to_dw_deadline <= self.dw_remaining and hours_to_dw_deadline > 0:
                    comfort_penalty += 3.0
        else:
            self.device_states['dw'] = 0
        
        # --- C. EV Charging ---
        ev_power = act_ev * DEVICE_CONFIG['shiftable']['ev']['power_max']
        ev_capacity = DEVICE_CONFIG['shiftable']['ev']['capacity']
        
        # Update EV SOC
        energy_added = ev_power * self.time_step_h  # kWh
        self.ev_soc = clamp(self.ev_soc + energy_added / ev_capacity, 0.0, 1.0)
        
        total_load += ev_power
        self.device_states['ev'] = round(act_ev, 2)  # Store as continuous value
        
        # EV deadline penalty
        if hour < self.ev_deadline and self.ev_soc < EV_CONFIG['min_target_soc']:
            hours_to_ev_deadline = self.ev_deadline - hour if hour < self.ev_deadline else 24 - hour + self.ev_deadline
            energy_needed = (EV_CONFIG['min_target_soc'] - self.ev_soc) * ev_capacity
            max_possible = DEVICE_CONFIG['shiftable']['ev']['power_max'] * hours_to_ev_deadline
            if energy_needed > max_possible:
                comfort_penalty += 10.0  # Severe penalty for missing EV deadline
        
        # --- D. Fixed Devices (Lights) - Rule-based on Occupancy ---
        for light_key, cfg in DEVICE_CONFIG['fixed'].items():
            if cfg.get('always_on'):  # Fridge
                # Already included in must_run
                continue
            
            room = cfg['room']
            is_occupied = is_room_occupied(room, hour)
            
            # Add random factor for toilet (50% chance when occupied hours)
            if room == 'toilet' and is_occupied:
                is_occupied = np.random.random() < 0.3  # 30% chance during active hours
            
            if is_occupied:
                total_load += cfg['power']
                self.device_states[light_key] = 1
            else:
                self.device_states[light_key] = 0
        
        # Add fixed load (fridge)
        total_load += must_run
        
        # --- 3. BATTERY & GRID LOGIC ---
        pv_gen = float(self.pv_profile[idx])
        surplus = pv_gen - total_load
        
        # Battery action interpretation:
        # act_battery > 0: Prefer charging
        # act_battery < 0: Prefer discharging
        p_charge_max = float(self.battery.get('p_charge_max_kw', 3.0))
        p_discharge_max = float(self.battery.get('p_discharge_max_kw', 3.0))
        
        if surplus >= 0:
            # More PV than load - charge battery with surplus
            p_ch = min(surplus, p_charge_max)
            eta = float(self.battery.get('eta_ch', 0.95))
            self.soc = clamp(self.soc + (p_ch * eta) / self.C_bat, 
                           self.battery['soc_min'], self.battery['soc_max'])
            grid = -(surplus - p_ch)  # Export excess
            battery_state = 'charge' if p_ch > 0.01 else 'idle'
        else:
            deficit = -surplus
            
            # Check if we should discharge (act_battery < 0 or auto-discharge)
            if act_battery < 0:
                # Agent wants to discharge
                discharge_intent = abs(act_battery)  # 0 to 1
            else:
                # Default: discharge to cover deficit
                discharge_intent = 1.0
            
            # Available energy in battery
            kwh_avail = (self.soc - self.battery['soc_min']) * self.C_bat
            p_dis_avail = (kwh_avail * float(self.battery.get('eta_dis', 0.95))) / self.time_step_h
            
            p_dis = min(deficit, p_discharge_max, p_dis_avail) * discharge_intent
            
            eta = float(self.battery.get('eta_dis', 0.95))
            self.soc = clamp(self.soc - (p_dis / eta) / self.C_bat, 
                           self.battery['soc_min'], self.battery['soc_max'])
            grid = deficit - p_dis  # Import from grid
            battery_state = 'discharge' if p_dis > 0.01 else 'idle'
        
        self.device_states['battery'] = battery_state

        current_weather = self.weather_series[idx] if idx < len(self.weather_series) else "unknown"
        
        # --- 4. BILLING LOGIC (TIERED) ---
        grid_kwh = grid * self.time_step_h
        
        if grid_kwh > 0:
            self.cumulative_import_kwh += grid_kwh
        else:
            self.cumulative_export_kwh += abs(grid_kwh)
            
        import_bill = calculate_vietnam_tiered_bill(self.cumulative_import_kwh)
        export_revenue = self.cumulative_export_kwh * 2000.0  # Feed-in Tariff
        
        self.total_cost = import_bill - export_revenue
        cost = 0  # Step cost is implicit in total
        
        # --- 5. REWARD CALCULATION ---
        reward = -cost / 1000.0 - comfort_penalty
        
        # Penalty for low battery
        if self.soc <= self.battery['soc_min'] + 0.01:
            reward -= 0.5
        
        self.t += 1
        done = self.t >= self.sim_steps
        
        # End-of-day penalties for incomplete tasks
        if done:
            # Washing machine didn't complete
            if self.wm_remaining > 0:
                reward -= 20.0
            # Dishwasher didn't complete
            if self.dw_remaining > 0:
                reward -= 15.0
            # EV not charged enough
            if self.ev_soc < EV_CONFIG['min_target_soc']:
                reward -= 25.0
        
        # --- 6. BUILD INFO DICT ---
        info = {
            'cost': cost,
            'total_cost': int(self.total_cost),
            'soc': self.soc,
            'temp': self.temp_out,
            'n_home': n_home,
            'pv': pv_gen,
            'load': total_load,
            'weather': current_weather,
            'cumulative_import': self.cumulative_import_kwh,
            'cumulative_export': self.cumulative_export_kwh,
            'hour': hour,
            # Device-specific states
            'ac_living': self.device_states.get('ac_living', 0),
            'ac_master': self.device_states.get('ac_master', 0),
            'ac_bed2': self.device_states.get('ac_bed2', 0),
            'light_living': self.device_states.get('light_living', 0),
            'light_master': self.device_states.get('light_master', 0),
            'light_bed2': self.device_states.get('light_bed2', 0),
            'light_kitchen': self.device_states.get('light_kitchen', 0),
            'light_toilet': self.device_states.get('light_toilet', 0),
            'wm': self.device_states.get('wm', 0),
            'dw': self.device_states.get('dw', 0),
            'ev': self.device_states.get('ev', 0),
            'battery': self.device_states.get('battery', 'idle'),
            # Internal states for hybrid agent
            'wm_remaining': self.wm_remaining,
            'dw_remaining': self.dw_remaining,
            'ev_soc': self.ev_soc,
            'room_temps': self.room_temps.copy()
        }
        
        return self._get_obs() if not done else np.zeros(13, dtype=np.float32), float(reward), done, False, info