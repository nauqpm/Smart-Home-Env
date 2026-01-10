"""
Expert Utilities for Smart Home Energy Management
Provides expert heuristic actions for AC, EV, Battery control.

FIXED ISSUES:
- Use ACTION_INDICES instead of hardcoded numbers
- Removed dead code
- Fixed bare except clauses
- Made EV charging proportional
- Improved AC thermostat thresholds
"""

import numpy as np
from simulation.device_config import DEVICE_CONFIG, EV_CONFIG, ROOM_OCCUPANCY_HOURS, ACTION_INDICES


def is_room_occupied(room: str, hour: int) -> bool:
    """Check if room is occupied at specific hour based on schedule."""
    ranges = ROOM_OCCUPANCY_HOURS.get(room, [])
    for start, end in ranges:
        if start <= end:
            if start <= hour < end:
                return True
        else:
            if hour >= start or hour < end:
                return True
    return False
    # FIXED: Removed dead code (duplicate return False)


def calculate_expert_action(obs, hour, outdoor_temp, n_home, ev_soc):
    """
    Calculates a baseline action based on SMART expert rules.
    
    IMPORTANT: Uses ACTION_INDICES for robust index mapping.
    
    Args:
        obs: Current observation 
             - obs[0] = battery SOC (0-1)
             - obs[1] = PV generation (kW, raw)
             - obs[2] = must-run load (kW)
             - obs[8-10] = room temps (RAW °C, NOT normalized!)
        hour (int): Current hour
        outdoor_temp (float): Outdoor temperature (unused, kept for API compat)
        n_home (int): Number of people home
        ev_soc (float): EV State of Charge (0-1)
    
    Returns:
        np.array: 7-dim action vector
    """
    action = np.zeros(7, dtype=np.float32)
    
    # Get indices from config (not hardcoded!)
    idx_bat = ACTION_INDICES.get('battery', 0)
    idx_ac_living = ACTION_INDICES.get('ac_living', 1)
    idx_ac_master = ACTION_INDICES.get('ac_master', 2)
    idx_ac_bed2 = ACTION_INDICES.get('ac_bed2', 3)
    idx_ev = ACTION_INDICES.get('ev', 4)
    
    # Extract observation values safely
    try:
        soc = float(obs[0])
        pv_gen = float(obs[1])
        net_load = float(obs[2])
        # Room temps are RAW (NOT normalized) in SmartHomeEnv._get_obs()
        t_living = float(obs[8])
        t_master = float(obs[9])
        t_bed2 = float(obs[10])
    except (IndexError, TypeError) as e:
        # Proper exception handling (not bare except)
        soc = 0.5
        pv_gen = 0.0
        net_load = 0.3
        t_living = t_master = t_bed2 = 25.0

    # =========================================================
    # 1. EV CHARGING (Priority 1) - Proportional, not binary
    # =========================================================
    is_off_peak = (hour >= 22 or hour < 4)
    ev_deficit = max(0.0, 0.9 - ev_soc)  # How much more SOC needed
    
    if is_off_peak and ev_soc < 0.9:
        # Proportional charging: charge harder if more deficit
        action[idx_ev] = min(1.0, ev_deficit * 3.0)
    elif pv_gen > net_load + 0.5 and ev_soc < 0.9:
        # Use excess solar for EV (regardless of absolute PV value)
        excess_ratio = (pv_gen - net_load) / max(pv_gen, 0.1)
        action[idx_ev] = min(1.0, excess_ratio * ev_deficit * 2.0)
    else:
        action[idx_ev] = 0.0
        
    # =========================================================
    # 2. BATTERY RULES (Priority 2) - Economic + Technical Safety
    # =========================================================
    is_peak_hour = (9 <= hour <= 11) or (17 <= hour <= 20)
    
    if pv_gen > net_load + 0.5:
        # Charge from excess solar (always good)
        action[idx_bat] = 0.8
    elif is_peak_hour and soc > 0.2:
        # Discharge during peak to save money
        action[idx_bat] = -0.6
    elif is_peak_hour and soc <= 0.2:
        # Low SOC during peak: don't discharge, but also don't charge (expensive!)
        action[idx_bat] = 0.0
    else:
        action[idx_bat] = 0.0

    # =========================================================
    # 3. AC THERMOSTAT (Priority 3) - Relaxed for Pre-cooling
    # =========================================================
    # Comfort Range: 24 - 27°C
    # NEW: Wider threshold to allow PPO pre-cooling decisions
    # Only force ON when very hot (>27.5), force OFF when not occupied
    # In comfort zone, return moderate value for blending
    
    def thermostat_action(temp, room, hour):
        """Smart thermostat with pre-cooling allowance."""
        if not is_room_occupied(room, hour):
            return -1.0  # Force OFF if empty
        
        if temp > 27.5:
            return 1.0   # Force High Cool (safety)
        elif temp > 26.5:
            return 0.5   # Medium cool
        elif temp > 25.5:
            return 0.2   # Light cool (allow PPO to optimize)
        else:
            return -1.0  # Comfortable, OFF
    
    action[idx_ac_living] = thermostat_action(t_living, "living", hour)
    action[idx_ac_master] = thermostat_action(t_master, "master", hour)
    action[idx_ac_bed2] = thermostat_action(t_bed2, "bed2", hour)
            
    return action


def get_residual_action(model, obs, hour, outdoor_temp, pv_gen, w_rl=0.1):
    """Blend expert and RL actions."""
    try:
        n_home = int(obs[6])
        ev_soc = float(obs[12]) if len(obs) > 12 else 0.5
    except (IndexError, TypeError):
        n_home = 0
        ev_soc = 0.5
    
    expert_act = calculate_expert_action(obs, hour, outdoor_temp, n_home, ev_soc)
    
    rl_act, _ = model.predict(obs, deterministic=True)
    rl_act = np.array(rl_act).flatten()
    final_act = (1 - w_rl) * expert_act + w_rl * rl_act
    return np.clip(final_act, -1.0, 1.0)


def expert_heuristic_action(obs, hour, price):
    """
    Adapter for train_il_bc.py
    
    Note: obs[7] is temp_out (RAW temperature from SmartHomeEnv)
    """
    try:
        n_home = int(obs[6])
        outdoor_temp = float(obs[7])  # RAW temp, not normalized
        ev_soc = float(obs[12]) if len(obs) > 12 else 0.5 
        
        return calculate_expert_action(obs, hour, outdoor_temp, n_home, ev_soc)
    except (IndexError, TypeError, ValueError) as e:
        return np.zeros(7)
