# backend/expert_utils.py
"""
Centralized Expert Logic for Smart Home RL
FIXED: Strict AC OFF when cold to prevent over-cooling.
"""

import numpy as np
from device_config import ROOM_OCCUPANCY_HOURS, EV_CONFIG, THERMAL_CONSTANTS, DEVICE_CONFIG


def is_room_occupied(room, hour):
    """Check if room is occupied at given hour"""
    for start, end in ROOM_OCCUPANCY_HOURS.get(room, []):
        if start <= end and start <= hour < end:
            return True
        if start > end and (hour >= start or hour < end):
            return True
    return False


def expert_heuristic_action(obs, hour, price):
    """
    FIXED: Ensure AC is strictly OFF (-1.0) when cold.
    """
    action = np.zeros(7, dtype=np.float32)
    
    soc = obs[0]
    n_home = obs[6]
    room_temps = obs[8:11]
    ev_soc = obs[12]
    
    # [0] Battery: Aggressive arbitrage
    if price[hour] < 0.12:
        action[0] = 1.0 if soc < 0.9 else 0.0  # Charge full speed
    elif price[hour] > 0.20:
        action[0] = -1.0 if soc > 0.2 else 0.0  # Discharge full speed
    else:
        action[0] = 0.0
    
    # [1-3] ACs: Strict Comfort Logic
    comfort_temp = THERMAL_CONSTANTS["comfort_temp"]
    for idx, room in enumerate(["living", "master", "bed2"]):
        if is_room_occupied(room, hour) and n_home > 0:
            temp_diff = room_temps[idx] - comfort_temp
            
            if temp_diff > 1.0:  # > 26°C -> Cooling
                # Proportional but capped at 1.0
                action[1 + idx] = min(1.0, temp_diff / 3.0)
            elif temp_diff < -0.5:  # < 24.5°C -> STRICT OFF
                # CRITICAL FIX: Was -0.5 (25%), now -1.0 (0%)
                action[1 + idx] = -1.0 
            else:
                # Maintain range [24.5, 26]: Run gentle low power
                # -0.2 maps to ~40% capacity (sufficient to maintain)
                action[1 + idx] = -0.2 
        else:
            action[1 + idx] = -1.0  # Off if empty
    
    # [4] EV
    deadline = EV_CONFIG["deadline_hour"]
    target_soc = EV_CONFIG["min_target_soc"]
    hours_left = (deadline - hour) % 24
    if hours_left == 0:
        hours_left = 24
    
    if (hour >= 22 or hour < 5) and ev_soc < 0.95:
        action[4] = 1.0
    elif ev_soc < target_soc and hours_left < 4:
        action[4] = 1.0
    else:
        action[4] = -1.0
    
    # [5-6] Shiftable
    if price[hour] < 0.15:  # Run when cheap
        action[5] = 1.0
        action[6] = 1.0
    else:
        action[5] = -1.0
        action[6] = -1.0
    
    return np.clip(action, -1, 1)
