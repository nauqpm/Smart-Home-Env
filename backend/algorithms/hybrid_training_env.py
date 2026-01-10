"""
Hybrid Training Environment Wrapper
Applies SOFT expert rules to actions DURING TRAINING.

This wrapper MUST MATCH hybrid_wrapper.py logic exactly!
Otherwise training-inference mismatch will occur.

SYNCED WITH: hybrid_wrapper.py (same soft constraint logic)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulation.device_config import (
    DEVICE_CONFIG, EV_CONFIG, ROOM_OCCUPANCY_HOURS, ACTION_INDICES
)
from algorithms.expert_utils import is_room_occupied


class HybridTrainingEnvWrapper(gym.Wrapper):
    """
    Gymnasium Wrapper that applies SOFT expert rules to actions before 
    passing to the underlying SmartHomeEnv.
    
    CRITICAL: Logic MUST match hybrid_wrapper.py exactly!
    """
    
    def __init__(self, env):
        super().__init__(env)
        # Use dynamic indices from config
        self.idx_bat = ACTION_INDICES.get('battery', 0)
        self.idx_ac_living = ACTION_INDICES.get('ac_living', 1)
        self.idx_ac_master = ACTION_INDICES.get('ac_master', 2)
        self.idx_ac_bed2 = ACTION_INDICES.get('ac_bed2', 3)
        self.idx_ev = ACTION_INDICES.get('ev', 4)
        self.idx_wm = ACTION_INDICES.get('wm', 5)
        self.idx_dw = ACTION_INDICES.get('dw', 6)
        
    def step(self, action):
        """Apply expert rules to action, then step environment."""
        # Get current state from underlying env
        try:
            hour = self.env.t % 24 if hasattr(self.env, 't') else 0
            soc = getattr(self.env, 'soc', 0.5)
            ev_soc = getattr(self.env, 'ev_soc', 0.5)
            n_home = 0
            if hasattr(self.env, 'load_schedules') and self.env.t < len(self.env.load_schedules):
                n_home = self.env.load_schedules[self.env.t].get('n_home', 0)
            
            # Get room temperatures (RAW, not normalized)
            room_temps = getattr(self.env, 'room_temps', {
                'living': 25.0, 'master': 25.0, 'bed2': 25.0
            })
        except (AttributeError, IndexError) as e:
            hour, soc, ev_soc, n_home = 12, 0.5, 0.5, 0
            room_temps = {'living': 25.0, 'master': 25.0, 'bed2': 25.0}
        
        # Apply soft expert overrides (matches hybrid_wrapper.py)
        modified_action = self._apply_soft_expert_rules(
            action.copy(), hour, soc, ev_soc, n_home, room_temps
        )
        
        return self.env.step(modified_action)
    
    def _apply_soft_expert_rules(self, action, hour, soc, ev_soc, n_home, room_temps):
        """
        Apply SOFT expert rules - same logic as hybrid_wrapper.py
        
        Philosophy: Trust PPO in comfort zone, override only for safety
        """
        a = np.array(action, dtype=np.float32).flatten()
        
        t_living = room_temps.get('living', 25.0)
        t_master = room_temps.get('master', 25.0)
        t_bed2 = room_temps.get('bed2', 25.0)
        
        # ==========================================================
        # A. SMART AC OVERRIDE (Soft Constraint)
        # ==========================================================
        def smart_ac_override(ppo_action, temp, room):
            if not is_room_occupied(room, hour):
                return -1.0  # Force OFF if empty
            
            if temp > 28.0:
                return 1.0   # SAFETY: Too hot
            elif temp > 27.5:
                return max(ppo_action, 0.5)  # Ensure cooling
            elif temp < 22.0:
                return -1.0  # Too cold
            else:
                return ppo_action  # Trust PPO (pre-cooling allowed!)
        
        a[self.idx_ac_living] = smart_ac_override(a[self.idx_ac_living], t_living, "living")
        a[self.idx_ac_master] = smart_ac_override(a[self.idx_ac_master], t_master, "master")
        a[self.idx_ac_bed2] = smart_ac_override(a[self.idx_ac_bed2], t_bed2, "bed2")
        
        # ==========================================================
        # B. EV SMART CHARGING (Priority Override)
        # ==========================================================
        is_off_peak = (hour >= 22 or hour < 4)
        
        if is_off_peak and ev_soc < 0.9:
            ev_deficit = 0.9 - ev_soc
            a[self.idx_ev] = max(a[self.idx_ev], min(1.0, ev_deficit * 3.0))
        
        # Deadline urgency
        deadline = EV_CONFIG.get("deadline_hour", 7)
        hours_left = max(1, (deadline - hour) % 24)
        capacity = DEVICE_CONFIG["shiftable"]["ev"]["capacity"]
        pmax = DEVICE_CONFIG["shiftable"]["ev"]["power_max"]
        energy_needed = (0.9 - ev_soc) * capacity
        
        if energy_needed > 0 and energy_needed > pmax * hours_left * 0.8:
            a[self.idx_ev] = 1.0
        
        # ==========================================================
        # C. BATTERY SAFETY (Technical + Economic)
        # ==========================================================
        is_peak_hour = (9 <= hour <= 11) or (17 <= hour <= 20)
        
        # Technical: Prevent deep discharge
        if soc < 0.15:
            a[self.idx_bat] = max(0.0, a[self.idx_bat])
        
        # Economic: No charging during peak
        if is_peak_hour and soc > 0.2:
            a[self.idx_bat] = min(0.0, a[self.idx_bat])
        
        # Prevent over-discharge during peak if low SOC
        if is_peak_hour and soc < 0.25:
            a[self.idx_bat] = max(-0.3, a[self.idx_bat])
        
        # ==========================================================
        # D. WM/DW - Let PPO decide (no override)
        # ==========================================================
        
        return np.clip(a, -1, 1)
    
    def reset(self, **kwargs):
        """Reset underlying environment."""
        return self.env.reset(**kwargs)
