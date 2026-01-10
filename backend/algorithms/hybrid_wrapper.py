"""
Hybrid Agent Wrapper (Inference Only)
Applies SOFT expert constraints on top of PPO decisions.

DESIGN PHILOSOPHY:
- Trust PPO for optimization within comfort zone
- Only override for SAFETY (too hot, too cold, low battery)
- Protect both TECHNICAL safety and ECONOMIC efficiency

FIXED ISSUES:
- Soft AC override (allow pre-cooling)
- Economic battery protection (no charging during peak)
- Proper exception handling
- Dynamic indices from ACTION_INDICES
"""

import numpy as np
import logging

from simulation.device_config import DEVICE_CONFIG, EV_CONFIG, ROOM_OCCUPANCY_HOURS, ACTION_INDICES
from algorithms.expert_utils import is_room_occupied

logger = logging.getLogger('HybridWrapper')


class HybridAgentWrapper:
    """
    PPO + Soft Expert Constraints
    
    Key difference from pure Expert:
    - Expert rules are SOFT constraints, not hard overrides
    - PPO can optimize within safe boundaries
    - Only override when safety/economics require it
    """

    def __init__(self, ppo_model):
        self.model = ppo_model
        logger.info("✅ HybridAgentWrapper initialized with SOFT Expert Constraints")
        
        # Use dynamic indices from config
        self.idx_bat = ACTION_INDICES.get('battery', 0)
        self.idx_ac_living = ACTION_INDICES.get('ac_living', 1)
        self.idx_ac_master = ACTION_INDICES.get('ac_master', 2)
        self.idx_ac_bed2 = ACTION_INDICES.get('ac_bed2', 3)
        self.idx_ev = ACTION_INDICES.get('ev', 4)
        self.idx_wm = ACTION_INDICES.get('wm', 5)
        self.idx_dw = ACTION_INDICES.get('dw', 6)

    def predict(self, obs, env_state=None, deterministic=True):
        """
        Predict action using PPO brain + SOFT Expert Constraints.
        
        Constraint Philosophy:
        - AC: Only override if too hot (>28°C) or room empty
        - Battery: Block discharge when low, block charge during peak
        - EV: Priority charge during off-peak or deadline urgency
        - WM/DW: Let PPO decide freely
        """
        # 1. Get Base Action from PPO Brain
        action, states = self.model.predict(obs, deterministic=deterministic)
        a = np.array(action, dtype=np.float32).flatten()
        
        # If no env_state, run in pure PPO mode (log warning)
        if env_state is None:
            logger.warning("⚠️ No env_state provided, running pure PPO mode")
            return np.clip(a, -1, 1), states

        # 2. Extract Context with proper error handling
        try:
            hour = int(env_state.get("hour", 12))
            soc = float(env_state.get("soc", obs[0]))
            ev_soc = float(env_state.get("ev_soc", 0.5))
            n_home = int(env_state.get("n_home", 0))
            
            # Room temps: prefer env_state, fallback to obs
            # obs[8-10] are RAW temps (NOT normalized)
            t_living = float(env_state.get("temp_living", obs[8] if len(obs) > 8 else 25.0))
            t_master = float(env_state.get("temp_master", obs[9] if len(obs) > 9 else 25.0))
            t_bed2 = float(env_state.get("temp_bed2", obs[10] if len(obs) > 10 else 25.0))
        except (IndexError, TypeError, ValueError) as e:
            logger.error(f"Error extracting context: {e}")
            hour, soc, ev_soc, n_home = 12, 0.5, 0.5, 0
            t_living = t_master = t_bed2 = 25.0

        # ==========================================================
        # A. SMART AC OVERRIDE (Soft Constraint)
        # ==========================================================
        # Philosophy: Trust PPO in comfort zone, override only for safety
        
        def smart_ac_override(ppo_action, temp, room):
            """Apply soft override to AC action."""
            if not is_room_occupied(room, hour):
                # Room empty: Force OFF (no comfort needed)
                return -1.0
            
            if temp > 28.0:
                # SAFETY: Too hot, force high cool
                return 1.0
            elif temp > 27.5:
                # Warm: ensure cooling, but PPO can fine-tune
                return max(ppo_action, 0.5)
            elif temp < 22.0:
                # Too cold: force OFF to prevent over-cooling
                return -1.0
            else:
                # Comfort zone (22-27.5): TRUST PPO (allow pre-cooling!)
                return ppo_action
        
        a[self.idx_ac_living] = smart_ac_override(a[self.idx_ac_living], t_living, "living")
        a[self.idx_ac_master] = smart_ac_override(a[self.idx_ac_master], t_master, "master")
        a[self.idx_ac_bed2] = smart_ac_override(a[self.idx_ac_bed2], t_bed2, "bed2")
        
        # ==========================================================
        # B. EV SMART CHARGING (Priority Override)
        # ==========================================================
        # Priority 1: Off-peak charging
        # Priority 2: Deadline urgency
        
        is_off_peak = (hour >= 22 or hour < 4)
        
        if is_off_peak and ev_soc < 0.9:
            # Off-peak: Proportional charging (not always max!)
            ev_deficit = 0.9 - ev_soc
            a[self.idx_ev] = max(a[self.idx_ev], min(1.0, ev_deficit * 3.0))
        
        # Deadline urgency check
        deadline = EV_CONFIG.get("deadline_hour", 7)
        hours_left = (deadline - hour) % 24
        hours_left = max(1, hours_left)
        
        capacity = DEVICE_CONFIG["shiftable"]["ev"]["capacity"]
        pmax = DEVICE_CONFIG["shiftable"]["ev"]["power_max"]
        min_target = EV_CONFIG.get("min_target_soc", 0.9)
        energy_needed = (min_target - ev_soc) * capacity  # kWh
        
        if energy_needed > 0 and energy_needed > pmax * hours_left * 0.8:
            # Urgent: Force charge to meet deadline
            a[self.idx_ev] = 1.0
             
        # ==========================================================
        # C. BATTERY SAFETY (Technical + Economic)
        # ==========================================================
        # Vietnam EVN peak hours: 9-11h and 17-20h (synced with ws_server.py)
        is_peak_hour = (9 <= hour < 11) or (17 <= hour < 20)
        
        # Technical safety: Prevent deep discharge
        if soc < 0.15:
            a[self.idx_bat] = max(0.0, a[self.idx_bat])
        
        # Economic protection: Prevent charging during peak (expensive!)
        # Exception: Allow if SOC is critically low
        if is_peak_hour and soc > 0.2:
            # During peak: Only allow discharge or idle, no charging
            a[self.idx_bat] = min(0.0, a[self.idx_bat])
        
        # Prevent over-discharge during high price periods
        if is_peak_hour and soc < 0.25:
            # Low SOC during peak: hold (don't force discharge)
            a[self.idx_bat] = max(-0.3, a[self.idx_bat])
        
        # ==========================================================
        # D. WM/DW (Chores) - Let PPO decide freely
        # ==========================================================
        # PPO was trained with deadline penalties, trust its scheduling
        # (No override applied to indices 5, 6)

        return np.clip(a, -1, 1), states
