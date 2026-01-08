import numpy as np

def calculate_expert_action(obs, hour, outdoor_temp, pv_gen):
    """
    Calculates a baseline action based on expert rules.
    
    Args:
        obs: Current observation from environment (unused directly here, but good for signature)
        hour (int): Current hour of day (0-23)
        outdoor_temp (float): Outdoor temperature in Celsius
        pv_gen (float): PV generation in kW (or whatever unit env uses, approx)
        
    Returns:
        np.array: A 7-dimensional action vector [-1, 1]
        [Battery, AC1, AC2, AC3, EV, WM, DW]
    """
    # Initialize zero action (idle)
    action = np.zeros(7, dtype=np.float32)
    
    # --- 1. Battery Rules ---
    # Charge during high solar (e.g., > 3.0 kW)
    if pv_gen > 3.0:
        action[0] = 0.8  # Strong charge
    # Discharge during night (no sun) or peak hours (17-23 usually)
    # Simple heuristic: Discharge if dark or evening
    elif hour >= 18 or hour < 6:
        action[0] = -0.6 # Moderate discharge
    else:
        action[0] = 0.0  # Idle/Float
        
    # --- 2. AC Rules ---
    # Only run AC if really hot
    if outdoor_temp > 32.0:
        action[1] = 1.0 # AC Living ON
        action[2] = 0.5 # AC Master Eco
        # AC Bed2 (index 3) optional, keep 0
    elif outdoor_temp > 30.0:
        action[1] = 0.7 # AC Living Eco
    
    # --- 3. Shiftable Loads (WM, DW, EV) ---
    # Expert rules for these are tricky without knowing state (remaining tasks).
    # Ideally, we let the RL handle the precise timing, or simple rules:
    # Run WM/DW during day if solar is good?
    # For now, let's leave them 0 (let RL decide entirely, or mix with 0)
    # Or provide a safe default like "Don't run unless urgent" (which is 0)
    
    # EV: Charge if solar is high?
    if pv_gen > 4.0:
        action[4] = 0.5
        
    return action

def get_residual_action(model, obs, hour, outdoor_temp, pv_gen, w_rl=0.1):
    """
    Implements Residual RL: Action = (1 - w) * Expert + w * RL
    
    Args:
        model: Loaded PPO model
        obs: Environment observation
        hour: Current hour
        outdoor_temp: Outdoor temp
        pv_gen: PV generation
        w_rl (float): Weight for RL component (0.0 to 1.0)
        
    Returns:
        np.array: Final clipped action
    """
    # 1. Expert Action
    expert_act = calculate_expert_action(obs, hour, outdoor_temp, pv_gen)
    
    # 2. RL Action
    rl_act, _ = model.predict(obs, deterministic=True)
    # Ensure flat array
    rl_act = np.array(rl_act).flatten()
    
    # 3. Residual Combination
    # Formula: Final = (1 - w) * Expert + w * RL
    # This means we trust Expert by default, and let RL nudge it.
    final_act = (1 - w_rl) * expert_act + w_rl * rl_act
    
    # 4. Safety Clipping
    final_act = np.clip(final_act, -1.0, 1.0)
    
    return final_act

# Alias for compatibility with train_il_bc.py
# Note: train_il_bc expects (obs, hour, price), but calculate_expert_action expects (obs, hour, outdoor_temp, pv_gen).
# We need an adapter wrapper.

def expert_heuristic_action(obs, hour, price):
    """
    Adapter to match signature expected by train_il_bc.py:
    func(obs, hour, price) -> action
    
    We need to extract outdoor_temp and pv_gen from obs or heuristics.
    SmartHomeEnv Observation structure:
    [0] soc, [1] pv, [2] must, [3] fut_pv, [4]-[5] time_sin/cos, [6] n_home, [7] temp_out, ...
    So temp_out is obs[7], pv is obs[1]
    """
    try:
        pv_gen = obs[1] * 10.0 # Un-normalize if needed? Or assumes raw? 
        # In collecting data, env.reset() returns obs. 
        # SmartHomeEnv _get_obs returns [soc, pv, ...] 
        # PV in obs is direct from pv_profile.
        # Temp is obs[7].
        
        outdoor_temp = obs[7]
        # Note: If standardized, these might be scaled. 
        # But SmartHomeEnv usually returns physical values in _get_obs unless wrapped.
        
        return calculate_expert_action(obs, hour, outdoor_temp, pv_gen)
    except Exception as e:
        print(f"Error in adapter: {e}")
        return np.zeros(7)
