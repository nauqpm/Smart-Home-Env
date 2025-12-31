"""
Detailed debug script - trace ALL energy components step by step
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smart_home_env import SmartHomeEnv, calculate_vietnam_tiered_bill
from device_config import DEVICE_CONFIG

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

def detailed_analysis():
    print("=" * 80)
    print("🔍 DETAILED ENERGY FLOW ANALYSIS")
    print("=" * 80)
    
    config = {
        'time_step_hours': 1.0,
        'sim_start': '2025-01-01',
        'sim_steps': 24,
        'sim_freq': '1h',
        'battery': {
            'capacity_kwh': 10.0,
            'soc_init': 0.5,
            'soc_min': 0.1,
            'soc_max': 0.9,
            'p_charge_max_kw': 3.0,
            'p_discharge_max_kw': 3.0,
            'eta_ch': 0.95,
            'eta_dis': 0.95
        }
    }
    
    env = SmartHomeEnv(None, None, config)
    obs, _ = env.reset(seed=42)
    
    print(f"\n📋 PV Profile: {env.pv_profile}")
    print(f"📋 Must Run Power: {DEVICE_CONFIG['fixed']['fridge']['power']} kW")
    
    # Load model
    if HAS_SB3 and os.path.exists("ppo_smart_home.zip"):
        model = PPO.load("ppo_smart_home.zip")
        print("✅ PPO model loaded")
    else:
        print("❌ PPO model not available")
        model = None
    
    print("\n" + "=" * 80)
    print(f"{'Hour':>4} {'Action':>50} {'Load':>8} {'PV':>6} {'Grid':>8} {'CumImp':>8} {'Bill':>10}")
    print("-" * 105)
    
    for t in range(24):
        # Get action
        if model:
            action, _ = model.predict(obs, deterministic=True)
            action = np.array(action, dtype=np.float32).flatten()
        else:
            action = np.zeros(7, dtype=np.float32)
        
        # Get state BEFORE step
        hour = env.times[env.t].hour
        must_run = env.load_schedules[env.t]["must_run"]
        pv = env.pv_profile[env.t]
        
        # Calculate what load WILL BE based on action
        act_bat = action[0]
        ac_vals = [(action[i] + 1) / 2 for i in [1, 2, 3]]
        act_ev = (action[4] + 1) / 2
        act_wm = action[5] > 0
        act_dw = action[6] > 0
        
        # Estimate total load
        total_load_est = must_run
        for ac in ac_vals:
            total_load_est += ac * 2.0  # assume 2kW max AC
        total_load_est += act_ev * 3.3  # EV charger
        if act_wm and env.wm_remaining > 0:
            total_load_est += 0.5
        if act_dw and env.dw_remaining > 0:
            total_load_est += 0.7
        
        # Battery
        if act_bat > 0:
            total_load_est += act_bat * 3.0  # charging adds load
        else:
            total_load_est -= (-act_bat) * 3.0  # discharging reduces load
        
        prev_cost = env.total_cost
        obs, reward, done, _, info = env.step(action)
        step_cost = env.total_cost - prev_cost
        
        action_str = f"[{action[0]:.2f},{action[1]:.2f},{action[2]:.2f},{action[3]:.2f},{action[4]:.2f},{action[5]:.2f},{action[6]:.2f}]"
        
        print(f"{hour:4d}h {action_str:>50} {total_load_est:8.2f} {pv:6.2f} {step_cost:8.0f} {env.cumulative_import_kwh:8.2f} {env.total_cost:10,.0f}")
        
        if done:
            break
    
    print("-" * 105)
    print(f"\n📊 FINAL SUMMARY:")
    print(f"  Cumulative Import: {env.cumulative_import_kwh:.4f} kWh")
    print(f"  Cumulative Export: {env.cumulative_export_kwh:.4f} kWh")
    print(f"  Total Bill: {env.total_cost:,.0f} VND")
    print(f"  Final SOC: {env.soc:.2f}")
    
    # Key insight
    print("\n💡 KEY ANALYSIS:")
    if env.cumulative_import_kwh < 0.001:
        print("  ⚠️ Import is essentially ZERO!")
        print("  Possible causes:")
        print("    1. PV profile is all zeros (no solar generation)")
        print("    2. Model learned to minimize ALL device usage")
        print("    3. Battery discharging covers all load")
    
    if all(env.pv_profile == 0):
        print("\n  🔴 CONFIRMED: PV profile is all zeros!")
        print("     This means no solar generation, so either:")
        print("     - All load must come from grid (should have bill)")
        print("     - Or model turned off everything (bill = 0)")
        
    # Check what the model is actually doing
    print("\n🔬 MODEL BEHAVIOR CHECK:")
    obs, _ = env.reset(seed=42)
    for t in range(3):
        action, _ = model.predict(obs, deterministic=True)
        print(f"  Step {t}: action = {action}")
        obs, _, _, _, _ = env.step(action)

if __name__ == "__main__":
    detailed_analysis()
