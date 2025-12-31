"""
Debug script to analyze model behavior and bill calculation
"""
import numpy as np
import sys
import os

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smart_home_env import SmartHomeEnv, calculate_vietnam_tiered_bill

# Try loading models
try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("⚠️ stable_baselines3 not installed")

def analyze_bill_behavior():
    print("=" * 60)
    print("🔍 ANALYZING BILL CALCULATION BEHAVIOR")
    print("=" * 60)
    
    # Test the tiered bill function
    print("\n📊 Testing calculate_vietnam_tiered_bill:")
    test_kwhs = [0, 1, 5, 10, 50, 100, 150, 200, 300]
    for kwh in test_kwhs:
        bill = calculate_vietnam_tiered_bill(kwh)
        print(f"  {kwh:5.1f} kWh → {bill:,.0f} VND")
    
    # Create environment
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
    
    # Run with heuristic actions
    print("\n" + "=" * 60)
    print("🏃 RUNNING 24-HOUR SIMULATION WITH HEURISTIC ACTIONS")
    print("=" * 60)
    
    obs, _ = env.reset(seed=42)
    
    print(f"\n{'Hour':>4} {'Load':>6} {'PV':>5} {'Grid':>6} {'Cum.Import':>10} {'Total Bill':>12} {'Step Cost':>10}")
    print("-" * 70)
    
    for t in range(24):
        hour = t % 24
        
        # Simple heuristic action
        action = np.zeros(7, dtype=np.float32)
        # Battery: charge during day, discharge at night
        action[0] = 0.5 if 6 <= hour <= 16 else -0.3
        # AC: on when hot hours
        if 11 <= hour <= 17:
            action[1] = 0.6  # AC living
            action[2] = 0.3  # AC master
            action[3] = 0.2  # AC bed2
        else:
            action[1:4] = -0.5
        # EV: charge at night
        action[4] = 0.8 if hour >= 22 or hour < 6 else -0.5
        # WM/DW: run in evening
        action[5] = 0.5 if 18 <= hour <= 20 else -0.5
        action[6] = 0.5 if 19 <= hour <= 21 else -0.5
        
        # Get state before step
        load = env.load_schedules[t]['must_run']
        pv = env.pv_profile[t]
        
        # Step
        prev_cost = env.total_cost
        obs, reward, done, _, info = env.step(action)
        step_cost = env.total_cost - prev_cost
        
        # Calculate grid
        grid_import = env.cumulative_import_kwh
        
        print(f"{hour:4d}h {load:6.2f} {pv:5.2f} {step_cost:>6.0f} {grid_import:10.2f} {env.total_cost:12,.0f} {step_cost:10,.0f}")
        
        if done:
            break
    
    print("-" * 70)
    print(f"\n📊 FINAL SUMMARY:")
    print(f"  Total Import: {env.cumulative_import_kwh:.2f} kWh")
    print(f"  Total Export: {env.cumulative_export_kwh:.2f} kWh")
    print(f"  Total Bill: {env.total_cost:,.0f} VND")
    
    # Load and test PPO model if available
    if HAS_SB3 and os.path.exists("ppo_smart_home.zip"):
        print("\n" + "=" * 60)
        print("🤖 RUNNING WITH PPO MODEL")
        print("=" * 60)
        
        model = PPO.load("ppo_smart_home.zip")
        
        env2 = SmartHomeEnv(None, None, config)
        obs, _ = env2.reset(seed=42)
        
        print(f"\n{'Hour':>4} {'Cum.Import':>10} {'Total Bill':>12} {'Action[0:3]':>20}")
        print("-" * 60)
        
        for t in range(24):
            hour = t % 24
            
            action, _ = model.predict(obs, deterministic=True)
            prev_cost = env2.total_cost
            obs, reward, done, _, info = env2.step(action)
            
            print(f"{hour:4d}h {env2.cumulative_import_kwh:10.2f} {env2.total_cost:12,.0f} {str(action[:3]):>20}")
            
            if done:
                break
        
        print(f"\n📊 PPO FINAL: Bill = {env2.total_cost:,.0f} VND, Import = {env2.cumulative_import_kwh:.2f} kWh")
    else:
        print("\n⚠️ PPO model not found or SB3 not installed")

if __name__ == "__main__":
    analyze_bill_behavior()
