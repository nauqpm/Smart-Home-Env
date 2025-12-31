"""Simple test to trace bill calculation"""
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smart_home_env import SmartHomeEnv, calculate_vietnam_tiered_bill

print("=== TIERED BILL TEST ===")
for kwh in [0, 1, 5, 10, 50, 100]:
    bill = calculate_vietnam_tiered_bill(kwh)
    print(f"  {kwh:5.1f} kWh -> {bill:,.0f} VND")

print("\n=== ENVIRONMENT TEST ===")
config = {
    'sim_steps': 24,
    'battery': {
        'capacity_kwh': 10, 'soc_init': 0.5, 'soc_min': 0.1, 'soc_max': 0.9,
        'p_charge_max_kw': 3, 'p_discharge_max_kw': 3, 'eta_ch': 0.95, 'eta_dis': 0.95
    }
}

env = SmartHomeEnv(None, None, config)
obs, _ = env.reset(seed=42)

print(f"PV profile: {env.pv_profile}")
print(f"Must run power: {env.load_schedules[0]['must_run']:.3f} kW")
print(f"Initial SOC: {env.soc}")

print("\n=== RUNNING WITH HEURISTIC ACTIONS ===")
print(f"{'Hour':>4} {'Act[0]':>8} {'Load':>8} {'PV':>6} {'Import':>8} {'CumImp':>10} {'Bill':>12}")
print("-" * 70)

for t in range(24):
    hour = t
    # Simple heuristic action
    action = np.zeros(7, dtype=np.float32)
    action[0] = 0.3 if 8 <= hour <= 16 else -0.3  # battery
    action[1] = 0.5 if 12 <= hour <= 15 else -0.8  # AC living
    action[4] = 0.8 if hour >= 22 or hour < 6 else -0.5  # EV
    
    must_run = env.load_schedules[env.t]['must_run']
    pv_now = env.pv_profile[env.t]
    
    prev_import = env.cumulative_import_kwh
    prev_bill = env.total_cost
    
    obs, reward, done, _, info = env.step(action)
    
    step_import = env.cumulative_import_kwh - prev_import
    step_bill = env.total_cost - prev_bill
    
    print(f"{hour:4d}h {action[0]:8.2f} {must_run:8.3f} {pv_now:6.2f} {step_import:8.3f} {env.cumulative_import_kwh:10.3f} {env.total_cost:12,.0f}")
    
    if done:
        break

print("\n=== FINAL ===")
print(f"Total Import: {env.cumulative_import_kwh:.3f} kWh")
print(f"Total Bill: {env.total_cost:,.0f} VND")

# Now try PPO model
print("\n" + "=" * 60)
print("=== PPO MODEL TEST ===")
try:
    from stable_baselines3 import PPO
    if os.path.exists("ppo_smart_home.zip"):
        model = PPO.load("ppo_smart_home.zip")
        print("PPO model loaded!")
        
        env2 = SmartHomeEnv(None, None, config)
        obs, _ = env2.reset(seed=42)
        
        print(f"\n{'Hour':>4} {'Act[0]':>8} {'Act[1]':>8} {'CumImp':>10} {'Bill':>12}")
        print("-" * 50)
        
        for t in range(24):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env2.step(action)
            print(f"{t:4d}h {action[0]:8.2f} {action[1]:8.2f} {env2.cumulative_import_kwh:10.3f} {env2.total_cost:12,.0f}")
            if done:
                break
        
        print(f"\nPPO Final Bill: {env2.total_cost:,.0f} VND, Import: {env2.cumulative_import_kwh:.3f} kWh")
    else:
        print("PPO model file not found")
except ImportError:
    print("stable_baselines3 not installed")
