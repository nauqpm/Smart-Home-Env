"""Quick test of improved reward function - ASCII only"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smart_home_env import SmartHomeEnv

print("="*60)
print("Testing Improved Reward Function")
print("="*60)

config = {
    'sim_steps': 24,
    'battery': {
        'capacity_kwh': 10, 'soc_init': 0.5, 'soc_min': 0.1, 'soc_max': 0.9,
        'p_charge_max_kw': 3, 'p_discharge_max_kw': 3, 'eta_ch': 0.95, 'eta_dis': 0.95
    }
}

# Test 1: All devices OFF (the degenerate policy)
print("\n[Test 1] All devices OFF (degenerate policy)")
env = SmartHomeEnv(None, None, config)
obs, _ = env.reset(seed=42)

total_reward_off = 0
for t in range(24):
    action = np.array([-1, -1, -1, -1, -1, -1, -1], dtype=np.float32)  # All off
    obs, reward, done, _, info = env.step(action)
    total_reward_off += reward
    if done:
        break

bill_off = env.total_cost
ev_off = env.ev_soc
wm_off = env.wm_remaining
dw_off = env.dw_remaining

print(f"  Final Bill: {bill_off:,.0f} VND")
print(f"  Final EV SOC: {ev_off:.2f}")
print(f"  WM Remaining: {wm_off}")
print(f"  DW Remaining: {dw_off}")
print(f"  Total Reward: {total_reward_off:.2f}")

# Test 2: Reasonable usage (heuristic)
print("\n[Test 2] Reasonable heuristic policy")
env = SmartHomeEnv(None, None, config)
obs, _ = env.reset(seed=42)

total_reward_on = 0
for t in range(24):
    hour = t
    action = np.zeros(7, dtype=np.float32)
    
    # Battery: charge during day
    action[0] = 0.5 if 8 <= hour <= 16 else -0.3
    
    # AC: on when hot hours
    if 11 <= hour <= 17:
        action[1] = 0.6
        action[2] = 0.4
        action[3] = 0.3
    
    # EV: charge at night
    action[4] = 0.8 if hour >= 22 or hour < 6 else 0.3
    
    # WM: run evening
    action[5] = 0.8 if 15 <= hour <= 18 else -0.5
    
    # DW: run evening  
    action[6] = 0.8 if 18 <= hour <= 21 else -0.5
    
    obs, reward, done, _, info = env.step(action)
    total_reward_on += reward
    if done:
        break

bill_on = env.total_cost
ev_on = env.ev_soc
wm_on = env.wm_remaining
dw_on = env.dw_remaining

print(f"  Final Bill: {bill_on:,.0f} VND")
print(f"  Final EV SOC: {ev_on:.2f}")
print(f"  WM Remaining: {wm_on}")
print(f"  DW Remaining: {dw_on}")
print(f"  Total Reward: {total_reward_on:.2f}")

# Comparison
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"  All OFF:      Reward = {total_reward_off:8.2f}, Bill = {bill_off:>10,.0f} VND")
print(f"  Heuristic:    Reward = {total_reward_on:8.2f}, Bill = {bill_on:>10,.0f} VND")
print("")
if total_reward_on > total_reward_off:
    print("  [OK] Heuristic has HIGHER reward than All-OFF!")
    print("  --> NEW REWARD FUNCTION WORKS CORRECTLY")
else:
    print("  [FAIL] All-OFF still has higher reward!")
    print("  --> Need stronger penalties")
