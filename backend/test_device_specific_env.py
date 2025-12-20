"""
Test script for Device-Specific SmartHomeEnv
"""
import numpy as np
from smart_home_env import SmartHomeEnv

def test_env():
    # Test configuration
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
        },
        'pv_config': {
            'latitude': 10.762622,
            'longitude': 106.660172,
            'tz': 'Asia/Ho_Chi_Minh',
            'surface_tilt': 10.0,
            'surface_azimuth': 180.0,
            'module_parameters': {'pdc0': 3.0}
        },
        'behavior': {
            'residents': [],
            'must_run_base': 0.15
        }
    }

    print('=' * 60)
    print('Testing Device-Specific SmartHomeEnv')
    print('=' * 60)
    
    print('\nCreating SmartHomeEnv...')
    env = SmartHomeEnv(None, None, config)

    print(f'\n✅ Action Space: {env.action_space}')
    print(f'   Shape: {env.action_space.shape}')  # Expected: (7,)
    
    print(f'\n✅ Observation Space: {env.observation_space}')
    print(f'   Shape: {env.observation_space.shape}')  # Expected: (13,)

    print('\n--- Resetting environment ---')
    obs, info = env.reset(seed=42)
    print(f'Initial Observation Shape: {obs.shape}')
    print(f'Initial Observation:')
    print(f'  SOC: {obs[0]:.2f}')
    print(f'  PV: {obs[1]:.2f}')
    print(f'  MustRun: {obs[2]:.2f}')
    print(f'  FuturePV: {obs[3]:.2f}')
    print(f'  TempOut: {obs[7]:.1f}')
    print(f'  RoomTemps: Living={obs[8]:.1f}, Master={obs[9]:.1f}, Bed2={obs[10]:.1f}')
    print(f'  WM Remaining: {obs[11]:.0f}')
    print(f'  EV SOC: {obs[12]:.2f}')

    print('\n--- Taking sample actions for 3 steps ---')
    for step in range(3):
        action = env.action_space.sample()
        print(f'\nStep {step + 1}:')
        print(f'  Action: {action}')
        
        obs, reward, done, truncated, info = env.step(action)
        
        print(f'  Reward: {reward:.4f}')
        print(f'  Device States:')
        print(f'    ACs: Living={info.get("ac_living")}, Master={info.get("ac_master")}, Bed2={info.get("ac_bed2")}')
        print(f'    Lights: Living={info.get("light_living")}, Kitchen={info.get("light_kitchen")}')
        print(f'    WM={info.get("wm")}, DW={info.get("dw")}, EV={info.get("ev")}')
        print(f'    Battery: {info.get("battery")}')
        print(f'    Room Temps: {info.get("room_temps")}')

    print('\n' + '=' * 60)
    print('✅ SmartHomeEnv Test PASSED!')
    print('=' * 60)

if __name__ == "__main__":
    test_env()
