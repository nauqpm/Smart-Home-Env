"""
Test script for Simulation Core with Device-Specific Actions
"""
import json
from simulation_core import SimulationEngine

def test_simulation_core():
    print('=' * 60)
    print('Testing Simulation Core with Device-Specific Actions')
    print('=' * 60)
    
    print('\nCreating SimulationEngine...')
    engine = SimulationEngine()
    
    print(f'\n✅ Engine created successfully')
    print(f'   Initial step count: {engine.step_count}')
    
    print('\n--- Running 3 simulation steps ---')
    for step in range(3):
        engine.update()
        packet = engine.get_data_packet()
        
        print(f'\nStep {step + 1}: {packet["timestamp"]}')
        print(f'  Environment:')
        print(f'    Weather: {packet["env"]["weather"]}')
        print(f'    Temp: {packet["env"]["temp"]}°C')
        print(f'    PV: {packet["env"]["pv"]} kW')
        print(f'    Price Tier: {packet["env"]["price_tier"]}')
        
        print(f'\n  PPO Agent:')
        print(f'    Bill: {packet["ppo"]["bill"]} VND')
        print(f'    SOC: {packet["ppo"]["soc"]}%')
        ppo_actions = packet["ppo"]["actions"]
        print(f'    Actions:')
        print(f'      ACs: living={ppo_actions["ac_living"]}, master={ppo_actions["ac_master"]}, bed2={ppo_actions["ac_bed2"]}')
        print(f'      Lights: living={ppo_actions["light_living"]}, kitchen={ppo_actions["light_kitchen"]}')
        print(f'      WM={ppo_actions["wm"]}, DW={ppo_actions["dw"]}, EV={ppo_actions["ev"]}')
        print(f'      Battery: {ppo_actions["battery"]}')
        
        print(f'\n  Hybrid Agent:')
        print(f'    Bill: {packet["hybrid"]["bill"]} VND')
        print(f'    SOC: {packet["hybrid"]["soc"]}%')
        hybrid_actions = packet["hybrid"]["actions"]
        print(f'    Actions:')
        print(f'      ACs: living={hybrid_actions["ac_living"]}, master={hybrid_actions["ac_master"]}, bed2={hybrid_actions["ac_bed2"]}')
        print(f'      Lights: living={hybrid_actions["light_living"]}, kitchen={hybrid_actions["light_kitchen"]}')
        print(f'      WM={hybrid_actions["wm"]}, DW={hybrid_actions["dw"]}, EV={hybrid_actions["ev"]}')
        print(f'      Battery: {hybrid_actions["battery"]}')
    
    print('\n--- Sample Data Packet JSON ---')
    print(json.dumps(engine.get_data_packet(), indent=2))
    
    print('\n' + '=' * 60)
    print('✅ Simulation Core Test PASSED!')
    print('=' * 60)

if __name__ == "__main__":
    test_simulation_core()
