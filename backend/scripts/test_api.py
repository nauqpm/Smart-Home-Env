import requests
import json

def test_simulation():
    url = "http://127.0.0.1:8000/simulate"
    payload = {
        "num_people": 2,
        "weather_condition": "sunny",
        "must_run_base": 0.2,
        "seed": 42
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        hybrid_data = data['hybrid']
        ppo_data = data['ppo']
        
        print(f"Hybrid Steps: {len(hybrid_data)}")
        print(f"PPO Steps: {len(ppo_data)}")
        
        # Check for non-zero battery actions in Hybrid (evidence of expert rules working)
        # Verify battery state changes
        soc_changes = []
        for i in range(1, len(hybrid_data)):
            diff = hybrid_data[i]['soc'] - hybrid_data[i-1]['soc']
            soc_changes.append(diff)
            
        print(f"Hybrid Max SOC Change: {max(soc_changes)}")
        print(f"Hybrid Min SOC Change: {min(soc_changes)}")
        
        # Check Total Result
        ppo_cost = ppo_data[-1]['total_bill']
        hybrid_cost = hybrid_data[-1]['total_bill']
        
        print(f"PPO Final Bill: {ppo_cost}")
        print(f"Hybrid Final Bill: {hybrid_cost}")
        
        if hybrid_cost < ppo_cost:
            print("SUCCESS: Hybrid cost is lower than PPO cost!")
        else:
            print("WARNING: Hybrid cost is NOT lower than PPO cost. Check weights and rules.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_simulation()
