import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'backend'))
from fastapi.testclient import TestClient
from backend.main import app, load_models
import json

# Manually load models to ensure they are available
load_models()

client = TestClient(app)

def test_internal():
    print("Testing /simulate internal...")
    payload = {
        "num_people": 2,
        "weather_condition": "sunny",
        "must_run_base": 0.2,
        "seed": 42
    }
    
    try:
        response = client.post("/simulate", json=payload)
        
        if response.status_code != 200:
            print(f"FAILED: Status {response.status_code}")
            print(f"Response: {response.text}")
            return
            
        data = response.json()
        hybrid_data = data['hybrid']
        ppo_data = data['ppo']
        
        print(f"SUCCESS: Hybrid Steps: {len(hybrid_data)}")
        
        ppo_cost = ppo_data[-1]['total_bill']
        hybrid_cost = hybrid_data[-1]['total_bill']
        
        print(f"PPO Bill: {ppo_cost}")
        print(f"Hybrid Bill: {hybrid_cost}")
        
        if hybrid_cost < ppo_cost:
            print("VERIFIED: Hybrid cost < PPO cost.")
        else:
             print("WARNING: Hybrid cost >= PPO cost.")

    except Exception as e:
        print(f"Internal Error: {e}")

if __name__ == "__main__":
    test_internal()
