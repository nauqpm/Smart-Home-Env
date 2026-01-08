
import numpy as np
from backend.expert_utils import calculate_expert_action

def test():
    # Mock Obs: [SOC, PV, ...]
    # PV at index 1 = 3.2
    obs = np.zeros(13)
    obs[1] = 3.2
    obs[7] = 25.0 # Temp
    
    hour = 12
    outdoor_temp = 25.0
    pv_gen = 3.2
    
    action = calculate_expert_action(obs, hour, outdoor_temp, pv_gen)
    print(f"PV={pv_gen}, Action[0]={action[0]}")
    
    if action[0] > 0.5:
        print("PASS: Expert logic charges correctly.")
    else:
        print("FAIL: Expert logic failing.")

if __name__ == "__main__":
    test()
