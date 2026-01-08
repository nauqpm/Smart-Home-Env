
import requests
import time

URL = "http://localhost:8012"

def trigger():
    print(f"Triggering Demo Mode on {URL}...")
    try:
        resp = requests.post(f"{URL}/set_mode", json={"demo_mode": True, "scenario": "ideal"})
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")
    except Exception as e:
        print(f"FAILED: {e}")

def step(n=10):
    print(f"Stepping {n} times...")
    for i in range(n):
        try:
            requests.get(f"{URL}/data")
        except:
            pass
        # time.sleep(0.1)

if __name__ == "__main__":
    trigger()
    step()
