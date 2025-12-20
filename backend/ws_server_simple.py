"""
Simple WebSocket Server for Testing
No heavy dependencies like SmartHomeEnv
"""
import asyncio
import json
import time
import math
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Smart Home WebSocket Test Server")

# CORS for HTTP endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimpleSimulation:
    """Lightweight simulation without heavy dependencies"""
    def __init__(self):
        self.step = 0
        self.ppo_soc = 50.0
        self.hybrid_soc = 50.0
        self.ppo_bill = 0
        self.hybrid_bill = 0
        
    def update(self):
        self.step += 1
        hour = self.step % 24
        
        # Simple PV generation (solar curve)
        if 6 <= hour <= 18:
            pv = 3.0 * math.sin(math.pi * (hour - 6) / 12)
        else:
            pv = 0.0
            
        # Random weather
        weathers = ['sunny', 'mild', 'cloudy', 'rainy']
        weather = random.choice(weathers)
        
        # Simple temp
        temp = 25 + 8 * math.sin(math.pi * (hour - 6) / 12) + random.uniform(-2, 2)
        
        # Price tier based on time
        if hour in [17, 18, 19, 20, 21]:
            price_tier = 5
        elif hour in [6, 7, 8, 9, 10]:
            price_tier = 3
        else:
            price_tier = 2
            
        # PPO: Random actions
        ppo_ac = random.randint(0, 1)
        ppo_wm = 1 if hour in [10, 14] else 0
        
        # Hybrid: Smarter actions
        hybrid_ac = 1 if temp > 30 else 0
        hybrid_wm = 1 if hour in [11, 15] and price_tier < 4 else 0
        
        # Update SOC
        delta_ppo = pv - (1.5 * ppo_ac + 0.5 * ppo_wm + 0.5)
        delta_hybrid = pv - (1.5 * hybrid_ac + 0.5 * hybrid_wm + 0.5)
        
        self.ppo_soc = max(10, min(90, self.ppo_soc + delta_ppo * 2))
        self.hybrid_soc = max(10, min(90, self.hybrid_soc + delta_hybrid * 2))
        
        # Update bill
        cost_per_kwh = [0, 1984, 2050, 2380, 2998, 3350, 3460][price_tier]
        self.ppo_bill += int((1.5 * ppo_ac + 0.5 * ppo_wm + 0.3) * cost_per_kwh / 24)
        self.hybrid_bill += int((1.5 * hybrid_ac + 0.5 * hybrid_wm + 0.3) * cost_per_kwh / 24)
        
        # Battery mode
        ppo_battery = 'charge' if delta_ppo > 0.5 else ('discharge' if delta_ppo < -0.5 else 'idle')
        hybrid_battery = 'charge' if delta_hybrid > 0.5 else ('discharge' if delta_hybrid < -0.5 else 'idle')
        
        return {
            "timestamp": f"{hour:02d}:00",
            "env": {
                "weather": weather,
                "temp": round(temp, 1),
                "pv": round(pv, 2),
                "price_tier": price_tier
            },
            "ppo": {
                "bill": self.ppo_bill,
                "soc": round(self.ppo_soc, 1),
                "grid": round(max(0, -delta_ppo), 2),
                "actions": {"ac": ppo_ac, "wm": ppo_wm, "ev": 0, "battery": ppo_battery},
                "comfort": 0.0
            },
            "hybrid": {
                "bill": self.hybrid_bill,
                "soc": round(self.hybrid_soc, 1),
                "grid": round(max(0, -delta_hybrid), 2),
                "actions": {"ac": hybrid_ac, "wm": hybrid_wm, "ev": 0, "battery": hybrid_battery},
                "comfort": 0.0
            }
        }


sim = SimpleSimulation()


@app.get("/")
async def root():
    return {"status": "running", "message": "WebSocket Test Server"}


@app.get("/data")
async def get_data():
    """HTTP polling endpoint"""
    return sim.update()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket streaming endpoint"""
    print(f"🔌 Connection attempt from {websocket.client}")
    
    try:
        await websocket.accept()
        print(f"✅ Client connected: {websocket.client}")
        
        while True:
            data = sim.update()
            await websocket.send_json(data)
            await asyncio.sleep(1.0)
            
    except WebSocketDisconnect:
        print(f"❌ Client disconnected")
    except Exception as e:
        print(f"⚠️ Error: {e}")


if __name__ == "__main__":
    print("🚀 Starting WebSocket Test Server...")
    print("📡 WebSocket: ws://localhost:8000/ws")
    print("🌐 HTTP: http://localhost:8000/data")
    uvicorn.run(app, host="0.0.0.0", port=8000)
