"""
WebSocket Server for Smart Home Simulation
Streams real-time PPO vs Hybrid agent comparison data to frontend
"""
import asyncio
import math
import random
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('WebSocketServer')

app = FastAPI(title="Smart Home HEMS WebSocket Server")

# CORS for HTTP endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LightweightSimulation:
    """
    Lightweight simulation that doesn't depend on SmartHomeEnv.
    This avoids numpy/torch dependency issues while providing realistic data.
    """
    def __init__(self):
        self.step = 0
        self.ppo_soc = 50.0
        self.hybrid_soc = 50.0
        self.ppo_bill = 0
        self.hybrid_bill = 0
        
    def update(self):
        self.step += 1
        hour = self.step % 24
        
        # PV generation curve (peak at noon)
        if 6 <= hour <= 18:
            pv = 3.0 * math.sin(math.pi * (hour - 6) / 12)
        else:
            pv = 0.0
            
        # Weather selection
        weather_probs = ['sunny', 'sunny', 'mild', 'cloudy', 'rainy']
        weather = random.choice(weather_probs)
        
        # Apply weather factor to PV
        weather_factors = {'sunny': 1.0, 'mild': 0.8, 'cloudy': 0.5, 'rainy': 0.3, 'stormy': 0.1}
        pv *= weather_factors.get(weather, 0.8)
        
        # Temperature curve
        temp = 28 + 6 * math.sin(math.pi * (hour - 9) / 12) + random.uniform(-1, 1)
        
        # Vietnam tiered pricing simulation
        if hour in [17, 18, 19, 20, 21]:  # Peak hours
            price_tier = 5
        elif hour in [6, 7, 8, 9, 10]:  # Morning peak
            price_tier = 4
        elif hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Night
            price_tier = 1
        else:
            price_tier = 3
        
        # --- ROOM-SPECIFIC DEVICE CONTROL ---
        
        # PPO Agent: Less optimized behavior (random-ish)
        # ACs - Living room always on when hot, others random
        ppo_ac_living = 1 if temp > 28 else 0
        ppo_ac_master = 1 if temp > 29 and random.random() > 0.4 else 0
        ppo_ac_bed2 = 1 if temp > 30 and random.random() > 0.5 else 0
        
        # Lights - On based on time (evening/night)
        is_dark = hour >= 18 or hour < 6
        ppo_light_living = 1 if is_dark else 0
        ppo_light_master = 1 if hour >= 21 or hour < 7 else 0
        ppo_light_bed2 = 1 if hour >= 20 or hour < 8 else 0
        ppo_light_kitchen = 1 if hour in [6, 7, 12, 18, 19] else 0
        ppo_light_toilet = 1 if random.random() > 0.7 else 0
        
        ppo_wm = 1 if hour in [10, 14, 16] else 0
        ppo_ev = 1 if hour in [22, 23, 0] else 0
        
        # Hybrid Agent: Smarter behavior (considers price + comfort)
        # ACs - Only run when needed AND price is reasonable
        hybrid_ac_living = 1 if temp > 29 and price_tier <= 4 else 0
        hybrid_ac_master = 1 if temp > 30 and price_tier <= 3 and hour >= 20 else 0
        hybrid_ac_bed2 = 1 if temp > 31 and price_tier <= 2 else 0
        
        # Lights - Smart scheduling to save energy
        hybrid_light_living = 1 if is_dark and hour < 23 else 0
        hybrid_light_master = 1 if hour >= 21 and hour < 23 else 0
        hybrid_light_bed2 = 1 if hour >= 20 and hour < 22 else 0
        hybrid_light_kitchen = 1 if hour in [6, 7, 18, 19] else 0  # Less than PPO
        hybrid_light_toilet = 1 if random.random() > 0.8 else 0
        
        hybrid_wm = 1 if hour in [11, 15] and price_tier <= 3 else 0
        hybrid_ev = 1 if hour in [1, 2, 3, 4] else 0  # Charge at lowest price
        
        # Calculate power consumption (sum of all room devices)
        ppo_ac_power = 1.5 * (ppo_ac_living + ppo_ac_master + ppo_ac_bed2)
        ppo_light_power = 0.015 * (ppo_light_living + ppo_light_master + ppo_light_bed2 + ppo_light_kitchen + ppo_light_toilet)
        ppo_load = ppo_ac_power + ppo_light_power + 0.5 * ppo_wm + 3.3 * ppo_ev + 0.3  # Base load
        
        hybrid_ac_power = 1.5 * (hybrid_ac_living + hybrid_ac_master + hybrid_ac_bed2)
        hybrid_light_power = 0.015 * (hybrid_light_living + hybrid_light_master + hybrid_light_bed2 + hybrid_light_kitchen + hybrid_light_toilet)
        hybrid_load = hybrid_ac_power + hybrid_light_power + 0.5 * hybrid_wm + 3.3 * hybrid_ev + 0.3
        
        # Net power (positive = need grid, negative = export)
        ppo_net = ppo_load - pv
        hybrid_net = hybrid_load - pv
        
        # Update SOC based on net power
        delta_ppo = -ppo_net * 0.8  # SOC change
        delta_hybrid = -hybrid_net * 0.8
        
        self.ppo_soc = max(10, min(90, self.ppo_soc + delta_ppo))
        self.hybrid_soc = max(10, min(90, self.hybrid_soc + delta_hybrid))
        
        # Calculate bill increment
        tier_prices = [0, 1984, 2050, 2380, 2998, 3350, 3460]
        cost_per_kwh = tier_prices[price_tier]
        
        # Only charge for grid import (positive net)
        if ppo_net > 0:
            self.ppo_bill += int(ppo_net * cost_per_kwh / 24)
        if hybrid_net > 0:
            self.hybrid_bill += int(hybrid_net * cost_per_kwh / 24)
        
        # Determine battery mode
        ppo_battery = 'charge' if delta_ppo > 0.3 else ('discharge' if delta_ppo < -0.3 else 'idle')
        hybrid_battery = 'charge' if delta_hybrid > 0.3 else ('discharge' if delta_hybrid < -0.3 else 'idle')
        
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
                "grid": round(max(0, ppo_net), 2),
                "actions": {
                    "ac_living": ppo_ac_living,
                    "ac_master": ppo_ac_master,
                    "ac_bed2": ppo_ac_bed2,
                    "light_living": ppo_light_living,
                    "light_master": ppo_light_master,
                    "light_bed2": ppo_light_bed2,
                    "light_kitchen": ppo_light_kitchen,
                    "light_toilet": ppo_light_toilet,
                    "wm": ppo_wm,
                    "ev": ppo_ev,
                    "battery": ppo_battery
                },
                "comfort": 0.0
            },
            "hybrid": {
                "bill": self.hybrid_bill,
                "soc": round(self.hybrid_soc, 1),
                "grid": round(max(0, hybrid_net), 2),
                "actions": {
                    "ac_living": hybrid_ac_living,
                    "ac_master": hybrid_ac_master,
                    "ac_bed2": hybrid_ac_bed2,
                    "light_living": hybrid_light_living,
                    "light_master": hybrid_light_master,
                    "light_bed2": hybrid_light_bed2,
                    "light_kitchen": hybrid_light_kitchen,
                    "light_toilet": hybrid_light_toilet,
                    "wm": hybrid_wm,
                    "ev": hybrid_ev,
                    "battery": hybrid_battery
                },
                "comfort": 0.0
            }
        }
    
    def reset(self):
        """Reset simulation for new day"""
        self.step = 0
        self.ppo_soc = 50.0
        self.hybrid_soc = 50.0
        self.ppo_bill = 0
        self.hybrid_bill = 0


# Global simulation instance
sim = LightweightSimulation()

# Track connected clients
connected_clients = []


@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Smart Home HEMS WebSocket Server",
        "websocket_endpoint": "/ws",
        "http_endpoint": "/data",
        "connected_clients": len(connected_clients)
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/data")
async def get_data():
    """HTTP polling endpoint as WebSocket fallback"""
    return JSONResponse(content=sim.update())


@app.get("/reset")
async def reset_sim():
    """Reset simulation"""
    sim.reset()
    return {"status": "reset"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming"""
    logger.info(f"🔌 Connection attempt from {websocket.client}")
    
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info(f"✅ Client connected. Total: {len(connected_clients)}")
    
    try:
        while True:
            # Update and get data
            data = sim.update()
            
            # Send to client
            await websocket.send_json(data)
            
            # 1 second interval (1 hour in simulation time)
            await asyncio.sleep(1.0)
            
    except WebSocketDisconnect:
        logger.info(f"❌ Client disconnected normally")
    except Exception as e:
        logger.error(f"⚠️ WebSocket error: {e}")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.info(f"Remaining clients: {len(connected_clients)}")


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 50)
    logger.info("🚀 Smart Home HEMS WebSocket Server")
    logger.info("=" * 50)
    logger.info("📡 WebSocket: ws://localhost:8001/ws")
    logger.info("🌐 HTTP Poll: http://localhost:8001/data")
    logger.info("=" * 50)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
