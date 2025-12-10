from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from simulation_core import sim_instance

app = FastAPI()

# --- CẤU HÌNH CORS ---
# Cho phép frontend (thường chạy ở localhost:5173) kết nối đến backend
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- QUẢN LÝ KẾT NỐI WEBSOCKET ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message_dict: dict):
        """Gửi tin nhắn JSON đến tất cả client đang kết nối"""
        if not self.active_connections:
            return
        
        json_data = json.dumps(message_dict)
        # Duyệt ngược để tránh lỗi khi remove connection chết
        for connection in reversed(self.active_connections):
            try:
                await connection.send_text(json_data)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

# --- WEBSOCKET ENDPOINT ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Giữ kết nối sống. Có thể nhận lệnh từ frontend tại đây.
            data = await websocket.receive_text()
            print(f"Received from client: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

# --- VÒNG LẶP MÔ PHỎNG CHẠY NGẦM ---
async def run_simulation_loop():
    print("--- Simulation Loop Started ---")
    while True:
        # 1. Cập nhật trạng thái mô phỏng
        sim_instance.update()
        
        # 2. Lấy gói dữ liệu
        data_packet = sim_instance.get_data_packet()
        
        # 3. Gửi xuống tất cả client
        await manager.broadcast(data_packet)
        
        # 4. Nghỉ một chút để kiểm soát tốc độ (FPS)
        # 0.05s = 20 lần/giây. Đủ mượt cho web và không quá tải server.
        await asyncio.sleep(0.05) 

@app.on_event("startup")
async def startup_event():
    # Chạy vòng lặp mô phỏng ngay khi server khởi động
    asyncio.create_task(run_simulation_loop())

if __name__ == "__main__":
    # Chạy server development
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)