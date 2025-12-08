import random
import math
import time

class SimulationEngine:
    def __init__(self):
        self.step_count = 0
        # Trạng thái thiết bị
        self.fan_speed = 0.0 # 0.0 đến 2.0 (rad/frame)
        self.light_on = False
        self.ac_target_temp = 25
        
        # Số liệu môi trường
        self.current_temp = 30.0
        self.power_consumption = 0.0
        self.start_time = time.time()

    def update(self):
        """Hàm này được gọi liên tục để cập nhật trạng thái mô phỏng"""
        self.step_count += 1
        elapsed = time.time() - self.start_time

        # --- GIẢ LẬP LOGIC HOẠT ĐỘNG ---
        
        # 1. Quạt: Tốc độ thay đổi theo hình sin (để thấy nó quay nhanh chậm)
        self.fan_speed = (math.sin(elapsed * 0.5) + 1) * 1.5 # Tốc độ từ 0 đến 3

        # 2. Đèn: Tự động bật tắt mỗi 3 giây
        if self.step_count % 60 == 0: # Giả sử chạy ở ~20fps thì 60 frames là ~3s
            self.light_on = not self.light_on

        # 3. Nhiệt độ: Dao động quanh 30 độ
        self.current_temp = 30 + math.sin(elapsed * 0.2) * 2 + random.uniform(-0.1, 0.1)

        # 4. Công suất tiêu thụ: Tính tổng giả định
        fan_power = self.fan_speed * 0.5 # Quạt quay càng nhanh càng tốn điện
        light_power = 1.0 if self.light_on else 0.0
        base_load = random.uniform(0.2, 0.5) # Tải nền
        self.power_consumption = round(fan_power + light_power + base_load, 2)

    def get_data_packet(self):
        """Đóng gói dữ liệu để gửi qua WebSocket"""
        return {
            # Timestamp để vẽ trục X biểu đồ
            "timestamp": int(time.time() * 1000), # miliseconds
            "step": self.step_count,
            # Dữ liệu điều khiển thiết bị 3D
            "devices": {
                "fan_speed": self.fan_speed,
                "light_on": self.light_on,
                "ac_target": self.ac_target_temp
            },
            # Dữ liệu số liệu cho biểu đồ
            "metrics": {
                "temperature": round(self.current_temp, 2),
                "power": self.power_consumption
            }
        }

# Tạo một instance duy nhất để dùng trong server
sim_instance = SimulationEngine()