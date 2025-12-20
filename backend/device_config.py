# backend/device_config.py
"""
Device Configuration for SmartHomeEnv
Defines all device specifications, action mappings, and occupancy schedules.
"""

DEVICE_CONFIG = {
    # Shiftable: Quan tâm đến hoàn thành công việc (Deadline)
    'shiftable': {
        'wm': {'name': 'Washing Machine', 'power': 0.5, 'duration': 2},  # 2 giờ
        'dw': {'name': 'Dishwasher', 'power': 1.2, 'duration': 1},       # 1 giờ
        'ev': {'name': 'EV Charger', 'power_max': 7.0, 'capacity': 40}   # 40kWh
    },
    
    # Adjustable: Quan tâm đến nhiệt độ (Comfort)
    'adjustable': {
        'ac_living': {'name': 'Living Room AC', 'power_max': 2.0, 'room': 'living'},
        'ac_master': {'name': 'Master Bedroom AC', 'power_max': 1.5, 'room': 'master'},
        'ac_bed2':   {'name': 'Bedroom 2 AC', 'power_max': 1.0, 'room': 'bed2'}
    },
    
    # Fixed: Tự động bật theo giờ sinh hoạt
    'fixed': {
        'light_living':  {'power': 0.05, 'room': 'living'},
        'light_master':  {'power': 0.03, 'room': 'master'},
        'light_bed2':    {'power': 0.03, 'room': 'bed2'},
        'light_kitchen': {'power': 0.04, 'room': 'kitchen'},
        'light_toilet':  {'power': 0.02, 'room': 'toilet'},
        'fridge':        {'power': 0.15, 'always_on': True}
    }
}

# Mapping Action Index cho PPO (Chỉ học những cái tốn điện/khó)
ACTION_INDICES = {
    'battery': 0,
    'ac_living': 1,
    'ac_master': 2,
    'ac_bed2': 3,
    'ev': 4,
    'wm': 5,
    'dw': 6
}
# Tổng Action Space = 7

# Khung giờ có người ở nhà (Giả định logic sinh hoạt)
# (Start Hour, End Hour)
ROOM_OCCUPANCY_HOURS = {
    'living':  [(18, 22), (7, 8)],   # Tối sinh hoạt chung, sáng chuẩn bị đi làm
    'master':  [(22, 24), (0, 6)],   # Ngủ đêm
    'bed2':    [(22, 24), (0, 6)],
    'kitchen': [(6, 7), (18, 19)],   # Nấu ăn sáng/tối
    'toilet':  [(6, 23)]             # Ngẫu nhiên (sẽ random xác suất trong code env)
}

# Thermal constants for room temperature simulation
THERMAL_CONSTANTS = {
    'k1': 0.1,   # Heat transfer coefficient (outdoor influence)
    'k2': 2.0,   # AC cooling efficiency
    'comfort_temp': 25.0,  # Target comfortable temperature
    'comfort_tolerance': 2.0  # Acceptable deviation from comfort_temp
}

# EV Charging constraints
EV_CONFIG = {
    'deadline_hour': 7,    # Must be fully charged by 7 AM
    'min_target_soc': 0.9  # Target SOC by deadline
}
