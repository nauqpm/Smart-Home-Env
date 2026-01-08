
import os
import numpy as np
from stable_baselines3 import PPO
from smart_home_env import SmartHomeEnv

# --- CONFIGURATION (Copied from backend/main.py) ---
# Keeping 24 hours simulation as standard
T = 24
PRICE_PROFILE = np.array([0.1] * 6 + [0.15] * 6 + [0.25] * 6 + [0.18] * 6) * 10000 

BASE_CONFIG = {
    "critical": [0.33] * 24,
    "adjustable": [
        {"P_min": 0.5, "P_max": 2.0, "P_com": 1.5, "alpha": 0.06}, # AC Living
        {"P_min": 0.0, "P_max": 2.0, "P_com": 1.5, "alpha": 0.08}  # AC Master
    ],
    "shiftable_su": [
        {"name": "washing_machine", "rate": 0.5, "L": 1}, 
        {"name": "dishwasher", "rate": 0.8, "L": 1}
    ],
    "shiftable_si": [
        {"name": "ev_charger", "rate": 3.3, "E": 7.0}
    ],
    "beta": 0.5,
    "battery": {
        "capacity_kwh": 10.0,
        "soc_init": 0.5,
        "soc_min": 0.1,
        "soc_max": 0.9,
        "p_charge_max_kw": 3.0,
        "p_discharge_max_kw": 3.0,
        "eta_ch": 0.95,
        "eta_dis": 0.95
    },
    "pv_config": {
        "latitude": 10.762622,
        "longitude": 106.660172,
        "tz": "Asia/Ho_Chi_Minh",
        "surface_tilt": 10.0,
        "surface_azimuth": 180.0,
        "module_parameters": {"pdc0": 3.0}
    },
    "sim_steps": T
}

# --- MODEL PATHS ---
# Using absolute paths or relative to script execution. 
# We assume script is run from project root, so paths are backend/ppo_...
PPO_PATH = "backend/ppo_smart_home.zip"
HYBRID_PATH = "backend/ppo_hybrid_smart_home.zip"

def evaluate_agent(model_path, agent_name, n_episodes=5):
    print(f"\n--- Đang đánh giá: {agent_name} ---")
    
    if not os.path.exists(model_path):
        # Try local path if running from backend dir
        if os.path.exists(os.path.basename(model_path)):
            model_path = os.path.basename(model_path)
            print(f"⚠️ Đã chuyển sang model path cục bộ: {model_path}")
        else:
            print(f"❌ Lỗi: Không tìm thấy file model tại {model_path}")
            return None

    # Load Model
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        return None

    # Khởi tạo môi trường với BASE_CONFIG và PRICE_PROFILE chuẩn
    # Note: passing np.zeros(T) as dummy PV, env will calc from config
    env = SmartHomeEnv(PRICE_PROFILE, np.zeros(T), BASE_CONFIG)
    
    total_rewards = []
    total_costs = []
    total_comforts = []
    battery_usage = [] 

    for ep in range(n_episodes):
        # Set Global Seed to ensure Env internals (weather gen) are consistent
        seed = 42 + ep
        np.random.seed(seed)
        
        # Reset Env
        obs, info = env.reset(seed=seed)
        
        done = False
        ep_reward = 0
        ep_cost = 0
        
        # Tracking variables
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            ep_reward += reward
            # ep_cost += info.get('total_cost', 0) # INCORRECT: this is usually cumulative or step cost depending on env.
            # User advised: Check info['total_cost'] at END of episode
            
            # Track Battery (Index 0 is SOC)
            battery_usage.append(obs[0]) 
            
            steps += 1
            done = terminated or truncated

        # Collect Final Episode Stats
        total_rewards.append(ep_reward)
        
        # Taking total_cost from the final info dict
        final_bill = info.get('total_cost', 0)
        total_costs.append(final_bill)
        
        # Comfort penalty is usually hidden in reward or info, assume env.comfort_penalty exists
        # Or calculate if needed. The user script used env.comfort_penalty
        comfort_pen = getattr(env, 'comfort_penalty', 0)
        total_comforts.append(comfort_pen)

    # Calculate Averages
    avg_reward = np.mean(total_rewards)
    avg_cost = np.mean(total_costs)
    avg_bat = np.mean(battery_usage)

    print(f"✅ Kết quả {agent_name}:")
    print(f"   - Avg Reward: {avg_reward:.4f} (Cao hơn là tốt)")
    print(f"   - Avg Bill:   {avg_cost:.4f} VND (Thấp hơn là tốt)")
    print(f"   - Avg Battery SOC: {avg_bat:.2f} (Nếu ~0.0 hoặc ~0.5 mãi là nghi vấn)")
    
    return {
        "reward": avg_reward,
        "cost": avg_cost,
        "soc": avg_bat
    }

if __name__ == "__main__":
    # 1. Chạy PPO Thuần
    ppo_stats = evaluate_agent(PPO_PATH, "PPO Pure")
    
    # 2. Chạy Hybrid (Imitation + PPO)
    hybrid_stats = evaluate_agent(HYBRID_PATH, "Hybrid Agent")
    
    # 3. So sánh
    if ppo_stats and hybrid_stats:
        output_lines = []
        output_lines.append("\n" + "="*40)
        output_lines.append("KẾT QUẢ SO SÁNH TRỰC TIẾP (PYTHON)")
        output_lines.append("="*40)
        
        cost_diff = ((hybrid_stats['cost'] - ppo_stats['cost']) / ppo_stats['cost']) * 100 if ppo_stats['cost'] != 0 else 0
        reward_diff = hybrid_stats['reward'] - ppo_stats['reward']
        
        output_lines.append(f"💰 Hóa đơn (Bill):")
        output_lines.append(f"   PPO: {ppo_stats['cost']:.2f} vs Hybrid: {hybrid_stats['cost']:.2f}")
        if hybrid_stats['cost'] < ppo_stats['cost']:
            output_lines.append(f"   => Hybrid RẺ HƠN {abs(cost_diff):.2f}% 🏆")
        else:
            output_lines.append(f"   => Hybrid ĐẮT HƠN {abs(cost_diff):.2f}% ⚠️")

        output_lines.append(f"\n🎯 Điểm thưởng (Reward - Bao gồm cả tiện nghi):")
        output_lines.append(f"   PPO: {ppo_stats['reward']:.2f} vs Hybrid: {hybrid_stats['reward']:.2f}")
        if hybrid_stats['reward'] > ppo_stats['reward']:
            output_lines.append(f"   => Hybrid TỐT HƠN")
        else:
            output_lines.append(f"   => Hybrid KÉM HƠN")
            
        output_lines.append("\n🔍 Phân tích Battery SOC trung bình:")
        output_lines.append(f"   PPO: {ppo_stats['soc']:.2f}")
        output_lines.append(f"   Hybrid: {hybrid_stats['soc']:.2f}")
        if hybrid_stats['soc'] < 0.1:
            output_lines.append("⚠️ CẢNH BÁO: Hybrid có vẻ không sạc pin (Eternal Night bug?)")

        # Print to console (best effort)
        try:
            print("\n".join(output_lines))
        except:
            pass
            
        # Write to file
        with open("backend/benchmark_report.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
