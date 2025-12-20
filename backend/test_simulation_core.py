"""Test script for simulation_core.py"""
import json
from simulation_core import sim_instance

print("=" * 50)
print("Testing SimulationEngine Data Packet")
print("=" * 50)

# First update
sim_instance.update()
packet = sim_instance.get_data_packet()

print("\n📦 Data Packet (Step 1):")
print(json.dumps(packet, indent=2, ensure_ascii=False))

# Verify structure
print("\n✅ Structure Validation:")
print(f"  - timestamp: {packet.get('timestamp')}")
print(f"  - env keys: {list(packet.get('env', {}).keys())}")
print(f"  - ppo keys: {list(packet.get('ppo', {}).keys())}")
print(f"  - hybrid keys: {list(packet.get('hybrid', {}).keys())}")

# Run a few more steps
print("\n🔄 Running 3 more steps...")
for i in range(3):
    sim_instance.update()
    p = sim_instance.get_data_packet()
    print(f"  Step {i+2}: timestamp={p['timestamp']}, ppo_soc={p['ppo']['soc']}%, hybrid_soc={p['hybrid']['soc']}%")

print("\n✅ Test Complete!")
