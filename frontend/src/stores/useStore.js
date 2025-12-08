import { create } from 'zustand'

const useStore = create((set, get) => ({
  deviceState: {
    fan_speed: 0,
    light_on: false,
    ac_target: 25,
  },

  metricsHistory: [],
  maxHistoryLength: 100, // Chỉ giữ lại 100 điểm dữ liệu gần nhất để tránh lag

  // --- Hành động: Nhận dữ liệu từ WebSocket ---
  ingestWebsocketData: (dataPacket) => {
    set((state) => {
      // 1. Cập nhật trạng thái thiết bị (ghi đè cái cũ)
      const newDeviceState = dataPacket.devices;

      // 2. Cập nhật lịch sử metrics (thêm vào mảng)
      const newMetricPoint = {
        time: new Date(dataPacket.timestamp).toLocaleTimeString(), // Format giờ phút giây
        timestamp: dataPacket.timestamp,
        temperature: dataPacket.metrics.temperature,
        power: dataPacket.metrics.power,
      };

      // Thêm cái mới vào cuối, và cắt bớt phần đầu nếu quá dài
      let newHistory = [...state.metricsHistory, newMetricPoint];
      if (newHistory.length > state.maxHistoryLength) {
        newHistory = newHistory.slice(newHistory.length - state.maxHistoryLength);
      }

      return {
        deviceState: newDeviceState,
        metricsHistory: newHistory,
      };
    });
  },
}))

export default useStore;