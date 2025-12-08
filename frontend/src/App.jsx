import React, { useEffect, useState } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';
import useStore from './stores/useStore';
import Scene3D from './components/Scene3D';
import Dashboard from './components/Dashboard';
import './App.css';

// Địa chỉ WebSocket Server Backend
const WS_URL = 'ws://127.0.0.1:8000/ws';

function App() {
  // Lấy action cập nhật store
  const ingestWebsocketData = useStore((state) => state.ingestWebsocketData);
  const [connectionStatus, setConnectionStatus] = useState('Connecting...');

  // --- Hook kết nối WebSocket ---
  const { lastJsonMessage, readyState } = useWebSocket(WS_URL, {
    onOpen: () => console.log('WS: Connected!'),
    onClose: () => console.log('WS: Disconnected!'),
    shouldReconnect: (closeEvent) => true, // Tự động kết nối lại nếu đứt
    reconnectAttempts: 10,
    reconnectInterval: 3000,
  });

  // --- Effect: Khi nhận được tin nhắn mới ---
  useEffect(() => {
    if (lastJsonMessage !== null) {
      // Đẩy dữ liệu vào Zustand store
      ingestWebsocketData(lastJsonMessage);
    }
  }, [lastJsonMessage, ingestWebsocketData]);

  // --- Effect: Theo dõi trạng thái kết nối ---
  useEffect(() => {
    const statusMap = {
      [ReadyState.CONNECTING]: 'Connecting...',
      [ReadyState.OPEN]: 'Connected (Live)',
      [ReadyState.CLOSING]: 'Closing...',
      [ReadyState.CLOSED]: 'Disconnected (Reconnecting...)',
      [ReadyState.UNINSTANTIATED]: 'Uninstantiated',
    };
    setConnectionStatus(statusMap[readyState]);
  }, [readyState]);


  return (
    <div className="app-container">
      {/* Phần 1: Khung cảnh 3D */}
      <div className="scene-container">
        <Scene3D />
        {/* Hiển thị trạng thái kết nối góc trên */}
        <div style={{position: 'absolute', top: 10, left: 10, background: 'rgba(0,0,0,0.5)', padding: '5px 10px', borderRadius: 4}}>
            Status: <span style={{color: readyState === ReadyState.OPEN ? '#55efc4' : '#ff7675'}}>{connectionStatus}</span>
        </div>
      </div>
      
      {/* Phần 2: Dashboard biểu đồ */}
      <div className="dashboard-container">
        <Dashboard />
      </div>
    </div>
  );
}

export default App;