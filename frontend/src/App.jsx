import React, { useEffect, useCallback } from 'react';
import Scene3D from './components/Scene3D';
import { useStore } from './stores/useStore';
import './App.css';

// WebSocket URL
const WS_URL = 'ws://localhost:8001/ws';

function App() {
  const { updateSimData, setIsConnected, isConnected } = useStore();

  // WebSocket connection with reconnection logic
  useEffect(() => {
    let ws = null;
    let reconnectTimeout = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 10;
    const reconnectDelay = 2000;

    const connect = () => {
      console.log('📡 Connecting to WebSocket...', WS_URL);

      ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setIsConnected(true);
        reconnectAttempts = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          // Data from backend matches SimulationPacket interface
          updateSimData(data);
        } catch (e) {
          console.error('Parse error:', e);
        }
      };

      ws.onclose = (event) => {
        console.log('❌ WebSocket closed:', event.code, event.reason);
        setIsConnected(false);

        // Attempt to reconnect
        if (reconnectAttempts < maxReconnectAttempts) {
          reconnectAttempts++;
          console.log(`🔄 Reconnecting in ${reconnectDelay}ms... (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
          reconnectTimeout = setTimeout(connect, reconnectDelay);
        } else {
          console.error('❌ Max reconnection attempts reached');
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    };

    // Initial connection
    connect();

    // Cleanup on unmount
    return () => {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      if (ws) {
        ws.close();
      }
    };
  }, [updateSimData, setIsConnected]);

  return (
    <div className="app-root">
      <Scene3D />

      {/* Fallback Connection Notice */}
      {!isConnected && (
        <div style={{
          position: 'fixed',
          bottom: 20,
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(239, 68, 68, 0.9)',
          color: '#fff',
          padding: '10px 24px',
          borderRadius: 8,
          fontSize: 13,
          fontWeight: 600,
          boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          gap: 10,
        }}>
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            background: '#fff',
            animation: 'pulse 1s infinite',
          }} />
          Connecting to simulation server...
        </div>
      )}
    </div>
  );
}

export default App;