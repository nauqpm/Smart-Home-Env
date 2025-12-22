import React, { useEffect, useCallback } from 'react';
import Scene3D from './components/Scene3D';
import { useStore } from './stores/useStore';
import './App.css';

// WebSocket URL
const WS_URL = 'ws://localhost:8000/ws';

function App() {
  const { updateSimData, setIsConnected, isConnected, setWsRef } = useStore();

  // WebSocket connection with exponential backoff reconnection
  useEffect(() => {
    let ws = null;
    let reconnectTimeout = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 10;
    const baseDelay = 2000; // Start with 2 seconds
    const maxDelay = 30000; // Cap at 30 seconds

    const connect = () => {
      // Only log detailed info for first few attempts to reduce console spam
      if (reconnectAttempts < 3) {
        console.log('📡 Connecting to WebSocket...', WS_URL);
      }

      ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setIsConnected(true);
        setWsRef(ws); // Store WebSocket reference for manual control
        reconnectAttempts = 0; // Reset on successful connection
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
        // Only log first close event to reduce spam
        if (reconnectAttempts === 0) {
          console.log('❌ WebSocket closed:', event.code, event.reason);
        }
        setIsConnected(false);

        // Exponential backoff: delay doubles each attempt, capped at maxDelay
        if (reconnectAttempts < maxReconnectAttempts) {
          const delay = Math.min(baseDelay * Math.pow(2, reconnectAttempts), maxDelay);
          reconnectAttempts++;

          if (reconnectAttempts <= 3) {
            console.log(`🔄 Reconnecting in ${delay / 1000}s... (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
          } else if (reconnectAttempts === 4) {
            console.log(`🔄 Continuing reconnection attempts silently...`);
          }

          reconnectTimeout = setTimeout(connect, delay);
        } else {
          console.error('❌ Max reconnection attempts reached. Please check if backend server is running.');
        }
      };

      ws.onerror = (error) => {
        // Only log first error to reduce spam
        if (reconnectAttempts === 0) {
          console.error('WebSocket error:', error);
        }
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