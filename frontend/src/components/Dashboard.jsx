import React from 'react';
import useStore from '../stores/useStore';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function Dashboard() {
  // Lấy lịch sử dữ liệu từ store
  const metricsHistory = useStore((state) => state.metricsHistory);
  
  // Lấy trạng thái tức thời để hiển thị số to
  const currentTemp = useStore((state) => state.metricsHistory.at(-1)?.temperature || '--');
  const currentPower = useStore((state) => state.metricsHistory.at(-1)?.power || '--');

  return (
    <div className="dashboard-content">
      <h2>Monitor Dashboard</h2>
      
      {/* Các thẻ số liệu nhanh */}
      <div className="metrics-cards">
        <div className="card temp">
          <h4>Nhiệt độ (Temp)</h4>
          <div className="value">{currentTemp} °C</div>
        </div>
        <div className="card power">
          <h4>Công suất (Power)</h4>
          <div className="value">{currentPower} kW</div>
        </div>
      </div>

      {/* Biểu đồ đường */}
      <div className="chart-container">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={metricsHistory}>
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis dataKey="time" stroke="#888" tick={{fontSize: 12}} />
            {/* Trục Y bên trái cho Nhiệt độ */}
            <YAxis yAxisId="left" stroke="#ff7675" label={{ value: 'Temp (°C)', angle: -90, position: 'insideLeft', fill: '#ff7675' }} domain={[25, 35]} />
            {/* Trục Y bên phải cho Công suất */}
            <YAxis yAxisId="right" orientation="right" stroke="#74b9ff" label={{ value: 'Power (kW)', angle: 90, position: 'insideRight', fill: '#74b9ff' }} domain={[0, 5]} />
            
            <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none' }} />
            <Legend />
            
            {/* isAnimationActive={false} để tắt animation mặc định giúp biểu đồ mượt khi update liên tục */}
            <Line yAxisId="left" type="monotone" dataKey="temperature" stroke="#ff7675" dot={false} isAnimationActive={false} name="Nhiệt độ" strokeWidth={2} />
            <Line yAxisId="right" type="monotone" dataKey="power" stroke="#74b9ff" dot={false} isAnimationActive={false} name="Công suất" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default Dashboard;