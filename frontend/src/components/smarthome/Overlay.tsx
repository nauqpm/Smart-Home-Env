import React, { useState } from 'react';
import { createPortal } from 'react-dom';
import { useStore } from '../../stores/useStore';
import ComparisonDashboard from './ComparisonDashboard';

export default function Overlay() {
    const {
        devices, isNight, batterySOC, gridImport, totalBill, weather, n_home,
        simData, currentViewMode, setViewMode, toggleNight, isConnected, resetSimulation
    } = useStore();

    const [activeTab, setActiveTab] = useState<'dashboard' | 'devices'>('dashboard');
    const [isCollapsed, setIsCollapsed] = useState(false);

    const weatherIcon: Record<string, string> = {
        sunny: '☀️', mild: '🌤️', cloudy: '☁️', rainy: '🌧️', stormy: '⛈️'
    };

    const timeDisplay = simData?.timestamp || '--:--';

    // Use createPortal to render outside Canvas, keeping overlay fixed on screen
    const overlayContent = (
        <div style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 100 }}>
            {/* Connection Status Indicator */}
            <div style={{
                position: 'absolute',
                top: 20,
                left: 20,
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                background: 'rgba(10, 15, 25, 0.9)',
                padding: '8px 14px',
                borderRadius: 20,
                pointerEvents: 'auto',
                border: isConnected ? '1px solid rgba(34, 197, 94, 0.4)' : '1px solid rgba(239, 68, 68, 0.4)',
            }}>
                <div style={{
                    width: 8, height: 8, borderRadius: '50%',
                    background: isConnected ? '#22c55e' : '#ef4444',
                    boxShadow: isConnected ? '0 0 8px #22c55e' : '0 0 8px #ef4444',
                }} />
                <span style={{ fontSize: 11, color: isConnected ? '#86efac' : '#fca5a5', fontWeight: 600 }}>
                    {isConnected ? 'LIVE' : 'DISCONNECTED'}
                </span>
            </div>

            {/* Collapsed State */}
            {isCollapsed && (
                <div
                    onClick={() => setIsCollapsed(false)}
                    style={{
                        position: 'absolute',
                        top: 20,
                        right: 20,
                        width: 50,
                        height: 50,
                        background: 'rgba(10, 15, 25, 0.95)',
                        backdropFilter: 'blur(16px)',
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        cursor: 'pointer',
                        pointerEvents: 'auto',
                        border: '1px solid rgba(0, 255, 255, 0.3)',
                        boxShadow: '0 4px 16px rgba(0, 229, 255, 0.15)',
                        fontSize: 20
                    }}
                    title="Mở Dashboard"
                >
                    📊
                </div>
            )}

            {/* Expanded State */}
            {!isCollapsed && (
                <div style={{
                    position: 'absolute',
                    top: 20,
                    right: 20,
                    width: 520,
                    maxHeight: '92vh',
                    background: 'rgba(10, 15, 25, 0.95)',
                    backdropFilter: 'blur(16px)',
                    borderRadius: 16,
                    padding: 20,
                    color: '#e0f7fa',
                    fontFamily: "'Inter', sans-serif",
                    boxShadow: '0 8px 32px rgba(0, 229, 255, 0.1)',
                    pointerEvents: 'auto',
                    border: '1px solid rgba(0, 255, 255, 0.2)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 16,
                    overflowY: 'auto'
                }}>
                    {/* Header */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                            <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: '#00e5ff' }}>
                                SMART HOME <span style={{ fontSize: 12, fontWeight: 400, color: '#fff' }}>COMPARISON DASHBOARD</span>
                            </h2>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
                                <span style={{ fontSize: 16 }}>{weatherIcon[weather] || '☀️'}</span>
                                <span style={{ fontSize: 11, color: '#aaa' }}>{weather.toUpperCase()} • {timeDisplay}</span>
                            </div>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            {/* New Day Button */}
                            <button
                                onClick={async () => {
                                    try {
                                        resetSimulation(); // Clear local state
                                        await fetch('http://localhost:8000/reset');
                                    } catch (e) {
                                        console.error('Reset failed:', e);
                                    }
                                }}
                                style={{
                                    display: 'flex', alignItems: 'center', gap: 4,
                                    background: 'linear-gradient(135deg, #10b981, #059669)',
                                    padding: '6px 12px', borderRadius: 6,
                                    border: 'none', cursor: 'pointer',
                                    fontSize: 10, fontWeight: 600, color: '#fff',
                                    boxShadow: '0 2px 8px rgba(16, 185, 129, 0.3)',
                                    transition: 'all 0.2s'
                                }}
                                title="Bắt đầu ngày mới (random episode)"
                            >
                                🔄 Sang Ngày Mới
                            </button>
                            <div style={{
                                display: 'flex', alignItems: 'center', gap: 6,
                                background: n_home > 0 ? 'rgba(76, 175, 80, 0.2)' : 'rgba(120, 144, 156, 0.2)',
                                padding: '4px 10px', borderRadius: 20,
                                border: n_home > 0 ? '1px solid rgba(76, 175, 80, 0.4)' : '1px solid rgba(120, 144, 156, 0.4)'
                            }}>
                                <div style={{ width: 6, height: 6, borderRadius: '50%', background: n_home > 0 ? '#4caf50' : '#b0bec5' }} />
                                <span style={{ fontSize: 9, fontWeight: 600, color: n_home > 0 ? '#81c784' : '#b0bec5' }}>
                                    {n_home > 0 ? `OCCUPIED (${n_home})` : 'EMPTY'}
                                </span>
                            </div>
                            {/* Collapse Button */}
                            <div
                                onClick={() => setIsCollapsed(true)}
                                style={{
                                    width: 28, height: 28, borderRadius: 6,
                                    background: 'rgba(255, 255, 255, 0.1)',
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                    cursor: 'pointer', fontSize: 14
                                }}
                                title="Thu nhỏ Dashboard"
                            >
                                ✕
                            </div>
                        </div>
                    </div>

                    {/* Tabs */}
                    <div style={{ display: 'flex', background: 'rgba(0,0,0,0.3)', borderRadius: 8, padding: 4 }}>
                        {['dashboard', 'devices'].map(tab => (
                            <div
                                key={tab}
                                onClick={() => setActiveTab(tab as any)}
                                style={{
                                    flex: 1, textAlign: 'center', padding: '8px 0', borderRadius: 6,
                                    cursor: 'pointer', fontSize: 11, fontWeight: 600, textTransform: 'uppercase',
                                    background: activeTab === tab ? 'rgba(0, 229, 255, 0.2)' : 'transparent',
                                    color: activeTab === tab ? '#fff' : '#666'
                                }}
                            >
                                {tab === 'dashboard' ? '📊 Comparison' : '🔌 Devices'}
                            </div>
                        ))}
                    </div>

                    {/* Dashboard Tab */}
                    {activeTab === 'dashboard' && <ComparisonDashboard />}

                    {/* Devices Tab */}
                    {activeTab === 'devices' && (
                        <div>
                            {/* View Mode Toggle */}
                            <div style={{ marginBottom: 16 }}>
                                <div style={{ fontSize: 11, color: '#aaa', marginBottom: 8, textTransform: 'uppercase' }}>
                                    Viewing: {currentViewMode.toUpperCase()} Agent Devices
                                </div>
                                <div style={{ display: 'flex', gap: 8 }}>
                                    <button
                                        onClick={() => setViewMode('ppo')}
                                        style={{
                                            flex: 1, padding: 8, borderRadius: 6, border: '1px solid',
                                            background: currentViewMode === 'ppo' ? 'rgba(244, 114, 182, 0.2)' : 'transparent',
                                            borderColor: currentViewMode === 'ppo' ? '#f472b6' : '#333',
                                            color: currentViewMode === 'ppo' ? '#f472b6' : '#666',
                                            cursor: 'pointer', fontSize: 12, fontWeight: 600
                                        }}
                                    >
                                        🤖 PPO
                                    </button>
                                    <button
                                        onClick={() => setViewMode('hybrid')}
                                        style={{
                                            flex: 1, padding: 8, borderRadius: 6, border: '1px solid',
                                            background: currentViewMode === 'hybrid' ? 'rgba(34, 211, 238, 0.2)' : 'transparent',
                                            borderColor: currentViewMode === 'hybrid' ? '#22d3ee' : '#333',
                                            color: currentViewMode === 'hybrid' ? '#22d3ee' : '#666',
                                            cursor: 'pointer', fontSize: 12, fontWeight: 600
                                        }}
                                    >
                                        🧠 Hybrid
                                    </button>
                                </div>
                            </div>

                            {/* Device List */}
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 8, maxHeight: 340, overflowY: 'auto', paddingRight: 4 }}>
                                {Object.values(devices).map((device) => (
                                    <div
                                        key={device.id}
                                        style={{
                                            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                            background: device.isOn ? 'rgba(0, 229, 255, 0.1)' : 'rgba(255,255,255,0.03)',
                                            padding: '10px 14px', borderRadius: 8,
                                            border: device.isOn ? '1px solid rgba(0, 229, 255, 0.3)' : '1px solid rgba(255,255,255,0.05)'
                                        }}
                                    >
                                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                                            <span style={{ fontSize: 12, fontWeight: 500, color: device.isOn ? '#fff' : '#aaa' }}>{device.name}</span>
                                            <span style={{ fontSize: 10, color: device.isOn ? '#00e5ff' : '#555' }}>
                                                {device.isOn ? `${device.currentPower} W` : 'Standby'}
                                            </span>
                                        </div>
                                        <div style={{
                                            width: 10, height: 10,
                                            background: device.isOn ? '#00e5ff' : '#333',
                                            borderRadius: '50%',
                                            boxShadow: device.isOn ? '0 0 8px #00e5ff' : 'none'
                                        }} />
                                    </div>
                                ))}
                            </div>

                            {/* Stats Footer */}
                            <div style={{
                                marginTop: 16, padding: 12, background: 'rgba(0,0,0,0.3)',
                                borderRadius: 8, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12
                            }}>
                                <div style={{ textAlign: 'center' }}>
                                    <div style={{ fontSize: 10, color: '#64748b' }}>Battery SOC</div>
                                    <div style={{ fontSize: 16, fontWeight: 700, color: batterySOC > 0.3 ? '#22c55e' : '#ef4444' }}>
                                        {(batterySOC * 100).toFixed(0)}%
                                    </div>
                                </div>
                                <div style={{ textAlign: 'center' }}>
                                    <div style={{ fontSize: 10, color: '#64748b' }}>Grid</div>
                                    <div style={{ fontSize: 16, fontWeight: 700, color: '#f59e0b' }}>
                                        {gridImport > 0 ? '+' : ''}{(gridImport / 1000).toFixed(1)} kW
                                    </div>
                                </div>
                                <div style={{ textAlign: 'center' }}>
                                    <div style={{ fontSize: 10, color: '#64748b' }}>Total Bill</div>
                                    <div style={{ fontSize: 16, fontWeight: 700, color: '#8b5cf6' }}>
                                        {(totalBill / 1000).toFixed(1)}K
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );

    // Portal to document.body so overlay stays fixed when 3D camera moves
    return createPortal(overlayContent, document.body);
}
