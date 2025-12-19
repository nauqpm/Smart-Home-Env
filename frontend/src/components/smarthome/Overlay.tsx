import React, { useState } from 'react';
import { Html } from '@react-three/drei';
import { useStore } from '../../stores/useStore';
import InputPanel from './InputPanel';
import ComparisonDashboard from './ComparisonDashboard';

export default function Overlay() {
    const {
        devices, isNight, batterySOC, gridImport, totalBill, weather, n_home,
        simData, simStep, isLoading, currentModelView,
        fetchSimulation, setSimStep, setModelView, toggleNight
    } = useStore();

    const [activeTab, setActiveTab] = useState<'sim' | 'dashboard' | 'devices'>('sim');
    const [isCollapsed, setIsCollapsed] = useState(false);

    const weatherIcon: Record<string, string> = {
        sunny: '☀️', mild: '🌤️', cloudy: '☁️', rainy: '🌧️', stormy: '⛈️'
    };

    const timeDisplay = `Day 1 - ${simStep.toString().padStart(2, '0')}:00`;

    return (
        <Html fullscreen style={{ pointerEvents: 'none' }}>
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
                    width: 400,
                    maxHeight: '90vh',
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
                                SMART HOME <span style={{ fontSize: 12, fontWeight: 400, color: '#fff' }}>TWIN SIM</span>
                            </h2>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
                                <span style={{ fontSize: 16 }}>{weatherIcon[weather] || '☀️'}</span>
                                <span style={{ fontSize: 11, color: '#aaa' }}>{weather.toUpperCase()} • {timeDisplay}</span>
                            </div>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
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
                        {['sim', 'dashboard', 'devices'].map(tab => (
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
                                {tab}
                            </div>
                        ))}
                    </div>

                    {activeTab === 'sim' && (
                        <>
                            <InputPanel onSimulate={fetchSimulation} isLoading={isLoading} />
                            {simData && (
                                <div style={{ marginTop: 8 }}>
                                    <div style={{ fontSize: 11, color: '#aaa', marginBottom: 8, textTransform: 'uppercase' }}>
                                        Time Travel <span style={{ color: '#fff' }}>{simStep}:00</span>
                                    </div>
                                    <input
                                        type="range" min="0" max="23" value={simStep}
                                        onChange={(e) => setSimStep(parseInt(e.target.value))}
                                        style={{ width: '100%', accentColor: '#00e5ff', cursor: 'pointer' }}
                                    />
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: 9, color: '#666' }}>
                                        <span>00:00</span><span>12:00</span><span>23:00</span>
                                    </div>
                                    <div style={{ marginTop: 16 }}>
                                        <div style={{ fontSize: 11, color: '#aaa', marginBottom: 8, textTransform: 'uppercase' }}>Current View Model</div>
                                        <div style={{ display: 'flex', gap: 8 }}>
                                            <button
                                                onClick={() => setModelView('ppo')}
                                                style={{
                                                    flex: 1, padding: 8, borderRadius: 6, border: '1px solid',
                                                    background: currentModelView === 'ppo' ? 'rgba(255, 82, 82, 0.2)' : 'transparent',
                                                    borderColor: currentModelView === 'ppo' ? '#ff5252' : '#333',
                                                    color: currentModelView === 'ppo' ? '#ff5252' : '#666',
                                                    cursor: 'pointer'
                                                }}
                                            >
                                                PPO (Baseline)
                                            </button>
                                            <button
                                                onClick={() => setModelView('hybrid')}
                                                style={{
                                                    flex: 1, padding: 8, borderRadius: 6, border: '1px solid',
                                                    background: currentModelView === 'hybrid' ? 'rgba(76, 175, 80, 0.2)' : 'transparent',
                                                    borderColor: currentModelView === 'hybrid' ? '#4caf50' : '#333',
                                                    color: currentModelView === 'hybrid' ? '#4caf50' : '#666',
                                                    cursor: 'pointer'
                                                }}
                                            >
                                                Hybrid (Proposed)
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </>
                    )}

                    {activeTab === 'dashboard' && <ComparisonDashboard data={simData} />}

                    {activeTab === 'devices' && (
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 8, maxHeight: 400, overflowY: 'auto', paddingRight: 4 }}>
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
                    )}
                </div>
            )}
        </Html>
    );
}
