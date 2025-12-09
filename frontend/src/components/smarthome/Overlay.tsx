import React, { useState } from 'react';
import { Html } from '@react-three/drei';
import { useStore } from '../../stores/useStore';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export default function Overlay() {
    const { devices, metricsHistory, toggleDevice, toggleNight, isNight, aiMode, toggleAIMode, batterySOC, gridImport, gridPrice, decisionLog } = useStore();

    const [activeTab, setActiveTab] = useState<'devices' | 'grid'>('grid');

    // Derived metrics
    const lastMetric = metricsHistory.length > 0 ? metricsHistory[metricsHistory.length - 1] : { power: 0, temperature: 28 };
    const totalPower = Math.round(lastMetric.power);
    const temperature = lastMetric.temperature.toFixed(1);

    return (
        <Html fullscreen style={{ pointerEvents: 'none' }}>
            <div style={{
                position: 'absolute',
                top: 20,
                right: 20,
                width: 380,
                background: 'rgba(10, 15, 25, 0.9)',
                backdropFilter: 'blur(12px)',
                borderRadius: 16,
                padding: 24,
                color: '#e0f7fa',
                fontFamily: "'Inter', sans-serif",
                boxShadow: '0 8px 32px rgba(0, 255, 255, 0.15)',
                pointerEvents: 'auto',
                border: '1px solid rgba(0, 255, 255, 0.3)',
                display: 'flex',
                flexDirection: 'column',
                gap: 20
            }}>
                {/* Header & Global Controls */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, letterSpacing: '0.5px', color: '#00e5ff', textShadow: '0 0 10px rgba(0,229,255,0.6)' }}>IOT HUB</h2>
                    <div style={{ display: 'flex', gap: 8 }}>
                        <div
                            onClick={toggleNight}
                            style={{
                                display: 'flex', alignItems: 'center', gap: 6,
                                background: 'rgba(255,255,255,0.1)', padding: '4px 10px', borderRadius: 20, cursor: 'pointer',
                                border: '1px solid rgba(255,255,255,0.2)'
                            }}
                        >
                            <span style={{ fontSize: 10, fontWeight: 600 }}>{isNight ? 'NIGHT' : 'DAY'}</span>
                            <div style={{ width: 8, height: 8, borderRadius: '50%', background: isNight ? '#4aa3df' : '#ffcc00', boxShadow: isNight ? '0 0 8px #4aa3df' : '0 0 8px #ffcc00' }} />
                        </div>
                    </div>
                </div>

                {/* Tabs */}
                <div style={{ display: 'flex', background: 'rgba(0,0,0,0.3)', borderRadius: 8, padding: 4 }}>
                    <div
                        onClick={() => setActiveTab('grid')}
                        style={{
                            flex: 1, textAlign: 'center', padding: '8px 0', borderRadius: 6, cursor: 'pointer', fontSize: 12, fontWeight: 600,
                            background: activeTab === 'grid' ? 'rgba(0, 229, 255, 0.2)' : 'transparent',
                            color: activeTab === 'grid' ? '#fff' : '#888',
                            transition: 'all 0.2s'
                        }}
                    >
                        SMART GRID (AI)
                    </div>
                    <div
                        onClick={() => setActiveTab('devices')}
                        style={{
                            flex: 1, textAlign: 'center', padding: '8px 0', borderRadius: 6, cursor: 'pointer', fontSize: 12, fontWeight: 600,
                            background: activeTab === 'devices' ? 'rgba(0, 229, 255, 0.2)' : 'transparent',
                            color: activeTab === 'devices' ? '#fff' : '#888',
                            transition: 'all 0.2s'
                        }}
                    >
                        DEVICES
                    </div>
                </div>

                {/* CONTENT: SMART GRID TAB */}
                {activeTab === 'grid' && (
                    <>
                        {/* AI Status */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(0, 255, 0, 0.05)', padding: 12, borderRadius: 8, border: '1px solid rgba(0, 255, 0, 0.2)' }}>
                            <div>
                                <div style={{ fontSize: 10, color: '#aaa', textTransform: 'uppercase' }}>AI Optimizer</div>
                                <div style={{ fontSize: 14, fontWeight: 600, color: aiMode ? '#4caf50' : '#ffa726' }}>
                                    {aiMode ? 'ACTIVE' : 'MANUAL OVERRIDE'}
                                </div>
                            </div>
                            <div
                                onClick={toggleAIMode}
                                style={{
                                    width: 40, height: 22, background: aiMode ? '#4caf50' : '#333', borderRadius: 22, position: 'relative', cursor: 'pointer', transition: 'background 0.3s'
                                }}
                            >
                                <div style={{ position: 'absolute', top: 2, left: aiMode ? 20 : 2, width: 18, height: 18, background: '#fff', borderRadius: '50%', transition: 'left 0.2s', boxShadow: '0 2px 4px rgba(0,0,0,0.3)' }} />
                            </div>
                        </div>

                        {/* KPIS */}
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                            <div style={{ background: 'rgba(255,255,255,0.05)', padding: 10, borderRadius: 8, textAlign: 'center' }}>
                                <div style={{ fontSize: 9, color: '#aaa' }}>Grid Price</div>
                                <div style={{ fontSize: 16, fontWeight: 700, color: '#ff5252' }}>${gridPrice}</div>
                                <div style={{ fontSize: 8, color: '#666' }}>/ kWh</div>
                            </div>
                            <div style={{ background: 'rgba(255,255,255,0.05)', padding: 10, borderRadius: 8, textAlign: 'center' }}>
                                <div style={{ fontSize: 9, color: '#aaa' }}>Importing</div>
                                <div style={{ fontSize: 16, fontWeight: 700, color: gridImport > 0 ? '#ff9800' : '#4caf50' }}>{Math.abs(gridImport)}</div>
                                <div style={{ fontSize: 8, color: '#666' }}>Watts</div>
                            </div>
                            <div style={{ background: 'rgba(255,255,255,0.05)', padding: 10, borderRadius: 8, textAlign: 'center' }}>
                                <div style={{ fontSize: 9, color: '#aaa' }}>Battery SOC</div>
                                <div style={{ fontSize: 16, fontWeight: 700, color: batterySOC > 0.2 ? '#00e5ff' : '#ff5252' }}>
                                    {(batterySOC * 100).toFixed(0)}%
                                </div>
                                <div style={{ width: '100%', height: 4, background: '#333', marginTop: 4, borderRadius: 2 }}>
                                    <div style={{ width: `${batterySOC * 100}%`, height: '100%', background: batterySOC > 0.2 ? '#00e5ff' : '#ff5252', borderRadius: 2 }} />
                                </div>
                            </div>
                        </div>

                        {/* Chart: Price vs Load */}
                        <div style={{ height: 140, background: 'rgba(0,0,0,0.2)', borderRadius: 8, padding: 8, border: '1px solid rgba(255,255,255,0.05)' }}>
                            <div style={{ fontSize: 10, color: '#aaa', marginBottom: 5 }}>Load History</div>
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={metricsHistory}>
                                    <defs>
                                        <linearGradient id="gradLoad" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#00e5ff" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#00e5ff" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <XAxis dataKey="time" hide />
                                    <YAxis hide domain={['auto', 'auto']} />
                                    <Tooltip contentStyle={{ background: '#222', border: 'none', fontSize: 11 }} />
                                    <Area type="monotone" dataKey="power" stroke="#00e5ff" fill="url(#gradLoad)" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>

                        {/* AI Logs */}
                        <div style={{ height: 100, background: 'rgba(0,0,0,0.4)', borderRadius: 8, padding: 10, overflowY: 'auto', border: '1px solid rgba(255,255,255,0.05)', fontFamily: 'monospace' }}>
                            <div style={{ fontSize: 10, color: '#888', marginBottom: 4, textTransform: 'uppercase' }}>Decision Stream</div>
                            {decisionLog.map((log, i) => (
                                <div key={i} style={{ fontSize: 10, color: i === 0 ? '#fff' : '#666', marginBottom: 2 }}>
                                    {i === 0 ? '> ' : '  '} {log}
                                </div>
                            ))}
                        </div>
                    </>
                )}

                {/* CONTENT: DEVICES TAB */}
                {activeTab === 'devices' && (
                    <>
                        {/* Big Stats Row */}
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                            <div style={{ background: 'rgba(0, 229, 255, 0.05)', padding: 12, borderRadius: 12, border: '1px solid rgba(0, 229, 255, 0.1)' }}>
                                <div style={{ fontSize: 10, color: '#88ccff', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Total Load</div>
                                <div style={{ fontSize: 24, fontWeight: 700, color: '#fff' }}>
                                    {totalPower} <span style={{ fontSize: 14, color: '#00e5ff', fontWeight: 400 }}>W</span>
                                </div>
                            </div>
                            <div style={{ background: 'rgba(0, 229, 255, 0.05)', padding: 12, borderRadius: 12, border: '1px solid rgba(0, 229, 255, 0.1)' }}>
                                <div style={{ fontSize: 10, color: '#88ccff', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Temperature</div>
                                <div style={{ fontSize: 24, fontWeight: 700, color: '#fff' }}>
                                    {temperature} <span style={{ fontSize: 14, color: '#00e5ff', fontWeight: 400 }}>°C</span>
                                </div>
                            </div>
                        </div>

                        {/* Device Control List */}
                        <div style={{ overflowY: 'auto', maxHeight: 350, paddingRight: 4 }}>
                            <div style={{ fontSize: 11, color: '#aaa', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Manual Control</div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 8 }}>
                                {Object.values(devices).map((device) => (
                                    <div
                                        key={device.id}
                                        style={{
                                            display: 'flex',
                                            justifyContent: 'space-between',
                                            alignItems: 'center',
                                            background: device.isOn ? 'rgba(0, 229, 255, 0.15)' : 'rgba(255,255,255,0.03)',
                                            padding: '10px 14px',
                                            borderRadius: 8,
                                            border: device.isOn ? '1px solid rgba(0, 229, 255, 0.4)' : '1px solid rgba(255,255,255,0.05)',
                                            transition: 'all 0.2s ease',
                                            opacity: aiMode ? 0.5 : 1, // Dim if AI Mode
                                            pointerEvents: aiMode ? 'none' : 'auto'
                                        }}
                                    >
                                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                                            <span style={{ fontSize: 13, fontWeight: 500, color: device.isOn ? '#fff' : '#aaa' }}>{device.name}</span>
                                            <span style={{ fontSize: 10, color: device.isOn ? '#00e5ff' : '#555' }}>
                                                {device.isOn ? `${device.currentPower} W` : 'Standby'}
                                            </span>
                                        </div>

                                        {/* Toggle Switch UI */}
                                        <div
                                            onClick={() => toggleDevice(device.id)}
                                            style={{
                                                width: 36, height: 20,
                                                background: device.isOn ? '#00e5ff' : '#333',
                                                borderRadius: 20,
                                                position: 'relative',
                                                cursor: 'pointer',
                                                transition: 'background 0.3s'
                                            }}
                                        >
                                            <div style={{
                                                position: 'absolute',
                                                top: 2, left: device.isOn ? 18 : 2,
                                                width: 16, height: 16,
                                                background: '#fff',
                                                borderRadius: '50%',
                                                transition: 'left 0.2s',
                                                boxShadow: '0 1px 3px rgba(0,0,0,0.3)'
                                            }} />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </>
                )}
            </div>
        </Html>
    );
}
