import React from 'react';
import { useStore, FinalReportData } from '../../stores/useStore';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    LineChart, Line, AreaChart, Area
} from 'recharts';

export default function ReportModal() {
    const { reportData, showReport, setShowReport } = useStore();

    if (!showReport || !reportData) return null;

    const { metrics, charts, scenario } = reportData;

    // Calculate savings percentage
    const savings = metrics.ppo_bill > 0
        ? ((metrics.ppo_bill - metrics.hybrid_bill) / metrics.ppo_bill) * 100
        : 0;

    // Bar chart data for cost comparison
    const costData = [
        { name: 'PPO Agent', cost: metrics.ppo_bill, fill: '#f472b6' },
        { name: 'Hybrid Agent', cost: metrics.hybrid_bill, fill: '#22d3ee' }
    ];

    const modalStyle: React.CSSProperties = {
        position: 'fixed', inset: 0, zIndex: 1000,
        background: 'rgba(0, 0, 0, 0.85)', backdropFilter: 'blur(8px)',
        display: 'flex', justifyContent: 'center', alignItems: 'center',
        fontFamily: 'Inter, sans-serif'
    };

    const contentStyle: React.CSSProperties = {
        background: '#0f172a', width: '90%', maxWidth: 1100, maxHeight: '90vh',
        borderRadius: 16, padding: 24, overflowY: 'auto',
        border: '1px solid #334155', boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)'
    };

    return (
        <div style={modalStyle} onClick={() => setShowReport(false)}>
            <div style={contentStyle} onClick={(e) => e.stopPropagation()}>
                {/* Header */}
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 24 }}>
                    <div>
                        <h2 style={{ margin: 0, color: '#38bdf8', fontSize: 24 }}>
                            📑 DEMO REPORT: {scenario.toUpperCase()}
                        </h2>
                        <p style={{ color: '#94a3b8', margin: '4px 0 0' }}>24-Hour Simulation Analysis</p>
                    </div>
                    <button
                        onClick={() => setShowReport(false)}
                        style={{
                            background: 'transparent', border: 'none',
                            color: '#fff', fontSize: 24, cursor: 'pointer',
                            width: 40, height: 40, borderRadius: 8,
                            display: 'flex', alignItems: 'center', justifyContent: 'center'
                        }}
                    >✕</button>
                </div>

                {/* 1. Metrics Comparison Cards - Row 1: Core Metrics */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 16 }}>
                    <MetricCard
                        title="Total Cost (VND)"
                        ppo={metrics.ppo_bill}
                        hybrid={metrics.hybrid_bill}
                        unit="₫"
                        inverse={true}
                    />
                    <MetricCard
                        title="Comfort Loss"
                        ppo={metrics.ppo_comfort}
                        hybrid={metrics.hybrid_comfort}
                        unit="°Ch"
                        inverse={true}
                    />
                    <MetricCard
                        title="Grid Import"
                        ppo={metrics.ppo_grid}
                        hybrid={metrics.hybrid_grid}
                        unit="kWh"
                        inverse={true}
                    />
                </div>

                {/* Row 2: Solar & Battery Metrics */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 16 }}>
                    <MetricCard
                        title="Solar Self-Consumption"
                        ppo={metrics.ppo_solar_self_consumption || 0}
                        hybrid={metrics.hybrid_solar_self_consumption || 0}
                        unit="%"
                        inverse={false}
                    />
                    <MetricCard
                        title="Battery Discharged"
                        ppo={metrics.ppo_battery_discharged || 0}
                        hybrid={metrics.hybrid_battery_discharged || 0}
                        unit="kWh"
                        inverse={false}
                    />
                    <MetricCard
                        title="Peak Load"
                        ppo={metrics.ppo_peak_load || 0}
                        hybrid={metrics.hybrid_peak_load || 0}
                        unit="kW"
                        inverse={true}
                    />
                </div>

                {/* Row 3: Temperature Stats */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 24 }}>
                    <MetricCard
                        title="Avg Indoor Temp"
                        ppo={metrics.ppo_avg_temp || 25}
                        hybrid={metrics.hybrid_avg_temp || 25}
                        unit="°C"
                        inverse={false}
                    />
                    <MetricCard
                        title="Min Indoor Temp"
                        ppo={metrics.ppo_min_temp || 25}
                        hybrid={metrics.hybrid_min_temp || 25}
                        unit="°C"
                        inverse={false}
                    />
                    <MetricCard
                        title="Max Indoor Temp"
                        ppo={metrics.ppo_max_temp || 25}
                        hybrid={metrics.hybrid_max_temp || 25}
                        unit="°C"
                        inverse={true}
                    />
                </div>

                {/* Conclusion Banner */}
                <div style={{
                    background: savings > 0 ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                    border: `1px solid ${savings > 0 ? '#22c55e' : '#ef4444'}`,
                    padding: 16, borderRadius: 8, marginBottom: 24, textAlign: 'center'
                }}>
                    <span style={{ fontSize: 18, color: savings > 0 ? '#4ade80' : '#f87171', fontWeight: 'bold' }}>
                        CONCLUSION: Hybrid Agent is {Math.abs(savings).toFixed(1)}% {savings > 0 ? 'Cheaper' : 'More Expensive'} than PPO
                    </span>
                </div>

                {/* 2. Charts Section */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 24 }}>

                    {/* Cost Comparison Bar Chart */}
                    <div style={{ background: '#1e293b', padding: 16, borderRadius: 12 }}>
                        <h4 style={{ color: '#e2e8f0', marginTop: 0, marginBottom: 16 }}>💰 Cost Comparison</h4>
                        <ResponsiveContainer width="100%" height={200}>
                            <BarChart data={costData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                                <YAxis stroke="#94a3b8" fontSize={12} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
                                <Tooltip
                                    contentStyle={{ background: '#0f172a', border: '1px solid #334155' }}
                                    formatter={(value: number) => [`${value.toLocaleString()} VND`, 'Cost']}
                                />
                                <Bar dataKey="cost" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Temperature Profile Line Chart */}
                    <div style={{ background: '#1e293b', padding: 16, borderRadius: 12 }}>
                        <h4 style={{ color: '#e2e8f0', marginTop: 0, marginBottom: 16 }}>🌡️ Thermal Comfort Profile</h4>
                        <ResponsiveContainer width="100%" height={200}>
                            <LineChart data={charts.temps}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis dataKey="hour" stroke="#94a3b8" fontSize={12} tickFormatter={(h) => `${h}h`} />
                                <YAxis stroke="#94a3b8" fontSize={12} domain={[20, 45]} unit="°C" />
                                <Tooltip
                                    contentStyle={{ background: '#0f172a', border: '1px solid #334155' }}
                                    labelFormatter={(h) => `Hour ${h}`}
                                    formatter={(value: number) => [`${value.toFixed(1)}°C`]}
                                />
                                <Legend />
                                <Line type="monotone" dataKey="outdoor" stroke="#94a3b8" strokeDasharray="5 5" name="Outdoor" dot={false} strokeWidth={1} />
                                <Line type="monotone" dataKey="ppo" stroke="#f472b6" name="PPO Indoor" strokeWidth={2} dot={false} />
                                <Line type="monotone" dataKey="hybrid" stroke="#22d3ee" name="Hybrid Indoor" strokeWidth={2} dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Hybrid Energy Source Stacked Area Chart - Full Width */}
                <div style={{ background: '#1e293b', padding: 16, borderRadius: 12 }}>
                    <h4 style={{ color: '#e2e8f0', marginTop: 0, marginBottom: 16 }}>⚡ Hybrid Energy Sources</h4>
                    <ResponsiveContainer width="100%" height={220}>
                        <AreaChart data={charts.energy_stack}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <XAxis dataKey="hour" stroke="#94a3b8" fontSize={12} tickFormatter={(h) => `${h}h`} />
                            <YAxis stroke="#94a3b8" fontSize={12} unit=" kW" />
                            <Tooltip
                                contentStyle={{ background: '#0f172a', border: '1px solid #334155' }}
                                labelFormatter={(h) => `Hour ${h}`}
                                formatter={(value: number, name: string) => [`${value.toFixed(2)} kW`, name]}
                            />
                            <Legend />
                            <Area type="monotone" dataKey="pv" stackId="1" stroke="#facc15" fill="#facc15" name="Solar PV" fillOpacity={0.8} />
                            <Area type="monotone" dataKey="battery" stackId="1" stroke="#22c55e" fill="#22c55e" name="Battery" fillOpacity={0.8} />
                            <Area type="monotone" dataKey="grid" stackId="1" stroke="#f87171" fill="#f87171" name="Grid Import" fillOpacity={0.8} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
}

// Metric Card Component
interface MetricCardProps {
    title: string;
    ppo: number;
    hybrid: number;
    unit: string;
    inverse?: boolean; // If true, lower is better
}

function MetricCard({ title, ppo, hybrid, unit, inverse = false }: MetricCardProps) {
    const diff = hybrid - ppo;
    const isBetter = inverse ? diff < 0 : diff > 0;
    const color = isBetter ? '#4ade80' : '#f87171';
    const percentage = ppo !== 0 ? ((diff / ppo) * 100).toFixed(1) : '0';

    return (
        <div style={{
            background: '#1e293b', padding: 16, borderRadius: 12,
            borderLeft: `4px solid ${color}`
        }}>
            <div style={{ color: '#94a3b8', fontSize: 11, textTransform: 'uppercase', marginBottom: 8 }}>
                {title}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
                <div>
                    <div style={{ fontSize: 12, color: '#f472b6', marginBottom: 4 }}>
                        PPO: {ppo.toLocaleString()} {unit}
                    </div>
                    <div style={{ fontSize: 12, color: '#22d3ee' }}>
                        Hybrid: {hybrid.toLocaleString()} {unit}
                    </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: 20, fontWeight: 'bold', color: color }}>
                        {diff > 0 ? '+' : ''}{diff.toLocaleString()}
                    </div>
                    <div style={{ fontSize: 11, color: color }}>
                        ({percentage}%)
                    </div>
                </div>
            </div>
        </div>
    );
}
