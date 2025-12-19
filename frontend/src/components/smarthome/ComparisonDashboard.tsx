import React from 'react';
import { AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface SimData {
    ppo: any[];
    hybrid: any[];
}

interface ComparisonDashboardProps {
    data: SimData | null;
}

export default function ComparisonDashboard({ data }: ComparisonDashboardProps) {
    if (!data || !data.ppo || data.ppo.length === 0) return null;

    // Prepare data directly for charts
    // We want to combine ppo and hybrid into single array for easy Recharts comparison
    const chartData = data.ppo.map((item, idx) => ({
        hour: item.hour,

        // PPO Metrics
        ppo_load: item.load,
        ppo_cost: item.total_bill,
        ppo_soc: item.soc * 100,

        // Hybrid Metrics
        hybrid_load: data.hybrid[idx]?.load || 0,
        hybrid_cost: data.hybrid[idx]?.total_bill || 0,
        hybrid_soc: (data.hybrid[idx]?.soc || 0) * 100,
    }));

    // Calculate Aggregates
    const finalPPO = data.ppo[data.ppo.length - 1];
    const finalHybrid = data.hybrid[data.hybrid.length - 1];

    const ppoTotalCost = finalPPO.total_bill;
    const hybridTotalCost = finalHybrid.total_bill;
    const savings = ((ppoTotalCost - hybridTotalCost) / ppoTotalCost) * 100;

    const BoxStyle = {
        background: 'rgba(0,0,0,0.3)',
        borderRadius: 8,
        padding: 10,
        border: '1px solid rgba(255,255,255,0.05)',
        display: 'flex',
        flexDirection: 'column' as const,
        alignItems: 'center'
    };

    const LabelStyle = { fontSize: 10, color: '#aaa', textTransform: 'uppercase' as const };
    const ValueStyle = { fontSize: 16, fontWeight: 700, color: '#fff', marginTop: 4 };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

            {/* KPI HEADERS */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                <div style={BoxStyle}>
                    <span style={LabelStyle}>PPO Cost</span>
                    <span style={{ ...ValueStyle, color: '#ff5252' }}>{ppoTotalCost.toLocaleString()} ₫</span>
                </div>
                <div style={BoxStyle}>
                    <span style={LabelStyle}>Hybrid Cost</span>
                    <span style={{ ...ValueStyle, color: '#4caf50' }}>{hybridTotalCost.toLocaleString()} ₫</span>
                </div>
                <div style={BoxStyle}>
                    <span style={LabelStyle}>Efficiency</span>
                    <span style={{ ...ValueStyle, color: savings > 0 ? '#4caf50' : '#ff5252' }}>
                        {savings > 0 ? '+' : ''}{savings.toFixed(1)}%
                    </span>
                </div>
            </div>

            {/* CHARTS */}
            <div style={{ height: 160, background: 'rgba(0,0,0,0.2)', borderRadius: 8, padding: 8 }}>
                <span style={{ ...LabelStyle, marginLeft: 4 }}>Battery SOC Comparison (%)</span>
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                        <defs>
                            <linearGradient id="gradPPO" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#ff5252" stopOpacity={0.2} />
                                <stop offset="95%" stopColor="#ff5252" stopOpacity={0} />
                            </linearGradient>
                            <linearGradient id="gradHybrid" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#4caf50" stopOpacity={0.2} />
                                <stop offset="95%" stopColor="#4caf50" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <XAxis dataKey="hour" hide />
                        <YAxis hide domain={[0, 100]} />
                        <Tooltip
                            contentStyle={{ background: '#1a1a1a', border: '1px solid #333', fontSize: 11 }}
                            itemStyle={{ fontSize: 11 }}
                        />
                        <Legend iconSize={8} wrapperStyle={{ fontSize: 10, bottom: 0 }} />

                        <Area type="monotone" dataKey="ppo_soc" name="PPO" stroke="#ff5252" fill="url(#gradPPO)" strokeWidth={2} />
                        <Area type="monotone" dataKey="hybrid_soc" name="Hybrid" stroke="#4caf50" fill="url(#gradHybrid)" strokeWidth={2} />
                    </AreaChart>
                </ResponsiveContainer>
            </div>

            <div style={{ height: 160, background: 'rgba(0,0,0,0.2)', borderRadius: 8, padding: 8 }}>
                <span style={{ ...LabelStyle, marginLeft: 4 }}>Power Load (Watts)</span>
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                        <defs>
                            <linearGradient id="gradLoadPPO" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#ff9800" stopOpacity={0.2} />
                                <stop offset="95%" stopColor="#ff9800" stopOpacity={0} />
                            </linearGradient>
                            <linearGradient id="gradLoadHybrid" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#00e5ff" stopOpacity={0.2} />
                                <stop offset="95%" stopColor="#00e5ff" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <XAxis dataKey="hour" hide />
                        <YAxis hide />
                        <Tooltip
                            contentStyle={{ background: '#1a1a1a', border: '1px solid #333', fontSize: 11 }}
                        />
                        <Area type="step" dataKey="ppo_load" name="PPO Load" stroke="#ff9800" fill="url(#gradLoadPPO)" strokeWidth={2} />
                        <Area type="step" dataKey="hybrid_load" name="Hybrid Load" stroke="#00e5ff" fill="url(#gradLoadHybrid)" strokeWidth={2} />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}
