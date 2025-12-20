import { useStore, SimulationPacket } from '../../stores/useStore';
import {
    LineChart, Line, AreaChart, Area, BarChart, Bar,
    XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

// --- Styles ---
const panelStyle: React.CSSProperties = {
    background: 'rgba(15, 23, 42, 0.95)',
    borderRadius: 12,
    padding: 16,
    color: '#e2e8f0',
    fontFamily: 'Inter, sans-serif',
};

const headerStyle: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
    padding: '12px 16px',
    background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(14, 165, 233, 0.2))',
    borderRadius: 8,
};

const chartContainerStyle: React.CSSProperties = {
    background: 'rgba(30, 41, 59, 0.6)',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
};

const buttonStyle = (active: boolean): React.CSSProperties => ({
    padding: '8px 20px',
    borderRadius: 6,
    border: 'none',
    cursor: 'pointer',
    fontWeight: 600,
    fontSize: 13,
    transition: 'all 0.2s ease',
    background: active ? 'linear-gradient(135deg, #6366f1, #0ea5e9)' : 'rgba(51, 65, 85, 0.8)',
    color: active ? '#fff' : '#94a3b8',
    boxShadow: active ? '0 4px 15px rgba(99, 102, 241, 0.4)' : 'none',
});

const weatherIcons: Record<string, string> = {
    sunny: '☀️',
    mild: '🌤️',
    cloudy: '☁️',
    rainy: '🌧️',
    stormy: '⛈️',
};

const tierColors: Record<number, string> = {
    1: '#22c55e', // Green - cheap
    2: '#84cc16',
    3: '#eab308',
    4: '#f97316',
    5: '#ef4444',
    6: '#dc2626', // Red - expensive
};

// --- Main Component ---
export default function ComparisonDashboard() {
    const { simData, history, currentViewMode, setViewMode, isConnected } = useStore();

    if (!simData) {
        return (
            <div style={{ ...panelStyle, textAlign: 'center', padding: 40 }}>
                <div style={{ fontSize: 48, marginBottom: 16 }}>📊</div>
                <div style={{ fontSize: 18, color: '#94a3b8' }}>
                    {isConnected ? 'Waiting for simulation data...' : 'Connecting to backend...'}
                </div>
                <div style={{
                    marginTop: 16, width: 40, height: 40,
                    border: '3px solid #6366f1', borderTopColor: 'transparent',
                    borderRadius: '50%', margin: '16px auto',
                    animation: 'spin 1s linear infinite'
                }} />
            </div>
        );
    }

    // Prepare chart data from history
    const chartData = history.map((packet, idx) => ({
        time: packet.timestamp,
        ppo_bill: packet.ppo.bill / 1000, // Convert to thousands VND
        hybrid_bill: packet.hybrid.bill / 1000,
        ppo_soc: packet.ppo.soc,
        hybrid_soc: packet.hybrid.soc,
        ppo_grid: packet.ppo.grid,
        hybrid_grid: packet.hybrid.grid,
    }));

    const selectedAgent = simData[currentViewMode];
    const otherAgent = simData[currentViewMode === 'ppo' ? 'hybrid' : 'ppo'];

    return (
        <div style={panelStyle}>
            {/* === HEADER: Environment Info === */}
            <div style={headerStyle}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
                    <div style={{ fontSize: 28, fontWeight: 700 }}>
                        🕐 {simData.timestamp}
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ fontSize: 24 }}>{weatherIcons[simData.env.weather] || '🌡️'}</span>
                        <span style={{ textTransform: 'capitalize' }}>{simData.env.weather}</span>
                    </div>
                    <div>🌡️ {simData.env.temp}°C</div>
                    <div>☀️ PV: {simData.env.pv} kW</div>
                </div>
                <div style={{
                    padding: '6px 12px',
                    borderRadius: 6,
                    background: tierColors[simData.env.price_tier] || '#6366f1',
                    fontWeight: 700,
                    fontSize: 13,
                }}>
                    TIER {simData.env.price_tier}
                </div>
            </div>

            {/* === CONTROL PANEL: View Mode Toggle === */}
            <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
                <button
                    style={buttonStyle(currentViewMode === 'ppo')}
                    onClick={() => setViewMode('ppo')}
                >
                    🤖 PPO Agent
                </button>
                <button
                    style={buttonStyle(currentViewMode === 'hybrid')}
                    onClick={() => setViewMode('hybrid')}
                >
                    🧠 Hybrid Agent
                </button>

                {/* Quick Stats */}
                <div style={{ marginLeft: 'auto', display: 'flex', gap: 16, alignItems: 'center' }}>
                    <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: 11, color: '#64748b' }}>PPO Bill</div>
                        <div style={{ fontWeight: 700, color: '#f472b6' }}>{(simData.ppo.bill / 1000).toFixed(1)}K</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: 11, color: '#64748b' }}>Hybrid Bill</div>
                        <div style={{ fontWeight: 700, color: '#22d3ee' }}>{(simData.hybrid.bill / 1000).toFixed(1)}K</div>
                    </div>
                    <div style={{
                        padding: '4px 10px',
                        borderRadius: 4,
                        background: simData.hybrid.bill < simData.ppo.bill ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                        color: simData.hybrid.bill < simData.ppo.bill ? '#22c55e' : '#ef4444',
                        fontWeight: 600,
                        fontSize: 12,
                    }}>
                        {simData.hybrid.bill < simData.ppo.bill ? '✓ Hybrid Wins' : '✓ PPO Wins'}
                    </div>
                </div>
            </div>

            {/* === CHARTS GRID === */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                {/* Bill Comparison Chart */}
                <div style={chartContainerStyle}>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: '#94a3b8' }}>
                        💰 Total Bill (VND x1000)
                    </div>
                    <ResponsiveContainer width="100%" height={140}>
                        <LineChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#64748b' }} />
                            <YAxis tick={{ fontSize: 10, fill: '#64748b' }} />
                            <Tooltip
                                contentStyle={{ background: '#1e293b', border: 'none', borderRadius: 6 }}
                                labelStyle={{ color: '#e2e8f0' }}
                            />
                            <Line type="monotone" dataKey="ppo_bill" stroke="#f472b6" strokeWidth={2} dot={false} name="PPO" />
                            <Line type="monotone" dataKey="hybrid_bill" stroke="#22d3ee" strokeWidth={2} dot={false} name="Hybrid" />
                            <Legend wrapperStyle={{ fontSize: 11 }} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                {/* Battery SOC Chart */}
                <div style={chartContainerStyle}>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: '#94a3b8' }}>
                        🔋 Battery SOC (%)
                    </div>
                    <ResponsiveContainer width="100%" height={140}>
                        <AreaChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#64748b' }} />
                            <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#64748b' }} />
                            <Tooltip
                                contentStyle={{ background: '#1e293b', border: 'none', borderRadius: 6 }}
                            />
                            <Area type="monotone" dataKey="ppo_soc" stroke="#f472b6" fill="#f472b6" fillOpacity={0.3} name="PPO" />
                            <Area type="monotone" dataKey="hybrid_soc" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.3} name="Hybrid" />
                            <Legend wrapperStyle={{ fontSize: 11 }} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>

                {/* Grid Import Chart */}
                <div style={chartContainerStyle}>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: '#94a3b8' }}>
                        ⚡ Grid Import/Export (kWh)
                    </div>
                    <ResponsiveContainer width="100%" height={140}>
                        <BarChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#64748b' }} />
                            <YAxis tick={{ fontSize: 10, fill: '#64748b' }} />
                            <Tooltip
                                contentStyle={{ background: '#1e293b', border: 'none', borderRadius: 6 }}
                            />
                            <Bar dataKey="ppo_grid" fill="#f472b6" name="PPO" />
                            <Bar dataKey="hybrid_grid" fill="#22d3ee" name="Hybrid" />
                            <Legend wrapperStyle={{ fontSize: 11 }} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                {/* Room Control Panel */}
                <div style={{ ...chartContainerStyle, gridColumn: '1 / -1' }}>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, color: '#94a3b8' }}>
                        🏠 Room Control Panel - {currentViewMode.toUpperCase()} Agent View
                    </div>

                    {/* Room Cards */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8 }}>
                        {/* Living Room */}
                        <RoomCard
                            name="Living Room"
                            icon="🛋️"
                            acOn={simData[currentViewMode].actions.ac_living === 1}
                            lightOn={simData[currentViewMode].actions.light_living === 1}
                        />

                        {/* Master Bedroom */}
                        <RoomCard
                            name="Master Bed"
                            icon="🛏️"
                            acOn={simData[currentViewMode].actions.ac_master === 1}
                            lightOn={simData[currentViewMode].actions.light_master === 1}
                        />

                        {/* 2nd Bedroom */}
                        <RoomCard
                            name="2nd Bedroom"
                            icon="🛌"
                            acOn={simData[currentViewMode].actions.ac_bed2 === 1}
                            lightOn={simData[currentViewMode].actions.light_bed2 === 1}
                        />

                        {/* Kitchen */}
                        <RoomCard
                            name="Kitchen"
                            icon="🍳"
                            lightOn={simData[currentViewMode].actions.light_kitchen === 1}
                        />

                        {/* Toilet */}
                        <RoomCard
                            name="Toilet"
                            icon="🚿"
                            lightOn={simData[currentViewMode].actions.light_toilet === 1}
                        />
                    </div>

                    {/* Other Devices */}
                    <div style={{ marginTop: 12, display: 'flex', gap: 16, justifyContent: 'center' }}>
                        <DeviceBadge
                            icon="🧺"
                            name="Washer"
                            ppoOn={simData.ppo.actions.wm === 1}
                            hybridOn={simData.hybrid.actions.wm === 1}
                        />
                        <DeviceBadge
                            icon="🔌"
                            name="EV Charger"
                            ppoOn={simData.ppo.actions.ev === 1}
                            hybridOn={simData.hybrid.actions.ev === 1}
                        />
                        <div style={{ display: 'flex', gap: 8 }}>
                            <BatteryBadge status={simData.ppo.actions.battery} />
                            <span style={{ color: '#64748b', fontSize: 11 }}>vs</span>
                            <BatteryBadge status={simData.hybrid.actions.battery} />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

// --- Helper Components ---

function ActionBadge({ on, color }: { on: boolean; color: string }) {
    return (
        <div style={{
            textAlign: 'center',
            padding: '4px 8px',
            borderRadius: 4,
            background: on ? `${color}33` : 'rgba(51, 65, 85, 0.5)',
            color: on ? color : '#64748b',
            fontWeight: 600,
            fontSize: 11,
        }}>
            {on ? 'ON' : 'OFF'}
        </div>
    );
}

function BatteryBadge({ status }: { status: 'charge' | 'discharge' | 'idle' }) {
    const config = {
        charge: { bg: 'rgba(34, 197, 94, 0.2)', color: '#22c55e', icon: '⚡', text: 'CHARGE' },
        discharge: { bg: 'rgba(239, 68, 68, 0.2)', color: '#ef4444', icon: '🔻', text: 'DISCHARGE' },
        idle: { bg: 'rgba(100, 116, 139, 0.2)', color: '#64748b', icon: '⏸️', text: 'IDLE' },
    }[status];

    return (
        <div style={{
            textAlign: 'center',
            padding: '4px 8px',
            borderRadius: 4,
            background: config.bg,
            color: config.color,
            fontWeight: 600,
            fontSize: 10,
        }}>
            {config.icon} {config.text}
        </div>
    );
}

// Room Card - Shows AC and Light status for a room
function RoomCard({ name, icon, acOn, lightOn }: { name: string; icon: string; acOn?: boolean; lightOn?: boolean }) {
    return (
        <div style={{
            background: 'rgba(30, 41, 59, 0.8)',
            borderRadius: 8,
            padding: 10,
            textAlign: 'center',
            border: (acOn || lightOn) ? '1px solid rgba(34, 197, 94, 0.4)' : '1px solid rgba(51, 65, 85, 0.4)',
        }}>
            <div style={{ fontSize: 20, marginBottom: 4 }}>{icon}</div>
            <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 6 }}>{name}</div>
            <div style={{ display: 'flex', gap: 4, justifyContent: 'center' }}>
                {acOn !== undefined && (
                    <div style={{
                        fontSize: 9,
                        padding: '2px 6px',
                        borderRadius: 4,
                        background: acOn ? 'rgba(56, 189, 248, 0.2)' : 'rgba(51, 65, 85, 0.5)',
                        color: acOn ? '#38bdf8' : '#64748b',
                        fontWeight: 600,
                    }}>
                        ❄️ {acOn ? 'ON' : 'OFF'}
                    </div>
                )}
                <div style={{
                    fontSize: 9,
                    padding: '2px 6px',
                    borderRadius: 4,
                    background: lightOn ? 'rgba(250, 204, 21, 0.2)' : 'rgba(51, 65, 85, 0.5)',
                    color: lightOn ? '#facc15' : '#64748b',
                    fontWeight: 600,
                }}>
                    💡 {lightOn ? 'ON' : 'OFF'}
                </div>
            </div>
        </div>
    );
}

// Device Badge - Shows PPO vs Hybrid status for other devices
function DeviceBadge({ icon, name, ppoOn, hybridOn }: { icon: string; name: string; ppoOn: boolean; hybridOn: boolean }) {
    return (
        <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            background: 'rgba(30, 41, 59, 0.8)',
            padding: '6px 12px',
            borderRadius: 6,
        }}>
            <span style={{ fontSize: 16 }}>{icon}</span>
            <span style={{ fontSize: 11, color: '#94a3b8' }}>{name}</span>
            <div style={{
                padding: '2px 6px',
                borderRadius: 4,
                background: ppoOn ? 'rgba(244, 114, 182, 0.2)' : 'rgba(51, 65, 85, 0.5)',
                color: ppoOn ? '#f472b6' : '#64748b',
                fontSize: 9,
                fontWeight: 600,
            }}>
                PPO: {ppoOn ? 'ON' : 'OFF'}
            </div>
            <div style={{
                padding: '2px 6px',
                borderRadius: 4,
                background: hybridOn ? 'rgba(34, 211, 238, 0.2)' : 'rgba(51, 65, 85, 0.5)',
                color: hybridOn ? '#22d3ee' : '#64748b',
                fontSize: 9,
                fontWeight: 600,
            }}>
                HYB: {hybridOn ? 'ON' : 'OFF'}
            </div>
        </div>
    );
}
