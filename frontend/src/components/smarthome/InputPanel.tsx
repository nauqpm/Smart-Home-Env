import React, { useState } from 'react';

interface SimulationConfig {
    num_people: number;
    weather_condition: string;
    must_run_base: number;
    seed: number;
}

interface InputPanelProps {
    onSimulate: (config: SimulationConfig) => void;
    isLoading: boolean;
}

export default function InputPanel({ onSimulate, isLoading }: InputPanelProps) {
    const [config, setConfig] = useState<SimulationConfig>({
        num_people: 4,
        weather_condition: 'sunny',
        must_run_base: 0.2,
        seed: 42
    });

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value } = e.target;
        setConfig(prev => ({
            ...prev,
            [name]: name === 'weather_condition' ? value : Number(value)
        }));
    };

    const handleSubmit = () => {
        onSimulate(config);
    };

    const inputStyle = {
        background: 'rgba(255,255,255,0.05)',
        border: '1px solid rgba(255,255,255,0.1)',
        color: '#fff',
        padding: '6px 10px',
        borderRadius: 6,
        width: '100%',
        fontSize: 12,
        outline: 'none',
        marginTop: 4
    };

    const labelStyle = {
        fontSize: 10,
        color: '#aaa',
        textTransform: 'uppercase' as const,
        letterSpacing: 0.5,
        display: 'block'
    };

    return (
        <div style={{
            background: 'rgba(0,0,0,0.3)',
            borderRadius: 12,
            padding: 16,
            border: '1px solid rgba(255,255,255,0.05)',
            marginBottom: 16
        }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#00e5ff', marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 }}>
                Parameters
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div>
                    <label style={labelStyle}>Residents</label>
                    <input
                        type="number"
                        name="num_people"
                        value={config.num_people}
                        onChange={handleChange}
                        min={1} max={10}
                        style={inputStyle}
                    />
                </div>
                <div>
                    <label style={labelStyle}>Weather</label>
                    <select
                        name="weather_condition"
                        value={config.weather_condition}
                        onChange={handleChange}
                        style={inputStyle}
                    >
                        <option value="sunny">Sunny</option>
                        <option value="mild">Mild</option>
                        <option value="cloudy">Cloudy</option>
                        <option value="rainy">Rainy</option>
                        <option value="stormy">Stormy</option>
                    </select>
                </div>
                <div>
                    <label style={labelStyle}>Base Load (kW)</label>
                    <input
                        type="number"
                        step="0.1"
                        name="must_run_base"
                        value={config.must_run_base}
                        onChange={handleChange}
                        style={inputStyle}
                    />
                </div>
                <div>
                    <label style={labelStyle}>Sim Seed</label>
                    <input
                        type="number"
                        name="seed"
                        value={config.seed}
                        onChange={handleChange}
                        style={inputStyle}
                    />
                </div>
            </div>

            <button
                onClick={handleSubmit}
                disabled={isLoading}
                style={{
                    width: '100%',
                    marginTop: 16,
                    padding: '10px',
                    background: isLoading ? '#333' : 'linear-gradient(90deg, #00e5ff, #2979ff)',
                    border: 'none',
                    borderRadius: 6,
                    color: '#fff',
                    fontWeight: 600,
                    fontSize: 12,
                    cursor: isLoading ? 'not-allowed' : 'pointer',
                    transition: 'opacity 0.2s'
                }}
            >
                {isLoading ? 'RUNNING SIM...' : 'RUN SIMULATION'}
            </button>
        </div>
    );
}
