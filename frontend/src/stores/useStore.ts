import { create } from 'zustand';

export interface Device {
    id: string;
    name: string;
    type: 'light' | 'ac' | 'tv' | 'fan' | 'washer' | 'fridge' | 'charger' | 'laptop' | 'solar';
    isOn: boolean;
    basePower: number; // Watts
    currentPower: number; // Watts
}

interface State {
    devices: Record<string, Device>;
    metricsHistory: { time: string; power: number; temperature: number }[];
    isNight: boolean;

    // AI / Backend State
    aiMode: boolean;
    batterySOC: number; // 0.0 - 1.0
    gridPrice: number; // $/kWh
    gridImport: number; // W. Positive = Buying, Negative = Selling
    decisionLog: string[];
    totalBill: number; // Accumulating Bill in VND

    // Environmental Context
    weather: 'sunny' | 'cloudy' | 'rainy' | 'stormy';
    n_home: number;

    // Data Playback
    simulationData: any[];
    fetchData: () => Promise<void>;
    simStep: number;

    // Actions
    toggleDevice: (id: string) => void;
    toggleNight: () => void;
    toggleAIMode: () => void;
    updateAIState: (data: Partial<State>) => void;
    tick: () => void;
}

const INITIAL_DEVICES: Record<string, Device> = {
    tv: { id: 'tv', name: 'Smart TV', type: 'tv', isOn: false, basePower: 150, currentPower: 0 },
    fridge: { id: 'fridge', name: 'Smart Fridge', type: 'fridge', isOn: true, basePower: 200, currentPower: 200 },
    washer: { id: 'washer', name: 'Washer', type: 'washer', isOn: false, basePower: 500, currentPower: 0 },
    ac_living: { id: 'ac_living', name: 'AC Living Room', type: 'ac', isOn: true, basePower: 1200, currentPower: 1200 },
    ac_master: { id: 'ac_master', name: 'AC Master Bed', type: 'ac', isOn: false, basePower: 1000, currentPower: 0 },
    fan: { id: 'fan', name: 'Ceiling Fan', type: 'fan', isOn: true, basePower: 75, currentPower: 75 },
    charger: { id: 'charger', name: 'EV Charger', type: 'charger', isOn: false, basePower: 7000, currentPower: 0 },
    laptop: { id: 'laptop', name: 'Guest Laptop', type: 'laptop', isOn: true, basePower: 65, currentPower: 65 },
    lamp: { id: 'lamp', name: 'Floor Lamp', type: 'light', isOn: false, basePower: 15, currentPower: 0 },
    solar: { id: 'solar', name: 'Solar Inverter', type: 'charger', isOn: true, basePower: 0, currentPower: 0 }, // Monitor only
};

export const useStore = create<State>((set) => ({
    devices: INITIAL_DEVICES,
    metricsHistory: [],
    isNight: false,

    // AI Defaults
    aiMode: false,
    batterySOC: 0.5,
    gridPrice: 2540, // Avg VND
    gridImport: 0,
    decisionLog: ['System Initialized'],
    totalBill: 0, // Accumulator
    weather: 'sunny',
    n_home: 2,
    simulationData: [],
    simStep: 0,

    fetchData: async () => {
        try {
            const response = await fetch('/data/agent_comparison.json');
            const json = await response.json();
            if (json && json.series) {
                set({ simulationData: json.series });
                console.log("Sim Data Loaded:", json.series.length);
            }
        } catch (e) {
            console.error("Failed to load sim data", e);
        }
    },

    toggleDevice: (id) =>
        set((state) => {
            if (state.aiMode) return state; // Disable manual control in AI Mode (optional choice)
            const device = state.devices[id];
            if (!device) return state;
            return {
                devices: {
                    ...state.devices,
                    [id]: { ...device, isOn: !device.isOn },
                },
            };
        }),

    toggleNight: () => set((state) => ({ isNight: !state.isNight })),
    toggleAIMode: () => set((state) => ({ aiMode: !state.aiMode })),
    updateAIState: (data) => set((state) => ({ ...state, ...data })),

    tick: () =>
        set((state) => {
            // 1. Update power for all active devices with noise
            const newDevices = { ...state.devices };
            let totalLoad = 0;

            Object.keys(newDevices).forEach((key) => {
                const device = newDevices[key];
                if (device.isOn) {
                    const noise = Math.random() * 10 - 5;
                    device.currentPower = Math.max(0, Math.floor(device.basePower + noise));
                } else {
                    device.currentPower = 0;
                }
                totalLoad += device.currentPower;
            });


            // 2. Playback or Mock Logic
            let newSOC = state.batterySOC;
            let newGridImport = totalLoad;
            let newGridPrice = state.gridPrice;
            const newLogs = [...state.decisionLog];
            let newTotalBill = state.totalBill || 0;
            let newWeather = state.weather;
            let newNHome = state.n_home;

            const nextStep = (state.simStep || 0) + 1;
            const { simulationData, metricsHistory } = state;

            let timeStr = "";

            if (simulationData && simulationData.length > 0) {
                // --- PLAYBACK MODE ---
                const idx = (nextStep) % simulationData.length;
                const row = simulationData[idx];

                // Map JSON fields to State
                newWeather = row.weather || 'sunny';
                newNHome = row.n_home !== undefined ? row.n_home : 2;
                newSOC = row.ppo_soc;
                newGridImport = row.ppo_grid;
                totalLoad = row.load; // Overwrite random load with real sim load

                // Cost (Total Bill from JSON)
                newTotalBill = row.ppo_total_bill;

                if (idx === 0) newLogs.unshift("Restarting Cycle");

                // Format Time: Day X - HH:00
                const d = row.day || Math.floor(idx / 24) + 1;
                const h = row.hour !== undefined ? row.hour : idx % 24;
                timeStr = `D${d} ${h.toString().padStart(2, '0')}:00`;

            } else {
                // --- MOCK MODE (Fallback) ---
                // Mock Environmental Changes
                if (Math.random() > 0.98) {
                    const weathers: ('sunny' | 'cloudy' | 'rainy' | 'stormy')[] = ['sunny', 'cloudy', 'rainy', 'stormy'];
                    newWeather = weathers[Math.floor(Math.random() * weathers.length)];
                    newLogs.unshift(`Weather changed to ${newWeather.toUpperCase()}`);
                }
                if (Math.random() > 0.98) {
                    newNHome = Math.floor(Math.random() * 4);
                }

                // Simulate Grid Price (VND)
                if (Math.random() > 0.9) {
                    newGridPrice = 2500 + Math.random() * 500;
                }

                // Simulate Battery
                const batteryMaxPower = 3000;
                if (newGridPrice > 3000 && newSOC > 0.1) {
                    const dischargeAmount = Math.min(totalLoad, batteryMaxPower);
                    newGridImport = totalLoad - dischargeAmount;
                    newSOC -= 0.005;
                    if (Math.random() > 0.8) newLogs.unshift("High Rate: Discharging");
                } else if (newGridPrice < 2100 && newSOC < 0.95) {
                    newGridImport = totalLoad + 2000;
                    newSOC += 0.005;
                    if (Math.random() > 0.8) newLogs.unshift("Low Rate: Charging");
                } else {
                    newGridImport = totalLoad;
                }

                // Accumulate Bill Mock
                const kwh = newGridImport / 1000.0;
                if (kwh > 0) newTotalBill += kwh * 2500;
                else newTotalBill += kwh * 2000;

                const now = new Date();
                timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
            }

            // keep log size small
            if (newLogs.length > 5) newLogs.pop();

            // 3. Add to history
            const newMetric = {
                time: timeStr,
                power: totalLoad,
                temperature: (simulationData && simulationData.length > 0) ? (simulationData[(nextStep) % simulationData.length].temp || 28) : (28 + Math.random()),
            };

            const newHistory = [...state.metricsHistory, newMetric];
            if (newHistory.length > 20) {
                newHistory.shift();
            }

            return {
                devices: newDevices,
                metricsHistory: newHistory,
                batterySOC: Math.max(0, Math.min(1, newSOC)),
                gridImport: Math.round(newGridImport),
                gridPrice: (simulationData && simulationData.length > 0) ? 2540 : Math.round(newGridPrice),
                decisionLog: newLogs,
                totalBill: newTotalBill,
                weather: newWeather,
                n_home: newNHome,
                simStep: nextStep
            };
        }),
}));
