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
    gridPrice: 0.12,
    gridImport: 0,
    decisionLog: ['System Initialized'],

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

            // 2. Mock AI Logic (if no backend connected yet)
            let newSOC = state.batterySOC;
            let newGridImport = totalLoad;
            let newGridPrice = state.gridPrice;
            const newLogs = [...state.decisionLog];

            // Simulate Grid Price fluctuations
            if (Math.random() > 0.9) {
                newGridPrice = 0.10 + Math.random() * 0.20; // 0.10 to 0.30
                newLogs.unshift(`${new Date().toLocaleTimeString()} - Price Change: $${newGridPrice.toFixed(2)}`);
            }

            // Simulate Battery Logic
            // If Price > 0.25, Discharge Battery to reduce Grid Import
            // If Price < 0.15, Charge Battery
            const batteryMaxPower = 3000; // 3kW max

            if (newGridPrice > 0.25 && newSOC > 0.1) {
                // High Price: Discharge
                const dischargeAmount = Math.min(totalLoad, batteryMaxPower);
                newGridImport = totalLoad - dischargeAmount;
                newSOC -= 0.005; // Drain battery
                if (Math.random() > 0.8) newLogs.unshift("High Price: Discharging Battery");
            } else if (newGridPrice < 0.15 && newSOC < 0.95) {
                // Low Price: Charge
                newGridImport = totalLoad + 2000; // Load + Charging
                newSOC += 0.005; // Charge battery
                if (Math.random() > 0.8) newLogs.unshift("Low Price: Charging Battery");
            } else {
                // Idle
                newGridImport = totalLoad;
            }

            // keep log size small
            if (newLogs.length > 5) newLogs.pop();

            // 3. Add to history
            const now = new Date();
            const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;

            const newMetric = {
                time: timeStr,
                power: totalLoad,
                temperature: 28 + Math.random(),
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
                gridPrice: parseFloat(newGridPrice.toFixed(2)),
                decisionLog: newLogs
            };
        }),
}));
