import { create } from 'zustand';

export interface Device {
    id: string;
    name: string;
    type: 'light' | 'ac' | 'tv' | 'fan' | 'washer' | 'fridge' | 'charger' | 'laptop' | 'solar';
    isOn: boolean;
    basePower: number; // Watts
    currentPower: number; // Watts
}

// Data structure returned by Backend API
export interface SimStepData {
    hour: number;
    soc: number;
    grid: number; // net import/export
    load: number;
    pv: number;
    temp: number;
    reward: number;
    total_bill: number;
    n_home: number;
    weather: string;
    devices: {
        washer: boolean;
        dishwasher: boolean;
        charger: boolean;
        ac_living: boolean;
        ac_master: boolean;
        tv: boolean;
        fridge: boolean;
        lights: boolean;
    };
}

interface SimDataPacket {
    ppo: SimStepData[];
    hybrid: SimStepData[];
}

interface State {
    devices: Record<string, Device>;
    metricsHistory: { time: string; power: number; temperature: number }[];
    isNight: boolean;

    // AI / Backend State
    aiMode: boolean; // Not used as much in Replay, but kept for UI toggle
    batterySOC: number; // 0.0 - 1.0
    gridPrice: number; // $/kWh or VND
    gridImport: number; // W
    decisionLog: string[];
    totalBill: number; // Accumulating Bill in VND

    // Environmental Context
    weather: string;
    n_home: number;

    // Simulation & Replay
    simData: SimDataPacket | null;
    currentModelView: 'ppo' | 'hybrid';
    simStep: number; // 0..23
    isLoading: boolean;

    // Actions
    fetchSimulation: (config: any) => Promise<void>;
    setSimStep: (step: number) => void;
    setModelView: (model: 'ppo' | 'hybrid') => void;
    toggleDevice: (id: string) => void;
    toggleNight: () => void;
    tick: () => void; // Used for auto-play loop if we keep it
}

const INITIAL_DEVICES: Record<string, Device> = {
    tv: { id: 'tv', name: 'Smart TV', type: 'tv', isOn: false, basePower: 150, currentPower: 0 },
    fridge: { id: 'fridge', name: 'Smart Fridge', type: 'fridge', isOn: true, basePower: 200, currentPower: 200 },
    washer: { id: 'washer', name: 'Washer', type: 'washer', isOn: false, basePower: 500, currentPower: 0 },
    ac_living: { id: 'ac_living', name: 'AC Living', type: 'ac', isOn: false, basePower: 1500, currentPower: 0 },
    ac_master: { id: 'ac_master', name: 'AC Master', type: 'ac', isOn: false, basePower: 1000, currentPower: 0 },
    fan: { id: 'fan', name: 'Ceiling Fan', type: 'fan', isOn: true, basePower: 75, currentPower: 75 },
    charger: { id: 'charger', name: 'EV Charger', type: 'charger', isOn: false, basePower: 7000, currentPower: 0 },
    laptop: { id: 'laptop', name: 'Laptop', type: 'laptop', isOn: false, basePower: 65, currentPower: 0 },
    lamp: { id: 'lamp', name: 'Smart Light', type: 'light', isOn: false, basePower: 15, currentPower: 0 },
};

export const useStore = create<State>((set, get) => ({
    devices: INITIAL_DEVICES,
    metricsHistory: [],
    isNight: false,

    aiMode: true,
    batterySOC: 0.5,
    gridPrice: 2540,
    gridImport: 0,
    decisionLog: ['System Ready'],
    totalBill: 0,
    weather: 'sunny',
    n_home: 2,

    simData: null,
    currentModelView: 'hybrid', // Default to showing the better model
    simStep: 0,
    isLoading: false,

    fetchSimulation: async (config) => {
        set({ isLoading: true });
        try {
            const response = await fetch('http://localhost:8000/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            const data = await response.json();
            if (data && data.ppo && data.hybrid) {
                set({
                    simData: data,
                    simStep: 0,
                    isLoading: false,
                    decisionLog: [`Simulation Completed. Loaded ${data.ppo.length} steps.`]
                });
                // Trigger initial update
                get().setSimStep(0);
            }
        } catch (e) {
            console.error("Simulation API Error", e);
            set({ isLoading: false, decisionLog: ["Simulation Failed. Check Backend."] });
        }
    },

    setSimStep: (step) => {
        const { simData, currentModelView } = get();
        if (!simData) return;

        // Clamp step
        const safeStep = Math.max(0, Math.min(step, simData[currentModelView].length - 1));
        const dataPoint = simData[currentModelView][safeStep];

        // Update Environment State
        set((state) => {
            const newDevices = { ...state.devices };

            // Map API device states to local devices
            if (dataPoint.devices) {
                newDevices.washer.isOn = dataPoint.devices.washer;
                newDevices.washer.currentPower = dataPoint.devices.washer ? newDevices.washer.basePower : 0;

                newDevices.ac_living.isOn = dataPoint.devices.ac_living;
                newDevices.ac_living.currentPower = dataPoint.devices.ac_living ? newDevices.ac_living.basePower : 0;

                newDevices.ac_master.isOn = dataPoint.devices.ac_master;
                newDevices.ac_master.currentPower = dataPoint.devices.ac_master ? newDevices.ac_master.basePower : 0;

                newDevices.charger.isOn = dataPoint.devices.charger;
                newDevices.charger.currentPower = dataPoint.devices.charger ? newDevices.charger.basePower : 0;

                newDevices.tv.isOn = dataPoint.devices.tv;
                newDevices.tv.currentPower = dataPoint.devices.tv ? newDevices.tv.basePower : 0;

                newDevices.lamp.isOn = dataPoint.devices.lights;
                newDevices.lamp.currentPower = dataPoint.devices.lights ? newDevices.lamp.basePower : 0;

                // Keep fridge always on/visual consistency
                newDevices.fridge.isOn = true;
                newDevices.fridge.currentPower = 200;
            }

            return {
                devices: newDevices,
                simStep: safeStep,
                batterySOC: dataPoint.soc,
                gridImport: Math.round(dataPoint.load - dataPoint.pv), // Simplified visual
                totalBill: dataPoint.total_bill,
                n_home: dataPoint.n_home,
                weather: dataPoint.weather,
                isNight: safeStep < 6 || safeStep > 18, // Auto day/night based on hour
            };
        });
    },

    setModelView: (model) => {
        set({ currentModelView: model });
        get().setSimStep(get().simStep); // Refresh view with new model data
    },

    toggleDevice: (id) =>
        set((state) => {
            if (state.simData) return state; // Locked in Sim Mode
            const device = state.devices[id];
            return {
                devices: { ...state.devices, [id]: { ...device, isOn: !device.isOn } }
            };
        }),

    toggleNight: () => set((state) => ({ isNight: !state.isNight })),

    tick: () => {
        // Optional: Auto-play logic could go here if we want a Play button later
    }
}));
