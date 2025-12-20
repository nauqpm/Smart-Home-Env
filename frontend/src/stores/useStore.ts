import { create } from 'zustand';

// --- 1. Define Data Contract Interfaces ---

interface EnvironmentData {
    weather: 'sunny' | 'mild' | 'cloudy' | 'rainy' | 'stormy';
    temp: number;
    pv: number;
    price_tier: number;
}

// Room-specific device actions
interface AgentAction {
    // ACs by room
    ac_living: number;
    ac_master: number;
    ac_bed2: number;
    // Lights by room
    light_living: number;
    light_master: number;
    light_bed2: number;
    light_kitchen: number;
    light_toilet: number;
    // Other devices
    wm: number;
    dw: number;  // Dishwasher
    ev: number;
    battery: 'charge' | 'discharge' | 'idle';
}

interface AgentData {
    bill: number;
    soc: number;
    grid: number;
    actions: AgentAction;
    comfort: number;
}

export interface SimulationPacket {
    timestamp: string;
    env: EnvironmentData;
    ppo: AgentData;
    hybrid: AgentData;
}

// --- 2. Device State for 3D Visualization ---

export interface Device {
    id: string;
    name: string;
    room?: string;
    type: 'light' | 'ac' | 'tv' | 'fan' | 'washer' | 'fridge' | 'charger' | 'laptop' | 'solar';
    isOn: boolean;
    basePower: number;
    currentPower: number;
}

// --- 3. Define Store State ---

interface AppState {
    isConnected: boolean;
    setIsConnected: (status: boolean) => void;

    simData: SimulationPacket | null;
    updateSimData: (data: SimulationPacket) => void;

    currentViewMode: 'ppo' | 'hybrid';
    setViewMode: (mode: 'ppo' | 'hybrid') => void;

    history: SimulationPacket[];
    devices: Record<string, Device>;

    isNight: boolean;
    toggleNight: () => void;

    // Legacy compatibility
    batterySOC: number;
    gridImport: number;
    totalBill: number;
    weather: string;
    n_home: number;
}

const INITIAL_DEVICES: Record<string, Device> = {
    // ACs by room
    ac_living: { id: 'ac_living', name: 'AC Living Room', room: 'living', type: 'ac', isOn: false, basePower: 1500, currentPower: 0 },
    ac_master: { id: 'ac_master', name: 'AC Master Bedroom', room: 'master', type: 'ac', isOn: false, basePower: 1200, currentPower: 0 },
    ac_bed2: { id: 'ac_bed2', name: 'AC 2nd Bedroom', room: 'bed2', type: 'ac', isOn: false, basePower: 1000, currentPower: 0 },

    // Lights by room
    light_living: { id: 'light_living', name: 'Living Room Light', room: 'living', type: 'light', isOn: false, basePower: 20, currentPower: 0 },
    light_master: { id: 'light_master', name: 'Master Bedroom Light', room: 'master', type: 'light', isOn: false, basePower: 15, currentPower: 0 },
    light_bed2: { id: 'light_bed2', name: '2nd Bedroom Light', room: 'bed2', type: 'light', isOn: false, basePower: 15, currentPower: 0 },
    light_kitchen: { id: 'light_kitchen', name: 'Kitchen Light', room: 'kitchen', type: 'light', isOn: false, basePower: 20, currentPower: 0 },
    light_toilet: { id: 'light_toilet', name: 'Toilet Light', room: 'toilet', type: 'light', isOn: false, basePower: 10, currentPower: 0 },

    // Other appliances
    tv: { id: 'tv', name: 'Smart TV', room: 'living', type: 'tv', isOn: false, basePower: 150, currentPower: 0 },
    fridge: { id: 'fridge', name: 'Smart Fridge', room: 'kitchen', type: 'fridge', isOn: true, basePower: 200, currentPower: 200 },
    washer: { id: 'washer', name: 'Washing Machine', room: 'utility', type: 'washer', isOn: false, basePower: 500, currentPower: 0 },
    dishwasher: { id: 'dishwasher', name: 'Dishwasher', room: 'kitchen', type: 'washer', isOn: false, basePower: 1200, currentPower: 0 },
    charger: { id: 'charger', name: 'EV Charger', room: 'garage', type: 'charger', isOn: false, basePower: 7000, currentPower: 0 },
};

export const useStore = create<AppState>((set, get) => ({
    isConnected: false,
    setIsConnected: (status) => set({ isConnected: status }),

    simData: null,
    history: [],

    updateSimData: (data) => set((state) => {
        const newHistory = [...state.history, data].slice(-24);
        const agentData = data[state.currentViewMode];
        const actions = agentData?.actions;

        // Update device states based on room-specific agent actions
        const newDevices = { ...state.devices };

        if (actions) {
            // Update ACs
            newDevices.ac_living.isOn = actions.ac_living === 1;
            newDevices.ac_living.currentPower = actions.ac_living === 1 ? newDevices.ac_living.basePower : 0;

            newDevices.ac_master.isOn = actions.ac_master === 1;
            newDevices.ac_master.currentPower = actions.ac_master === 1 ? newDevices.ac_master.basePower : 0;

            newDevices.ac_bed2.isOn = actions.ac_bed2 === 1;
            newDevices.ac_bed2.currentPower = actions.ac_bed2 === 1 ? newDevices.ac_bed2.basePower : 0;

            // Update Lights
            newDevices.light_living.isOn = actions.light_living === 1;
            newDevices.light_living.currentPower = actions.light_living === 1 ? newDevices.light_living.basePower : 0;

            newDevices.light_master.isOn = actions.light_master === 1;
            newDevices.light_master.currentPower = actions.light_master === 1 ? newDevices.light_master.basePower : 0;

            newDevices.light_bed2.isOn = actions.light_bed2 === 1;
            newDevices.light_bed2.currentPower = actions.light_bed2 === 1 ? newDevices.light_bed2.basePower : 0;

            newDevices.light_kitchen.isOn = actions.light_kitchen === 1;
            newDevices.light_kitchen.currentPower = actions.light_kitchen === 1 ? newDevices.light_kitchen.basePower : 0;

            newDevices.light_toilet.isOn = actions.light_toilet === 1;
            newDevices.light_toilet.currentPower = actions.light_toilet === 1 ? newDevices.light_toilet.basePower : 0;

            // Update other devices
            newDevices.washer.isOn = actions.wm === 1;
            newDevices.washer.currentPower = actions.wm === 1 ? newDevices.washer.basePower : 0;

            newDevices.dishwasher.isOn = actions.dw === 1;
            newDevices.dishwasher.currentPower = actions.dw === 1 ? newDevices.dishwasher.basePower : 0;

            newDevices.charger.isOn = actions.ev === 1;
            newDevices.charger.currentPower = actions.ev === 1 ? newDevices.charger.basePower : 0;
        }

        // Determine day/night from timestamp
        const hour = parseInt(data.timestamp?.split(':')[0] || '12');
        const isNight = hour < 6 || hour >= 18;

        return {
            simData: data,
            history: newHistory,
            devices: newDevices,
            isNight,
            batterySOC: agentData?.soc / 100 || 0.5,
            gridImport: agentData?.grid * 1000 || 0,
            totalBill: agentData?.bill || 0,
            weather: data.env?.weather || 'sunny',
            n_home: hour >= 17 || hour <= 7 ? 2 : 0,
        };
    }),

    currentViewMode: 'ppo',
    setViewMode: (mode) => {
        set({ currentViewMode: mode });
        const { simData } = get();
        if (simData) {
            const agentData = simData[mode];
            const actions = agentData?.actions;
            const devices = { ...get().devices };

            if (actions) {
                // Update all room-specific devices when view mode changes
                devices.ac_living.isOn = actions.ac_living === 1;
                devices.ac_living.currentPower = actions.ac_living === 1 ? devices.ac_living.basePower : 0;
                devices.ac_master.isOn = actions.ac_master === 1;
                devices.ac_master.currentPower = actions.ac_master === 1 ? devices.ac_master.basePower : 0;
                devices.ac_bed2.isOn = actions.ac_bed2 === 1;
                devices.ac_bed2.currentPower = actions.ac_bed2 === 1 ? devices.ac_bed2.basePower : 0;

                devices.light_living.isOn = actions.light_living === 1;
                devices.light_living.currentPower = actions.light_living === 1 ? devices.light_living.basePower : 0;
                devices.light_master.isOn = actions.light_master === 1;
                devices.light_master.currentPower = actions.light_master === 1 ? devices.light_master.basePower : 0;
                devices.light_bed2.isOn = actions.light_bed2 === 1;
                devices.light_bed2.currentPower = actions.light_bed2 === 1 ? devices.light_bed2.basePower : 0;
                devices.light_kitchen.isOn = actions.light_kitchen === 1;
                devices.light_kitchen.currentPower = actions.light_kitchen === 1 ? devices.light_kitchen.basePower : 0;
                devices.light_toilet.isOn = actions.light_toilet === 1;
                devices.light_toilet.currentPower = actions.light_toilet === 1 ? devices.light_toilet.basePower : 0;

                devices.washer.isOn = actions.wm === 1;
                devices.washer.currentPower = actions.wm === 1 ? devices.washer.basePower : 0;
                devices.dishwasher.isOn = actions.dw === 1;
                devices.dishwasher.currentPower = actions.dw === 1 ? devices.dishwasher.basePower : 0;
                devices.charger.isOn = actions.ev === 1;
                devices.charger.currentPower = actions.ev === 1 ? devices.charger.basePower : 0;
            }

            set({
                devices,
                batterySOC: agentData?.soc / 100 || 0.5,
                gridImport: agentData?.grid * 1000 || 0,
                totalBill: agentData?.bill || 0,
            });
        }
    },

    devices: INITIAL_DEVICES,

    isNight: false,
    toggleNight: () => set((state) => ({ isNight: !state.isNight })),

    batterySOC: 0.5,
    gridImport: 0,
    totalBill: 0,
    weather: 'sunny',
    n_home: 2,
}));
