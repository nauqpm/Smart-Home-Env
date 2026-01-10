import { create } from 'zustand';

// --- 1. Define Data Contract Interfaces ---

interface EnvironmentData {
    weather: 'sunny' | 'mild' | 'cloudy' | 'rainy' | 'stormy';
    temp: number;
    pv: number;
    price_tier: number;
    // Demo mode fields (optional)
    demo_mode?: boolean;
    scenario_name?: string;
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
    // Room temperatures
    temp_living?: number;
    temp_master?: number;
    temp_bed2?: number;
}

export interface SimulationPacket {
    timestamp: string;
    env: EnvironmentData;
    ppo: AgentData;
    hybrid: AgentData;
}

// FINAL_REPORT data structure from backend
export interface FinalReportData {
    scenario: string;
    metrics: {
        // Cost metrics
        ppo_bill: number;
        hybrid_bill: number;
        // Comfort metrics
        ppo_comfort: number;
        hybrid_comfort: number;
        // Grid metrics
        ppo_grid: number;
        hybrid_grid: number;
        // Solar metrics
        solar_generated: number;
        ppo_solar_used: number;
        hybrid_solar_used: number;
        ppo_solar_self_consumption: number;
        hybrid_solar_self_consumption: number;
        // Battery metrics
        ppo_battery_discharged: number;
        hybrid_battery_discharged: number;
        // Temperature metrics
        ppo_avg_temp: number;
        hybrid_avg_temp: number;
        ppo_min_temp: number;
        hybrid_min_temp: number;
        ppo_max_temp: number;
        hybrid_max_temp: number;
        // Peak load
        ppo_peak_load: number;
        hybrid_peak_load: number;
    };
    charts: {
        temps: Array<{ hour: number; outdoor: number; ppo: number; hybrid: number }>;
        energy_stack: Array<{ hour: number; pv: number; grid: number; battery: number }>;
    };
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
    resetSimulation: () => void;

    currentViewMode: 'ppo' | 'hybrid';
    setViewMode: (mode: 'ppo' | 'hybrid') => void;

    history: SimulationPacket[];
    devices: Record<string, Device>;

    isNight: boolean;
    toggleNight: () => void;

    // Manual Override System
    manualOverride: Record<string, boolean>; // deviceId -> isManualMode
    manualDeviceState: Record<string, boolean>; // deviceId -> isOn (when manual)
    toggleManualMode: (deviceId: string) => void;
    toggleDevice: (deviceId: string) => void;

    // Room Energy for 3D Heatmap (kW)
    roomEnergy: Record<string, number>;

    // WebSocket reference for sending manual commands
    wsRef: WebSocket | null;
    setWsRef: (ws: WebSocket | null) => void;

    // Legacy compatibility
    batterySOC: number;
    gridImport: number;
    totalBill: number;
    weather: string;
    n_home: number;

    // Demo Mode
    isDemoMode: boolean;
    currentScenario: 'ideal' | 'erratic' | 'heatwave';
    setDemoMode: (enabled: boolean, scenario?: string) => Promise<void>;

    // Demo Report
    reportData: FinalReportData | null;
    showReport: boolean;
    setShowReport: (show: boolean) => void;
    setReportData: (data: FinalReportData) => void;

    // Simulation Control
    isPaused: boolean;
    pauseSimulation: () => void;
    resumeSimulation: () => void;
    resetAndRestart: () => Promise<void>;
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

    resetSimulation: () => set({
        simData: null,
        history: [],
        batterySOC: 0.5,
        gridImport: 0,
        totalBill: 0,
        roomEnergy: { living: 0, master: 0, bed2: 0, kitchen: 0, toilet: 0 },
    }),

    // Manual Override System
    manualOverride: {},
    manualDeviceState: {},

    toggleManualMode: (deviceId) => set((state) => {
        const newOverride = { ...state.manualOverride };
        const newManualState = { ...state.manualDeviceState };

        if (newOverride[deviceId]) {
            // Switching back to Auto
            delete newOverride[deviceId];
            delete newManualState[deviceId];
        } else {
            // Switching to Manual - keep current state
            newOverride[deviceId] = true;
            newManualState[deviceId] = state.devices[deviceId]?.isOn || false;
        }

        return { manualOverride: newOverride, manualDeviceState: newManualState };
    }),

    toggleDevice: (deviceId) => set((state) => {
        if (!state.manualOverride[deviceId]) return {}; // Only work in manual mode

        const newManualState = { ...state.manualDeviceState };
        newManualState[deviceId] = !newManualState[deviceId];

        const newDevices = { ...state.devices };
        if (newDevices[deviceId]) {
            newDevices[deviceId] = {
                ...newDevices[deviceId],
                isOn: newManualState[deviceId],
                currentPower: newManualState[deviceId] ? newDevices[deviceId].basePower : 0
            };
        }

        // Send to backend via WebSocket
        if (state.wsRef && state.wsRef.readyState === WebSocket.OPEN) {
            state.wsRef.send(JSON.stringify({
                type: 'manual_control',
                device: deviceId,
                state: newManualState[deviceId]
            }));
        }

        return { manualDeviceState: newManualState, devices: newDevices };
    }),

    // Room Energy Tracking
    roomEnergy: { living: 0, master: 0, bed2: 0, kitchen: 0, toilet: 0 },

    // WebSocket Reference
    wsRef: null,
    setWsRef: (ws) => set({ wsRef: ws }),

    updateSimData: (data) => set((state) => {
        const newHistory = [...state.history, data].slice(-24);
        const agentData = data[state.currentViewMode];
        const actions = agentData?.actions;

        // Update device states based on room-specific agent actions
        // BUT respect manual override - don't update devices that are in manual mode
        const newDevices = { ...state.devices };
        const manualOverride = state.manualOverride;
        const manualDeviceState = state.manualDeviceState;

        // Helper to update device only if not in manual mode
        const updateDevice = (deviceId: string, actionValue: number) => {
            if (manualOverride[deviceId]) {
                // In manual mode - use manual state
                newDevices[deviceId].isOn = manualDeviceState[deviceId] || false;
                newDevices[deviceId].currentPower = manualDeviceState[deviceId] ? newDevices[deviceId].basePower : 0;
            } else {
                // In auto mode - use AI decision
                newDevices[deviceId].isOn = actionValue === 1;
                newDevices[deviceId].currentPower = actionValue === 1 ? newDevices[deviceId].basePower : 0;
            }
        };

        if (actions) {
            // Update ACs
            updateDevice('ac_living', actions.ac_living);
            updateDevice('ac_master', actions.ac_master);
            updateDevice('ac_bed2', actions.ac_bed2);

            // Update Lights
            updateDevice('light_living', actions.light_living);
            updateDevice('light_master', actions.light_master);
            updateDevice('light_bed2', actions.light_bed2);
            updateDevice('light_kitchen', actions.light_kitchen);
            updateDevice('light_toilet', actions.light_toilet);

            // Update other devices
            updateDevice('washer', actions.wm);
            updateDevice('dishwasher', actions.dw);
            updateDevice('charger', actions.ev);
        }

        // Calculate Room Energy for 3D Heatmap (in kW)
        const roomEnergy: Record<string, number> = {
            living: (newDevices.ac_living.currentPower + newDevices.light_living.currentPower + newDevices.tv.currentPower) / 1000,
            master: (newDevices.ac_master.currentPower + newDevices.light_master.currentPower) / 1000,
            bed2: (newDevices.ac_bed2.currentPower + newDevices.light_bed2.currentPower) / 1000,
            kitchen: (newDevices.light_kitchen.currentPower + newDevices.fridge.currentPower + newDevices.dishwasher.currentPower) / 1000,
            toilet: newDevices.light_toilet.currentPower / 1000,
        };

        // Determine day/night from timestamp
        const hour = parseInt(data.timestamp?.split(':')[0] || '12');
        const isNight = hour < 6 || hour >= 18;

        return {
            simData: data,
            history: newHistory,
            devices: newDevices,
            isNight,
            roomEnergy,
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

    // Demo Mode
    isDemoMode: false,
    currentScenario: 'ideal',
    setDemoMode: async (enabled, scenario) => {
        const newScenario = scenario || get().currentScenario;
        set({ isDemoMode: enabled, currentScenario: newScenario as any });

        try {
            const response = await fetch('http://localhost:8012/set_mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ demo_mode: enabled, scenario: newScenario })
            });
            // Reset local state when toggling demo mode
            if (enabled) {
                get().resetSimulation();
            }
        } catch (e) {
            console.error('Failed to set demo mode:', e);
        }
    },

    // Demo Report
    reportData: null,
    showReport: false,
    setShowReport: (show) => set({ showReport: show }),
    setReportData: (data) => set({ reportData: data, showReport: true }),  // Auto-show when data received

    // Simulation Control
    isPaused: false,
    pauseSimulation: () => {
        const { wsRef } = get();
        if (wsRef && wsRef.readyState === WebSocket.OPEN) {
            wsRef.send(JSON.stringify({ type: 'pause' }));
        }
        set({ isPaused: true });
    },
    resumeSimulation: () => {
        const { wsRef } = get();
        if (wsRef && wsRef.readyState === WebSocket.OPEN) {
            wsRef.send(JSON.stringify({ type: 'resume' }));
        }
        set({ isPaused: false });
    },
    resetAndRestart: async () => {
        const { wsRef, isDemoMode, currentScenario } = get();
        // Reset local state
        get().resetSimulation();
        set({ isPaused: false });

        // Tell backend to reset
        if (wsRef && wsRef.readyState === WebSocket.OPEN) {
            wsRef.send(JSON.stringify({ type: 'reset' }));
        }

        // Re-trigger demo mode if active
        if (isDemoMode) {
            try {
                await fetch('http://localhost:8012/set_mode', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ demo_mode: true, scenario: currentScenario })
                });
            } catch (e) {
                console.error('Failed to restart demo:', e);
            }
        }
    },
}));
