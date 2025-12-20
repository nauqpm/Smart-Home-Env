import React, { useRef, useMemo, useEffect } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { useStore } from '../../stores/useStore';
import Battery3D from './Battery3D';

// ==========================================
// MATERIALS
// ==========================================
const plasticWhite = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4, metalness: 0.1 });
const plasticBlack = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.4, metalness: 0.1 });
const metalSilver = new THREE.MeshStandardMaterial({ color: 0xaaaaaa, roughness: 0.3, metalness: 0.6 });
const screenOff = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.2 });
const screenOnTv = new THREE.MeshStandardMaterial({ color: 0x4aa3df, emissive: 0x4aa3df, emissiveIntensity: 2.0 });
const lampShade = new THREE.MeshStandardMaterial({ color: 0xfff8e7, roughness: 0.8, side: THREE.DoubleSide, transparent: true, opacity: 0.9 });

// LED Materials
const ledOn = new THREE.MeshStandardMaterial({ color: 0x00ff00, emissive: 0x00ff00, emissiveIntensity: 2.0 });
const ledOff = new THREE.MeshStandardMaterial({ color: 0x550000 });
const panelGlow = new THREE.MeshStandardMaterial({ color: 0xccffcc, emissive: 0xccffcc, emissiveIntensity: 1.0 });

// AC Active Materials (cold blue glow)
const acActiveGlow = new THREE.MeshStandardMaterial({
    color: 0x88ccff,
    emissive: 0x88ccff,
    emissiveIntensity: 1.5,
    transparent: true,
    opacity: 0.8
});

// ==========================================
// HELPER: Get active agent data from store
// ==========================================
function useActiveAgentData() {
    const { simData, currentViewMode, devices } = useStore();
    const activeAgentData = simData ? simData[currentViewMode] : null;
    const actions = activeAgentData?.actions || {
        ac_living: 0, ac_master: 0, ac_bed2: 0,
        light_living: 0, light_master: 0, light_bed2: 0, light_kitchen: 0, light_toilet: 0,
        wm: 0, ev: 0, battery: 'idle'
    };
    const soc = activeAgentData?.soc || 50;

    return { activeAgentData, actions, soc, devices, simData };
}

// ==========================================
// SMART TV - Living Room
// ==========================================
function SmartTV({ position }: { position: [number, number, number] }) {
    const { devices } = useActiveAgentData();
    const isOn = devices.tv?.isOn;

    return (
        <group position={position}>
            {/* Screen */}
            <mesh castShadow receiveShadow>
                <boxGeometry args={[1.4, 0.8, 0.08]} />
                <primitive object={isOn ? screenOnTv : screenOff} attach="material" />
            </mesh>
            {/* Bezel/Back */}
            <mesh position={[0, 0, -0.05]} castShadow>
                <boxGeometry args={[1.45, 0.85, 0.04]} />
                <primitive object={plasticBlack} attach="material" />
            </mesh>
            {/* LED glow when ON */}
            {isOn && (
                <pointLight position={[0, 0, 0.2]} intensity={0.5} distance={3} color={0x4aa3df} />
            )}
        </group>
    );
}

// ==========================================
// SMART FRIDGE - Kitchen
// ==========================================
function SmartFridge({ position, rotation = [0, 0, 0] }: { position: [number, number, number], rotation?: [number, number, number] }) {
    const { devices } = useActiveAgentData();
    const isOn = devices.fridge?.isOn;

    return (
        <group position={position} rotation={rotation}>
            {/* Main Body */}
            <mesh castShadow receiveShadow>
                <boxGeometry args={[0.7, 1.8, 0.65]} />
                <primitive object={metalSilver} attach="material" />
            </mesh>
            {/* Divider Line */}
            <mesh position={[0, 0.2, 0.33]}>
                <boxGeometry args={[0.68, 0.02, 0.02]} />
                <primitive object={plasticBlack} attach="material" />
            </mesh>
            {/* Handle */}
            <mesh position={[-0.25, 0.35, 0.35]}>
                <boxGeometry args={[0.04, 0.7, 0.04]} />
                <primitive object={plasticBlack} attach="material" />
            </mesh>
            {/* Smart Panel */}
            <mesh position={[0.18, 0.35, 0.33]}>
                <planeGeometry args={[0.18, 0.25]} />
                <primitive object={isOn ? panelGlow : screenOff} attach="material" />
            </mesh>
        </group>
    );
}

// ==========================================
// AIR CONDITIONER with Fan Animation
// ==========================================
function AirConditioner({ position, rotation = [0, 0, 0], id }: { position: [number, number, number], rotation?: [number, number, number], id: string }) {
    const { devices } = useActiveAgentData();
    // Read from devices state which is already mapped from room-specific actions
    const isOn = devices[id]?.isOn ?? false;

    const ventRef = useRef<THREE.Mesh>(null);
    const fanRef = useRef<THREE.Group>(null);

    // Vent swing animation
    useFrame((state, delta) => {
        if (ventRef.current && isOn) {
            // Subtle swing motion for vent
            ventRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
        }
        if (fanRef.current && isOn) {
            // Internal fan spin
            fanRef.current.rotation.z += delta * 8;
        }
    });

    return (
        <group position={position} rotation={rotation}>
            {/* Main body */}
            <mesh castShadow receiveShadow>
                <boxGeometry args={[1.0, 0.35, 0.25]} />
                <primitive object={plasticWhite} attach="material" />
            </mesh>

            {/* Animated Vent */}
            <mesh ref={ventRef} position={[0, -0.08, 0.14]}>
                <boxGeometry args={[0.9, 0.08, 0.02]} />
                <primitive object={isOn ? acActiveGlow : plasticBlack} attach="material" />
            </mesh>

            {/* Internal Fan (visible through vent) */}
            <group ref={fanRef} position={[0, -0.02, 0.1]}>
                {[0, 1, 2, 3].map((i) => (
                    <mesh key={i} rotation={[0, 0, (Math.PI / 2) * i]}>
                        <boxGeometry args={[0.3, 0.03, 0.01]} />
                        <meshStandardMaterial color={0x666666} transparent opacity={isOn ? 0.6 : 0.3} />
                    </mesh>
                ))}
            </group>

            {/* LED Status Light */}
            <mesh position={[0.4, 0.08, 0.14]}>
                <sphereGeometry args={[0.02, 16, 16]} />
                <primitive object={isOn ? ledOn : ledOff} attach="material" />
            </mesh>

            {/* Cold air glow when ON */}
            {isOn && (
                <>
                    <pointLight position={[0, -0.3, 0.2]} intensity={0.4} distance={2.5} color={0x88ccff} />
                    {/* Cold air particles effect placeholder */}
                    <mesh position={[0, -0.4, 0.1]}>
                        <planeGeometry args={[0.6, 0.3]} />
                        <meshBasicMaterial color={0xaaddff} transparent opacity={0.2} />
                    </mesh>
                </>
            )}
        </group>
    );
}

// ==========================================
// WASHING MACHINE with Drum Animation
// ==========================================
function WashingMachine({ position }: { position: [number, number, number] }) {
    const { actions, devices } = useActiveAgentData();
    // Use both agent action and store device state
    const isOn = devices.washer?.isOn || actions.wm === 1;

    const drumRef = useRef<THREE.Group>(null);
    const waterGlowRef = useRef<THREE.Mesh>(null);

    useFrame((state, delta) => {
        if (drumRef.current && isOn) {
            // Fast drum rotation with wobble
            drumRef.current.rotation.z -= delta * 8;
            // Subtle vibration
            drumRef.current.position.x = Math.sin(state.clock.elapsedTime * 20) * 0.003;
        }
        if (waterGlowRef.current && isOn) {
            // Water swirl effect
            const pulse = Math.sin(state.clock.elapsedTime * 3) * 0.3 + 0.5;
            (waterGlowRef.current.material as THREE.MeshStandardMaterial).opacity = pulse;
        }
    });

    return (
        <group position={position}>
            {/* Body */}
            <mesh castShadow receiveShadow>
                <boxGeometry args={[0.6, 0.75, 0.6]} />
                <primitive object={plasticWhite} attach="material" />
            </mesh>

            {/* Control Panel with LED */}
            <mesh position={[0, 0.3, 0.31]}>
                <planeGeometry args={[0.5, 0.08]} />
                <primitive object={plasticBlack} attach="material" />
            </mesh>
            <mesh position={[0.2, 0.3, 0.315]}>
                <sphereGeometry args={[0.015, 8, 8]} />
                <meshStandardMaterial
                    color={isOn ? 0x00ff00 : 0x333333}
                    emissive={isOn ? 0x00ff00 : 0x000000}
                    emissiveIntensity={isOn ? 2 : 0}
                />
            </mesh>

            {/* Drum Window Rim */}
            <mesh position={[0, 0, 0.31]} rotation={[Math.PI / 2, 0, 0]}>
                <cylinderGeometry args={[0.22, 0.22, 0.04, 32]} />
                <primitive object={metalSilver} attach="material" />
            </mesh>

            {/* Spinning Drum */}
            <group position={[0, 0, 0.32]} ref={drumRef}>
                <mesh rotation={[Math.PI / 2, 0, 0]}>
                    <cylinderGeometry args={[0.18, 0.18, 0.02, 16]} />
                    <meshStandardMaterial color={0x333333} />
                </mesh>
                {/* Drum pattern */}
                {[0, 1, 2, 3, 4, 5].map((i) => (
                    <mesh key={i} rotation={[Math.PI / 2, 0, (Math.PI / 3) * i]} position={[0, 0, 0.005]}>
                        <boxGeometry args={[0.02, 0.16, 0.005]} />
                        <meshStandardMaterial color={0x555555} />
                    </mesh>
                ))}
                {/* Water/Glass effect */}
                <mesh ref={waterGlowRef} rotation={[Math.PI / 2, 0, 0]} position={[0, 0, 0.01]}>
                    <circleGeometry args={[0.17, 32]} />
                    <meshStandardMaterial
                        color={isOn ? 0x66ccff : 0x88ccff}
                        transparent
                        opacity={isOn ? 0.6 : 0.3}
                        emissive={isOn ? 0x4488cc : 0x000000}
                        emissiveIntensity={isOn ? 0.5 : 0}
                    />
                </mesh>
            </group>

            {/* Vibration glow when running */}
            {isOn && (
                <pointLight position={[0, 0, 0.4]} intensity={0.3} distance={1} color={0x88ccff} />
            )}
        </group>
    );
}

// ==========================================
// SOLAR INVERTER
// ==========================================
function SolarInverter({ position, rotation = [0, 0, 0] }: { position: [number, number, number], rotation?: [number, number, number] }) {
    const { simData } = useActiveAgentData();
    const pvPower = simData?.env?.pv || 0;
    const isGenerating = pvPower > 0.1;

    return (
        <group position={position} rotation={rotation}>
            {/* Main Box */}
            <mesh castShadow>
                <boxGeometry args={[0.4, 0.6, 0.15]} />
                <primitive object={plasticWhite} attach="material" />
            </mesh>
            {/* Status Display Panel */}
            <mesh position={[0, 0.08, 0.08]}>
                <planeGeometry args={[0.25, 0.12]} />
                <meshStandardMaterial
                    color={isGenerating ? 0x00ff00 : 0x333333}
                    emissive={isGenerating ? 0x00ff00 : 0x000000}
                    emissiveIntensity={isGenerating ? 1.0 : 0}
                />
            </mesh>
            {/* Status LEDs */}
            <mesh position={[-0.08, -0.15, 0.08]}>
                <sphereGeometry args={[0.015]} />
                <primitive object={isGenerating ? ledOn : ledOff} attach="material" />
            </mesh>
            <mesh position={[0.08, -0.15, 0.08]}>
                <sphereGeometry args={[0.015]} />
                <primitive object={ledOn} attach="material" />
            </mesh>
        </group>
    );
}

// ==========================================
// FLOOR LAMP with Glow Animation
// ==========================================
function FloorLamp({ position }: { position: [number, number, number] }) {
    const { devices } = useActiveAgentData();
    const isOn = devices.lamp?.isOn;
    const bulbRef = useRef<THREE.Mesh>(null);

    // Warm glow pulse when on
    useFrame((state) => {
        if (bulbRef.current && isOn) {
            const pulse = Math.sin(state.clock.elapsedTime * 2) * 0.15 + 0.85;
            (bulbRef.current.material as THREE.MeshBasicMaterial).color.setHSL(0.1, 0.8, pulse * 0.5);
        }
    });

    return (
        <group position={position}>
            {/* Base */}
            <mesh position={[0, 0.04, 0]} castShadow>
                <cylinderGeometry args={[0.12, 0.12, 0.08, 32]} />
                <primitive object={metalSilver} attach="material" />
            </mesh>
            {/* Pole */}
            <mesh position={[0, 0.65, 0]} castShadow>
                <cylinderGeometry args={[0.015, 0.015, 1.3, 32]} />
                <primitive object={metalSilver} attach="material" />
            </mesh>
            {/* Shade */}
            <mesh position={[0, 1.2, 0]} castShadow>
                <cylinderGeometry args={[0.12, 0.25, 0.35, 32, 1, true]} />
                <meshStandardMaterial
                    color={0xfff8e7}
                    roughness={0.8}
                    side={THREE.DoubleSide}
                    transparent
                    opacity={isOn ? 0.95 : 0.7}
                    emissive={isOn ? 0xffcc00 : 0x000000}
                    emissiveIntensity={isOn ? 0.3 : 0}
                />
            </mesh>
            {/* Bulb */}
            <mesh ref={bulbRef} position={[0, 1.15, 0]}>
                <sphereGeometry args={[0.04, 16, 16]} />
                <meshBasicMaterial color={isOn ? 0xffaa00 : 0xaa6600} />
            </mesh>
            {/* Light emission when ON */}
            {isOn && (
                <>
                    <pointLight position={[0, 1.15, 0]} intensity={0.8} distance={4} color={0xffaa00} />
                    <pointLight position={[0, 1.0, 0]} intensity={0.4} distance={2} color={0xffcc66} />
                </>
            )}
        </group>
    );
}

// ==========================================
// EV CHARGER
// ==========================================
function EVCharger({ position }: { position: [number, number, number] }) {
    const { actions, devices } = useActiveAgentData();
    const isOn = devices.charger?.isOn || actions.ev === 1;
    const cableRef = useRef<THREE.Mesh>(null);

    // Cable pulse animation when charging
    useFrame((state) => {
        if (cableRef.current && isOn) {
            const pulse = Math.sin(state.clock.elapsedTime * 4) * 0.3 + 0.7;
            (cableRef.current.material as THREE.MeshStandardMaterial).emissiveIntensity = pulse * 2;
        }
    });

    return (
        <group position={position}>
            {/* Charger Box */}
            <mesh castShadow>
                <boxGeometry args={[0.3, 0.5, 0.15]} />
                <meshStandardMaterial color={0x444444} roughness={0.5} />
            </mesh>

            {/* Status LED */}
            <mesh position={[0, 0.15, 0.08]}>
                <sphereGeometry args={[0.02, 16, 16]} />
                <meshStandardMaterial
                    color={isOn ? 0x00ff00 : 0xff0000}
                    emissive={isOn ? 0x00ff00 : 0x550000}
                    emissiveIntensity={isOn ? 2 : 0.5}
                />
            </mesh>

            {/* Cable */}
            <mesh ref={cableRef} position={[0.4, -0.1, 0]} rotation={[0, 0, Math.PI / 4]}>
                <cylinderGeometry args={[0.02, 0.02, 0.8, 8]} />
                <meshStandardMaterial
                    color={isOn ? 0x00ff00 : 0x333333}
                    emissive={isOn ? 0x00ff00 : 0x000000}
                    emissiveIntensity={isOn ? 1 : 0}
                />
            </mesh>

            {/* Charging indicator */}
            {isOn && (
                <pointLight position={[0, 0, 0.2]} intensity={0.5} distance={1.5} color={0x00ff00} />
            )}
        </group>
    );
}

// ==========================================
// MAIN EXPORT
// ==========================================
export default function Appliances() {
    return (
        <group>
            {/* ============================================ */}
            {/* ZONE A: LIVING ROOM */}
            {/* ============================================ */}
            <SmartTV position={[2, 1.15, 3.55]} />
            <AirConditioner position={[3.8, 2.1, 2]} rotation={[0, -Math.PI / 2, 0]} id="ac_living" />
            <FloorLamp position={[0.3, 0, 3.3]} />

            {/* ============================================ */}
            {/* ZONE A: KITCHEN */}
            {/* ============================================ */}
            <SmartFridge position={[0.6, 0.9, -3.3]} rotation={[0, 0, 0]} />

            {/* ============================================ */}
            {/* UTILITY CLOSET */}
            {/* ============================================ */}
            <SolarInverter position={[-2.15, 1.3, -3.4]} rotation={[0, Math.PI, 0]} />
            <Battery3D position={[-2.15, 0.45, -3.55]} rotation={[0, Math.PI, 0]} />

            {/* Washing Machine - in utility area */}
            <WashingMachine position={[-2.85, 0.375, -3.4]} />

            {/* EV Charger - Garage area */}
            <EVCharger position={[3.5, 0.3, -3.5]} />

            {/* ============================================ */}
            {/* ZONE B: MASTER BEDROOM */}
            {/* ============================================ */}
            <AirConditioner position={[-3.8, 2.1, 2.5]} rotation={[0, Math.PI / 2, 0]} id="ac_master" />

            {/* ============================================ */}
            {/* ZONE B: 2ND BEDROOM */}
            {/* ============================================ */}
            <AirConditioner position={[-2.4, 2.1, -1.5]} rotation={[0, Math.PI / 2, 0]} id="ac_bed2" />
        </group>
    );
}
