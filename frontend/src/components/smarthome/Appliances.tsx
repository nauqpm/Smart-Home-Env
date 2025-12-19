import React, { useRef } from 'react';
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

// ==========================================
// SMART TV - Living Room
// Position: x:3, z:3.5 (back wall)
// ==========================================
function SmartTV({ position }: { position: [number, number, number] }) {
    const { devices, toggleDevice } = useStore();
    const isOn = devices.tv?.isOn;

    return (
        <group position={position} onClick={(e) => { e.stopPropagation(); toggleDevice('tv'); }}>
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
            {/* Stand LED when ON */}
            {isOn && (
                <pointLight position={[0, 0, 0.2]} intensity={0.5} distance={3} color={0x4aa3df} />
            )}
        </group>
    );
}

// ==========================================
// SMART FRIDGE - Kitchen
// Position: x:3.5, z:-2.5
// ==========================================
function SmartFridge({ position, rotation = [0, 0, 0] }: { position: [number, number, number], rotation?: [number, number, number] }) {
    const { devices, toggleDevice } = useStore();
    const isOn = devices.fridge?.isOn;

    return (
        <group position={position} rotation={rotation} onClick={(e) => { e.stopPropagation(); toggleDevice('fridge'); }}>
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
// AIR CONDITIONER - Living Room / Bedroom
// Position: Wall mounted
// ==========================================
function AirConditioner({ position, rotation = [0, 0, 0], id }: { position: [number, number, number], rotation?: [number, number, number], id: string }) {
    const { devices, toggleDevice } = useStore();
    const isOn = devices[id]?.isOn;

    return (
        <group position={position} rotation={rotation} onClick={(e) => { e.stopPropagation(); toggleDevice(id); }}>
            <mesh castShadow receiveShadow>
                <boxGeometry args={[1.0, 0.35, 0.25]} />
                <primitive object={plasticWhite} attach="material" />
            </mesh>
            {/* Vents */}
            <mesh position={[0, -0.08, 0.14]}>
                <boxGeometry args={[0.9, 0.08, 0.02]} />
                <primitive object={plasticBlack} attach="material" />
            </mesh>
            {/* LED Status Light */}
            <mesh position={[0.4, 0.08, 0.14]}>
                <sphereGeometry args={[0.02, 16, 16]} />
                <primitive object={isOn ? ledOn : ledOff} attach="material" />
            </mesh>
            {/* Cold air glow when ON */}
            {isOn && (
                <pointLight position={[0, -0.3, 0.2]} intensity={0.3} distance={2} color={0x88ccff} />
            )}
        </group>
    );
}

// ==========================================
// WASHING MACHINE - Loggia
// Position: x:-3.25, z:-3.4
// ==========================================
function WashingMachine({ position }: { position: [number, number, number] }) {
    const { devices, toggleDevice } = useStore();
    const isOn = devices.washer?.isOn;
    const drumRef = useRef<THREE.Group>(null);

    useFrame((state, delta) => {
        if (isOn && drumRef.current) {
            drumRef.current.rotation.z -= delta * 5;
        }
    });

    return (
        <group position={position} onClick={(e) => { e.stopPropagation(); toggleDevice('washer'); }}>
            {/* Body */}
            <mesh castShadow receiveShadow>
                <boxGeometry args={[0.6, 0.75, 0.6]} />
                <primitive object={plasticWhite} attach="material" />
            </mesh>
            {/* Control Panel */}
            <mesh position={[0, 0.3, 0.31]}>
                <planeGeometry args={[0.5, 0.08]} />
                <primitive object={plasticBlack} attach="material" />
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
                {/* Glass tint */}
                <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, 0.01]}>
                    <circleGeometry args={[0.18, 32]} />
                    <meshStandardMaterial color={0x88ccff} transparent opacity={0.3} />
                </mesh>
            </group>
        </group>
    );
}

// ==========================================
// SOLAR INVERTER - Loggia (wall mounted)
// Position: x:-3.85, z:-3.4
// ==========================================
function SolarInverter({ position, rotation = [0, 0, 0] }: { position: [number, number, number], rotation?: [number, number, number] }) {
    // Always on for simulation purposes
    const isOn = true;

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
                <primitive object={isOn ? screenOnTv : screenOff} attach="material" />
            </mesh>
            {/* Status LEDs */}
            <mesh position={[-0.08, -0.15, 0.08]}>
                <sphereGeometry args={[0.015]} />
                <primitive object={isOn ? ledOn : ledOff} attach="material" />
            </mesh>
            <mesh position={[0.08, -0.15, 0.08]}>
                <sphereGeometry args={[0.015]} />
                <primitive object={ledOn} attach="material" />
            </mesh>
        </group>
    );
}

// ==========================================
// FLOOR LAMP - Living Room
// ==========================================
function FloorLamp({ position }: { position: [number, number, number] }) {
    const { devices, toggleDevice } = useStore();
    const isOn = devices.lamp?.isOn;

    return (
        <group position={position} onClick={(e) => { e.stopPropagation(); toggleDevice('lamp'); }}>
            <mesh position={[0, 0.04, 0]} castShadow>
                <cylinderGeometry args={[0.12, 0.12, 0.08, 32]} />
                <primitive object={metalSilver} attach="material" />
            </mesh>
            <mesh position={[0, 0.65, 0]} castShadow>
                <cylinderGeometry args={[0.015, 0.015, 1.3, 32]} />
                <primitive object={metalSilver} attach="material" />
            </mesh>
            <mesh position={[0, 1.2, 0]} castShadow>
                <cylinderGeometry args={[0.12, 0.25, 0.35, 32, 1, true]} />
                <primitive object={lampShade} attach="material" />
            </mesh>
            <mesh position={[0, 1.15, 0]}>
                <sphereGeometry args={[0.04, 16, 16]} />
                <meshBasicMaterial color={isOn ? 0xffaa00 : 0xaa6600} />
            </mesh>
            {isOn && <pointLight position={[0, 1.15, 0]} intensity={0.8} distance={4} color={0xffaa00} />}
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
            {/* SmartTV - centered on back wall (north) */}
            <SmartTV position={[2, 1.15, 3.55]} />

            {/* AC Unit - Living room, mounted on east wall (x=4), rotated to face room */}
            <AirConditioner position={[3.8, 2.1, 2]} rotation={[0, -Math.PI / 2, 0]} id="ac_living" />

            {/* Floor Lamp - near TV wall corner */}
            <FloorLamp position={[0.3, 0, 3.3]} />

            {/* ============================================ */}
            {/* ZONE A: KITCHEN */}
            {/* ============================================ */}
            {/* SmartFridge - kitchen corner, moved to west side to avoid clipping */}
            <SmartFridge position={[0.6, 0.9, -3.3]} rotation={[0, 0, 0]} />

            {/* ============================================ */}
            {/* UTILITY CLOSET (in Small Bedroom corner) */}
            {/* Battery + Solar Inverter storage */}
            {/* ============================================ */}
            {/* Solar Inverter - inside utility closet walls */}
            <SolarInverter position={[-2.15, 1.3, -3.4]} rotation={[0, Math.PI, 0]} />

            {/* Battery3D - next to inverter in utility closet */}
            <Battery3D position={[-2.15, 0.45, -3.55]} rotation={[0, Math.PI, 0]} />

            {/* ============================================ */}
            {/* ZONE B: MASTER BEDROOM */}
            {/* ============================================ */}
            {/* AC Unit - Master bedroom, mounted on west wall */}
            <AirConditioner position={[-3.8, 2.1, 2.5]} rotation={[0, Math.PI / 2, 0]} id="ac_master" />

            {/* ============================================ */}
            {/* ZONE B: SMALL BEDROOM */}
            {/* ============================================ */}
            {/* AC Unit - Small bedroom, on west wall near utility */}
            <AirConditioner position={[-2.4, 2.1, -1.5]} rotation={[0, Math.PI / 2, 0]} id="ac_small" />
        </group>
    );
}
