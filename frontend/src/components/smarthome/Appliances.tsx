import React, { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { useStore } from '../../stores/useStore';

// --- MATERIALS HELPER ---
const plasticWhite = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4, metalness: 0.1 });
const plasticBlack = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.4, metalness: 0.1 });
const metalSilver = new THREE.MeshStandardMaterial({ color: 0xaaaaaa, roughness: 0.3, metalness: 0.6 });
const screenOff = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.2 });
const screenOnTv = new THREE.MeshStandardMaterial({ color: 0x4aa3df, emissive: 0x4aa3df, emissiveIntensity: 2.0 });
const screenOnLaptop = new THREE.MeshStandardMaterial({ color: 0x88ccff, emissive: 0x88ccff, emissiveIntensity: 1.5 });
const lampShade = new THREE.MeshStandardMaterial({ color: 0xfff8e7, roughness: 0.8, side: THREE.DoubleSide, transparent: true, opacity: 0.9 });

// LED Materials
const ledOn = new THREE.MeshStandardMaterial({ color: 0x00ff00, emissive: 0x00ff00, emissiveIntensity: 2.0 });
const ledOff = new THREE.MeshStandardMaterial({ color: 0x550000 });
const ledCharging = new THREE.MeshStandardMaterial({ color: 0x00ff00, emissive: 0x00ff00, emissiveIntensity: 2.0 });
const ledStandby = new THREE.MeshStandardMaterial({ color: 0xaaaa00, emissive: 0xffff00, emissiveIntensity: 0.5 });
const panelGlow = new THREE.MeshStandardMaterial({ color: 0xccffcc, emissive: 0xccffcc, emissiveIntensity: 1.0 });

function SmartTV({ position, rotation }: { position: [number, number, number], rotation: [number, number, number] }) {
    const { devices, toggleDevice } = useStore();
    const isOn = devices.tv?.isOn;

    return (
        <group position={position} rotation={rotation} onClick={(e) => { e.stopPropagation(); toggleDevice('tv'); }}>
            {/* Screen */}
            <mesh position={[0, 0, 0]} castShadow receiveShadow>
                <boxGeometry args={[1.6, 0.9, 0.1]} />
                <primitive object={isOn ? screenOnTv : screenOff} attach="material" />
            </mesh>
            {/* Bezel/Back */}
            <mesh position={[0, 0, -0.06]} castShadow>
                <boxGeometry args={[1.65, 0.95, 0.05]} />
                <primitive object={plasticBlack} attach="material" />
            </mesh>
        </group>
    );
}

function SmartFridge({ position }: { position: [number, number, number] }) {
    const { devices, toggleDevice } = useStore();
    const isOn = devices.fridge?.isOn;

    return (
        <group position={position} onClick={(e) => { e.stopPropagation(); toggleDevice('fridge'); }}>
            {/* Main Body */}
            <mesh castShadow receiveShadow>
                <boxGeometry args={[0.8, 2.0, 0.7]} />
                <primitive object={metalSilver} attach="material" />
            </mesh>
            {/* Divider Line */}
            <mesh position={[0, 0.2, 0.36]}>
                <boxGeometry args={[0.78, 0.02, 0.02]} />
                <primitive object={plasticBlack} attach="material" />
            </mesh>
            {/* Handle Vertical */}
            <mesh position={[-0.3, 0.4, 0.38]}>
                <boxGeometry args={[0.05, 0.8, 0.05]} />
                <primitive object={plasticBlack} attach="material" />
            </mesh>
            {/* Smart Panel */}
            <mesh position={[0.2, 0.4, 0.36]}>
                <planeGeometry args={[0.2, 0.3]} />
                <primitive object={isOn ? panelGlow : screenOff} attach="material" />
            </mesh>
        </group>
    )
}

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
                <boxGeometry args={[0.7, 0.85, 0.7]} />
                <primitive object={plasticWhite} attach="material" />
            </mesh>
            {/* Control Panel Area */}
            <mesh position={[0, 0.35, 0.36]}>
                <planeGeometry args={[0.6, 0.1]} />
                <primitive object={plasticBlack} attach="material" />
            </mesh>
            {/* Drum Window Rim */}
            <mesh position={[0, 0, 0.36]} rotation={[Math.PI / 2, 0, 0]}>
                <cylinderGeometry args={[0.28, 0.28, 0.05, 32]} />
                <primitive object={metalSilver} attach="material" />
            </mesh>
            {/* Spinning Drum Internals */}
            <group position={[0, 0, 0.37]} ref={drumRef}>
                <mesh rotation={[Math.PI / 2, 0, 0]}>
                    <cylinderGeometry args={[0.22, 0.22, 0.02, 16]} />
                    <meshStandardMaterial color={0x333333} />
                </mesh>
                {/* Glass tint */}
                <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, 0.01]}>
                    <circleGeometry args={[0.22, 32]} />
                    <meshStandardMaterial color={0x88ccff} transparent opacity={0.3} />
                </mesh>
            </group>
        </group>
    )
}

function AirConditioner({ position, rotation = [0, 0, 0], id }: { position: [number, number, number], rotation?: [number, number, number], id: string }) {
    const { devices, toggleDevice } = useStore();
    const isOn = devices[id]?.isOn;

    return (
        <group position={position} rotation={rotation} onClick={(e) => { e.stopPropagation(); toggleDevice(id); }}>
            <mesh castShadow receiveShadow>
                <boxGeometry args={[1.2, 0.4, 0.3]} />
                <primitive object={plasticWhite} attach="material" />
            </mesh>
            {/* Vents or detail */}
            <mesh position={[0, -0.1, 0.16]}>
                <boxGeometry args={[1.1, 0.1, 0.02]} />
                <primitive object={plasticBlack} attach="material" />
            </mesh>
            {/* LED Status Light */}
            <mesh position={[0.5, 0.1, 0.16]}>
                <sphereGeometry args={[0.02, 16, 16]} />
                <primitive object={isOn ? ledOn : ledOff} attach="material" />
            </mesh>
        </group>
    );
}

function CeilingFan({ position }: { position: [number, number, number] }) {
    const { devices, toggleDevice } = useStore();
    const isOn = devices.fan?.isOn;
    const fanRef = useRef<THREE.Group>(null);

    useFrame((state, delta) => {
        if (isOn && fanRef.current) {
            fanRef.current.rotation.y += delta * 5;
        }
    });

    return (
        <group position={position} onClick={(e) => { e.stopPropagation(); toggleDevice('fan'); }}>
            {/* Hub */}
            <mesh castShadow receiveShadow>
                <cylinderGeometry args={[0.15, 0.15, 0.2, 32]} />
                <primitive object={plasticWhite} attach="material" />
            </mesh>
            {/* Blades Group - Rotates */}
            <group ref={fanRef}>
                {[0, (Math.PI * 2) / 3, (Math.PI * 4) / 3].map((angle, i) => (
                    <mesh key={i} rotation={[0, angle, 0]} position={[Math.sin(angle) * 0.6, 0, Math.cos(angle) * 0.6]} castShadow receiveShadow>
                        <boxGeometry args={[0.15, 0.02, 1.2]} />
                        <primitive object={plasticWhite} attach="material" />
                    </mesh>
                ))}
            </group>
        </group>
    );
}

function EVCharger({ position, rotation = [0, 0, 0] }: { position: [number, number, number], rotation?: [number, number, number] }) {
    const { devices, toggleDevice } = useStore();
    const isOn = devices.charger?.isOn;

    return (
        <group position={position} rotation={rotation} onClick={(e) => { e.stopPropagation(); toggleDevice('charger'); }}>
            <mesh castShadow receiveShadow>
                <boxGeometry args={[0.4, 0.6, 0.15]} />
                <primitive object={plasticWhite} attach="material" />
            </mesh>
            <mesh position={[0, 0.1, 0.08]}>
                <planeGeometry args={[0.2, 0.1]} />
                <primitive object={screenOnTv} attach="material" />
            </mesh>
            {/* Cable/Plug holder */}
            <mesh position={[0, -0.15, 0.08]} rotation={[Math.PI / 2, 0, 0]}>
                <cylinderGeometry args={[0.05, 0.05, 0.1]} />
                <primitive object={plasticBlack} attach="material" />
            </mesh>

            {/* LED Charging Status */}
            <mesh position={[0.15, 0.2, 0.08]}>
                <sphereGeometry args={[0.02, 16, 16]} />
                <primitive object={isOn ? ledCharging : ledStandby} attach="material" />
            </mesh>
        </group>
    )
}

function Laptop({ position, rotation = [0, 0, 0] }: { position: [number, number, number], rotation?: [number, number, number] }) {
    const { devices, toggleDevice } = useStore();
    const isOn = devices.laptop?.isOn;

    return (
        <group position={position} rotation={rotation} onClick={(e) => { e.stopPropagation(); toggleDevice('laptop'); }}>
            {/* Base */}
            <mesh position={[0, 0.01, 0.1]} castShadow>
                <boxGeometry args={[0.4, 0.02, 0.3]} />
                <primitive object={metalSilver} attach="material" />
            </mesh>
            {/* Screen Lid (Opened 100 deg) */}
            <group position={[0, 0.02, -0.05]} rotation={[-1.7, 0, 0]}>
                <mesh position={[0, 0.15, 0]}>
                    <boxGeometry args={[0.4, 0.3, 0.02]} />
                    <primitive object={metalSilver} attach="material" />
                </mesh>
                {/* Display */}
                <mesh position={[0, 0.15, 0.011]}>
                    <planeGeometry args={[0.38, 0.28]} />
                    <primitive object={isOn ? screenOnLaptop : screenOff} attach="material" />
                </mesh>
            </group>
        </group>
    )
}

function FloorLamp({ position }: { position: [number, number, number] }) {
    const { devices, toggleDevice } = useStore();
    const isOn = devices.lamp?.isOn;

    return (
        <group position={position} onClick={(e) => { e.stopPropagation(); toggleDevice('lamp'); }}>
            <group>
                <mesh position={[0, 0.05, 0]} castShadow receiveShadow>
                    <cylinderGeometry args={[0.15, 0.15, 0.1, 32]} />
                    <primitive object={metalSilver} attach="material" />
                </mesh>
                <mesh position={[0, 0.75, 0]} castShadow receiveShadow>
                    <cylinderGeometry args={[0.02, 0.02, 1.5, 32]} />
                    <primitive object={metalSilver} attach="material" />
                </mesh>
                <mesh position={[0, 1.4, 0]} castShadow>
                    <cylinderGeometry args={[0.15, 0.3, 0.4, 32, 1, true]} />
                    <primitive object={lampShade} attach="material" />
                </mesh>
                <mesh position={[0, 1.35, 0]}>
                    <sphereGeometry args={[0.05, 16, 16]} />
                    <meshBasicMaterial color={isOn ? 0xffaa00 : 0xaa6600} />
                </mesh>
                {/* Actual Point Light if On - Optional, expensive */}
                {isOn && <pointLight position={[0, 1.35, 0]} intensity={1} distance={5} color={0xffaa00} />}
            </group>
        </group>
    );
}


function SolarInverter({ position, rotation = [0, 0, 0] }: { position: [number, number, number], rotation?: [number, number, number] }) {
    const { devices, toggleDevice } = useStore();
    const isOn = devices.solar?.isOn;

    return (
        <group position={position} rotation={rotation} onClick={(e) => { e.stopPropagation(); toggleDevice('solar'); }}>
            {/* Main Box */}
            <mesh castShadow>
                <boxGeometry args={[0.5, 0.8, 0.2]} />
                <primitive object={plasticWhite} attach="material" />
            </mesh>
            {/* Status Display Panel */}
            <mesh position={[0, 0.1, 0.11]}>
                <planeGeometry args={[0.3, 0.15]} />
                <primitive object={isOn ? screenOnTv : screenOff} attach="material" />
            </mesh>
            {/* Status LEDs */}
            <mesh position={[-0.1, -0.2, 0.11]}>
                <sphereGeometry args={[0.02]} />
                <primitive object={isOn ? ledOn : ledOff} attach="material" />
            </mesh>
            <mesh position={[0.1, -0.2, 0.11]}>
                <sphereGeometry args={[0.02]} />
                <primitive object={ledCharging} attach="material" />
            </mesh>
            {/* Conduit/Pipe */}
            <mesh position={[0, -0.4, 0]} rotation={[0, 0, 0]}>
                <cylinderGeometry args={[0.02, 0.02, 0.2]} />
                <primitive object={plasticBlack} attach="material" />
            </mesh>
        </group>
    )
}

import Battery3D from './Battery3D';

// ... (Existing Imports)

// ...

export default function Appliances() {
    return (
        <group>
            <SmartTV position={[1.2, 1.25, 0.5]} rotation={[0, -Math.PI / 2, 0]} />
            <SmartFridge position={[-5.3, 1.0, -1.5]} />
            <WashingMachine position={[-6.5, 0.545, 0.5]} />
            {/* Living Room AC */}
            <AirConditioner position={[2, 2.5, -5.7]} id="ac_living" />
            {/* Master Bedroom AC */}
            <AirConditioner position={[7.7, 2.5, 3.5]} rotation={[0, -Math.PI / 2, 0]} id="ac_master" />
            <CeilingFan position={[5, 2.7, 0]} />
            <Laptop position={[7.0, 0.41, -5.0]} rotation={[0, Math.PI, 0]} />
            <EVCharger position={[-4, 1.2, 6.08]} />
            {/* Exterior Solar Inverter - Side Balcony Wall x=8.2, z=0 region */}
            <SolarInverter position={[8.2, 1.5, 2.0]} rotation={[0, -Math.PI / 2, 0]} />
            {/* AI Battery Storage - Near Solar Inverse but distinct */}
            <Battery3D position={[9.2, 0.8, 2.0]} rotation={[0, -Math.PI / 2, 0]} />
            <FloorLamp position={[4.5, 0, -5.5]} />
        </group>
    );
}
