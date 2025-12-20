import React, { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { useStore } from '../../stores/useStore';

// Materials
const caseWhite = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3, metalness: 0.1 });
const caseBlack = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.5 });
const ledGreen = new THREE.MeshStandardMaterial({ color: 0x00ff00, emissive: 0x00ff00, emissiveIntensity: 2 });
const ledYellow = new THREE.MeshStandardMaterial({ color: 0xffff00, emissive: 0xffff00, emissiveIntensity: 2 });
const ledRed = new THREE.MeshStandardMaterial({ color: 0xff0000, emissive: 0xff0000, emissiveIntensity: 2 });
const ledOff = new THREE.MeshStandardMaterial({ color: 0x333333 });

// Flow arrow materials with dynamic intensity
const createArrowMaterial = (color: number) => new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.8
});

// ==========================================
// FLOW ARROWS - Animated charge/discharge indicator
// ==========================================
function FlowArrows({ mode }: { mode: 'charge' | 'discharge' | 'idle' }) {
    const groupRef = useRef<THREE.Group>(null);
    const arrow1Ref = useRef<THREE.Mesh>(null);
    const arrow2Ref = useRef<THREE.Mesh>(null);

    useFrame((state, delta) => {
        if (!groupRef.current || mode === 'idle') return;

        const speed = mode === 'charge' ? 1 : -1;
        groupRef.current.position.y += speed * delta * 0.8;

        // Loop animation
        if (groupRef.current.position.y > 0.6) groupRef.current.position.y = -0.6;
        if (groupRef.current.position.y < -0.6) groupRef.current.position.y = 0.6;

        // Pulse opacity
        const pulse = Math.sin(state.clock.elapsedTime * 4) * 0.3 + 0.7;
        if (arrow1Ref.current) {
            (arrow1Ref.current.material as THREE.MeshBasicMaterial).opacity = pulse;
        }
        if (arrow2Ref.current) {
            (arrow2Ref.current.material as THREE.MeshBasicMaterial).opacity = pulse;
        }
    });

    if (mode === 'idle') return null;

    const arrowColor = mode === 'charge' ? 0x00ff00 : 0xff4444;
    const rotation = mode === 'charge' ? 0 : Math.PI;

    return (
        <group ref={groupRef} position={[0.3, 0, 0.15]}>
            <mesh ref={arrow1Ref} position={[0, 0.25, 0]} rotation={[0, 0, rotation]}>
                <coneGeometry args={[0.06, 0.12, 3]} />
                <meshBasicMaterial color={arrowColor} transparent opacity={0.8} />
            </mesh>
            <mesh ref={arrow2Ref} position={[0, -0.25, 0]} rotation={[0, 0, rotation]}>
                <coneGeometry args={[0.06, 0.12, 3]} />
                <meshBasicMaterial color={arrowColor} transparent opacity={0.8} />
            </mesh>
        </group>
    );
}

// ==========================================
// LOW SOC BLINK EFFECT
// ==========================================
function LowSOCBlinker({ soc }: { soc: number }) {
    const meshRef = useRef<THREE.Mesh>(null);

    useFrame((state) => {
        if (meshRef.current && soc < 20) {
            // Fast blink when critically low
            const blink = Math.sin(state.clock.elapsedTime * 8) > 0;
            (meshRef.current.material as THREE.MeshStandardMaterial).emissiveIntensity = blink ? 3 : 0;
        }
    });

    if (soc >= 20) return null;

    return (
        <mesh ref={meshRef} position={[0, -0.45, 0.12]}>
            <planeGeometry args={[0.6, 0.08]} />
            <meshStandardMaterial
                color={0xff0000}
                emissive={0xff0000}
                emissiveIntensity={2}
                transparent
                opacity={0.9}
            />
        </mesh>
    );
}

// ==========================================
// MAIN BATTERY COMPONENT
// ==========================================
export default function Battery3D({ position, rotation = [0, 0, 0] }: { position: [number, number, number], rotation?: [number, number, number] }) {
    const { simData, currentViewMode, batterySOC } = useStore();

    // Get active agent's data
    const activeAgentData = simData ? simData[currentViewMode] : null;
    const actions = activeAgentData?.actions || { battery: 'idle' };
    const soc = activeAgentData?.soc ?? batterySOC * 100; // Use agent SOC or legacy store SOC

    // Determine battery mode from agent actions
    const mode = useMemo(() => {
        if (!actions.battery) return 'idle';
        return actions.battery as 'charge' | 'discharge' | 'idle';
    }, [actions.battery]);

    // SOC Bar segments (5 levels)
    const bars = [0.2, 0.4, 0.6, 0.8, 1.0];
    const normalizedSOC = soc / 100; // Convert to 0-1 range

    // Determine LED color based on SOC level
    const getLEDMaterial = (threshold: number) => {
        if (normalizedSOC >= threshold - 0.1) {
            if (normalizedSOC < 0.2) return ledRed;
            if (normalizedSOC < 0.4) return ledYellow;
            return ledGreen;
        }
        return ledOff;
    };

    // Main body glow based on mode
    const bodyGlow = useMemo(() => {
        switch (mode) {
            case 'charge':
                return { color: 0xaaffaa, intensity: 0.3 };
            case 'discharge':
                return { color: 0xffaaaa, intensity: 0.3 };
            default:
                return { color: 0xffffff, intensity: 0 };
        }
    }, [mode]);

    return (
        <group position={position} rotation={rotation}>
            {/* Main Body (Tesla Powerwall-style) */}
            <mesh castShadow receiveShadow>
                <boxGeometry args={[0.8, 1.2, 0.2]} />
                <meshStandardMaterial
                    color={0xffffff}
                    roughness={0.3}
                    metalness={0.1}
                    emissive={bodyGlow.color}
                    emissiveIntensity={bodyGlow.intensity}
                />
            </mesh>

            {/* Side trims */}
            <mesh position={[0.4, 0, 0]}>
                <boxGeometry args={[0.05, 1.2, 0.21]} />
                <primitive object={caseBlack} attach="material" />
            </mesh>
            <mesh position={[-0.4, 0, 0]}>
                <boxGeometry args={[0.05, 1.2, 0.21]} />
                <primitive object={caseBlack} attach="material" />
            </mesh>

            {/* Status Label */}
            <mesh position={[-0.1, 0.45, 0.11]}>
                <planeGeometry args={[0.35, 0.1]} />
                <meshStandardMaterial
                    color={mode === 'charge' ? 0x00cc00 : mode === 'discharge' ? 0xcc0000 : 0x666666}
                    emissive={mode !== 'idle' ? (mode === 'charge' ? 0x00cc00 : 0xcc0000) : 0x000000}
                    emissiveIntensity={mode !== 'idle' ? 1 : 0}
                />
            </mesh>

            {/* SOC Percentage Display */}
            <mesh position={[0, 0.3, 0.11]}>
                <planeGeometry args={[0.5, 0.15]} />
                <meshStandardMaterial color={0x111111} />
            </mesh>

            {/* LED Strip (Vertical) */}
            <group position={[0.2, -0.1, 0.11]}>
                {bars.map((limit, i) => (
                    <mesh key={i} position={[0, (i - 2) * 0.18, 0]}>
                        <planeGeometry args={[0.08, 0.12]} />
                        <primitive
                            object={getLEDMaterial(limit)}
                            attach="material"
                        />
                    </mesh>
                ))}
            </group>

            {/* Mode indicator light */}
            <mesh position={[-0.25, -0.45, 0.11]}>
                <sphereGeometry args={[0.04, 16, 16]} />
                <meshStandardMaterial
                    color={mode === 'charge' ? 0x00ff00 : mode === 'discharge' ? 0xff4444 : 0x888888}
                    emissive={mode === 'charge' ? 0x00ff00 : mode === 'discharge' ? 0xff0000 : 0x000000}
                    emissiveIntensity={mode !== 'idle' ? 2 : 0}
                />
            </mesh>

            {/* Flow Arrows */}
            <FlowArrows mode={mode} />

            {/* Low SOC Warning Blinker */}
            <LowSOCBlinker soc={soc} />

            {/* Ambient glow when active */}
            {mode !== 'idle' && (
                <pointLight
                    position={[0, 0, 0.3]}
                    intensity={0.4}
                    distance={2}
                    color={mode === 'charge' ? 0x00ff00 : 0xff4444}
                />
            )}
        </group>
    );
}
