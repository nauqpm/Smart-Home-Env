import React, { useRef, useMemo } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';

// Shared materials
const metalDark = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.3, metalness: 0.9 });
const metalBronze = new THREE.MeshStandardMaterial({ color: 0x8b7355, roughness: 0.3, metalness: 0.7 });

interface SmartLightProps {
    position: [number, number, number];
    isOn: boolean;
    roomName?: string;
    type?: 'linear' | 'pendant' | 'ceiling' | 'wall';
    color?: number;
    length?: number;
    rotation?: [number, number, number];
}

/**
 * SmartLight - Modern 3D light fixtures with on/off state visualization
 * 
 * Types:
 * - linear: Modern LED bar light (like the reference image)
 * - pendant: Hanging pendant lamp
 * - ceiling: Flush mount ceiling light
 * - wall: Wall sconce
 * 
 * Features:
 * - Glowing LED strip when ON (emissive material)
 * - PointLight/RectAreaLight bound to isOn state
 * - Subtle pulse animation when lit
 */
export default function SmartLight({
    position,
    isOn,
    roomName = 'Light',
    type = 'linear',
    color = 0xffffff,
    length = 1.2,
    rotation = [0, 0, 0]
}: SmartLightProps) {
    const ledRef = useRef<THREE.Mesh>(null);
    const glowRef = useRef<THREE.PointLight>(null);

    // Subtle glow pulse animation when on
    useFrame((state) => {
        if (ledRef.current && isOn) {
            const pulse = Math.sin(state.clock.elapsedTime * 1.5) * 0.1 + 0.9;
            (ledRef.current.material as THREE.MeshStandardMaterial).emissiveIntensity = 2.5 * pulse;
        }
        if (glowRef.current && isOn) {
            const pulse = Math.sin(state.clock.elapsedTime * 1.5) * 0.05 + 0.95;
            glowRef.current.intensity = 2 * pulse;
        }
    });

    // ===========================================
    // LINEAR LED BAR (Modern style from reference)
    // ===========================================
    if (type === 'linear') {
        const ledMaterial = useMemo(() => new THREE.MeshStandardMaterial({
            color: isOn ? color : 0x333333,
            emissive: isOn ? color : 0x000000,
            emissiveIntensity: isOn ? 5.0 : 0, // Boosted from 2.5
            roughness: 0.1,
            metalness: 0.0,
        }), [isOn, color]);

        return (
            <group position={position} rotation={rotation}>
                {/* Main housing (dark aluminum profile) */}
                <mesh castShadow>
                    <boxGeometry args={[length, 0.04, 0.06]} />
                    <primitive object={metalDark} attach="material" />
                </mesh>

                {/* LED strip (bottom facing) */}
                <mesh ref={ledRef} position={[0, -0.025, 0]}>
                    <boxGeometry args={[length - 0.02, 0.01, 0.04]} />
                    <primitive object={ledMaterial} attach="material" />
                </mesh>

                {/* Diffuser cover */}
                <mesh position={[0, -0.03, 0]}>
                    <boxGeometry args={[length - 0.01, 0.005, 0.045]} />
                    <meshStandardMaterial
                        color={isOn ? 0xffffff : 0x888888}
                        transparent
                        opacity={isOn ? 0.9 : 0.6}
                        roughness={0.1}
                    />
                </mesh>

                {/* End caps */}
                <mesh position={[-length / 2, 0, 0]}>
                    <boxGeometry args={[0.01, 0.045, 0.065]} />
                    <primitive object={metalDark} attach="material" />
                </mesh>
                <mesh position={[length / 2, 0, 0]}>
                    <boxGeometry args={[0.01, 0.045, 0.065]} />
                    <primitive object={metalDark} attach="material" />
                </mesh>

                {/* Light sources along the bar - Boosted Intensity */}
                <pointLight
                    ref={glowRef}
                    position={[0, -0.1, 0]}
                    intensity={isOn ? 3.0 : 0}
                    distance={8}
                    color={color}
                    castShadow
                    shadow-mapSize-width={256}
                    shadow-mapSize-height={256}
                />
                {/* Additional fill lights for even distribution */}
                <pointLight
                    position={[-length / 3, -0.08, 0]}
                    intensity={isOn ? 1.5 : 0}
                    distance={5}
                    color={color}
                />
                <pointLight
                    position={[length / 3, -0.08, 0]}
                    intensity={isOn ? 1.5 : 0}
                    distance={5}
                    color={color}
                />
            </group>
        );
    }

    // ===========================================
    // PENDANT LIGHT (Hanging lamp)
    // ===========================================
    if (type === 'pendant') {
        const bulbMaterial = useMemo(() => new THREE.MeshStandardMaterial({
            color: isOn ? 0xffdd44 : 0x444444,
            emissive: isOn ? 0xffdd44 : 0x000000,
            emissiveIntensity: isOn ? 5.0 : 0, // Boosted
            roughness: 0.2,
            metalness: 0.1,
            transparent: true,
            opacity: isOn ? 0.95 : 0.7,
        }), [isOn]);

        return (
            <group position={position} rotation={rotation}>
                {/* Ceiling mount */}
                <mesh position={[0, 0, 0]}>
                    <cylinderGeometry args={[0.04, 0.04, 0.02, 16]} />
                    <primitive object={metalDark} attach="material" />
                </mesh>

                {/* Power cord */}
                <mesh position={[0, -0.12, 0]}>
                    <cylinderGeometry args={[0.006, 0.006, 0.22, 8]} />
                    <meshBasicMaterial color={0x111111} />
                </mesh>

                {/* Shade */}
                <mesh position={[0, -0.28, 0]} rotation={[Math.PI, 0, 0]}>
                    <coneGeometry args={[0.12, 0.1, 16, 1, true]} />
                    <meshStandardMaterial
                        color={0x222222}
                        roughness={0.4}
                        metalness={0.7}
                        side={THREE.DoubleSide}
                    />
                </mesh>

                {/* Bulb */}
                <mesh ref={ledRef} position={[0, -0.34, 0]}>
                    <sphereGeometry args={[0.05, 16, 16]} />
                    <primitive object={bulbMaterial} attach="material" />
                </mesh>

                {/* Light */}
                <pointLight
                    ref={glowRef}
                    position={[0, -0.34, 0]}
                    intensity={isOn ? 1.5 : 0}
                    distance={5}
                    color={0xffdd44}
                    castShadow
                />
            </group>
        );
    }

    // ===========================================
    // CEILING FLUSH MOUNT
    // ===========================================
    if (type === 'ceiling') {
        const diffuserMaterial = useMemo(() => new THREE.MeshStandardMaterial({
            color: isOn ? 0xffffee : 0x555555,
            emissive: isOn ? color : 0x000000,
            emissiveIntensity: isOn ? 2 : 0,
            transparent: true,
            opacity: 0.85,
            side: THREE.DoubleSide,
        }), [isOn, color]);

        return (
            <group position={position} rotation={rotation}>
                {/* Base plate */}
                <mesh>
                    <cylinderGeometry args={[0.1, 0.1, 0.02, 24]} />
                    <primitive object={metalDark} attach="material" />
                </mesh>

                {/* Diffuser dome */}
                <mesh ref={ledRef} position={[0, -0.04, 0]}>
                    <sphereGeometry args={[0.08, 24, 24, 0, Math.PI * 2, 0, Math.PI / 2]} />
                    <primitive object={diffuserMaterial} attach="material" />
                </mesh>

                {/* Light */}
                <pointLight
                    ref={glowRef}
                    position={[0, -0.06, 0]}
                    intensity={isOn ? 1.8 : 0}
                    distance={6}
                    color={color}
                    castShadow
                />
            </group>
        );
    }

    // ===========================================
    // WALL SCONCE (Default fallback)
    // ===========================================
    return (
        <group position={position} rotation={rotation}>
            {/* Wall bracket */}
            <mesh position={[0, 0, 0.04]}>
                <boxGeometry args={[0.06, 0.1, 0.03]} />
                <primitive object={metalBronze} attach="material" />
            </mesh>

            {/* Glass globe */}
            <mesh ref={ledRef} position={[0, 0, 0.12]}>
                <sphereGeometry args={[0.06, 16, 16]} />
                <meshStandardMaterial
                    color={isOn ? 0xffffee : 0x666666}
                    emissive={isOn ? color : 0x000000}
                    emissiveIntensity={isOn ? 1.5 : 0}
                    transparent
                    opacity={0.7}
                    roughness={0.1}
                />
            </mesh>

            {/* Light */}
            <pointLight
                ref={glowRef}
                position={[0, 0, 0.12]}
                intensity={isOn ? 1 : 0}
                distance={4}
                color={color}
            />
        </group>
    );
}

// ==========================================
// Room-specific light presets
// ==========================================

export function LivingRoomLight({ position, isOn }: { position: [number, number, number]; isOn: boolean }) {
    return <SmartLight position={position} isOn={isOn} type="linear" color={0xfff8dc} length={1.5} />;
}

export function KitchenLight({ position, isOn }: { position: [number, number, number]; isOn: boolean }) {
    return <SmartLight position={position} isOn={isOn} type="linear" color={0xffffff} length={1.2} />;
}

export function BedroomLight({ position, isOn }: { position: [number, number, number]; isOn: boolean }) {
    return <SmartLight position={position} isOn={isOn} type="linear" color={0xffe4b5} length={1.0} />;
}

export function ToiletLight({ position, isOn }: { position: [number, number, number]; isOn: boolean }) {
    return <SmartLight position={position} isOn={isOn} type="ceiling" color={0xffffff} />;
}
