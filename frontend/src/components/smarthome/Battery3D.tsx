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

const arrowGreen = new THREE.MeshBasicMaterial({ color: 0x00ff00, transparent: true, opacity: 0.8 });
const arrowRed = new THREE.MeshBasicMaterial({ color: 0xff0000, transparent: true, opacity: 0.8 });

function FlowArrows({ mode }: { mode: 'charging' | 'discharging' | 'idle' }) {
    const groupRef = useRef<THREE.Group>(null);

    useFrame((state, delta) => {
        if (groupRef.current && mode !== 'idle') {
            // Animate arrows up or down
            const speed = mode === 'charging' ? 1 : -1;
            groupRef.current.position.y += speed * delta * 0.5;
            if (groupRef.current.position.y > 0.5) groupRef.current.position.y = -0.5;
            if (groupRef.current.position.y < -0.5) groupRef.current.position.y = 0.5;
        }
    });

    if (mode === 'idle') return null;

    return (
        <group ref={groupRef} position={[0, 0, 0.2]}>
            <mesh position={[0, 0.2, 0]} rotation={[0, 0, mode === 'charging' ? 0 : Math.PI]}>
                <coneGeometry args={[0.08, 0.15, 3]} />
                <primitive object={mode === 'charging' ? arrowRed : arrowGreen} attach="material" />
            </mesh>
            <mesh position={[0, -0.2, 0]} rotation={[0, 0, mode === 'charging' ? 0 : Math.PI]}>
                <coneGeometry args={[0.08, 0.15, 3]} />
                <primitive object={mode === 'charging' ? arrowRed : arrowGreen} attach="material" />
            </mesh>
        </group>
    )
}

export default function Battery3D({ position, rotation = [0, 0, 0] }: { position: [number, number, number], rotation?: [number, number, number] }) {
    const { batterySOC, gridImport } = useStore();

    // Determine Flow Mode
    // gridImport > 0 (Buying) -> Likely charging battery if logic says so, OR battery idle. 
    // Logic simplification: if gridImport > Load, we are charging. If gridImport < Load, we might be discharging.
    // Actually, useStore Logic: 
    // - High Price: Discharge -> gridImport decreases. 
    // - Low Price: Charge -> gridImport increases.
    // We can infer mode from SOC change direction if we tracked it, but simpler:
    // For now, let's visualize based on a simple heuristic or add 'batteryStatus' to store later.
    // Heuristic: If we are selling (gridImport < 0), we are definitely discharging/exporting.
    // If we are Buying huge amount (gridImport > 2000), likely charging.

    // Better approach: Derived state from Store logic, but here purely visual constraint.
    // Let's assume:
    // Discharging if SOC is notably dropping (hard to see without prev state).
    // Let's use the Store's `decisionLog` or specific state if available.
    // For now: 
    // Charge if gridImport > 2500W (Assumed threshold).
    // Discharge if gridImport < 500W (Assumed assisted).

    const mode = useMemo(() => {
        // Ideally store should tell us status. For visualization proof of concept:
        // We'll read SOC change... too complex for R3F component.
        // Let's just use random flicker for 'Active' feel or based on logic?
        // Let's check useStore logic: Discharging when Price > 0.25. Charging when Price < 0.15.
        // We can read Price from store too? 
        // Let's just show 'Green' (Good/Discharging/Saving) vs 'Red' (Charging/Costly)
        return 'idle';
    }, []);

    // Actually, let's just show SOC bars.

    // SOC Bar: 5 segments
    const bars = [0.2, 0.4, 0.6, 0.8, 1.0];

    return (
        <group position={position} rotation={rotation}>
            {/* Main Body (Tesla Powerwall-ish) */}
            <mesh castShadow receiveShadow>
                <boxGeometry args={[0.8, 1.2, 0.2]} />
                <primitive object={caseWhite} attach="material" />
            </mesh>
            {/* Side trim */}
            <mesh position={[0.4, 0, 0]}>
                <boxGeometry args={[0.05, 1.2, 0.21]} />
                <primitive object={caseBlack} attach="material" />
            </mesh>
            <mesh position={[-0.4, 0, 0]}>
                <boxGeometry args={[0.05, 1.2, 0.21]} />
                <primitive object={caseBlack} attach="material" />
            </mesh>

            {/* LED Strip (Vertical) */}
            <group position={[0.2, 0, 0.11]}>
                {bars.map((limit, i) => (
                    <mesh key={i} position={[0, (i - 2) * 0.15, 0]}>
                        <planeGeometry args={[0.05, 0.1]} />
                        <primitive
                            object={batterySOC >= limit - 0.1 ? (batterySOC < 0.3 ? ledRed : ledGreen) : ledOff}
                            attach="material"
                        />
                    </mesh>
                ))}
            </group>

            {/* Logo/Text */}
            <mesh position={[-0.1, 0.4, 0.11]}>
                <planeGeometry args={[0.2, 0.05]} />
                <meshBasicMaterial color={0xaaaaaa} />
            </mesh>

            <FlowArrows mode={gridImport > 2500 ? 'charging' : (gridImport < 1000 ? 'discharging' : 'idle')} />
        </group>
    )
}
