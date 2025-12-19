import React from 'react';
import { useStore } from '../../stores/useStore';

export default function RoomLights() {
    const { devices } = useStore();

    return (
        <group>
            {/* ============================================ */}
            {/* ZONE A: COMMON AREA */}
            {/* ============================================ */}

            {/* Living Room ceiling light */}
            <pointLight
                position={[2, 2.2, 2]}
                intensity={1.5}
                color={0xfff8dc}
                distance={8}
                castShadow
            />

            {/* Dining area light */}
            <pointLight
                position={[2, 2.2, -0.6]}
                intensity={1.2}
                color={0xffffff}
                distance={6}
            />

            {/* Kitchen light */}
            <pointLight
                position={[2, 2.2, -2.5]}
                intensity={1.5}
                color={0xffffff}
                distance={7}
            />

            {/* ============================================ */}
            {/* ZONE B: PRIVATE AREA */}
            {/* ============================================ */}

            {/* Master Bedroom spotlight */}
            <spotLight
                position={[-2, 2.2, 2.5]}
                angle={Math.PI / 3}
                intensity={1.2}
                color={0xffe4b5}
                distance={8}
                castShadow
            />

            {/* Small Bedroom light */}
            <pointLight
                position={[-2, 2.2, -1.8]}
                intensity={1.0}
                color={0xfff8dc}
                distance={6}
            />

            {/* Bathroom light */}
            <pointLight
                position={[-3.25, 2.0, 0.25]}
                intensity={1.0}
                color={0xffffff}
                distance={4}
            />

            {/* ============================================ */}
            {/* ZONE C: LOGGIA */}
            {/* ============================================ */}

            {/* Loggia light (utility area) */}
            <pointLight
                position={[-3.25, 2.0, -3.4]}
                intensity={0.8}
                color={0xccffcc}
                distance={3}
            />

            {/* ============================================ */}
            {/* DYNAMIC LIGHTS (State-dependent) */}
            {/* ============================================ */}

            {/* Floor Lamp effect (when on) */}
            {devices.lamp?.isOn && (
                <pointLight
                    position={[0.4, 0.9, 0.8]}
                    intensity={1.5}
                    color={0xffaa00}
                    distance={5}
                />
            )}
        </group>
    );
}
