import React from 'react';
import { useStore } from '../../stores/useStore';
import SmartLight from './SmartLight';

/**
 * RoomLights - Room-specific 3D LED light fixtures
 * 
 * All rooms use modern linear LED strip lights.
 */
export default function RoomLights() {
    const { devices, simData, currentViewMode } = useStore();
    const actions = simData?.[currentViewMode]?.actions || {};

    // LED light height - positioned on upper wall
    const LED_Y = 2.0;

    return (
        <group>
            {/* ============================================ */}
            {/* LIVING ROOM - LED bar along front wall */}
            {/* ============================================ */}
            <SmartLight
                position={[2, LED_Y, 3.6]}
                isOn={devices.light_living?.isOn ?? actions.light_living === 1}
                type="linear"
                color={0xfff8dc}
                length={2.3}
                rotation={[0, 0, 0]}
            />

            {/* ============================================ */}
            {/* MASTER BEDROOM - LED bar along front wall */}
            {/* ============================================ */}
            <SmartLight
                position={[-2, LED_Y, 3.6]}
                isOn={devices.light_master?.isOn ?? actions.light_master === 1}
                type="linear"
                color={0xffe4b5}
                length={2.3}
                rotation={[0, 0, 0]}
            />

            {/* ============================================ */}
            {/* 2ND BEDROOM - LED bar on east wall (opposite AC) */}
            {/* ============================================ */}
            <SmartLight
                position={[-0.1, LED_Y, -2]}
                isOn={devices.light_bed2?.isOn ?? actions.light_bed2 === 1}
                type="linear"
                color={0xffe4b5}
                length={2.0}
                rotation={[0, Math.PI / 2, 0]}
            />

            {/* ============================================ */}
            {/* KITCHEN - LED bar along south wall */}
            {/* ============================================ */}
            <SmartLight
                position={[2, LED_Y, -3.6]}
                isOn={devices.light_kitchen?.isOn ?? actions.light_kitchen === 1}
                type="linear"
                color={0xffffff}
                length={2.3}
                rotation={[0, 0, 0]}
            />

            {/* ============================================ */}
            {/* TOILET - Ceiling light (inside bathroom) */}
            {/* Bathroom is at x: -4 to -2.5, z: -3 to 1 */}
            {/* ============================================ */}
            <SmartLight
                position={[-3.25, LED_Y, -1]}
                isOn={devices.light_toilet?.isOn ?? actions.light_toilet === 1}
                type="ceiling"
                color={0xffffff}
            />

            {/* ============================================ */}
            {/* UTILITY/LOGGIA - Dim ambient */}
            {/* ============================================ */}
            <pointLight
                position={[-2.5, 2.0, -3.4]}
                intensity={0.3}
                color={0xccffcc}
                distance={3}
            />
        </group>
    );
}
