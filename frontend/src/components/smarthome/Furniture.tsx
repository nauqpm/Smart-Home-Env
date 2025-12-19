import React, { useMemo } from 'react';
import * as THREE from 'three';
import { makeCeramicMat, makeChromeMat } from './materials';

// ==========================================
// MATERIALS
// ==========================================
const sofaMat = new THREE.MeshStandardMaterial({ color: 0x5D4E37, roughness: 0.8 });
const tableMat = new THREE.MeshStandardMaterial({ color: 0xDEB887, roughness: 0.6 });
const woodDarkMat = new THREE.MeshStandardMaterial({ color: 0x654321, roughness: 0.7 });
const fabricMat = new THREE.MeshStandardMaterial({ color: 0xF5F5DC, roughness: 0.9 });
const kitchenMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3 });
const countertopMat = new THREE.MeshStandardMaterial({ color: 0x2c3e50, roughness: 0.2, metalness: 0.3 });
const glassMat = new THREE.MeshStandardMaterial({ color: 0x88ccff, transparent: true, opacity: 0.3, roughness: 0.1 });
const doorMat = new THREE.MeshStandardMaterial({ color: 0x8B4513, roughness: 0.6 }); // Brown wood door
const doorFrameMat = new THREE.MeshStandardMaterial({ color: 0x444444, roughness: 0.5 }); // Dark door frame

// ==========================================
// ZONE A: COMMON AREA (Right Side, x: 0 to 4)
// Living Room, Dining, Kitchen, Entrance, Balcony
// ==========================================
function CommonArea() {
    const chairPositions: [number, number, number][] = [
        [1.3, 0.45, -0.3], [2.7, 0.45, -0.3], [1.3, 0.45, -0.9], [2.7, 0.45, -0.9]
    ];

    return (
        <group>
            {/* ============================================ */}
            {/* LIVING ROOM (x: 0 to 4, z: 0 to 3.75) */}
            {/* ============================================ */}

            {/* L-Shaped Sofa - facing TV (north wall z=3.75) */}
            {/* Main sofa section - runs along x-axis, faces north */}
            <mesh position={[2, 0.55, 2]} castShadow>
                <boxGeometry args={[2.4, 0.7, 0.9]} />
                <primitive object={sofaMat} attach="material" />
            </mesh>
            {/* Side section - smaller to avoid clipping with TV stand */}
            <mesh position={[0.9, 0.55, 2.5]} castShadow>
                <boxGeometry args={[0.7, 0.7, 1.0]} />
                <primitive object={sofaMat} attach="material" />
            </mesh>
            {/* Sofa back - behind main section (south side) */}
            <mesh position={[2, 0.85, 1.6]} castShadow>
                <boxGeometry args={[2.4, 0.35, 0.15]} />
                <primitive object={sofaMat} attach="material" />
            </mesh>
            {/* Sofa back - left side section */}
            <mesh position={[0.6, 0.85, 2.5]} castShadow>
                <boxGeometry args={[0.15, 0.35, 1.0]} />
                <primitive object={sofaMat} attach="material" />
            </mesh>

            {/* Coffee Table - between sofa and TV */}
            <mesh position={[2.2, 0.4, 2.8]} castShadow>
                <boxGeometry args={[0.8, 0.4, 0.5]} />
                <primitive object={tableMat} attach="material" />
            </mesh>

            {/* TV Stand - centered with TV */}
            <mesh position={[2, 0.45, 3.4]} castShadow>
                <boxGeometry args={[1.8, 0.6, 0.3]} />
                <primitive object={woodDarkMat} attach="material" />
            </mesh>

            {/* Rug under sofa area */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[1.5, 0.27, 2.2]} receiveShadow>
                <planeGeometry args={[2.5, 2]} />
                <meshStandardMaterial color={0x8B7355} roughness={0.9} />
            </mesh>

            {/* Plant corner */}
            <group position={[0.4, 0.15, 3.4]}>
                <mesh castShadow>
                    <cylinderGeometry args={[0.12, 0.14, 0.3]} />
                    <primitive object={woodDarkMat} attach="material" />
                </mesh>
                <mesh position={[0, 0.4, 0]} castShadow>
                    <coneGeometry args={[0.3, 0.6, 8]} />
                    <meshStandardMaterial color={0x2d5016} />
                </mesh>
            </group>

            {/* ============================================ */}
            {/* BALCONY RAILING (z: 3.75 to 4.25) */}
            {/* ============================================ */}
            <mesh position={[3, 0.15, 4.1]} castShadow>
                <boxGeometry args={[1.9, 1.0, 0.05]} />
                <primitive object={glassMat} attach="material" />
            </mesh>

            {/* ============================================ */}
            {/* DINING AREA (x: 0 to 4, z: -1.5 to 0) */}
            {/* ============================================ */}

            {/* Dining Table */}
            <mesh position={[2, 0.87, -0.6]} castShadow>
                <boxGeometry args={[1.6, 0.05, 0.8]} />
                <primitive object={tableMat} attach="material" />
            </mesh>
            {/* Table legs */}
            {[[1.3, -0.3], [2.7, -0.3], [1.3, -0.9], [2.7, -0.9]].map(([x, z], i) => (
                <mesh key={`tleg-${i}`} position={[x, 0.5, z]}>
                    <cylinderGeometry args={[0.03, 0.03, 0.65]} />
                    <primitive object={tableMat} attach="material" />
                </mesh>
            ))}

            {/* 4 Chairs */}
            {chairPositions.map((pos, i) => (
                <group key={`chair-${i}`} position={pos}>
                    <mesh castShadow>
                        <boxGeometry args={[0.35, 0.06, 0.35]} />
                        <primitive object={sofaMat} attach="material" />
                    </mesh>
                    <mesh position={[0, 0.3, i < 2 ? 0.15 : -0.15]} castShadow>
                        <boxGeometry args={[0.35, 0.5, 0.06]} />
                        <primitive object={sofaMat} attach="material" />
                    </mesh>
                </group>
            ))}

            {/* ============================================ */}
            {/* KITCHEN (x: 0 to 4, z: -3.75 to -1.5) */}
            {/* ============================================ */}

            {/* L-shaped Cabinets - back wall */}
            <mesh position={[2, 0.55, -3.4]} castShadow>
                <boxGeometry args={[3.5, 0.8, 0.5]} />
                <primitive object={kitchenMat} attach="material" />
            </mesh>
            {/* Side cabinets */}
            <mesh position={[3.7, 0.55, -2.5]} castShadow>
                <boxGeometry args={[0.5, 0.8, 1.5]} />
                <primitive object={kitchenMat} attach="material" />
            </mesh>

            {/* Countertops */}
            <mesh position={[2, 0.96, -3.4]}>
                <boxGeometry args={[3.6, 0.04, 0.55]} />
                <primitive object={countertopMat} attach="material" />
            </mesh>
            <mesh position={[3.7, 0.96, -2.5]}>
                <boxGeometry args={[0.55, 0.04, 1.6]} />
                <primitive object={countertopMat} attach="material" />
            </mesh>

            {/* Sink */}
            <mesh position={[1.5, 0.99, -3.4]}>
                <boxGeometry args={[0.45, 0.15, 0.35]} />
                <meshStandardMaterial color={0xcccccc} metalness={0.8} roughness={0.2} />
            </mesh>

            {/* Induction Cooker */}
            <mesh position={[2.8, 0.98, -3.4]}>
                <boxGeometry args={[0.6, 0.02, 0.4]} />
                <meshStandardMaterial color={0x222222} />
            </mesh>
        </group>
    );
}

// ==========================================
// ZONE B: PRIVATE AREA (Left Side, x: -4 to 0)
// Master Bedroom, Small Bedroom, Bathroom
// ==========================================
function PrivateArea() {
    const ceramic = useMemo(() => makeCeramicMat(), []);
    const chrome = useMemo(() => makeChromeMat(), []);

    return (
        <group>
            {/* ============================================ */}
            {/* MASTER BEDROOM (x: -4 to 0, z: 1 to 3.75) */}
            {/* ============================================ */}

            {/* Large Bed - moved closer to AC wall (west wall) */}
            <group position={[-2.8, 0.15, 2.5]} rotation={[0, Math.PI, 0]}>
                <mesh position={[0, 0.25, 0]} castShadow>
                    <boxGeometry args={[1.8, 0.5, 2.2]} />
                    <primitive object={woodDarkMat} attach="material" />
                </mesh>
                <mesh position={[0, 0.55, 0]} castShadow>
                    <boxGeometry args={[1.6, 0.2, 2.0]} />
                    <primitive object={fabricMat} attach="material" />
                </mesh>
                {/* Pillows */}
                <mesh position={[-0.4, 0.7, -0.8]} castShadow>
                    <boxGeometry args={[0.45, 0.12, 0.4]} />
                    <meshStandardMaterial color={0xffffff} />
                </mesh>
                <mesh position={[0.4, 0.7, -0.8]} castShadow>
                    <boxGeometry args={[0.45, 0.12, 0.4]} />
                    <meshStandardMaterial color={0xffffff} />
                </mesh>
                {/* Headboard */}
                <mesh position={[0, 0.7, -1.05]} castShadow>
                    <boxGeometry args={[1.8, 0.8, 0.1]} />
                    <primitive object={woodDarkMat} attach="material" />
                </mesh>
            </group>

            {/* Nightstands - adjusted for moved bed */}
            <mesh position={[-3.6, 0.4, 2.5]} castShadow>
                <boxGeometry args={[0.4, 0.5, 0.4]} />
                <primitive object={woodDarkMat} attach="material" />
            </mesh>
            <mesh position={[-1.6, 0.4, 2.5]} castShadow>
                <boxGeometry args={[0.4, 0.5, 0.4]} />
                <primitive object={woodDarkMat} attach="material" />
            </mesh>

            {/* Small Wardrobe - inside room, away from wall edge */}
            <mesh position={[-0.5, 0.95, 1.7]} castShadow>
                <boxGeometry args={[0.5, 1.6, 1.0]} />
                <primitive object={woodDarkMat} attach="material" />
            </mesh>

            {/* Work Desk with Laptop - flush with north wall */}
            <group position={[-0.75, 0.15, 3.5]} rotation={[0, Math.PI, 0]}>
                {/* Desk surface */}
                <mesh position={[0, 0.72, 0]} castShadow>
                    <boxGeometry args={[1.0, 0.04, 0.5]} />
                    <primitive object={woodDarkMat} attach="material" />
                </mesh>
                {/* Desk legs */}
                <mesh position={[-0.45, 0.36, 0.2]} castShadow>
                    <boxGeometry args={[0.04, 0.72, 0.04]} />
                    <primitive object={woodDarkMat} attach="material" />
                </mesh>
                <mesh position={[0.45, 0.36, 0.2]} castShadow>
                    <boxGeometry args={[0.04, 0.72, 0.04]} />
                    <primitive object={woodDarkMat} attach="material" />
                </mesh>
                <mesh position={[-0.45, 0.36, -0.2]} castShadow>
                    <boxGeometry args={[0.04, 0.72, 0.04]} />
                    <primitive object={woodDarkMat} attach="material" />
                </mesh>
                <mesh position={[0.45, 0.36, -0.2]} castShadow>
                    <boxGeometry args={[0.04, 0.72, 0.04]} />
                    <primitive object={woodDarkMat} attach="material" />
                </mesh>
                {/* Laptop base */}
                <mesh position={[0, 0.76, 0]} castShadow>
                    <boxGeometry args={[0.35, 0.02, 0.25]} />
                    <meshStandardMaterial color={0x333333} />
                </mesh>
                {/* Laptop screen */}
                <mesh position={[0, 0.88, -0.1]} rotation={[-0.3, 0, 0]} castShadow>
                    <boxGeometry args={[0.35, 0.22, 0.01]} />
                    <meshStandardMaterial color={0x222222} />
                </mesh>
                {/* Screen glow */}
                <mesh position={[0, 0.88, -0.09]} rotation={[-0.3, 0, 0]}>
                    <planeGeometry args={[0.32, 0.18]} />
                    <meshStandardMaterial color={0x4488ff} emissive={0x4488ff} emissiveIntensity={0.5} />
                </mesh>
                {/* Desk Chair */}
                <mesh position={[0, 0.4, 0.4]} castShadow>
                    <cylinderGeometry args={[0.2, 0.2, 0.05, 16]} />
                    <primitive object={sofaMat} attach="material" />
                </mesh>
                <mesh position={[0, 0.65, 0.55]} castShadow>
                    <boxGeometry args={[0.38, 0.45, 0.06]} />
                    <primitive object={sofaMat} attach="material" />
                </mesh>
            </group>

            {/* ============================================ */}
            {/* SMALL BEDROOM (x: -4 to 0, z: -3 to -0.5) */}
            {/* ============================================ */}

            {/* Single Bed - against AC wall (west side) */}
            <group position={[-2.0, 0.15, -1.5]}>
                <mesh position={[0, 0.2, 0]} castShadow>
                    <boxGeometry args={[1.2, 0.4, 1.9]} />
                    <primitive object={woodDarkMat} attach="material" />
                </mesh>
                <mesh position={[0, 0.45, 0]} castShadow>
                    <boxGeometry args={[1.0, 0.18, 1.7]} />
                    <primitive object={fabricMat} attach="material" />
                </mesh>
                {/* Pillow */}
                <mesh position={[0, 0.58, -0.65]} castShadow>
                    <boxGeometry args={[0.45, 0.1, 0.35]} />
                    <meshStandardMaterial color={0xaaccff} />
                </mesh>
            </group>

            {/* Work Desk for small bedroom - against south wall (opposite door) */}
            <group position={[-0.5, 0.15, -3.3]}>
                {/* Desk surface */}
                <mesh position={[0, 0.72, 0]} castShadow>
                    <boxGeometry args={[0.9, 0.04, 0.45]} />
                    <primitive object={tableMat} attach="material" />
                </mesh>
                {/* Desk legs */}
                <mesh position={[-0.4, 0.36, 0.18]} castShadow>
                    <boxGeometry args={[0.04, 0.72, 0.04]} />
                    <primitive object={tableMat} attach="material" />
                </mesh>
                <mesh position={[0.4, 0.36, 0.18]} castShadow>
                    <boxGeometry args={[0.04, 0.72, 0.04]} />
                    <primitive object={tableMat} attach="material" />
                </mesh>
                <mesh position={[-0.4, 0.36, -0.18]} castShadow>
                    <boxGeometry args={[0.04, 0.72, 0.04]} />
                    <primitive object={tableMat} attach="material" />
                </mesh>
                <mesh position={[0.4, 0.36, -0.18]} castShadow>
                    <boxGeometry args={[0.04, 0.72, 0.04]} />
                    <primitive object={tableMat} attach="material" />
                </mesh>
            </group>

            {/* Desk Chair for small bedroom */}
            <group position={[-0.5, 0.15, -2.7]}>
                <mesh position={[0, 0.4, 0]} castShadow>
                    <cylinderGeometry args={[0.18, 0.18, 0.05, 16]} />
                    <primitive object={sofaMat} attach="material" />
                </mesh>
                <mesh position={[0, 0.65, 0.12]} castShadow>
                    <boxGeometry args={[0.32, 0.4, 0.06]} />
                    <primitive object={sofaMat} attach="material" />
                </mesh>
            </group>

            {/* Small Wardrobe for small bedroom - against south wall */}
            <mesh position={[-1.5, 0.95, -3.3]} castShadow>
                <boxGeometry args={[0.5, 1.6, 1.0]} />
                <primitive object={woodDarkMat} attach="material" />
            </mesh>

            {/* ============================================ */}
            {/* BATHROOM (x: -4 to -2.5, z: -3 to 1) - EXPANDED */}
            {/* New layout: Vanity opposite door, shower+toilet near bedroom 2 */}
            {/* ============================================ */}

            {/* ============================================ */}
            {/* VANITY CABINET with SINK + MIRROR - opposite door (west wall) */}
            {/* ============================================ */}
            <group position={[-3.75, 0.15, 0.3]}>
                {/* Vanity cabinet base */}
                <mesh position={[0, 0.35, 0]} castShadow>
                    <boxGeometry args={[0.45, 0.7, 0.6]} />
                    <meshStandardMaterial color={0x5c4033} roughness={0.4} />
                </mesh>
                {/* Cabinet doors */}
                <mesh position={[0.23, 0.35, 0]}>
                    <boxGeometry args={[0.01, 0.6, 0.55]} />
                    <meshStandardMaterial color={0x4a3728} roughness={0.3} />
                </mesh>
                {/* Countertop */}
                <mesh position={[0, 0.72, 0]} castShadow>
                    <boxGeometry args={[0.5, 0.04, 0.65]} />
                    <primitive object={ceramic} attach="material" />
                </mesh>
                {/* Sink basin (inset) */}
                <mesh position={[0, 0.73, 0]}>
                    <cylinderGeometry args={[0.18, 0.15, 0.08, 32]} />
                    <primitive object={ceramic} attach="material" />
                </mesh>
                {/* Faucet */}
                <mesh position={[0, 0.85, -0.2]}>
                    <cylinderGeometry args={[0.02, 0.02, 0.15, 8]} />
                    <primitive object={chrome} attach="material" />
                </mesh>
                <mesh position={[0, 0.9, -0.1]} rotation={[Math.PI / 3, 0, 0]}>
                    <cylinderGeometry args={[0.015, 0.015, 0.15, 8]} />
                    <primitive object={chrome} attach="material" />
                </mesh>
                {/* Large Mirror - rotated to face into room, flush against wall */}
                <mesh position={[-0.22, 1.3, 0]} rotation={[0, Math.PI / 2, 0]}>
                    <planeGeometry args={[0.55, 0.7]} />
                    <meshStandardMaterial color={0xeeeeee} metalness={0.95} roughness={0.05} side={2} />
                </mesh>
                {/* Mirror frame - on wall */}
                <mesh position={[-0.21, 1.3, 0]}>
                    <boxGeometry args={[0.02, 0.75, 0.6]} />
                    <meshStandardMaterial color={0x333333} />
                </mesh>
            </group>

            {/* ============================================ */}
            {/* WASHING MACHINE - next to vanity */}
            {/* ============================================ */}
            <group position={[-3.75, 0.60, -0.4]}>
                {/* Body */}
                <mesh castShadow>
                    <boxGeometry args={[0.5, 0.8, 0.55]} />
                    <meshStandardMaterial color={0xffffff} roughness={0.3} />
                </mesh>
                {/* Door circle */}
                <mesh position={[0.26, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
                    <cylinderGeometry args={[0.18, 0.18, 0.02, 32]} />
                    <meshStandardMaterial color={0x333333} metalness={0.5} />
                </mesh>
                {/* Control panel */}
                <mesh position={[0.26, 0.35, 0]}>
                    <boxGeometry args={[0.02, 0.1, 0.45]} />
                    <meshStandardMaterial color={0x222222} />
                </mesh>
            </group>

            {/* ============================================ */}
            {/* SHOWER ENCLOSURE - against south wall (near bedroom 2) */}
            {/* ============================================ */}
            <group position={[-3.5, 0.15, -2.5]}>
                {/* Shower base/tray */}
                <mesh position={[0, 0.05, 0]} castShadow receiveShadow>
                    <boxGeometry args={[0.8, 0.1, 0.8]} />
                    <meshStandardMaterial color={0xffffff} roughness={0.2} />
                </mesh>
                {/* Glass walls - back (south) */}
                <mesh position={[0, 1.0, -0.39]}>
                    <boxGeometry args={[0.78, 2.0, 0.02]} />
                    <meshStandardMaterial color={0xaaddff} transparent opacity={0.3} roughness={0.1} />
                </mesh>
                {/* Glass walls - side (west) */}
                <mesh position={[-0.39, 1.0, 0]}>
                    <boxGeometry args={[0.02, 2.0, 0.8]} />
                    <meshStandardMaterial color={0xaaddff} transparent opacity={0.3} roughness={0.1} />
                </mesh>
                {/* Glass door (front/east) */}
                <mesh position={[0.39, 1.0, 0]}>
                    <boxGeometry args={[0.02, 2.0, 0.8]} />
                    <meshStandardMaterial color={0xaaddff} transparent opacity={0.2} roughness={0.1} />
                </mesh>
                {/* Shower head */}
                <mesh position={[0, 1.9, -0.35]}>
                    <cylinderGeometry args={[0.08, 0.06, 0.03, 16]} />
                    <primitive object={chrome} attach="material" />
                </mesh>
                {/* Shower pipe */}
                <mesh position={[0, 1.5, -0.37]}>
                    <cylinderGeometry args={[0.015, 0.015, 0.8, 8]} />
                    <primitive object={chrome} attach="material" />
                </mesh>
                {/* Frame - top */}
                <mesh position={[0, 2.0, 0]}>
                    <boxGeometry args={[0.8, 0.03, 0.8]} />
                    <meshStandardMaterial color={0xcccccc} metalness={0.5} />
                </mesh>
            </group>

            {/* ============================================ */}
            {/* TOILET - next to shower (north of shower) */}
            {/* ============================================ */}
            <group position={[-3.5, 0.50, -1.6]}>
                {/* Toilet base */}
                <mesh castShadow>
                    <cylinderGeometry args={[0.18, 0.22, 0.3, 32]} />
                    <primitive object={ceramic} attach="material" />
                </mesh>
                {/* Toilet tank */}
                <mesh position={[-0.2, 0.15, 0]} castShadow>
                    <boxGeometry args={[0.15, 0.35, 0.35]} />
                    <primitive object={ceramic} attach="material" />
                </mesh>
                {/* Toilet seat */}
                <mesh position={[0.05, 0.18, 0]} castShadow>
                    <boxGeometry args={[0.35, 0.05, 0.3]} />
                    <primitive object={ceramic} attach="material" />
                </mesh>
            </group>
        </group>
    );
}

// ==========================================
// UTILITY CLOSET (in Small Bedroom corner)
// Battery + Solar Inverter - placed in Appliances.tsx
// ==========================================
function UtilityCloset() {
    return (
        <group>
            {/* Utility closet floor - tile */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[-2.15, 0.12, -3.375]} receiveShadow>
                <planeGeometry args={[0.7, 0.75]} />
                <meshStandardMaterial color={0xaaaaaa} roughness={0.8} />
            </mesh>
        </group>
    );
}

// ==========================================
// DOORS - All doors CLOSED for clear identification
// ==========================================
function Doors() {
    const DOOR_W = 0.8;  // Door width
    const DOOR_H = 2.1;  // Door height
    const DOOR_T = 0.06; // Door thickness
    const FRAME_T = 0.08; // Frame thickness

    return (
        <group>
            {/* ============================================ */}
            {/* MAIN ENTRANCE DOOR (x = 4, z: 0) */}
            {/* East wall - between living room and kitchen */}
            {/* ============================================ */}
            <group position={[4, DOOR_H / 2, 0]} rotation={[0, Math.PI / 2, 0]}>
                {/* Door frame */}
                <mesh position={[-0.45, 0, 0]} castShadow receiveShadow>
                    <boxGeometry args={[FRAME_T, DOOR_H + 0.1, 0.2]} />
                    <primitive object={doorFrameMat} attach="material" />
                </mesh>
                <mesh position={[0.45, 0, 0]} castShadow receiveShadow>
                    <boxGeometry args={[FRAME_T, DOOR_H + 0.1, 0.2]} />
                    <primitive object={doorFrameMat} attach="material" />
                </mesh>
                <mesh position={[0, DOOR_H / 2 + FRAME_T, 0]} castShadow receiveShadow>
                    <boxGeometry args={[0.9 + FRAME_T, FRAME_T, 0.2]} />
                    <primitive object={doorFrameMat} attach="material" />
                </mesh>
                {/* Door panel - CLOSED */}
                <mesh castShadow receiveShadow>
                    <boxGeometry args={[0.9, DOOR_H, DOOR_T]} />
                    <meshStandardMaterial color={0x4a3728} roughness={0.5} />
                </mesh>
                {/* Door handle */}
                <mesh position={[0.35, 0, 0.05]}>
                    <boxGeometry args={[0.08, 0.03, 0.04]} />
                    <meshStandardMaterial color={0xcccccc} metalness={0.8} />
                </mesh>
            </group>

            {/* ============================================ */}
            {/* MASTER BEDROOM DOOR (z = 1, x: -1.1) */}
            {/* ============================================ */}
            <group position={[-1.1, DOOR_H / 2, 1]}>
                {/* Door frame */}
                <mesh position={[-DOOR_W / 2 - FRAME_T / 2, 0, 0]} castShadow receiveShadow>
                    <boxGeometry args={[FRAME_T, DOOR_H, 0.18]} />
                    <primitive object={doorFrameMat} attach="material" />
                </mesh>
                <mesh position={[DOOR_W / 2 + FRAME_T / 2, 0, 0]} castShadow receiveShadow>
                    <boxGeometry args={[FRAME_T, DOOR_H, 0.18]} />
                    <primitive object={doorFrameMat} attach="material" />
                </mesh>
                <mesh position={[0, DOOR_H / 2 + FRAME_T / 2, 0]} castShadow receiveShadow>
                    <boxGeometry args={[DOOR_W + FRAME_T * 2, FRAME_T, 0.18]} />
                    <primitive object={doorFrameMat} attach="material" />
                </mesh>
                {/* Door panel - CLOSED */}
                <mesh castShadow receiveShadow>
                    <boxGeometry args={[DOOR_W, DOOR_H - 0.05, DOOR_T]} />
                    <primitive object={doorMat} attach="material" />
                </mesh>
                {/* Door handle */}
                <mesh position={[0.3, 0, 0.05]}>
                    <boxGeometry args={[0.08, 0.03, 0.04]} />
                    <meshStandardMaterial color={0xcccccc} metalness={0.8} />
                </mesh>
            </group>

            {/* ============================================ */}
            {/* SMALL BEDROOM DOOR (z = -0.5, x: -0.5) */}
            {/* ============================================ */}
            <group position={[-0.5, DOOR_H / 2, -0.5]}>
                {/* Door frame */}
                <mesh position={[-DOOR_W / 2 - FRAME_T / 2, 0, 0]} castShadow receiveShadow>
                    <boxGeometry args={[FRAME_T, DOOR_H, 0.18]} />
                    <primitive object={doorFrameMat} attach="material" />
                </mesh>
                <mesh position={[DOOR_W / 2 + FRAME_T / 2, 0, 0]} castShadow receiveShadow>
                    <boxGeometry args={[FRAME_T, DOOR_H, 0.18]} />
                    <primitive object={doorFrameMat} attach="material" />
                </mesh>
                <mesh position={[0, DOOR_H / 2 + FRAME_T / 2, 0]} castShadow receiveShadow>
                    <boxGeometry args={[DOOR_W + FRAME_T * 2, FRAME_T, 0.18]} />
                    <primitive object={doorFrameMat} attach="material" />
                </mesh>
                {/* Door panel - CLOSED */}
                <mesh castShadow receiveShadow>
                    <boxGeometry args={[DOOR_W, DOOR_H - 0.05, DOOR_T]} />
                    <primitive object={doorMat} attach="material" />
                </mesh>
            </group>

            {/* ============================================ */}
            {/* BATHROOM DOOR (x = -2.5, z: 0.325) */}
            {/* Narrow door 0.5m, centered in opening z=0.05 to z=0.6 */}
            {/* ============================================ */}
            <group position={[-2.5, DOOR_H / 2, 0.325]} rotation={[0, Math.PI / 2, 0]}>
                {/* Door frame - 0.5m wide */}
                <mesh position={[-0.26, 0, 0]} castShadow receiveShadow>
                    <boxGeometry args={[FRAME_T, DOOR_H, 0.16]} />
                    <primitive object={doorFrameMat} attach="material" />
                </mesh>
                <mesh position={[0.26, 0, 0]} castShadow receiveShadow>
                    <boxGeometry args={[FRAME_T, DOOR_H, 0.16]} />
                    <primitive object={doorFrameMat} attach="material" />
                </mesh>
                <mesh position={[0, DOOR_H / 2 + FRAME_T / 2, 0]} castShadow receiveShadow>
                    <boxGeometry args={[0.52 + FRAME_T * 2, FRAME_T, 0.16]} />
                    <primitive object={doorFrameMat} attach="material" />
                </mesh>
                {/* Door panel - 0.5m wide */}
                <mesh position={[0, 0, 0.02]} castShadow receiveShadow>
                    <boxGeometry args={[0.5, DOOR_H - 0.05, DOOR_T]} />
                    <primitive object={doorMat} attach="material" />
                </mesh>
            </group>
        </group>
    );
}

// ==========================================
// WINDOWS - Add windows to bedrooms and living room
// ==========================================
function Windows() {
    const WIN_W = 1.2;   // Window width
    const WIN_H = 1.2;   // Window height
    const FRAME_T = 0.08; // Frame thickness - thicker for visibility

    // Bright white aluminum frame
    const windowFrameMat = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.2,
        metalness: 0.3
    });
    // Clear glass with light blue tint
    const windowGlassMat = new THREE.MeshStandardMaterial({
        color: 0xadd8e6,
        transparent: true,
        opacity: 0.25,
        roughness: 0.05,
        metalness: 0.1,
        side: THREE.DoubleSide
    });

    return (
        <group>
            {/* ============================================ */}
            {/* MASTER BEDROOM WINDOW (north wall, z = 3.75) */}
            {/* ============================================ */}
            <group position={[-2, 1.5, 3.85]}>
                {/* Window frame */}
                <mesh position={[0, WIN_H / 2 + FRAME_T / 2, 0]}>
                    <boxGeometry args={[WIN_W + FRAME_T * 2, FRAME_T, 0.1]} />
                    <primitive object={windowFrameMat} attach="material" />
                </mesh>
                <mesh position={[0, -WIN_H / 2 - FRAME_T / 2, 0]}>
                    <boxGeometry args={[WIN_W + FRAME_T * 2, FRAME_T, 0.1]} />
                    <primitive object={windowFrameMat} attach="material" />
                </mesh>
                <mesh position={[-WIN_W / 2 - FRAME_T / 2, 0, 0]}>
                    <boxGeometry args={[FRAME_T, WIN_H, 0.1]} />
                    <primitive object={windowFrameMat} attach="material" />
                </mesh>
                <mesh position={[WIN_W / 2 + FRAME_T / 2, 0, 0]}>
                    <boxGeometry args={[FRAME_T, WIN_H, 0.1]} />
                    <primitive object={windowFrameMat} attach="material" />
                </mesh>
                {/* Center divider */}
                <mesh>
                    <boxGeometry args={[FRAME_T, WIN_H, 0.08]} />
                    <primitive object={windowFrameMat} attach="material" />
                </mesh>
                {/* Glass panes */}
                <mesh position={[-WIN_W / 4, 0, 0]}>
                    <planeGeometry args={[WIN_W / 2 - FRAME_T, WIN_H - FRAME_T]} />
                    <primitive object={windowGlassMat} attach="material" />
                </mesh>
                <mesh position={[WIN_W / 4, 0, 0]}>
                    <planeGeometry args={[WIN_W / 2 - FRAME_T, WIN_H - FRAME_T]} />
                    <primitive object={windowGlassMat} attach="material" />
                </mesh>
            </group>

            {/* ============================================ */}
            {/* SMALL BEDROOM WINDOW (south wall, z = -3.75) */}
            {/* ============================================ */}
            <group position={[-2, 1.5, -3.85]}>
                {/* Window frame */}
                <mesh position={[0, WIN_H / 2 + FRAME_T / 2, 0]}>
                    <boxGeometry args={[WIN_W + FRAME_T * 2, FRAME_T, 0.1]} />
                    <primitive object={windowFrameMat} attach="material" />
                </mesh>
                <mesh position={[0, -WIN_H / 2 - FRAME_T / 2, 0]}>
                    <boxGeometry args={[WIN_W + FRAME_T * 2, FRAME_T, 0.1]} />
                    <primitive object={windowFrameMat} attach="material" />
                </mesh>
                <mesh position={[-WIN_W / 2 - FRAME_T / 2, 0, 0]}>
                    <boxGeometry args={[FRAME_T, WIN_H, 0.1]} />
                    <primitive object={windowFrameMat} attach="material" />
                </mesh>
                <mesh position={[WIN_W / 2 + FRAME_T / 2, 0, 0]}>
                    <boxGeometry args={[FRAME_T, WIN_H, 0.1]} />
                    <primitive object={windowFrameMat} attach="material" />
                </mesh>
                {/* Center divider */}
                <mesh>
                    <boxGeometry args={[FRAME_T, WIN_H, 0.08]} />
                    <primitive object={windowFrameMat} attach="material" />
                </mesh>
                {/* Glass panes */}
                <mesh position={[-WIN_W / 4, 0, 0]}>
                    <planeGeometry args={[WIN_W / 2 - FRAME_T, WIN_H - FRAME_T]} />
                    <primitive object={windowGlassMat} attach="material" />
                </mesh>
                <mesh position={[WIN_W / 4, 0, 0]}>
                    <planeGeometry args={[WIN_W / 2 - FRAME_T, WIN_H - FRAME_T]} />
                    <primitive object={windowGlassMat} attach="material" />
                </mesh>
            </group>

            {/* ============================================ */}
            {/* LIVING ROOM WINDOW (north wall, z = 3.75) */}
            {/* Large window for balcony view */}
            {/* ============================================ */}
            <group position={[2, 1.5, 3.85]}>
                {/* Window frame - larger */}
                <mesh position={[0, 0.75 + FRAME_T / 2, 0]}>
                    <boxGeometry args={[2.5 + FRAME_T * 2, FRAME_T, 0.1]} />
                    <primitive object={windowFrameMat} attach="material" />
                </mesh>
                <mesh position={[0, -0.75 - FRAME_T / 2, 0]}>
                    <boxGeometry args={[2.5 + FRAME_T * 2, FRAME_T, 0.1]} />
                    <primitive object={windowFrameMat} attach="material" />
                </mesh>
                <mesh position={[-1.25 - FRAME_T / 2, 0, 0]}>
                    <boxGeometry args={[FRAME_T, 1.5, 0.1]} />
                    <primitive object={windowFrameMat} attach="material" />
                </mesh>
                <mesh position={[1.25 + FRAME_T / 2, 0, 0]}>
                    <boxGeometry args={[FRAME_T, 1.5, 0.1]} />
                    <primitive object={windowFrameMat} attach="material" />
                </mesh>
                {/* Center dividers */}
                <mesh position={[-0.42, 0, 0]}>
                    <boxGeometry args={[FRAME_T, 1.5, 0.08]} />
                    <primitive object={windowFrameMat} attach="material" />
                </mesh>
                <mesh position={[0.42, 0, 0]}>
                    <boxGeometry args={[FRAME_T, 1.5, 0.08]} />
                    <primitive object={windowFrameMat} attach="material" />
                </mesh>
                {/* Glass panes */}
                <mesh position={[-0.84, 0, 0]}>
                    <planeGeometry args={[0.8, 1.45]} />
                    <primitive object={windowGlassMat} attach="material" />
                </mesh>
                <mesh>
                    <planeGeometry args={[0.8, 1.45]} />
                    <primitive object={windowGlassMat} attach="material" />
                </mesh>
                <mesh position={[0.84, 0, 0]}>
                    <planeGeometry args={[0.8, 1.45]} />
                    <primitive object={windowGlassMat} attach="material" />
                </mesh>
            </group>
        </group>
    );
}

// ==========================================
// MAIN EXPORT
// ==========================================
export default function Furniture() {
    return (
        <group>
            <CommonArea />
            <PrivateArea />
            <UtilityCloset />
            <Doors />
        </group>
    );
}
