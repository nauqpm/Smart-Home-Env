import React, { useMemo } from 'react';
import * as THREE from 'three';
import { makeTileMat, makeWallMat, makeWoodFloorMat } from './materials';

// ==========================================
// CONSTANTS
// ==========================================
const WALL_H = 2.5;   // Wall height (meters)
const WALL_T = 0.15;  // Wall thickness (meters)

// Footprint: 8m (x: -4 to 4) × 7.5m (z: -3.75 to 3.75)
// Origin at center [0, 0, 0]

// ==========================================
// TYPE DEFINITIONS
// ==========================================
type WallDef = {
    id: string;
    w: number; h: number; d: number;
    x: number; y: number; z: number;
    ry?: number;
    isWindow?: boolean;
};

type FloorDef = {
    id: string;
    size: [number, number, number];
    pos: [number, number, number];
    matType: 'wood' | 'tile';
};

// ==========================================
// WALLS ARRAY - Seamless corners (overlap by WALL_T/2)
// ==========================================
const WALLS: WallDef[] = [
    // ========================================
    // PERIMETER WALLS
    // ========================================
    // North wall (z = 3.75)
    { id: 'perimeter-north', w: 8 + WALL_T, h: WALL_H, d: WALL_T, x: 0, y: WALL_H / 2, z: 3.75 + WALL_T / 2 },
    // South wall (z = -3.75)
    { id: 'perimeter-south', w: 8 + WALL_T, h: WALL_H, d: WALL_T, x: 0, y: WALL_H / 2, z: -3.75 - WALL_T / 2 },
    // West wall (x = -4)
    { id: 'perimeter-west', w: WALL_T, h: WALL_H, d: 7.5 + WALL_T, x: -4 - WALL_T / 2, y: WALL_H / 2, z: 0 },
    // East wall (x = 4)
    { id: 'perimeter-east', w: WALL_T, h: WALL_H, d: 7.5 + WALL_T, x: 4 + WALL_T / 2, y: WALL_H / 2, z: 0 },

    // ========================================
    // ZONE A/B DIVIDER (x = 0, Private/Common separation)
    // ========================================
    // Living/Dining divider from Master Bedroom (z: 1 to 3.75)
    // NO door - full wall to living room
    { id: 'divider-ab-top', w: WALL_T, h: WALL_H, d: 2.75, x: 0, y: WALL_H / 2, z: 2.375 },

    // Divider Small Bedroom / Kitchen-Dining (z: -3.75 to -0.5)
    // NO door - full wall separating bedroom from kitchen
    { id: 'divider-ab-bot', w: WALL_T, h: WALL_H, d: 3.25, x: 0, y: WALL_H / 2, z: -2.125 },

    // ========================================
    // BATHROOM WALLS (x: -4 to -2.5, z: -3 to 1) - EXPANDED
    // ========================================
    // North wall (z = 1)
    { id: 'bath-north', w: 1.5 + WALL_T, h: WALL_H, d: WALL_T, x: -3.25, y: WALL_H / 2, z: 1 },
    // South wall (z = -3) - expanded from -0.5
    { id: 'bath-south', w: 1.5 + WALL_T, h: WALL_H, d: WALL_T, x: -3.25, y: WALL_H / 2, z: -3 },
    // East wall (x = -2.5) - door opening from z=0.05 to z=0.6 (width 0.55m)
    { id: 'bath-east-top', w: WALL_T, h: WALL_H, d: 0.4, x: -2.5, y: WALL_H / 2, z: 0.8 },
    { id: 'bath-east-mid', w: WALL_T, h: WALL_H, d: 2.45, x: -2.5, y: WALL_H / 2, z: -1.725 },
    { id: 'bath-east-bot', w: WALL_T, h: WALL_H, d: 0.55, x: -2.5, y: WALL_H / 2, z: -0.225 },

    // ========================================
    // MASTER BEDROOM SOUTH WALL (z = 1, x: -2.5 to 0)
    // Door opening at x: -1.5 to -0.7 (access from hallway, not living room)
    // ========================================
    { id: 'master-south-left', w: 1.0, h: WALL_H, d: WALL_T, x: -2.0, y: WALL_H / 2, z: 1 },
    { id: 'master-south-right', w: 0.7, h: WALL_H, d: WALL_T, x: -0.35, y: WALL_H / 2, z: 1 },

    // ========================================
    // SMALL BEDROOM WALLS (x: -2.5 to 0, z: -3.75 to -0.5)
    // Shifted to the east, loggia area removed
    // ========================================
    // North wall (z = -0.5) - door at x: -0.9 to -0.1
    { id: 'small-north-left', w: 1.6, h: WALL_H, d: WALL_T, x: -1.7, y: WALL_H / 2, z: -0.5 },
    { id: 'small-north-right', w: 0.1, h: WALL_H, d: WALL_T, x: -0.05, y: WALL_H / 2, z: -0.5 },

    // ========================================
    // UTILITY CLOSET IN SMALL BEDROOM (corner for Battery + Solar Inverter)
    // Small walled corner: x: -2.5 to -1.8, z: -3.75 to -3
    // ========================================
    { id: 'utility-north', w: 0.7, h: WALL_H, d: WALL_T, x: -2.15, y: WALL_H / 2, z: -3 },
    { id: 'utility-east', w: WALL_T, h: WALL_H, d: 0.75, x: -1.8, y: WALL_H / 2, z: -3.375 },

    // ========================================
    // KITCHEN/DINING DIVIDER (z = -1.5, x: 0 to 4)
    // Open plan - no wall, just floor transition
    // ========================================

    // ========================================
    // BALCONY RAILING (z = 3.75, x: 2 to 4)
    // Glass railing, shorter height
    // ========================================
    { id: 'balcony-rail', w: 2, h: 1.0, d: WALL_T, x: 3, y: 0.5, z: 4.25 },
];

// ==========================================
// FLOORS ARRAY - Zone-specific textures
// ==========================================
const FLOORS: FloorDef[] = [
    // Zone A: Common Area (Right Side) - TILE
    { id: 'living-dining-kitchen', size: [4, 0.02, 7.5], pos: [2, 0.11, 0], matType: 'tile' },

    // Balcony - TILE
    { id: 'balcony', size: [2, 0.02, 0.5], pos: [3, 0.11, 4], matType: 'tile' },

    // Zone B: Private Area - WOOD for bedrooms, TILE for bathroom
    { id: 'master-bedroom', size: [4, 0.15, 2.75], pos: [-2, 0.075, 2.375], matType: 'wood' },
    // Small bedroom shifted: x: -2.5 to 0, z: -3.75 to -0.5 (minus utility corner)
    { id: 'small-bedroom', size: [2.5, 0.15, 3.25], pos: [-1.25, 0.075, -2.125], matType: 'wood' },
    // Bathroom expanded: x: -4 to -2.5, z: -3 to 1
    { id: 'bathroom', size: [1.5, 0.02, 4.0], pos: [-3.25, 0.11, -1.0], matType: 'tile' },
    // Utility closet in small bedroom corner
    { id: 'utility-closet', size: [0.7, 0.02, 0.75], pos: [-2.15, 0.11, -3.375], matType: 'tile' },
];

// ==========================================
// WALL COMPONENT
// ==========================================
function Wall({ def, material }: { def: WallDef; material: THREE.Material }) {
    return (
        <mesh
            position={[def.x, def.y, def.z]}
            rotation={[0, def.ry || 0, 0]}
            material={material}
            castShadow
            receiveShadow
        >
            <boxGeometry args={[def.w, def.h, def.d]} />
        </mesh>
    );
}

// ==========================================
// FLOOR COMPONENT
// ==========================================
function Floor({ def, woodMat, tileMat }: { def: FloorDef; woodMat: THREE.Material; tileMat: THREE.Material }) {
    const material = def.matType === 'wood' ? woodMat : tileMat;
    return (
        <mesh position={def.pos} receiveShadow>
            <boxGeometry args={def.size} />
            <primitive object={material} attach="material" />
        </mesh>
    );
}

// ==========================================
// MAIN FLOORPLAN COMPONENT
// ==========================================
export default function FloorPlan() {
    const woodMat = useMemo(() => makeWoodFloorMat(), []);
    const tileMat = useMemo(() => makeTileMat(), []);
    const wallMat = useMemo(() => makeWallMat(), []);

    // Ceiling trim material (off-white)
    const trimMat = useMemo(() => new THREE.MeshStandardMaterial({
        color: 0xf5f5f0,
        roughness: 0.3,
        metalness: 0.0,
    }), []);

    // Ceiling trim dimensions
    const TRIM_H = 0.08;  // Height (8cm)
    const TRIM_D = 0.04;  // Depth/thickness (4cm)
    const TRIM_Y = WALL_H - TRIM_H / 2;  // Position at top of wall

    return (
        <group>
            {/* Render all floor slabs */}
            {FLOORS.map((floor) => (
                <Floor key={floor.id} def={floor} woodMat={woodMat} tileMat={tileMat} />
            ))}

            {/* Render all walls */}
            {WALLS.map((wall) => (
                <Wall key={wall.id} def={wall} material={wallMat} />
            ))}

            {/* ============================================ */}
            {/* CEILING TRIM - Continuous molding at wall-ceiling junction */}
            {/* ============================================ */}
            <group name="ceiling-trim">
                {/* Perimeter trim - North wall */}
                <mesh position={[0, TRIM_Y, 3.75 + WALL_T / 2 - TRIM_D / 2]}>
                    <boxGeometry args={[8 + WALL_T, TRIM_H, TRIM_D]} />
                    <primitive object={trimMat} attach="material" />
                </mesh>

                {/* Perimeter trim - South wall */}
                <mesh position={[0, TRIM_Y, -3.75 - WALL_T / 2 + TRIM_D / 2]}>
                    <boxGeometry args={[8 + WALL_T, TRIM_H, TRIM_D]} />
                    <primitive object={trimMat} attach="material" />
                </mesh>

                {/* Perimeter trim - West wall */}
                <mesh position={[-4 - WALL_T / 2 + TRIM_D / 2, TRIM_Y, 0]}>
                    <boxGeometry args={[TRIM_D, TRIM_H, 7.5 + WALL_T]} />
                    <primitive object={trimMat} attach="material" />
                </mesh>

                {/* Perimeter trim - East wall */}
                <mesh position={[4 + WALL_T / 2 - TRIM_D / 2, TRIM_Y, 0]}>
                    <boxGeometry args={[TRIM_D, TRIM_H, 7.5 + WALL_T]} />
                    <primitive object={trimMat} attach="material" />
                </mesh>

                {/* Zone divider trim - Upper section */}
                <mesh position={[TRIM_D / 2, TRIM_Y, 2.375]}>
                    <boxGeometry args={[TRIM_D, TRIM_H, 2.75]} />
                    <primitive object={trimMat} attach="material" />
                </mesh>
                <mesh position={[-TRIM_D / 2, TRIM_Y, 2.375]}>
                    <boxGeometry args={[TRIM_D, TRIM_H, 2.75]} />
                    <primitive object={trimMat} attach="material" />
                </mesh>

                {/* Zone divider trim - Lower section */}
                <mesh position={[TRIM_D / 2, TRIM_Y, -2.125]}>
                    <boxGeometry args={[TRIM_D, TRIM_H, 3.25]} />
                    <primitive object={trimMat} attach="material" />
                </mesh>
                <mesh position={[-TRIM_D / 2, TRIM_Y, -2.125]}>
                    <boxGeometry args={[TRIM_D, TRIM_H, 3.25]} />
                    <primitive object={trimMat} attach="material" />
                </mesh>

                {/* Bathroom trim - North */}
                <mesh position={[-3.25, TRIM_Y, 1 - TRIM_D / 2]}>
                    <boxGeometry args={[1.5 + WALL_T, TRIM_H, TRIM_D]} />
                    <primitive object={trimMat} attach="material" />
                </mesh>

                {/* Bathroom trim - South */}
                <mesh position={[-3.25, TRIM_Y, -3 + TRIM_D / 2]}>
                    <boxGeometry args={[1.5 + WALL_T, TRIM_H, TRIM_D]} />
                    <primitive object={trimMat} attach="material" />
                </mesh>

                {/* Bathroom trim - East side */}
                <mesh position={[-2.5 + TRIM_D / 2, TRIM_Y, -1]}>
                    <boxGeometry args={[TRIM_D, TRIM_H, 4]} />
                    <primitive object={trimMat} attach="material" />
                </mesh>

                {/* Master bedroom south wall trim */}
                <mesh position={[-1.25, TRIM_Y, 1 + TRIM_D / 2]}>
                    <boxGeometry args={[2.5, TRIM_H, TRIM_D]} />
                    <primitive object={trimMat} attach="material" />
                </mesh>

                {/* Small bedroom north wall trim */}
                <mesh position={[-1.25, TRIM_Y, -0.5 - TRIM_D / 2]}>
                    <boxGeometry args={[2.5, TRIM_H, TRIM_D]} />
                    <primitive object={trimMat} attach="material" />
                </mesh>

                {/* Utility closet trim */}
                <mesh position={[-2.15, TRIM_Y, -3 - TRIM_D / 2]}>
                    <boxGeometry args={[0.7, TRIM_H, TRIM_D]} />
                    <primitive object={trimMat} attach="material" />
                </mesh>
                <mesh position={[-1.8 + TRIM_D / 2, TRIM_Y, -3.375]}>
                    <boxGeometry args={[TRIM_D, TRIM_H, 0.75]} />
                    <primitive object={trimMat} attach="material" />
                </mesh>
            </group>
        </group>
    );
}
