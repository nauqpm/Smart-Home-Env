import React, { useMemo } from 'react';
import * as THREE from 'three';
import { makeWoodFloorMat } from './materials';

// --- MATERIALS HELPER ---
const grayMat = (tone = 0x888888) =>
  new THREE.MeshStandardMaterial({ color: tone, roughness: 0.6, metalness: 0.1 });
const woodMat = () => makeWoodFloorMat();
const fabricMat = (color: number) =>
  new THREE.MeshStandardMaterial({ color, roughness: 0.9, metalness: 0.05 });
const plantGreen = new THREE.MeshStandardMaterial({ color: 0x4caf50, roughness: 0.8 });
const potClay = new THREE.MeshStandardMaterial({ color: 0x8d6e63, roughness: 0.9 });
const metalSilver = new THREE.MeshStandardMaterial({ color: 0xcccccc, roughness: 0.3, metalness: 0.7 });
const blackPlastic = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.5 });

// --- FURNITURE COMPONENTS ---

function SofaSet() {
  const base = useMemo(() => grayMat(0x8d8d93), []);
  const back = useMemo(() => grayMat(0x7a7a80), []);
  const rugMat = useMemo(() => fabricMat(0x555555), []);

  return (
    // FIX: Moved Sofa to x=1.0 to avoid clipping wall at x=0
    <group position={[1.0, 0.25, 0.5]} rotation={[0, Math.PI / 2, 0]}>
      {/* Rug */}
      <mesh position={[0, -0.24, 0.5]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[3.5, 2.5]} />
        <primitive attach="material" object={rugMat} />
      </mesh>

      {/* Sofa */}
      <mesh position={[0, 0.25, 0]} castShadow receiveShadow>
        <boxGeometry args={[2.4, 0.5, 1]} />
        <primitive attach="material" object={base} />
      </mesh>
      <mesh position={[0, 0.75, -0.45]} castShadow receiveShadow>
        <boxGeometry args={[2.4, 0.5, 0.1]} />
        <primitive attach="material" object={back} />
      </mesh>

      {/* Potted Plant */}
      <group position={[1.8, 0, 0.5]}>
        <mesh position={[0, 0.3, 0]} castShadow>
          <cylinderGeometry args={[0.2, 0.15, 0.4, 16]} />
          <primitive attach="material" object={potClay} />
        </mesh>
        <mesh position={[0, 0.7, 0]} castShadow>
          <dodecahedronGeometry args={[0.3]} />
          <primitive attach="material" object={plantGreen} />
        </mesh>
      </group>
    </group>
  );
}

function TVStand() {
  const mat = useMemo(() => grayMat(0x555555), []);
  // FIX: Moved to x=1.2 to probably avoid wall at x=2 clipping?
  // Wall x=2. TV Stand at x=1.2, width 1.8 (rotated). depth 0.5 along X.
  // x range: 1.2 +/- 0.25 = 0.95 to 1.45. Safe from x=2.
  return (
    <group position={[1.2, 0.35, 0.5]} rotation={[0, -Math.PI / 2, 0]}>
      <mesh castShadow receiveShadow>
        <boxGeometry args={[1.8, 0.3, 0.5]} />
        <primitive attach="material" object={mat} />
      </mesh>
    </group>
  );
}

function Bed({ position, coverColor, rotation = [0, 0, 0] }: any) {
  const frame = useMemo(() => woodMat(), []);
  const cover = useMemo(() => fabricMat(coverColor), [coverColor]);
  const pillow = useMemo(() => fabricMat(0xf5f5f5), []);
  return (
    <group position={position} rotation={rotation}>
      <mesh castShadow receiveShadow position={[0, 0.25, 0]}>
        <boxGeometry args={[2, 0.5, 3]} />
        <primitive attach="material" object={frame} />
      </mesh>
      <mesh castShadow receiveShadow position={[0, 0.6, 0]}>
        <boxGeometry args={[1.8, 0.25, 2.2]} />
        <primitive attach="material" object={cover} />
      </mesh>
      <mesh castShadow receiveShadow position={[0, 0.75, -0.7]}>
        <boxGeometry args={[1.6, 0.15, 0.5]} />
        <primitive attach="material" object={pillow} />
      </mesh>
    </group>
  );
}

function Wardrobe({ position }: { position: [number, number, number] }) {
  const mat = useMemo(() => woodMat(), []);
  return (
    <mesh position={position} castShadow receiveShadow>
      <boxGeometry args={[0.6, 2.4, 1.2]} />
      <primitive attach="material" object={mat} />
    </mesh>
  );
}

function DiningSet() {
  const topMat = useMemo(() => woodMat(), []);
  const chairMat = useMemo(() => grayMat(0xb0b0b0), []);
  return (
    <group position={[-5, 0.35, 2.4]}>
      <mesh castShadow><boxGeometry args={[1.8, 0.15, 0.9]} /><primitive attach="material" object={topMat} /></mesh>
      <mesh position={[0, -0.35, 0]}><boxGeometry args={[0.2, 0.7, 0.2]} /><primitive attach="material" object={chairMat} /></mesh>
    </group>
  )
}
function KitchenCounters() {
  const counterMat = useMemo(() => grayMat(0xd8d8d8), []);
  const cabinetMat = useMemo(() => woodMat(), []);
  return (
    <group>
      {/* 
          FIX: Moved x from -6.5 to -6.0 to avoid exterior wall clipping (x=-8).
          Width 3.5 -> Extent [-7.75, -4.25]. Wall is at -8. Safe.
      */}
      {/* Base Counter */}
      <mesh position={[-6.0, 0.45, -1.5]} castShadow><boxGeometry args={[3.5, 0.9, 0.6]} /><primitive attach="material" object={counterMat} /></mesh>

      {/* Refrigerator Removed - Moving to Appliances.tsx */}

      {/* Upper Cabinets - Floating on wall x=-8? */}
      {/* Aligned with counter */}
      <mesh position={[-6.0, 2.2, -1.5]} castShadow>
        <boxGeometry args={[3.5, 0.8, 0.4]} />
        <primitive attach="material" object={cabinetMat} />
      </mesh>
    </group>
  );
}

function BathFixtures() {
  const tubMat = useMemo(() => grayMat(0xe5e5e5), []);
  return (
    <group position={[-6.75, 0.45, 0.25]}>
      <mesh castShadow position={[0, 0.35, -0.8]}><boxGeometry args={[1.2, 0.6, 0.8]} /><primitive attach="material" object={tubMat} /></mesh>
    </group>
  )
}

function DoorFrame({ pos, rot = [0, 0, 0], size = [1, 2.2, 0.25] }: any) {
  const frameMat = useMemo(() => grayMat(0x333333), []);
  // Simple 3-piece frame? Just a box outline for low poly
  // Top
  return (
    <group position={pos} rotation={rot}>
      {/* Top Lintle */}
      <mesh position={[0, size[1] / 2, 0]} castShadow>
        <boxGeometry args={[size[0] + 0.1, 0.1, size[2]]} />
        <primitive attach="material" object={frameMat} />
      </mesh>
      {/* Sides */}
      <mesh position={[-(size[0] / 2), 0, 0]} castShadow>
        <boxGeometry args={[0.05, size[1], size[2]]} />
        <primitive attach="material" object={frameMat} />
      </mesh>
      <mesh position={[(size[0] / 2), 0, 0]} castShadow>
        <boxGeometry args={[0.05, size[1], size[2]]} />
        <primitive attach="material" object={frameMat} />
      </mesh>
    </group>
  )
}

function OfficeSetup() {
  const deskMat = useMemo(() => woodMat(), []);
  const chairMat = useMemo(() => grayMat(0x222222), []);
  // Guest Room Corner: x=7.5, z=-5.5
  return (
    <group position={[7.0, 0, -5.0]} rotation={[0, -Math.PI / 2, 0]}>
      <mesh position={[0, 0.37, 0]} castShadow>
        <boxGeometry args={[1.4, 0.05, 0.6]} />
        <primitive attach="material" object={deskMat} />
      </mesh>
      {/* Legs */}
      <mesh position={[-0.6, 0.18, 0.25]}><boxGeometry args={[0.05, 0.36, 0.05]} /><primitive attach="material" object={deskMat} /></mesh>
      <mesh position={[0.6, 0.18, 0.25]}><boxGeometry args={[0.05, 0.36, 0.05]} /><primitive attach="material" object={deskMat} /></mesh>
      <mesh position={[-0.6, 0.18, -0.25]}><boxGeometry args={[0.05, 0.36, 0.05]} /><primitive attach="material" object={deskMat} /></mesh>
      <mesh position={[0.6, 0.18, -0.25]}><boxGeometry args={[0.05, 0.36, 0.05]} /><primitive attach="material" object={deskMat} /></mesh>

      {/* Monitor */}
      <mesh position={[0, 0.55, -0.2]} rotation={[0, 0, 0]}>
        <boxGeometry args={[0.6, 0.35, 0.02]} />
        <primitive attach="material" object={blackPlastic} />
      </mesh>

      {/* Chair */}
      <mesh position={[0, 0.25, 0.5]} rotation={[0, Math.PI, 0]}>
        <boxGeometry args={[0.4, 0.05, 0.4]} />
        <primitive attach="material" object={chairMat} />
      </mesh>
      <mesh position={[0, 0.5, 0.7]} rotation={[0, Math.PI, 0]}>
        <boxGeometry args={[0.4, 0.5, 0.05]} />
        <primitive attach="material" object={chairMat} />
      </mesh>
    </group>
  )
}

function BalconySet() {
  const tableMat = useMemo(() => grayMat(0xffffff), []);
  // Balcony x=9 area
  return (
    <group position={[9.0, 0, 0]}>
      <mesh position={[0, 0.35, 0]} castShadow>
        <cylinderGeometry args={[0.4, 0.4, 0.05, 32]} />
        <primitive attach="material" object={tableMat} />
      </mesh>
      <mesh position={[0, 0.17, 0]}>
        <cylinderGeometry args={[0.05, 0.05, 0.35]} />
        <primitive attach="material" object={tableMat} />
      </mesh>
      {/* Chairs */}
      <mesh position={[0.6, 0.2, 0]}><boxGeometry args={[0.3, 0.05, 0.3]} /><primitive attach="material" object={tableMat} /></mesh>
      <mesh position={[-0.6, 0.2, 0]}><boxGeometry args={[0.3, 0.05, 0.3]} /><primitive attach="material" object={tableMat} /></mesh>
    </group>
  )
}

function PictureFrame({ pos, rot, size = [0.8, 1, 0.05], imgColor = 0x88ccff }: any) {
  const frameMat = useMemo(() => grayMat(0x222222), []);
  const canvasMat = useMemo(() => new THREE.MeshStandardMaterial({ color: imgColor, roughness: 0.8 }), [imgColor]);

  return (
    <group position={pos} rotation={rot}>
      {/* Frame */}
      <mesh castShadow>
        <boxGeometry args={[size[0], size[1], size[2]]} />
        <primitive attach="material" object={frameMat} />
      </mesh>
      {/* Canvas */}
      <mesh position={[0, 0, size[2] / 2 + 0.005]}>
        <planeGeometry args={[size[0] - 0.1, size[1] - 0.1]} />
        <primitive attach="material" object={canvasMat} />
      </mesh>
    </group>
  )
}

export default function Furniture() {
  return (
    <group>
      <SofaSet />
      <TVStand />
      <DiningSet />
      <KitchenCounters />
      <BathFixtures />
      <OfficeSetup />
      <BalconySet />

      {/* --- PHÒNG NGỦ MASTER (Góc dưới phải) --- */}
      {/* Nằm dưới tường z=0.8. Tọa độ an toàn: x=6.5, z=3.5 */}
      <Bed position={[6.5, 0.25, 3.5]} coverColor={0x7fcad3} />
      <Wardrobe position={[7.5, 1.2, 5.2]} />

      {/* --- PHÒNG NGỦ KHÁCH (Góc trên phải) --- */}
      {/* Nằm trên tường z=0.8. Tọa độ an toàn: x=6.5, z=-2.5 */}
      {/* FIX: Moved z from -2.5 to -2.0 (or -1.8) to avoid partition wall clipping?
          Partition is at z=-1 (between kitchen/wc) maybe? 
          Wait, partition "Ngăn Hành lang | Khu Ngủ (x=5)" is vertical.
          Partition "Tường ngăn ngang" x=6.5, z=0.8.
          Bed at z=-2.5 is safely away from z=0.8.
          Maybe clipping wall z=-6? Bounds Z: -2.5 +/- 1.5 = [-4, -1]. Wall -6 is far.
          Maybe clipping wall x=5? Bed X: 6.5 +/- 1 = [5.5, 7.5]. Safe from x=5.
          Maybe user meant "Room divider"?
          Let's try shifting away from wall, maybe z=-2.2.
      */}
      <Bed position={[6.5, 0.25, -2.2]} coverColor={0x8bcf8f} />
      <Wardrobe position={[7.5, 1.2, -5.2]} />

      {/* --- DOOR FRAMES --- */}
      {/* Main Door (z=6, x=0?) - House wall is z=6. Entrance probably around x=2 area? 
          Checking FloorPlan: Walls are Perim around x=0..16? No.
          Perim Walls: [-8 to 8]. Front z=6.
          Corridor is x=2 and x=5 lines.
          Door to Master Bed (x=5 wall, z=3.5? No, hole is z=0.8-ish?)
          Let's place a few logical frames.
      */}
      {/* Master Bedroom Door - Wall x=5, z=3.5 hole */}
      <DoorFrame pos={[5, 1.1, 3.5]} rot={[0, Math.PI / 2, 0]} size={[1, 2.2, 0.25]} />

      {/* Guest Bedroom Door - Wall x=5, z=-3.5 hole */}
      <DoorFrame pos={[5, 1.1, -3.5]} rot={[0, Math.PI / 2, 0]} size={[1, 2.2, 0.25]} />

      {/* Main Entrance - Front Wall z=6, maybe x=2 corridor? */}
      {/*<DoorFrame pos={[2, 1.1, 6]} size={[1.2, 2.2, 0.25]} />*/}

      {/* --- PICTURE FRAMES --- */}
      {/* Corridor Wall x=2, facing Living Room (x>2) */}
      <PictureFrame pos={[2.05, 1.6, 0]} rot={[0, Math.PI / 2, 0]} size={[0.8, 1.0, 0.05]} imgColor={0xffaa88} />

      {/* Living Room Back Wall z=-6, facing room (z>-6) */}
      <PictureFrame pos={[5, 1.6, -5.95]} rot={[0, 0, 0]} size={[1.5, 1.0, 0.05]} imgColor={0x88ccff} />
    </group>
  );
}
