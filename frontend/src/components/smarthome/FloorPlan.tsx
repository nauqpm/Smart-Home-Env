import React, { useMemo } from 'react';
import * as THREE from 'three';
import { makeTileMat, makeWallMat, makeWoodFloorMat } from './materials';

type WallSeg = {
  w: number; h: number; d: number;
  x: number; y: number; z: number;
  ry?: number;
};

type FloorSlab = {
  size: [number, number, number]; // Bắt buộc là mảng 3 số
  pos: [number, number, number];  // Bắt buộc là mảng 3 số
  mat: THREE.Material;
};

const WALL_H = 3;
const WALL_T = 0.2;

function Wall({ seg, material }: { seg: WallSeg; material: THREE.Material }) {
  return (
    <mesh position={[seg.x, seg.y, seg.z]} rotation={[0, seg.ry || 0, 0]} material={material} castShadow receiveShadow>
      <boxGeometry args={[seg.w, seg.h, seg.d]} />
    </mesh>
  );
}

export default function FloorPlan() {
  const woodMat = useMemo(() => makeWoodFloorMat(), []);
  const tileMat = useMemo(() => makeTileMat(), []);
  const wallMat = useMemo(() => makeWallMat(), []);

  // Sàn nhà
  const floorSlabs: FloorSlab[] = [
    { size: [16, 0.15, 12], pos: [0, 0.075, 0], mat: woodMat }, // Sàn chính
    { size: [6, 0.02, 4.5], pos: [-5, 0.11, 2.25], mat: tileMat }, // Bếp
    { size: [2.5, 0.02, 2.5], pos: [-6.75, 0.12, 0.25], mat: tileMat }, // WC
    { size: [8, 0.02, 1.5], pos: [0, 0.11, 6.0], mat: tileMat }, // Ban công trước
    { size: [2, 0.02, 8], pos: [9, 0.11, 0], mat: tileMat }, // Ban công bên
  ];

  const walls: WallSeg[] = [
    // --- TƯỜNG BAO (PERIMETER) ---
    { w: 16, h: WALL_H, d: WALL_T, x: 0, y: WALL_H / 2, z: -6 }, // Sau
    { w: 16, h: WALL_H, d: WALL_T, x: 0, y: WALL_H / 2, z: 6 }, // Trước
    { w: WALL_T, h: WALL_H, d: 12, x: -8, y: WALL_H / 2, z: 0 }, // Trái
    { w: WALL_T, h: WALL_H, d: 12, x: 8, y: WALL_H / 2, z: 0 }, // Phải

    // --- TƯỜNG NGĂN DỌC (VERTICAL) ---
    // Ngăn Phòng Khách | Hành lang (x=2)
    { w: WALL_T, h: WALL_H, d: 12, x: 2, y: WALL_H / 2, z: 0 },

    // Ngăn Hành lang | Khu Ngủ (x=5) - QUAN TRỌNG: Chừa lối đi
    // Đoạn 1: Từ trên xuống cửa
    { w: WALL_T, h: WALL_H, d: 5, x: 5, y: WALL_H / 2, z: -3.5 },
    // Đoạn 2: Từ cửa xuống dưới
    { w: WALL_T, h: WALL_H, d: 5, x: 5, y: WALL_H / 2, z: 3.5 },

    // --- TƯỜNG NGĂN NGANG (HORIZONTAL) - CÁI BẠN ĐANG THIẾU ---
    // Bức tường chia đôi 2 phòng ngủ bên phải.
    // Tọa độ z=0.8. Nối từ tường x=5 đến x=8.
    { w: 3, h: WALL_H, d: WALL_T, x: 6.5, y: WALL_H / 2, z: 0.8 },

    // --- VÁCH BẾP & WC ---
    { w: 4, h: WALL_H, d: WALL_T, x: 0, y: WALL_H / 2, z: -2.5 }, // Hành lang trên
    { w: 4, h: WALL_H, d: WALL_T, x: 0, y: WALL_H / 2, z: 1.5 }, // Hành lang dưới
    { w: WALL_T, h: WALL_H, d: 4.5, x: -2, y: WALL_H / 2, z: 1.25 }, // Ngăn Bếp/Khách
    { w: 3, h: WALL_H, d: WALL_T, x: -6.5, y: WALL_H / 2, z: -1 }, // Ngăn Bếp/WC
    { w: 2.5, h: WALL_H, d: WALL_T, x: -6.75, y: WALL_H / 2, z: 1.5 }, // Lưng WC
    { w: WALL_T, h: WALL_H, d: 2.5, x: -8, y: WALL_H / 2, z: 0.25 }, // Hông WC

    // Lan can ban công
    { w: 8, h: 1.0, d: WALL_T, x: 0, y: 0.5, z: 6.0 },
    { w: WALL_T, h: 1.0, d: 8, x: 8, y: 0.5, z: 0 },
  ];

  return (
    <group>
      {floorSlabs.map((slab, i) => (
        <mesh key={`slab-${i}`} position={slab.pos} receiveShadow>
          <boxGeometry args={slab.size} />
          <primitive object={slab.mat} attach="material" />
        </mesh>
      ))}
      {walls.map((seg, i) => (
        <Wall key={`wall-${i}`} seg={seg} material={wallMat} />
      ))}
    </group>
  );
}