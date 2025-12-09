import React, { useMemo } from 'react';
import { Html } from '@react-three/drei';
import * as THREE from 'three';

export default function Walls({ occupancy }) {
  const wallMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: 0xe0e0e0,
        transparent: true,
        opacity: 0.15,
        side: THREE.DoubleSide,
      }),
    []
  );
  const wallHeight = 3.0;
  const wallThick = 0.15;
  const walls = [
    { w: 12 + wallThick, d: wallThick, x: 0, z: -5, r: 0 },
    { w: 12 + wallThick, d: wallThick, x: 0, z: 5, r: 0 },
    { w: 10 + wallThick, d: wallThick, x: -6, z: 0, r: Math.PI / 2 },
    { w: 10 + wallThick, d: wallThick, x: 6, z: 0, r: Math.PI / 2 },
    { w: 5, d: wallThick, x: -3.5, z: 2.5, r: 0 },
    { w: 5, d: wallThick, x: -1, z: 0, r: Math.PI / 2 },
    { w: 5, d: wallThick, x: -2, z: -2.5, r: 0 },
    { w: 5, d: wallThick, x: 2, z: -2.5, r: 0 },
    { w: 2, d: wallThick, x: -5, z: -4, r: Math.PI / 2 },
    { w: 2, d: wallThick, x: -4, z: -3, r: 0 },
  ];
  return (
    <group>
      {walls.map((w, i) => (
        <mesh key={i} material={wallMat} position={[w.x, wallHeight / 2 + 0.15, w.z]} rotation-y={w.r}>
          <boxGeometry args={[w.w, wallHeight, w.d]} />
        </mesh>
      ))}
      <Html position={[-6.5, 2, 0]}>
        <div className="label label-occupancy">{`Số người: ${occupancy}`}</div>
      </Html>
    </group>
  );
}

