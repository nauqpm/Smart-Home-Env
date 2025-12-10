import React, { useMemo } from 'react';
import { Html } from '@react-three/drei';
import * as THREE from 'three';

export default function Laptop({ on, power, createMat }) {
  const baseMat = useMemo(() => createMat(0x555555), [createMat]);
  const screenMat = useMemo(() => createMat(0x222222), [createMat]);

  useMemo(() => {
    screenMat.emissive = new THREE.Color(on ? 0x9999ff : 0x000000);
  }, [screenMat, on]);

  return (
    <group position={[4.0, 0.15 + 0.7 + 0.03 / 2, -4.0]} rotation={[0, Math.PI / 4, 0]}>
      <mesh castShadow>
        <boxGeometry args={[0.4, 0.03, 0.3]} />
        <primitive object={baseMat} attach="material" />
      </mesh>
      <mesh position={[0, 0.15, -0.14]} rotation={[-Math.PI / 6, 0, 0]} castShadow>
        <boxGeometry args={[0.4, 0.3, 0.02]} />
        <primitive object={screenMat} attach="material" />
      </mesh>
      <mesh scale={[1.2, 1.2, 1.2]}>
        <boxGeometry args={[0.4, 0.3, 0.02]} />
        <meshBasicMaterial color={0x64ffda} side={THREE.BackSide} />
      </mesh>
      <Html position={[0, 0.4, 0]} style={{ visibility: 'visible' }}>
        <div className="label">{on ? `Laptop: ${power.toFixed(2)} kW` : 'Laptop: OFF'}</div>
      </Html>
    </group>
  );
}

