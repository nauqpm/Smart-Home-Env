import React from 'react';
import { Html } from '@react-three/drei';

export default function WashingMachine({ on, power, createMat }) {
  return (
    <group position={[-4.0, 0, -4.5]}>
      <group position={[0, 0.8 / 2 + 0.15, 0]}>
        <mesh castShadow>
          <boxGeometry args={[0.6, 0.8, 0.6]} />
          <primitive object={createMat(0x95a5a6)} attach="material" />
        </mesh>
        <mesh position={[0, 0, 0.6 / 2 + 0.01]}>
          <circleGeometry args={[0.2, 16]} />
          <primitive object={createMat(0x333333)} attach="material" />
        </mesh>
        <mesh scale={[1.1, 1.1, 1.1]}>
          <boxGeometry args={[0.6, 0.8, 0.6]} />
          <meshBasicMaterial color={0x64ffda} side={2} />
        </mesh>
      </group>
      <Html position={[0, 0.5, 0]} style={{ visibility: 'visible' }}>
        <div className="label">{on ? `Máy giặt: ${power.toFixed(2)} kW` : 'Máy giặt: OFF'}</div>
      </Html>
    </group>
  );
}

