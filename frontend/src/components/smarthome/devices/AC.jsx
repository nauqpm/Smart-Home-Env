import React from 'react';
import { Html } from '@react-three/drei';

export default function AC({ on, power, createMat }) {
  return (
    <group position={[5.8, 2.8, -2.5]} rotation={[0, -Math.PI / 2, 0]}>
      <mesh castShadow>
        <boxGeometry args={[1.0, 0.35, 0.25]} />
        <primitive object={createMat(0xf0f0f0)} attach="material" />
        <mesh scale={[1.1, 1.1, 1.1]}>
          <boxGeometry args={[1.0, 0.35, 0.25]} />
          <meshBasicMaterial color={0x64ffda} side={2} />
        </mesh>
        <Html position={[0, 0.3, 0]} style={{ visibility: 'visible' }}>
          <div className="label">{on ? `AC: ${power.toFixed(2)} kW` : 'AC: OFF'}</div>
        </Html>
      </mesh>
    </group>
  );
}

