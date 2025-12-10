import React from 'react';
import { Html } from '@react-three/drei';

export default function Dishwasher({ on, power, createMat }) {
  return (
    <group position={[-4.0, 0, 0.3]}>
      <mesh castShadow position={[0, 0.75 / 2 + 0.15, 0]}>
        <boxGeometry args={[0.7, 0.75, 0.55]} />
        <primitive object={createMat(0xf9f9f9)} attach="material" />
        <mesh scale={[1.1, 1.1, 1.1]}>
          <boxGeometry args={[0.7, 0.75, 0.55]} />
          <meshBasicMaterial color={0x64ffda} side={2} />
        </mesh>
        <Html position={[0, 0.5, 0]} style={{ visibility: 'visible' }}>
          <div className="label">{on ? `Rửa bát: ${power.toFixed(2)} kW` : 'Rửa bát: OFF'}</div>
        </Html>
      </mesh>
    </group>
  );
}

