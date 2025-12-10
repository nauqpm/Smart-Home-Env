import React from 'react';
import { Html } from '@react-three/drei';

export default function Heater({ on, power, createMat }) {
  return (
    <group position={[-5.5, 0, -1.5]}>
      <mesh castShadow position={[0, 0.45 / 2 + 0.15, 0]}>
        <boxGeometry args={[0.6, 0.45, 0.35]} />
        <primitive object={createMat(0xc0392b)} attach="material" />
        <mesh scale={[1.1, 1.1, 1.1]}>
          <boxGeometry args={[0.6, 0.45, 0.35]} />
          <meshBasicMaterial color={0xff8c00} side={2} />
        </mesh>
        <Html position={[0, 0.4, 0]} style={{ visibility: 'visible' }}>
          <div className="label">{on ? `Sưởi: ${power.toFixed(2)} kW` : 'Sưởi: OFF'}</div>
        </Html>
      </mesh>
    </group>
  );
}

