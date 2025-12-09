import React from 'react';
import { Html } from '@react-three/drei';

export default function Fridge({ on, power, createMat }) {
  return (
    <group position={[-5.5, 0, 1.5]}>
      <mesh castShadow position={[0, 1.7 / 2 + 0.15, 0]}>
        <boxGeometry args={[0.8, 1.7, 0.7]} />
        <primitive object={createMat(0xeeeeee)} attach="material" />
        <mesh scale={[1.1, 1.1, 1.1]}>
          <boxGeometry args={[0.8, 1.7, 0.7]} />
          <meshBasicMaterial color={0x64ffda} side={2} />
        </mesh>
        <Html position={[0, 1.0, 0]} style={{ visibility: 'visible' }}>
          <div className="label">{on ? `Tủ lạnh: ${power.toFixed(2)} kW` : 'Tủ lạnh: OFF'}</div>
        </Html>
      </mesh>
    </group>
  );
}

