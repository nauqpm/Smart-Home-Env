import React from 'react';
import { Html } from '@react-three/drei';

export default function EVCharger({ on, power, createMat }) {
  return (
    <group position={[8.5, 0, 3]}>
      <mesh castShadow position={[-1, 0.8, 0]}>
        <boxGeometry args={[0.25, 0.5, 0.15]} />
        <primitive object={createMat(0x666666)} attach="material" />
        <mesh scale={[1.2, 1.2, 1.2]}>
          <boxGeometry args={[0.25, 0.5, 0.15]} />
          <meshBasicMaterial color={0xff6b6b} side={2} />
        </mesh>
        <Html position={[0.5, 1.0, 0]} style={{ visibility: 'visible' }}>
          <div className="label">{on ? `Sạc EV: ${power.toFixed(2)} kW` : 'Sạc EV: OFF'}</div>
        </Html>
      </mesh>
      <mesh castShadow position={[0.5, 0.7 / 2 + 0.1, 0]}>
        <boxGeometry args={[1.8, 0.7, 0.9]} />
        <primitive object={createMat(0xbf4040)} attach="material" />
      </mesh>
    </group>
  );
}

