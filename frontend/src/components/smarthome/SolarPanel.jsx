import React from 'react';
import { Html } from '@react-three/drei';

export default function SolarPanel({ pvPower }) {
  return (
    <group position={[-3.5, 4.4, -2.5]} rotation={[-0.4, Math.PI / 4, 0]}>
      <mesh castShadow>
        <boxGeometry args={[4, 0.1, 2.5]} />
        <meshStandardMaterial color={0x1a2536} metalness={0.8} roughness={0.2} />
      </mesh>
      <Html position={[0, 0.5, 0]} style={{ visibility: pvPower > 0.01 ? 'visible' : 'hidden' }}>
        <div className="label label-pv">{`PV: ${pvPower.toFixed(2)} kW`}</div>
      </Html>
    </group>
  );
}

