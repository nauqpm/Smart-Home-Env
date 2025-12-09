import React, { useMemo } from 'react';
import { Html } from '@react-three/drei';
import * as THREE from 'three';

export default function TV({ on, power, createMat }) {
  const mat = useMemo(() => createMat(0x111111), [createMat]);
  useMemo(() => {
    mat.emissive = new THREE.Color(on ? 0x222255 : 0x000000);
  }, [mat, on]);

  return (
    <group>
      <mesh castShadow position={[0, 0.15 + 0.4 + 0.9 / 2, -4.7]} material={mat}>
        <boxGeometry args={[1.6, 0.9, 0.05]} />
        <mesh scale={[1.05, 1.05, 1.05]}>
          <boxGeometry args={[1.6, 0.9, 0.05]} />
          <meshBasicMaterial color={0x64ffda} side={THREE.BackSide} />
        </mesh>
        <Html position={[0, 0.6, 0.1]} style={{ visibility: 'visible' }}>
          <div className="label">{on ? `TV: ${power.toFixed(2)} kW` : 'TV: OFF'}</div>
        </Html>
      </mesh>
    </group>
  );
}

