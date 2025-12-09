import React, { useMemo } from 'react';
import { Html } from '@react-three/drei';
import * as THREE from 'three';

export default function Lights({ on, power }) {
  const geo = useMemo(() => new THREE.SphereGeometry(0.2, 16, 8), []);
  const mat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: 0xffd700,
        emissive: 0x000000,
        emissiveIntensity: 0,
      }),
    []
  );

  const renderLight = (pos) => (
    <group position={pos}>
      <mesh geometry={geo} material={mat} castShadow>
        <mesh scale={[1.15, 1.15, 1.15]}>
          <sphereGeometry args={[0.2, 16, 8]} />
          <meshBasicMaterial color={0x64ffda} side={THREE.BackSide} />
        </mesh>
        <pointLight intensity={on ? 1.5 : 0} color={0xffd700} distance={6} />
      </mesh>
    </group>
  );

  mat.emissiveIntensity = on ? 1 : 0;

  return (
    <group>
      {renderLight([2, 2.8, -2.5])}
      {renderLight([-4, 2.8, 2.5])}
      {renderLight([-4, 2.8, -2.5])}
      {renderLight([4, 2.8, 2.5])}
      <Html position={[2, 3.2, -2.5]} style={{ visibility: 'visible' }}>
        <div className="label label-soc">
          {on ? `Đèn (Tổng): ${power.toFixed(2)} kW` : 'Đèn: OFF'}
        </div>
      </Html>
    </group>
  );
}

