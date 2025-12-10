import React from 'react';
import { useMemo } from 'react';
import * as THREE from 'three';

export default function Furniture({ createMat }) {
  const furnMat = useMemo(() => createMat(0x8b4513), [createMat]);
  const sofaMat = useMemo(() => createMat(0x555555), [createMat]);
  const bedMat = useMemo(() => createMat(0x444466), [createMat]);
  const tableMat = useMemo(() => createMat(0x967969), [createMat]);
  const deskMat = useMemo(() => createMat(0x7a5a4a), [createMat]);

  const Chair = ({ x, z, r }) => (
    <group position={[x, 0.15, z]} rotation={[0, r, 0]}>
      <mesh castShadow position={[0, 0.45 / 2, 0]}>
        <boxGeometry args={[0.4, 0.45, 0.4]} />
        <meshStandardMaterial color={0x967969} />
      </mesh>
      <mesh position={[0, 0.45 / 2 + 0.5 / 2, -0.175]}>
        <boxGeometry args={[0.4, 0.5, 0.05]} />
        <meshStandardMaterial color={0x967969} />
      </mesh>
    </group>
  );

  return (
    <group>
      <mesh castShadow position={[4, 0.7 / 2 + 0.15, -4]} material={sofaMat}>
        <boxGeometry args={[2.5, 0.7, 0.8]} />
      </mesh>
      <mesh castShadow position={[0, 0.4 / 2 + 0.15, -2.5]} material={tableMat}>
        <boxGeometry args={[1.2, 0.4, 0.6]} />
      </mesh>
      <mesh castShadow position={[-0.5, 0.7 / 2 + 0.15, -1.0]} material={sofaMat}>
        <boxGeometry args={[0.8, 0.7, 0.8]} />
      </mesh>
      <mesh castShadow position={[0.5, 0.7 / 2 + 0.15, -1.0]} material={sofaMat}>
        <boxGeometry args={[0.8, 0.7, 0.8]} />
      </mesh>
      <mesh castShadow position={[-4, 0.75 / 2 + 0.15, 3.5]} material={tableMat}>
        <boxGeometry args={[1.8, 0.75, 1.0]} />
      </mesh>
      <Chair x={-3.2} z={3.5} r={Math.PI / 2} />
      <Chair x={-4.8} z={3.5} r={-Math.PI / 2} />
      <Chair x={-4} z={2.8} r={0} />
      <Chair x={-4} z={4.2} r={Math.PI} />
      <mesh castShadow position={[-4.75, 0.8 / 2 + 0.15, 0.3]} material={deskMat}>
        <boxGeometry args={[2.5, 0.8, 0.6]} />
      </mesh>
      <mesh castShadow position={[-4, 0.5 / 2 + 0.15, -3.5]} material={bedMat}>
        <boxGeometry args={[1.5, 0.5, 2.0]} />
      </mesh>
      <mesh castShadow position={[-5.5, 0.7 / 2 + 0.15, -1.0]} material={furnMat}>
        <boxGeometry args={[0.8, 0.7, 0.5]} />
      </mesh>
      <mesh castShadow position={[4.0, 0.7 / 2 + 0.15, -4.0]} material={deskMat}>
        <boxGeometry args={[1.2, 0.7, 0.6]} />
      </mesh>
      <Chair x={5.0} z={-4.0} r={Math.PI / 1.5} />
      <mesh castShadow position={[4, 0.5 / 2 + 0.15, 3.5]} material={bedMat}>
        <boxGeometry args={[1.5, 0.5, 2.0]} />
      </mesh>
      <mesh castShadow position={[-5.5, 0.6 / 2 + 0.15, -4.5]}>
        <boxGeometry args={[0.4, 0.6, 0.5]} />
        <meshStandardMaterial color={0xffffff} />
      </mesh>
    </group>
  );
}

