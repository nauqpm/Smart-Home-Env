import React from 'react';

export default function GridConnection() {
  return (
    <mesh castShadow position={[12, 4, -8]}>
      <cylinderGeometry args={[0.15, 0.15, 8, 12]} />
      <meshStandardMaterial color={0x593d2b} />
    </mesh>
  );
}

