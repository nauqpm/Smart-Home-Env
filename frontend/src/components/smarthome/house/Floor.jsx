import React from 'react';

export default function Floor() {
  return (
    <mesh receiveShadow position={[0, 0.15, 0]}>
      <boxGeometry args={[12, 0.3, 10]} />
      <meshStandardMaterial color={0x9b7653} />
    </mesh>
  );
}

