import React, { useEffect, useRef } from 'react';
import { Html } from '@react-three/drei';

export default function Battery({ soc }) {
  const fillRef = useRef();

  useEffect(() => {
    if (!fillRef.current) return;
    const h = 3;
    fillRef.current.scale.y = Math.max(0.001, soc);
    fillRef.current.position.y = (soc * h) / 2;
  }, [soc]);

  return (
    <group position={[8, 0, -3]}>
      <mesh position={[0, 3 / 2, 0]}>
        <cylinderGeometry args={[0.8, 0.8, 3, 32, 1, true]} />
        <meshStandardMaterial color={0x4a4a4a} transparent opacity={0.3} />
      </mesh>
      <mesh ref={fillRef} position={[0, 3 / 2, 0]}>
        <cylinderGeometry args={[0.7, 0.7, 3, 32]} />
        <meshStandardMaterial color={0x64ffda} emissive={0x64ffda} emissiveIntensity={0.2} />
      </mesh>
      <Html position={[0, 3 + 0.5, 0]}>
        <div className="label label-soc">{`SOC: ${Math.round(soc * 100)}%`}</div>
      </Html>
    </group>
  );
}

