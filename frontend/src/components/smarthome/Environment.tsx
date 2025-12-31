import React, { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useStore } from '../../stores/useStore';

// Rain Particle System
function Rain() {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const count = 1000;
  const dummy = useMemo(() => new THREE.Object3D(), []);

  // Random Initial Positions
  const particles = useMemo(() => {
    const temp = [];
    for (let i = 0; i < count; i++) {
      const x = (Math.random() - 0.5) * 60;
      const y = Math.random() * 40;
      const z = (Math.random() - 0.5) * 60;
      const speed = 0.5 + Math.random() * 0.5;
      temp.push({ x, y, z, speed });
    }
    return temp;
  }, []);

  useFrame(() => {
    if (!meshRef.current) return;
    particles.forEach((p, i) => {
      // Fall down
      p.y -= p.speed;
      if (p.y < 0) p.y = 40; // Reset height

      dummy.position.set(p.x, p.y, p.z);
      dummy.scale.set(0.05, 0.8, 0.05); // Thin streaks
      dummy.updateMatrix();
      meshRef.current!.setMatrixAt(i, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
      <boxGeometry />
      <meshBasicMaterial color="#aabccd" transparent opacity={0.6} />
    </instancedMesh>
  );
}

export default function Environment() {
  const { isNight, weather } = useStore();
  const sunRef = useRef<THREE.Mesh>(null);

  // Lighting & Background
  // Sunny: Blue Sky, Bright Sun
  // Cloudy: Grey-Blue Sky, Soft Sun
  // Rainy/Stormy: Dark Grey Sky, Dim Light

  const intensity = useMemo(() => {
    if (isNight) return 0;
    switch (weather) {
      case 'sunny': return 1.5;
      case 'cloudy': return 0.8;
      case 'rainy': return 0.3;
      case 'stormy': return 0.1;
      default: return 1.0;
    }
  }, [weather, isNight]);

  // Sky Color
  const skyColor = useMemo(() => {
    if (isNight) return '#050510';
    switch (weather) {
      case 'sunny': return '#87CEEB';
      case 'cloudy': return '#bdc3c7';
      case 'rainy': return '#34495e';
      case 'stormy': return '#2c3e50';
      default: return '#87CEEB';
    }
  }, [weather, isNight]);

  return (
    <group>
      {/* GLOBAL BACKGROUND - SKY */}
      <color attach="background" args={[skyColor]} />

      {/* 1. Ground - Mặt đất (Fixed Green) */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
        <circleGeometry args={[100, 64]} />
        <meshStandardMaterial color="#4f772d" roughness={1} />
      </mesh>

      {/* 2. Synced Sun & Light System */}
      <group position={[50, 80, 50]}>
        {/* Sun Mesh */}
        <mesh ref={sunRef} visible={!isNight && weather === 'sunny'}>
          <sphereGeometry args={[8, 32, 32]} />
          <meshBasicMaterial color={0xffe28a} />
        </mesh>

        {/* Directional Light */}
        <directionalLight
          intensity={intensity}
          castShadow
          shadow-mapSize={[2048, 2048]}
          shadow-camera-left={-50}
          shadow-camera-right={50}
          shadow-camera-top={50}
          shadow-camera-bottom={-50}
        />
        <ambientLight intensity={isNight ? 0.2 : (weather === 'sunny' ? 0.6 : 0.2)} />
      </group>

      {/* Background (Sky) handled by parent or default for now, skipping direct scene.background modification to avoid side effects in verification */}

      {/* Rain Effect */}
      {(weather === 'rainy' || weather === 'stormy') && <Rain />}
    </group>
  );
}




