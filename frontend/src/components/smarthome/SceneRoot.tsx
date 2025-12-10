import React, { Suspense, useMemo, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import FloorPlan from './FloorPlan';
import Furniture from './Furniture';
import Appliances from './Appliances';
import Environment from './Environment';
import Overlay from './Overlay';
import { useStore } from '../../stores/useStore';

const cameraPos: [number, number, number] = [14, 12, 14];

export default function SceneRoot() {
  const { isNight, tick } = useStore();

  // Heartbeat System: Simulates fluctuation every 1 second
  useEffect(() => {
    const interval = setInterval(() => {
      tick();
    }, 1000);
    return () => clearInterval(interval);
  }, [tick]);

  const lightSettings = useMemo(
    () => ({
      ambient: isNight ? 0.35 : 0.6,
      hemi: isNight ? 0.3 : 0.6,
      sun: isNight ? 0 : 1.1,
      background: isNight ? '#0f1424' : '#cfe5ff',
    }),
    [isNight]
  );

  return (
    <Canvas
      shadows
      camera={{ position: cameraPos, fov: 45, near: 0.1, far: 100 }}
      onCreated={({ gl }) => {
        gl.setClearColor(lightSettings.background);
      }}
    >
      <color attach="background" args={[lightSettings.background]} />
      <ambientLight intensity={lightSettings.ambient} />
      <hemisphereLight args={['#ffffff', '#dddddd', lightSettings.hemi]} />

      {/* Global Dashboard Overlay */}
      <Overlay />

      <Suspense fallback={null}>
        <Environment />
        <group position={[0, -0.5, 0]}>
          <FloorPlan />
          <Furniture />
          <Appliances />
        </group>
      </Suspense>

      <OrbitControls
        enableDamping
        dampingFactor={0.08}
        minDistance={8}
        maxDistance={28}
        maxPolarAngle={Math.PI / 2.2}
      />
    </Canvas>
  );
}

