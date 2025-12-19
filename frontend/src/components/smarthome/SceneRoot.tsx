import React, { Suspense, useMemo, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, SoftShadows, OrthographicCamera } from '@react-three/drei';
import FloorPlan from './FloorPlan';
import Furniture from './Furniture';
import Appliances from './Appliances';
import Environment from './Environment';
import Overlay from './Overlay';
import RoomLights from './RoomLights';
import { useStore } from '../../stores/useStore';

// Orthographic camera settings for isometric view
// Zoom level calculated to frame 8x7.5m rectangle
const ZOOM = 55;

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
      orthographic
      camera={{
        zoom: ZOOM,
        position: [12, 10, 12],
        near: 0.1,
        far: 100,
      }}
      onCreated={({ gl }) => {
        gl.setClearColor(lightSettings.background);
      }}
    >
      <color attach="background" args={[lightSettings.background]} />
      <ambientLight intensity={lightSettings.ambient} />
      <hemisphereLight args={['#ffffff', '#dddddd', lightSettings.hemi]} />
      <SoftShadows size={12} focus={0.5} samples={12} />

      {/* Directional sunlight for shadows */}
      <directionalLight
        position={[10, 15, 10]}
        intensity={lightSettings.sun}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-camera-far={50}
        shadow-camera-left={-10}
        shadow-camera-right={10}
        shadow-camera-top={10}
        shadow-camera-bottom={-10}
      />

      {/* Global Dashboard Overlay */}
      <Overlay />

      <Suspense fallback={null}>
        <Environment />
        <RoomLights />
        <group position={[0, -0.5, 0]}>
          <FloorPlan />
          <Furniture />
          <Appliances />
        </group>
      </Suspense>

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minZoom={30}
        maxZoom={100}
        maxPolarAngle={Math.PI / 2.1}
        enableRotate={true}
        enablePan={true}
      />
    </Canvas>
  );
}
