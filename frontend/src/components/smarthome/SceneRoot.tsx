import React, { Suspense, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, SoftShadows, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import FloorPlan from './FloorPlan';
import Furniture from './Furniture';
import Appliances from './Appliances';
import Environment from './Environment';
import Overlay from './Overlay';
import RoomLights from './RoomLights';
import { useStore } from '../../stores/useStore';

export default function SceneRoot() {
  const { isNight } = useStore();

  // Light settings based on day/night
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
    <>
      {/* 3D Canvas */}
      <Canvas
        shadows
        gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}
        onCreated={({ gl }) => {
          gl.setClearColor(lightSettings.background);
        }}
      >
        {/* Perspective Camera for better 3D feel */}
        <PerspectiveCamera
          makeDefault
          position={[10, 10, 10]}
          fov={45}
          near={0.1}
          far={200}
        />

        {/* Interactive Camera Controls */}
        <OrbitControls
          makeDefault
          enableDamping={true}
          dampingFactor={0.05}
          minDistance={5}
          maxDistance={30}
          minPolarAngle={Math.PI / 6}
          maxPolarAngle={Math.PI / 2.2}
          target={[0, 0, 0]}
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          panSpeed={0.8}
          rotateSpeed={0.6}
          zoomSpeed={1.0}
          screenSpacePanning={true}
        />

        {/* Background */}
        <color attach="background" args={[lightSettings.background]} />

        {/* Lighting */}
        <ambientLight intensity={lightSettings.ambient} />
        <hemisphereLight args={['#ffffff', '#dddddd', lightSettings.hemi]} />
        {/* Reduced samples from 12 to 6 for better performance */}
        <SoftShadows size={8} focus={0.5} samples={6} />

        {/* Directional sunlight for shadows - reduced shadow map for performance */}
        <directionalLight
          position={[10, 15, 10]}
          intensity={lightSettings.sun}
          castShadow
          shadow-mapSize-width={1024}
          shadow-mapSize-height={1024}
          shadow-camera-far={50}
          shadow-camera-left={-10}
          shadow-camera-right={10}
          shadow-camera-top={10}
          shadow-camera-bottom={-10}
        />

        {/* 3D Scene Content */}
        <Suspense fallback={null}>
          <Environment />
          <RoomLights />
          <group position={[0, -0.5, 0]}>
            <FloorPlan />
            <Furniture />
            <Appliances />
          </group>
        </Suspense>
      </Canvas>

      {/* UI Overlay - rendered OUTSIDE Canvas so it stays fixed when camera moves */}
      <Overlay />
    </>
  );
}
