import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, ContactShadows } from '@react-three/drei';
import HouseModel from './HouseModel';

function Scene3D() {
  return (
    // Canvas là nơi mọi thứ 3D diễn ra
    <Canvas shadows camera={{ position: [5, 4, 5], fov: 50 }}>
      <color attach="background" args={['#1e1e2e']} /> {/* Màu nền tối */}
      
      {/* --- Ánh sáng môi trường --- */}
      <ambientLight intensity={0.4} />
      <directionalLight
        castShadow
        position={[10, 10, 5]}
        intensity={1}
        shadow-mapSize={[1024, 1024]}
      />
      {/* Môi trường giả lập thành phố để tạo phản xạ đẹp */}
      <Environment preset="night" />

      {/* --- Điều khiển Camera (xoay, zoom) --- */}
      <OrbitControls makeDefault minPolarAngle={0} maxPolarAngle={Math.PI / 2} />

      {/* --- Model ngôi nhà --- */}
      <Suspense fallback={null}>
        <HouseModel />
      </Suspense>
      
      {/* Tạo bóng đổ dưới sàn cho thật */}
      <ContactShadows position={[0, -2, 0]} opacity={0.5} scale={10} blur={1.5} far={4.5} />
    </Canvas>
  );
}

export default Scene3D;