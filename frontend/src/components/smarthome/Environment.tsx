import React, { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useStore } from '../../stores/useStore';

export default function Environment() {
  const { isNight } = useStore();
  const sunRef = useRef<THREE.Mesh>(null);

  // Tạo danh sách các nhóm mây
  const clouds = useMemo(() => {
    const list: THREE.Group[] = [];

    // Vật liệu mây (Màu trắng)
    const cloudMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.8, // Tăng lên 0.8 cho rõ
      depthWrite: false,
    });

    for (let i = 0; i < 20; i++) { // Tăng lên 20 đám mây
      const g = new THREE.Group();

      // Tạo hình dáng mây từ 6 khối cầu
      for (let j = 0; j < 6; j++) {
        const size = 3 + Math.random() * 3;
        const m = new THREE.Mesh(new THREE.SphereGeometry(size, 8, 8), cloudMat);
        m.position.set(
          (Math.random() - 0.5) * 10,
          (Math.random() - 0.5) * 5,
          (Math.random() - 0.5) * 8
        );
        g.add(m);
      }

      // --- THUẬT TOÁN DONUT (VÙNG CẤM BAY) ---
      // 1. Góc ngẫu nhiên
      const angle = Math.random() * Math.PI * 2;

      // 2. Bán kính RẤT XA: Từ 45m đến 95m (Nhà an toàn tuyệt đối)
      const radius = 45 + Math.random() * 50;

      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;

      // 3. Độ cao: Bay tít trên cao (20m - 40m)
      const y = 20 + Math.random() * 20;

      g.position.set(x, y, z);

      // Xoay ngẫu nhiên đám mây cho tự nhiên
      g.rotation.y = Math.random() * Math.PI;

      list.push(g);
    }
    return list;
  }, []); // Bỏ dependency [lightsOn] để mây không bị reset liên tục

  // Hiệu ứng bầu trời xoay nhẹ
  const groupRef = useRef<THREE.Group>(null);
  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.rotation.y += 0.0002;
    }
  });

  return (
    <group>
      {/* 1. Ground - Mặt đất */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
        <circleGeometry args={[100, 64]} />
        <meshStandardMaterial color="#588157" />
      </mesh>

      {/* 2. Synced Sun & Light System */}
      <group position={[50, 80, 50]}>
        {/* Sun Mesh */}
        <mesh ref={sunRef} visible={!isNight}>
          <sphereGeometry args={[8, 32, 32]} />
          <meshBasicMaterial color={0xffe28a} />
        </mesh>

        {/* Directional Light */}
        <directionalLight
          intensity={isNight ? 0 : 1.1}
          castShadow
          shadow-mapSize={[2048, 2048]}
          shadow-camera-left={-50}
          shadow-camera-right={50}
          shadow-camera-top={50}
          shadow-camera-bottom={-50}
        />
      </group>

      {/* Render Mây - Dùng primitive object={c} SẼ KHÔNG BỊ LỖI MẤT NHÀ */}
      <group ref={groupRef} visible={!isNight}>
        {clouds.map((c, i) => (
          <primitive key={i} object={c} />
        ))}
      </group>
    </group>
  );
}
