import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import useStore from '../stores/useStore';

function HouseModel() {
  // Ref để truy cập trực tiếp vào mesh 3D
  const fanRef = useRef();
  const lightBulbRef = useRef();

  // Lấy trạng thái từ Zustand store
  // Chỉ lấy những gì cần thiết để tối ưu hiệu năng
  const fanSpeed = useStore((state) => state.deviceState.fan_speed);
  const lightOn = useStore((state) => state.deviceState.light_on);

  // --- ANIMATION LOOP (Chạy mỗi khung hình - 60fps) ---
  useFrame((state, delta) => {
    // 1. Xoay quạt
    if (fanRef.current) {
      // Cộng dồn góc quay. Delta giúp tốc độ ổn định trên mọi màn hình.
      fanRef.current.rotation.y += fanSpeed * delta; 
    }

    // 2. Đổi màu đèn
    if (lightBulbRef.current) {
        // Nếu lightOn = true thì màu vàng sáng, false thì màu xám tối
        const targetColor = lightOn ? '#ffcc00' : '#333333';
        const intensity = lightOn ? 2 : 0;
        
        lightBulbRef.current.material.color.set(targetColor);
        lightBulbRef.current.material.emissive.set(targetColor);
        lightBulbRef.current.material.emissiveIntensity = intensity;
    }
  });

  return (
    <group dispose={null}>
      {/* --- Sàn nhà giả định --- */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -2, 0]}>
        <planeGeometry args={[10, 10]} />
        <meshStandardMaterial color="#555555" />
      </mesh>

      {/* --- Giả lập CÁI QUẠT TRẦN (Khối hộp màu xanh) --- */}
      <group position={[0, 1, 0]}>
         {/* Trục quạt */}
        <mesh position={[0, 0.2, 0]}>
            <cylinderGeometry args={[0.1, 0.1, 0.5]} />
            <meshStandardMaterial color="gray" />
        </mesh>
        {/* Cánh quạt quay (Gán ref vào đây) */}
        <mesh ref={fanRef} scale={[1, 0.1, 0.2]}>
            <boxGeometry />
            <meshStandardMaterial color="#00cec9" />
        </mesh>
      </group>

      {/* --- Giả lập CÁI ĐÈN (Khối cầu) --- */}
      <mesh ref={lightBulbRef} position={[3, 1, 2]}>
        <sphereGeometry args={[0.3, 32, 32]} />
        <meshStandardMaterial color="#333333" />
        {/* Thêm point light để nó chiếu sáng thật sự */}
        {lightOn && <pointLight color="#ffcc00" intensity={1.5} distance={5} />}
      </mesh>
    </group>
  );
}

export default HouseModel;