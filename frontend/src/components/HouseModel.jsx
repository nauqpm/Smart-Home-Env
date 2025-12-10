// frontend/src/components/HouseModel.jsx
import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import useStore from '../stores/useStore';

function HouseModel() {
  const fanGroupRef = useRef();
  const lightBulbRef = useRef();

  // Lấy state từ store
  const fanSpeed = useStore((state) => state.deviceState.fan_speed);
  const lightOn = useStore((state) => state.deviceState.light_on);
  // Giả sử bạn sẽ thêm trạng thái này vào backend sau này
  const acOn = useStore((state) => state.deviceState.ac_on || false); 
  const evCharging = useStore((state) => state.deviceState.ev_charging || false);

  // --- ANIMATION LOOP ---
  useFrame((state, delta) => {
    // Xoay cả nhóm quạt
    if (fanGroupRef.current) {
      fanGroupRef.current.rotation.y -= fanSpeed * delta * 3; // Quay ngược chiều kim đồng hồ
    }

    // Đổi màu đèn
    if (lightBulbRef.current) {
        const targetColor = lightOn ? '#ffff00' : '#333333'; // Vàng hoặc xám tối
        const intensity = lightOn ? 1 : 0;
        
        lightBulbRef.current.material.color.set(targetColor);
        lightBulbRef.current.material.emissive.set(targetColor);
        lightBulbRef.current.material.emissiveIntensity = intensity;
    }
  });

  return (
    <group dispose={null}>
      {/* --- A. QUẠT TRẦN CHI TIẾT (Thay cho khối hộp cũ) --- */}
      <group ref={fanGroupRef} position={[0, 2.4, 0]}> {/* Treo sát trần */}
         {/* Trục giữa */}
        <mesh>
            <cylinderGeometry args={[0.1, 0.1, 0.2, 16]} />
            <meshStandardMaterial color="#333" />
        </mesh>
        {/* 4 Cánh quạt (Dùng vòng lặp để tạo) */}
        {[0, 1, 2, 3].map((i) => (
            <mesh key={i} rotation={[0, (Math.PI / 2) * i, 0]} position={[0.8 * Math.cos((Math.PI / 2) * i), 0, -0.8 * Math.sin((Math.PI / 2) * i)]}>
                <boxGeometry args={[1.6, 0.05, 0.3]} />
                <meshStandardMaterial color="#666" />
            </mesh>
        ))}
      </group>

      {/* --- B. ĐÈN TRẦN (Bóng vàng) --- */}
      <group position={[0, 2.4, 0]}> {/* Treo giữa trần */}
        <mesh ref={lightBulbRef}>
            <sphereGeometry args={[0.15, 16, 16]} />
            <meshStandardMaterial color="#333333" />
        </mesh>
        <pointLight color="#ffff00" intensity={lightOn ? 1.5 : 0} distance={10} castShadow />
      </group>

      {/* --- C. ĐIỀU HÒA (AC UNIT) - Mới --- */}
      <group position={[-1.5, 1.8, 2.05]}> {/* Gắn trên tường trước */}
        {/* Cục lạnh */}
        <mesh castShadow>
            <boxGeometry args={[1.2, 0.5, 0.3]} />
            <meshStandardMaterial color="white" />
        </mesh>
        {/* Khe gió */}
        <mesh position={[0, -0.1, 0.16]}>
             <boxGeometry args={[1, 0.1, 0.05]} />
             <meshStandardMaterial color="#3498db" />
        </mesh>
        {/* Đèn báo trạng thái (Xanh=On, Đỏ=Off) */}
        <mesh position={[0.5, 0.15, 0.16]}>
            <sphereGeometry args={[0.05]} />
            <meshStandardMaterial color={acOn ? "#00ff00" : "#ff0000"} emissive={acOn ? "#00ff00" : "#ff0000"} emissiveIntensity={0.5} />
        </mesh>
      </group>

      {/* --- D. XE ĐIỆN & SẠC - Mới --- */}
      <group position={[3.5, 0.5, 0]}> {/* Đặt bên cạnh nhà */}
          {/* Cái xe giả định */}
          <mesh castShadow receiveShadow>
              <boxGeometry args={[2, 1, 3.5]} />
              <meshStandardMaterial color="#34495e" />
          </mesh>
          
          {/* Dây sạc (Chỉ hiện khi evCharging = true) */}
          {evCharging && (
            <mesh position={[-1.5, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
                {/* Nối từ xe vào tường nhà */}
                <cylinderGeometry args={[0.05, 0.05, 3]} /> 
                <meshStandardMaterial color="orange" />
            </mesh>
          )}
      </group>
    </group>
  );
}

export default HouseModel;