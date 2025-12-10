import React from 'react';

export default function SmartHomeLights({ sunLightRef, ambientRef, hemiRef }) {
  return (
    <>
      <ambientLight ref={ambientRef} intensity={0.8} />
      <hemisphereLight ref={hemiRef} args={[0xffffff, 0xdddddd, 0.6]} />
      <directionalLight
        ref={sunLightRef}
        color={0xffffff}
        intensity={0.7}
        position={[30, 40, 20]}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-camera-left={-25}
        shadow-camera-right={25}
        shadow-camera-top={25}
        shadow-camera-bottom={-25}
      />
    </>
  );
}

