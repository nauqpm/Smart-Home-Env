// frontend/src/components/Scene3D.jsx
import React from 'react';
import SceneRoot from './smarthome/SceneRoot';
import '../App.css';

function Scene3D() {
  return (
    <div className="scene-wrapper">
      <div id="canvas-container">
        <SceneRoot />
      </div>
    </div>
  );
}

export default Scene3D;