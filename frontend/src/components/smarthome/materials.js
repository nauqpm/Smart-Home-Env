import * as THREE from 'three';

export const palette = {
  woodFloor: 0xcaa073,
  tileLight: 0xe8e8e8,
  wall: 0xf4f4f4,
  trim: 0xdadada,
  carpetTeal: 0x7cc6c3,
  carpetGreen: 0x9bcf8f,
  ceramicWhite: 0xffffff,
  metalChrome: 0xaaaaaa,
};

// --- PROCEDURAL TEXTURE GENERATORS ---

const createWoodTexture = () => {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  if (!ctx) return null;

  // Background
  ctx.fillStyle = '#C19A6B';
  ctx.fillRect(0, 0, 512, 512);

  // Grain
  ctx.strokeStyle = '#A07040';
  ctx.lineWidth = 2;
  for (let i = 0; i < 60; i++) {
    const x = Math.random() * 512;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    // Wavy line
    ctx.bezierCurveTo(x + Math.random() * 50 - 25, 170, x + Math.random() * 50 - 25, 340, x + Math.random() * 20 - 10, 512);
    ctx.stroke();
  }

  // Planks
  ctx.strokeStyle = '#805030';
  ctx.lineWidth = 3;
  for (let i = 0; i <= 512; i += 64) {
    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(512, i);
    ctx.stroke();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(4, 4);
  return texture;
};

const createTileTexture = () => {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  if (!ctx) return null;

  // Background
  ctx.fillStyle = '#E8E8E8';
  ctx.fillRect(0, 0, 512, 512);

  // Grid
  ctx.strokeStyle = '#CCCCCC';
  ctx.lineWidth = 4;

  // 4x4 Tiles
  const step = 128;
  for (let x = 0; x <= 512; x += step) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 512); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, x); ctx.lineTo(512, x); ctx.stroke();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(2, 2);
  return texture;
};

// Singleton Textures
let woodTex = null;
let tileTex = null;

const getWoodTex = () => {
  if (!woodTex) woodTex = createWoodTexture();
  return woodTex;
}

const getTileTex = () => {
  if (!tileTex) tileTex = createTileTexture();
  return tileTex;
}

export const makeWoodFloorMat = () => {
  const tex = getWoodTex();
  return new THREE.MeshStandardMaterial({
    map: tex,
    color: 0xdddddd, // Tint slightly
    roughness: 0.7,
    metalness: 0.05
  });
}

export const makeTileMat = () => {
  const tex = getTileTex();
  return new THREE.MeshStandardMaterial({
    map: tex,
    roughness: 0.5,
    metalness: 0.1
  });
}

export const makeWallMat = () =>
  new THREE.MeshStandardMaterial({
    color: palette.wall,
    roughness: 0.9,
    metalness: 0.05,
    side: THREE.DoubleSide,
  });

export const makeTrimMat = () =>
  new THREE.MeshStandardMaterial({
    color: palette.trim,
    roughness: 0.8,
    metalness: 0.05,
  });

export const makeCeramicMat = () =>
  new THREE.MeshStandardMaterial({
    color: palette.ceramicWhite,
    roughness: 0.2, // Smooth, glossy
    metalness: 0.1,
  });

export const makeChromeMat = () =>
  new THREE.MeshStandardMaterial({
    color: palette.metalChrome,
    roughness: 0.1,
    metalness: 0.9,
  });




