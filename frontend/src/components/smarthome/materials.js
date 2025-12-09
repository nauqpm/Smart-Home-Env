import * as THREE from 'three';

export const palette = {
  woodFloor: 0xcaa073,
  tileLight: 0xe8e8e8,
  wall: 0xf4f4f4,
  trim: 0xdadada,
  carpetTeal: 0x7cc6c3,
  carpetGreen: 0x9bcf8f,
};

export const makeWoodFloorMat = () =>
  new THREE.MeshStandardMaterial({ color: palette.woodFloor, roughness: 0.6, metalness: 0.05 });

export const makeTileMat = () =>
  new THREE.MeshStandardMaterial({ color: palette.tileLight, roughness: 0.8, metalness: 0.05 });

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

