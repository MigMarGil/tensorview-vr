// TensorView VR — entry point. Built with the Immersive Web SDK (IWSDK): World provides renderer, scene,
// camera, XR session handling and the frame loop; TensorView's app logic runs as an ECS system inside it.
import { World, createSystem, SessionMode } from '@iwsdk/core';
import { createTensorView } from './app.js';

World.create(document.getElementById('scene-container'), {
  xr: {
    sessionMode: SessionMode.ImmersiveVR,
    // 'local': origin at the user's head when the session starts -> ideal for seated use
    referenceSpace: { type: 'local', fallbackOrder: ['local-floor', 'viewer'] },
    features: { handTracking: true },
    offer: 'always',
  },
  render: { fov: 60, near: 0.05, far: 60 },
  features: { locomotion: false, grabbing: false, physics: false, sceneUnderstanding: false, environmentRaycast: false, spatialUI: false },
}).then((world) => {
  const app = createTensorView(world);
  class TensorViewSystem extends createSystem({}) {
    update(delta) { app.update(delta); }
  }
  world.registerSystem(TensorViewSystem);
  window.__world = world;
});
