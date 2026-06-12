import * as THREE from "three";

/**
 * Connection flow material — ported from v4
 * (src/components/canvas/shaders/connectionMaterial.js), upgraded with
 * per-vertex color (skill-category → project color gradient), per-vertex
 * activation (Phase 3), and the design-elevation line art direction:
 * idle opacity scales with aLenFade (long cross-section edges whisper
 * until activated) and fades with camera depth, so no edge ever reads as
 * a full-frame streak.
 */
export function createConnectionMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uTime: { value: 0 },
      uOpacity: { value: 0.05 }, // restraint: edges read as field, not wires
      uFlowSpeed: { value: 1.0 },
    },
    vertexShader: /* glsl */ `
      attribute float aProgress;
      attribute vec3 aColor;
      attribute float aActive;
      attribute float aLenFade;
      varying float vProgress;
      varying vec3 vColor;
      varying float vActive;
      varying float vLenFade;
      varying float vViewZ;
      void main() {
        vProgress = aProgress;
        vColor = aColor;
        vActive = aActive;
        vLenFade = aLenFade;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vViewZ = -mvPosition.z;
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uTime;
      uniform float uOpacity;
      uniform float uFlowSpeed;
      varying float vProgress;
      varying vec3 vColor;
      varying float vActive;
      varying float vLenFade;
      varying float vViewZ;

      void main() {
        // Animated energy pulse traveling along the connection (v4 heritage);
        // active edges flow visibly faster.
        float speed = uFlowSpeed * (1.0 + vActive * 1.5);
        float flow = sin(vProgress * 12.0 - uTime * speed * 3.0) * 0.5 + 0.5;
        flow = pow(flow, 3.0);

        float alpha = uOpacity + flow * 0.15 * (1.0 + vActive * 2.0);
        alpha *= (1.0 + vActive * 2.5);

        // Soft endpoints
        float edgeFade = smoothstep(0.0, 0.08, vProgress) * smoothstep(1.0, 0.92, vProgress);
        alpha *= edgeFade;

        // Length fade: long edges idle near-invisible; activation restores.
        alpha *= mix(vLenFade, 1.0, clamp(vActive, 0.0, 1.0));
        // Depth fade: edges dissolve with distance instead of streaking.
        alpha *= smoothstep(70.0, 22.0, vViewZ);

        // Active edges go HDR so the connection itself blooms.
        vec3 color = vColor * (1.0 + vActive * 1.8);
        gl_FragColor = vec4(color, alpha);
      }
    `,
  });
}
