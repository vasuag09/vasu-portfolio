import * as THREE from "three";

/**
 * Connection flow material — ported from v4
 * (src/components/canvas/shaders/connectionMaterial.js), upgraded with
 * per-vertex color so each edge fades from skill-category color to
 * project color. Renders as LineSegments: one aProgress 0→1 pair per edge.
 */
export function createConnectionMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uTime: { value: 0 },
      uOpacity: { value: 0.05 }, // restraint: edges read as field, not wires
      uActive: { value: 0.0 },
      uFlowSpeed: { value: 1.0 },
    },
    vertexShader: /* glsl */ `
      attribute float aProgress;
      attribute vec3 aColor;
      varying float vProgress;
      varying vec3 vColor;
      void main() {
        vProgress = aProgress;
        vColor = aColor;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uTime;
      uniform float uOpacity;
      uniform float uActive;
      uniform float uFlowSpeed;
      varying float vProgress;
      varying vec3 vColor;

      void main() {
        // Animated energy pulse traveling along the connection (v4 heritage)
        float flow = sin(vProgress * 12.0 - uTime * uFlowSpeed * 3.0) * 0.5 + 0.5;
        flow = pow(flow, 3.0);

        float alpha = uOpacity + flow * 0.15 * (1.0 + uActive * 2.0);
        alpha *= (1.0 + uActive * 1.5);

        // Soft endpoints
        float edgeFade = smoothstep(0.0, 0.08, vProgress) * smoothstep(1.0, 0.92, vProgress);
        alpha *= edgeFade;

        gl_FragColor = vec4(vColor, alpha);
      }
    `,
  });
}
