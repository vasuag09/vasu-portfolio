import * as THREE from "three";

/**
 * Custom ShaderMaterial for neural connections.
 * Renders animated energy flow along connection lines with glow.
 */
export function createConnectionMaterial() {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uTime: { value: 0 },
      uColor: { value: new THREE.Color(0x00f0ff) },
      uOpacity: { value: 0.12 },
      uActive: { value: 0.0 },
      uFlowSpeed: { value: 1.0 },
    },
    vertexShader: /* glsl */ `
      attribute float aProgress;
      varying float vProgress;
      void main() {
        vProgress = aProgress;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uTime;
      uniform vec3 uColor;
      uniform float uOpacity;
      uniform float uActive;
      uniform float uFlowSpeed;
      varying float vProgress;

      void main() {
        // Animated energy pulse traveling along the connection
        float flow = sin(vProgress * 12.0 - uTime * uFlowSpeed * 3.0) * 0.5 + 0.5;
        flow = pow(flow, 3.0); // Sharpen the pulse

        // Base opacity + flow contribution
        float alpha = uOpacity + flow * 0.15 * (1.0 + uActive * 2.0);

        // Active connections glow brighter
        alpha *= (1.0 + uActive * 1.5);

        // Fade at endpoints for soft look
        float edgeFade = smoothstep(0.0, 0.08, vProgress) * smoothstep(1.0, 0.92, vProgress);
        alpha *= edgeFade;

        gl_FragColor = vec4(uColor, alpha);
      }
    `,
  });
}
