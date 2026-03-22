import * as THREE from "three";

/**
 * Custom ShaderMaterial for neural network nodes.
 * Creates a Fresnel rim-glow sphere with pulsing emissive effect.
 */
export function createNodeMaterial() {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
    uniforms: {
      uTime: { value: 0 },
      uColor: { value: new THREE.Color(0x00f0ff) },
      uActive: { value: 0.0 },
      uHover: { value: 0.0 },
    },
    vertexShader: /* glsl */ `
      varying vec3 vNormal;
      varying vec3 vViewDir;
      void main() {
        vNormal = normalize(normalMatrix * normal);
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vViewDir = normalize(-mvPosition.xyz);
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uTime;
      uniform vec3 uColor;
      uniform float uActive;
      uniform float uHover;
      varying vec3 vNormal;
      varying vec3 vViewDir;

      void main() {
        // Fresnel rim glow — brighter at edges
        float fresnel = 1.0 - dot(vNormal, vViewDir);
        fresnel = pow(fresnel, 2.5);

        // Core brightness
        float core = dot(vNormal, vViewDir);
        core = pow(core, 1.5) * 0.4;

        // Pulse animation for active nodes
        float pulse = sin(uTime * 2.0) * 0.15 + 0.85;
        float activePulse = mix(1.0, pulse, uActive);

        // Combined intensity
        float intensity = (fresnel * 0.8 + core) * activePulse;
        intensity += uHover * 0.3;
        intensity += uActive * 0.4;

        // Alpha: solid core, glowing rim
        float alpha = (core * 0.8 + fresnel * 0.6) * (0.6 + uActive * 0.4 + uHover * 0.2);
        alpha = clamp(alpha, 0.0, 1.0);

        gl_FragColor = vec4(uColor * intensity, alpha);
      }
    `,
  });
}
