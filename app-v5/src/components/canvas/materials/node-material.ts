import * as THREE from "three";

/**
 * Fresnel rim-glow node material — ported from v4
 * (src/components/canvas/shaders/nodeMaterial.js) and upgraded for v5:
 *  - instanced rendering (instanceMatrix in the vertex stage)
 *  - per-instance color + emissive intensity (aColor, aIntensity)
 *    so flagship/accent nodes exceed bloom threshold (selective bloom)
 *  - uActive/uHover survive as global uniforms; Phase 3 moves activation
 *    to a per-instance attribute for edge highlighting.
 */
export function createNodeMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
    uniforms: {
      uTime: { value: 0 },
      uActive: { value: 0.0 },
      uHover: { value: 0.0 },
    },
    vertexShader: /* glsl */ `
      attribute vec3 aColor;
      attribute float aIntensity;
      varying vec3 vNormal;
      varying vec3 vViewDir;
      varying vec3 vColor;
      varying float vIntensity;

      void main() {
        vColor = aColor;
        vIntensity = aIntensity;
        // instanceMatrix is uniform-scale here, so normalMatrix stays valid
        vec4 worldPosition = instanceMatrix * vec4(position, 1.0);
        vNormal = normalize(normalMatrix * mat3(instanceMatrix) * normal);
        vec4 mvPosition = modelViewMatrix * worldPosition;
        vViewDir = normalize(-mvPosition.xyz);
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uTime;
      uniform float uActive;
      uniform float uHover;
      varying vec3 vNormal;
      varying vec3 vViewDir;
      varying vec3 vColor;
      varying float vIntensity;

      void main() {
        // Fresnel rim glow — brighter at edges (v4 heritage).
        // clamp/max before pow: interpolated normals can push these dots
        // slightly out of range, and pow(negative, x) = NaN — one NaN pixel
        // poisons the whole bloom mip chain black.
        float fresnel = clamp(1.0 - dot(vNormal, vViewDir), 0.0, 1.0);
        fresnel = pow(fresnel, 2.5);

        float core = max(dot(vNormal, vViewDir), 0.0);
        core = pow(core, 1.5) * 0.4;

        float pulse = sin(uTime * 2.0) * 0.15 + 0.85;
        float activePulse = mix(1.0, pulse, uActive);

        float intensity = (fresnel * 0.8 + core) * activePulse;
        intensity += uHover * 0.3 + uActive * 0.4;

        float alpha = (core * 0.8 + fresnel * 0.6) * (0.6 + uActive * 0.4 + uHover * 0.2);
        alpha = clamp(alpha, 0.0, 1.0);

        // aIntensity > 1 pushes accent nodes past the bloom luminance
        // threshold — this is what makes bloom selective.
        gl_FragColor = vec4(vColor * intensity * vIntensity, alpha);
      }
    `,
  });
}
