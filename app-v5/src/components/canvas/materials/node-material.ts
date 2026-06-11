import * as THREE from "three";

/**
 * Fresnel rim-glow node material — ported from v4
 * (src/components/canvas/shaders/nodeMaterial.js) and upgraded for v5:
 *  - instanced rendering (instanceMatrix in the vertex stage)
 *  - per-instance color + emissive intensity (aColor, aIntensity)
 *    so flagship/accent nodes exceed bloom threshold (selective bloom)
 *  - per-instance activation (aActive, Phase 3): hovering a skill in the
 *    DOM overlay glows exactly the connected nodes — active nodes pulse
 *    and get pushed past the bloom threshold.
 */
export function createNodeMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
    uniforms: {
      uTime: { value: 0 },
    },
    vertexShader: /* glsl */ `
      attribute vec3 aColor;
      attribute float aIntensity;
      attribute float aActive;
      varying vec3 vNormal;
      varying vec3 vViewDir;
      varying vec3 vColor;
      varying float vIntensity;
      varying float vActive;

      void main() {
        vColor = aColor;
        vIntensity = aIntensity;
        vActive = aActive;
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
      varying vec3 vNormal;
      varying vec3 vViewDir;
      varying vec3 vColor;
      varying float vIntensity;
      varying float vActive;

      void main() {
        // Fresnel rim glow — brighter at edges (v4 heritage).
        // clamp/max before pow: interpolated normals can push these dots
        // slightly out of range, and pow(negative, x) = NaN — one NaN pixel
        // poisons the whole bloom mip chain black.
        float fresnel = clamp(1.0 - dot(vNormal, vViewDir), 0.0, 1.0);
        fresnel = pow(fresnel, 2.5);

        float core = max(dot(vNormal, vViewDir), 0.0);
        core = pow(core, 1.5) * 0.4;

        // Active nodes pulse (v4's uActive behavior, now per instance)
        float pulse = sin(uTime * 2.0) * 0.15 + 0.85;
        float activePulse = mix(1.0, pulse, vActive);

        float intensity = (fresnel * 0.8 + core) * activePulse;
        intensity += vActive * 0.5;

        float alpha = (core * 0.8 + fresnel * 0.6) * (0.6 + vActive * 0.4);
        alpha = clamp(alpha, 0.0, 1.0);

        // aIntensity > 1 blooms at rest; activation pushes ANY node past
        // the threshold — the glow IS the connection feedback.
        float hdr = vIntensity * (1.0 + vActive * 1.6);
        gl_FragColor = vec4(vColor * intensity * hdr, alpha);
      }
    `,
  });
}
