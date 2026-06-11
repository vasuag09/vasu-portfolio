"use client";

import { useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { mulberry32 } from "@/lib/seeded-random";
import { SCENE_COLORS } from "@/lib/scene-colors";

/**
 * GPU curl-noise particle field (AWARD-RESEARCH §5 quality bar).
 *
 * Stateless design: every particle's position is a pure function of
 * (seed, uTime) computed in the vertex shader — no FBO ping-pong, no CPU
 * loop, 80k particles ≈ one draw call and zero per-frame JS work.
 * Per-particle lifetime: particles fade out, teleport to a fresh noise
 * offset, fade back in — the cycle is invisible because alpha hits zero
 * exactly at the wrap point.
 *
 * ~4% are HDR "signal" particles (accent color, intensity > bloom
 * threshold) — only these bloom. The rest stay a dim field (restraint).
 */

const PARTICLE_SEED = 0x5eed_c0de;
const SIGNAL_RATIO = 0.04;

// World volume covering all five section regions.
const CENTER: [number, number, number] = [1, -4, 2];
const HALF_EXTENTS: [number, number, number] = [34, 20, 11];

const vertexShader = /* glsl */ `
  attribute vec3 aSeed;
  attribute float aSize;
  attribute float aIntensity;
  uniform float uTime;
  uniform float uPixelRatio;
  uniform vec3 uCenter;
  uniform vec3 uHalfExtents;
  varying float vEnvelope;
  varying float vIntensity;

  // Simplex 3D noise — Ashima Arts / Stefan Gustavson (MIT), the standard
  // WebGL implementation, same family v4's shaders drew from.
  vec4 permute(vec4 x) { return mod(((x * 34.0) + 1.0) * x, 289.0); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
  float snoise(vec3 v) {
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + 1.0 * C.xxx;
    vec3 x2 = x0 - i2 + 2.0 * C.xxx;
    vec3 x3 = x0 - 1.0 + 3.0 * C.xxx;
    i = mod(i, 289.0);
    vec4 p = permute(permute(permute(
        i.z + vec4(0.0, i1.z, i2.z, 1.0))
      + i.y + vec4(0.0, i1.y, i2.y, 1.0))
      + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 1.0 / 7.0;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
  }

  // Curl of the noise field via finite differences — divergence-free flow,
  // the "physics-driven motion" that separates this from linear drift.
  vec3 curlNoise(vec3 p) {
    const float e = 0.1;
    float n1 = snoise(vec3(p.x, p.y + e, p.z));
    float n2 = snoise(vec3(p.x, p.y - e, p.z));
    float n3 = snoise(vec3(p.x, p.y, p.z + e));
    float n4 = snoise(vec3(p.x, p.y, p.z - e));
    float n5 = snoise(vec3(p.x + e, p.y, p.z));
    float n6 = snoise(vec3(p.x - e, p.y, p.z));
    float x = (n1 - n2) - (n3 - n4);
    float y = (n3 - n4) - (n5 - n6);
    float z = (n5 - n6) - (n1 - n2);
    // Safe normalize: normalize(0) is NaN, and ONE NaN fragment poisons the
    // entire bloom mip chain into a black screen.
    vec3 c = vec3(x, y, z);
    return c / (length(c) + 1e-5);
  }

  const float LIFETIME = 14.0;

  void main() {
    vIntensity = aIntensity;

    // Per-particle life phase, offset by seed so respawns never sync up.
    float phase = fract(uTime / LIFETIME + dot(aSeed, vec3(7.31, 3.17, 9.73)));
    // Each cycle picks a fresh spawn cell so respawn = relocation.
    float cycle = floor(uTime / LIFETIME + dot(aSeed, vec3(7.31, 3.17, 9.73)));
    vec3 cellSeed = fract(aSeed + vec3(cycle * 0.6180339887));

    vec3 spawn = uCenter + (cellSeed * 2.0 - 1.0) * uHalfExtents;

    // Curl flow: slow field evolution + per-particle travel over its life.
    vec3 flow = curlNoise(spawn * 0.06 + uTime * 0.015);
    vec3 displaced = spawn + flow * phase * 3.5;

    // Fade in 15%, fade out last 15% — zero alpha at the wrap point.
    vEnvelope = smoothstep(0.0, 0.15, phase) * smoothstep(1.0, 0.85, phase);

    vec4 mvPosition = modelViewMatrix * vec4(displaced, 1.0);
    // Perspective size, clamped so near-camera particles never blow out
    // into screen-filling blobs.
    float dist = max(-mvPosition.z, 4.0);
    gl_PointSize = min(aSize * uPixelRatio * vEnvelope * (80.0 / dist), 14.0);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = /* glsl */ `
  uniform vec3 uBaseColor;
  uniform vec3 uSignalColor;
  varying float vEnvelope;
  varying float vIntensity;

  void main() {
    // Soft round sprite
    float d = length(gl_PointCoord - 0.5);
    float disc = smoothstep(0.5, 0.12, d);
    if (disc < 0.001) discard;

    // Signal particles (vIntensity > 1) are accent-colored and HDR.
    vec3 color = mix(uBaseColor, uSignalColor, step(1.5, vIntensity));
    float alpha = disc * vEnvelope * 0.4;
    gl_FragColor = vec4(color * vIntensity, alpha);
  }
`;

export function Particles({ count }: { count: number }) {
  const pixelRatio = useThree((state) => state.gl.getPixelRatio());
  const pointsRef = useRef<THREE.Points>(null);

  const { geometry, material } = useMemo(() => {
    const rand = mulberry32(PARTICLE_SEED);
    const seeds = new Float32Array(count * 3);
    const sizes = new Float32Array(count);
    const intensities = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      seeds.set([rand(), rand(), rand()], i * 3);
      const isSignal = rand() < SIGNAL_RATIO;
      sizes[i] = isSignal ? 1.6 + rand() * 1.0 : 0.45 + rand() * 0.8;
      intensities[i] = isSignal ? 2.4 + rand() * 1.2 : 0.4 + rand() * 0.3;
    }
    const geometry = new THREE.BufferGeometry();
    // Real positions come from the vertex shader; this zero buffer only
    // satisfies the attribute contract.
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(count * 3), 3),
    );
    geometry.setAttribute("aSeed", new THREE.BufferAttribute(seeds, 3));
    geometry.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
    geometry.setAttribute(
      "aIntensity",
      new THREE.BufferAttribute(intensities, 1),
    );

    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uTime: { value: 0 },
        uPixelRatio: { value: 1 },
        uCenter: { value: new THREE.Vector3(...CENTER) },
        uHalfExtents: { value: new THREE.Vector3(...HALF_EXTENTS) },
        uBaseColor: { value: new THREE.Color(SCENE_COLORS.particleBase) },
        uSignalColor: { value: new THREE.Color(SCENE_COLORS.accentBright) },
      },
    });
    return { geometry, material };
  }, [count]);

  // Frame-loop mutation via ref — the React Compiler escape hatch.
  useFrame(({ clock }) => {
    const points = pointsRef.current;
    if (!points) return;
    const mat = points.material as THREE.ShaderMaterial;
    mat.uniforms.uTime.value = clock.getElapsedTime();
    mat.uniforms.uPixelRatio.value = pixelRatio;
  });

  return (
    <points
      ref={pointsRef}
      geometry={geometry}
      material={material}
      frustumCulled={false}
    />
  );
}
