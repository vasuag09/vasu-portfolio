"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { sections } from "@/data/sections-v5";
import { projectNodes } from "@/data/projects-v5";
import { skills, edges } from "@/data/skills-graph";
import { SCENE_COLORS, CATEGORY_HEX } from "@/lib/scene-colors";
import { getActiveIds, getGraphState } from "@/lib/graph-store";
import { buildActivationSet } from "@/lib/graph-adjacency";
import { createNodeMaterial } from "./materials/node-material";
import { createConnectionMaterial } from "./materials/connection-material";
import type { Vec3 } from "@/data/types";

/**
 * The data-driven neural world (Phase 2/3): every node IS a section core, a
 * project, or a skill from src/data — no decorative fakes. Skill→project
 * edges come straight from the skills graph; short intra-cluster links add
 * visual density, derived deterministically from node distances.
 *
 * Phase 3: hover/selection state flows DOM → graph-store → per-instance
 * aActive attributes here (damped every frame), so the canvas glow always
 * matches the DOM overlay exactly (shared adjacency module).
 */

interface NodeInstance {
  position: Vec3;
  scale: number;
  color: string;
  /** >1 pushes past bloom threshold (selective bloom). */
  intensity: number;
}

const SECTION_CORE_SCALE = 1.15;
const ACTIVATION_DAMPING = 9; // 1/s — exp-decay rate toward target

function buildNodeInstances(): {
  instances: NodeInstance[];
  indexById: Map<string, number>;
} {
  const instances: NodeInstance[] = [];
  const indexById = new Map<string, number>();

  sections.forEach((s) => {
    instances.push({
      position: s.cameraTarget,
      scale: SECTION_CORE_SCALE,
      color: SCENE_COLORS.accent,
      intensity: 1.8,
    });
    // Section cores never activate; not registered in indexById.
  });
  projectNodes.forEach((n) => {
    indexById.set(n.id, instances.length);
    instances.push({
      position: n.position,
      scale: n.scale * 0.32,
      color: n.flagship ? SCENE_COLORS.nodeFlagship : SCENE_COLORS.nodeProject,
      intensity: n.flagship ? 2.2 : 0.9,
    });
  });
  skills.forEach((s) => {
    indexById.set(s.id, instances.length);
    instances.push({
      position: s.position,
      scale: 0.22,
      color: CATEGORY_HEX[s.category],
      intensity: s.category === "genai" ? 1.6 : 0.85,
    });
  });
  return { instances, indexById };
}

/** k nearest neighbors within one cluster — deterministic, build-time cheap. */
function nearestNeighborPairs(
  positions: readonly Vec3[],
  k: number,
): [number, number][] {
  const pairs = new Set<string>();
  positions.forEach((a, i) => {
    const ranked = positions
      .map((b, j) => ({ j, d: distSq(a, b) }))
      .filter(({ j }) => j !== i)
      .sort((m, n) => m.d - n.d)
      .slice(0, k);
    ranked.forEach(({ j }) => {
      pairs.add(i < j ? `${i}:${j}` : `${j}:${i}`);
    });
  });
  return Array.from(pairs).map((key) => {
    const [i, j] = key.split(":").map(Number);
    return [i, j];
  });
}

function distSq(a: Vec3, b: Vec3): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

interface SegmentSpec {
  from: Vec3;
  to: Vec3;
  fromColor: string;
  toColor: string;
}

/** Data edges FIRST: segment index i < edges.length ⇔ edges[i] (activation). */
function buildSegments(): SegmentSpec[] {
  const projectById = new Map(projectNodes.map((n) => [n.id, n]));
  const skillById = new Map(skills.map((s) => [s.id, s]));

  const dataEdges: SegmentSpec[] = edges.map((edge) => {
    const skill = skillById.get(edge.skillId);
    const project = projectById.get(edge.projectId);
    if (!skill || !project) {
      throw new Error(`Dangling edge ${edge.skillId}→${edge.projectId}`);
    }
    return {
      from: skill.position,
      to: project.position,
      fromColor: CATEGORY_HEX[skill.category],
      toColor: project.flagship
        ? SCENE_COLORS.nodeFlagship
        : SCENE_COLORS.nodeProject,
    };
  });

  const projectLinks: SegmentSpec[] = nearestNeighborPairs(
    projectNodes.map((n) => n.position),
    2,
  ).map(([i, j]) => ({
    from: projectNodes[i].position,
    to: projectNodes[j].position,
    fromColor: SCENE_COLORS.nodeProject,
    toColor: SCENE_COLORS.nodeProject,
  }));
  const skillLinks: SegmentSpec[] = nearestNeighborPairs(
    skills.map((s) => s.position),
    2,
  ).map(([i, j]) => ({
    from: skills[i].position,
    to: skills[j].position,
    fromColor: CATEGORY_HEX[skills[i].category],
    toColor: CATEGORY_HEX[skills[j].category],
  }));

  return [...dataEdges, ...projectLinks, ...skillLinks];
}

export function NeuralNetwork() {
  const nodeMaterial = useMemo(() => createNodeMaterial(), []);
  const connectionMaterial = useMemo(() => createConnectionMaterial(), []);
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const lineRef = useRef<THREE.LineSegments>(null);

  const { nodeGeometry, instances, indexById } = useMemo(() => {
    const { instances, indexById } = buildNodeInstances();
    const geometry = new THREE.SphereGeometry(1, 24, 24);
    const colors = new Float32Array(instances.length * 3);
    const intensities = new Float32Array(instances.length);
    const color = new THREE.Color();
    instances.forEach((inst, i) => {
      color.set(inst.color);
      colors.set([color.r, color.g, color.b], i * 3);
      intensities[i] = inst.intensity;
    });
    geometry.setAttribute(
      "aColor",
      new THREE.InstancedBufferAttribute(colors, 3),
    );
    geometry.setAttribute(
      "aIntensity",
      new THREE.InstancedBufferAttribute(intensities, 1),
    );
    geometry.setAttribute(
      "aActive",
      new THREE.InstancedBufferAttribute(new Float32Array(instances.length), 1),
    );
    return { nodeGeometry: geometry, instances, indexById };
  }, []);

  const { connectionGeometry, segmentCount } = useMemo(() => {
    const segments = buildSegments();
    const positions = new Float32Array(segments.length * 6);
    const progress = new Float32Array(segments.length * 2);
    const colors = new Float32Array(segments.length * 6);
    const color = new THREE.Color();
    segments.forEach((seg, i) => {
      positions.set([...seg.from, ...seg.to], i * 6);
      progress.set([0, 1], i * 2);
      color.set(seg.fromColor);
      colors.set([color.r, color.g, color.b], i * 6);
      color.set(seg.toColor);
      colors.set([color.r, color.g, color.b], i * 6 + 3);
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("aProgress", new THREE.BufferAttribute(progress, 1));
    geometry.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute(
      "aActive",
      new THREE.BufferAttribute(new Float32Array(segments.length * 2), 1),
    );
    return { connectionGeometry: geometry, segmentCount: segments.length };
  }, []);

  // Activation targets, rebuilt only when the hovered/selected ids change.
  const activationRef = useRef({
    key: "",
    nodeTargets: new Float32Array(instances.length),
    edgeTargets: new Float32Array(segmentCount * 2),
  });

  // Write instance matrices once (positions are static this phase).
  // All frame-loop mutation goes through refs — the React Compiler treats
  // memoized values as frozen, and refs are the sanctioned escape hatch.
  const matricesSet = useRef(false);
  useFrame(({ clock }, delta) => {
    const t = clock.getElapsedTime();
    const mesh = meshRef.current;
    const lines = lineRef.current;
    if (mesh) {
      (mesh.material as THREE.ShaderMaterial).uniforms.uTime.value = t;
    }
    if (lines) {
      (lines.material as THREE.ShaderMaterial).uniforms.uTime.value = t;
    }

    if (mesh && !matricesSet.current) {
      const matrix = new THREE.Matrix4();
      instances.forEach((inst, i) => {
        matrix
          .makeScale(inst.scale, inst.scale, inst.scale)
          .setPosition(...inst.position);
        mesh.setMatrixAt(i, matrix);
      });
      mesh.instanceMatrix.needsUpdate = true;
      matricesSet.current = true;
    }

    // ---- Phase 3 activation: store → targets → damped attributes ----
    if (!mesh || !lines) return;
    const active = getActiveIds(getGraphState());
    const key = `${active.skillId ?? ""}|${active.projectId ?? ""}`;
    const activation = activationRef.current;

    if (key !== activation.key) {
      activation.key = key;
      activation.nodeTargets.fill(0);
      activation.edgeTargets.fill(0);
      const set = buildActivationSet(active);
      set.nodeIds.forEach((id) => {
        const index = indexById.get(id);
        if (index !== undefined) activation.nodeTargets[index] = 1;
      });
      set.edgeIndices.forEach((i) => {
        activation.edgeTargets[i * 2] = 1;
        activation.edgeTargets[i * 2 + 1] = 1;
      });
    }

    // Exponential damping toward targets; skip GPU upload once settled.
    const blend = 1 - Math.exp(-ACTIVATION_DAMPING * delta);
    const nodeAttr = nodeGeometry.getAttribute(
      "aActive",
    ) as THREE.InstancedBufferAttribute;
    const edgeAttr = connectionGeometry.getAttribute(
      "aActive",
    ) as THREE.BufferAttribute;
    const nodeArray = nodeAttr.array as Float32Array;
    const edgeArray = edgeAttr.array as Float32Array;

    let moved = false;
    for (let i = 0; i < nodeArray.length; i += 1) {
      const next =
        nodeArray[i] + (activation.nodeTargets[i] - nodeArray[i]) * blend;
      if (Math.abs(next - nodeArray[i]) > 1e-4) {
        nodeArray[i] = next;
        moved = true;
      }
    }
    if (moved) nodeAttr.needsUpdate = true;

    let edgeMoved = false;
    for (let i = 0; i < edgeArray.length; i += 1) {
      const next =
        edgeArray[i] + (activation.edgeTargets[i] - edgeArray[i]) * blend;
      if (Math.abs(next - edgeArray[i]) > 1e-4) {
        edgeArray[i] = next;
        edgeMoved = true;
      }
    }
    if (edgeMoved) edgeAttr.needsUpdate = true;
  });

  return (
    <group>
      {/* Instances span the whole world — geometry-bounds culling would
          wrongly drop them when the local-origin sphere leaves the frustum */}
      <instancedMesh
        ref={meshRef}
        args={[nodeGeometry, nodeMaterial, instances.length]}
        frustumCulled={false}
      />
      <lineSegments
        ref={lineRef}
        geometry={connectionGeometry}
        material={connectionMaterial}
        frustumCulled={false}
      />
    </group>
  );
}
