import { describe, expect, it } from "vitest";
import * as THREE from "three";

import { dampTowards, deriveDivePose } from "../camera-dive";

const NODE: [number, number, number] = [10, -2, 4];
const REST = {
  position: new THREE.Vector3(2, 0, 14),
  target: new THREE.Vector3(8, -2, 3),
};

describe("deriveDivePose", () => {
  it("targets the node center exactly", () => {
    const pose = deriveDivePose(NODE, 0.5, REST);
    expect(pose.target.toArray()).toEqual(NODE);
  });

  it("positions the camera between the rest pose and the node (dive-in feel)", () => {
    const pose = deriveDivePose(NODE, 0.5, REST);
    const restToNode = new THREE.Vector3(...NODE).sub(REST.position).length();
    const camToNode = new THREE.Vector3(...NODE).sub(pose.position).length();
    expect(camToNode).toBeLessThan(restToNode);
  });

  it("keeps a near-distance proportional to node scale, floored", () => {
    const small = deriveDivePose(NODE, 0.2, REST);
    const large = deriveDivePose(NODE, 0.6, REST);
    const d = (p: { position: THREE.Vector3 }) =>
      new THREE.Vector3(...NODE).sub(p.position).length();
    expect(d(large)).toBeGreaterThan(d(small));
    expect(d(small)).toBeGreaterThanOrEqual(1.2); // floor: never inside geometry
  });

  it("is deterministic", () => {
    const a = deriveDivePose(NODE, 0.5, REST);
    const b = deriveDivePose(NODE, 0.5, REST);
    expect(a.position.toArray()).toEqual(b.position.toArray());
    expect(a.target.toArray()).toEqual(b.target.toArray());
  });

  it("degenerate case: rest pose at the node still yields a valid offset", () => {
    const pose = deriveDivePose(
      NODE,
      0.5,
      { position: new THREE.Vector3(...NODE), target: REST.target },
    );
    expect(Number.isFinite(pose.position.x)).toBe(true);
    expect(pose.position.distanceTo(new THREE.Vector3(...NODE))).toBeGreaterThan(1);
  });
});

describe("dampTowards", () => {
  it("moves toward the target without overshooting", () => {
    const next = dampTowards(0, 1, 1 / 60, 9);
    expect(next).toBeGreaterThan(0);
    expect(next).toBeLessThan(1);
  });

  it("converges to the target", () => {
    let v = 0;
    for (let i = 0; i < 300; i += 1) v = dampTowards(v, 1, 1 / 60, 9);
    expect(v).toBeCloseTo(1, 3);
  });

  it("handles retargeting mid-flight (rapid open/close)", () => {
    let v = 0;
    for (let i = 0; i < 20; i += 1) v = dampTowards(v, 1, 1 / 60, 9);
    const mid = v;
    for (let i = 0; i < 300; i += 1) v = dampTowards(v, 0, 1 / 60, 9);
    expect(mid).toBeGreaterThan(0.5);
    expect(v).toBeCloseTo(0, 3);
  });
});
