import { describe, it, expect } from "vitest";
import { profile, stats, careerTrajectory } from "../profile";

describe("profile data", () => {
  it("has all required profile fields", () => {
    expect(profile.name).toBe("VASU AGRAWAL");
    expect(profile.title).toBeTruthy();
    expect(profile.status).toBe("OPEN_TO_WORK");
    expect(profile.location).toBeTruthy();
    expect(profile.bio).toBeTruthy();
  });

  it("has 4 stats entries", () => {
    expect(stats).toHaveLength(4);
    stats.forEach((stat) => {
      expect(stat).toHaveProperty("label");
      expect(stat).toHaveProperty("value");
    });
  });

  it("has career trajectory in chronological order", () => {
    expect(careerTrajectory.length).toBeGreaterThan(0);
    for (let i = 1; i < careerTrajectory.length; i++) {
      expect(Number(careerTrajectory[i].year)).toBeGreaterThan(
        Number(careerTrajectory[i - 1].year),
      );
    }
  });

  it("has increasing level values in trajectory", () => {
    for (let i = 1; i < careerTrajectory.length; i++) {
      expect(careerTrajectory[i].level).toBeGreaterThan(
        careerTrajectory[i - 1].level,
      );
    }
  });
});
