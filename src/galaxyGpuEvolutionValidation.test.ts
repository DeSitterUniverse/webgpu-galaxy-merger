import { describe, expect, it } from "vitest";
import {
  relativeVectorDrift,
  synchronizeBlockVelocity,
  vectorDifferenceMagnitude,
} from "./galaxyGpuEvolutionValidation";

describe("GPU evolution diagnostics", () => {
  it("synchronizes block velocity from both force endpoints", () => {
    expect(synchronizeBlockVelocity(10, 2, 4, 8, 0, 0.5)).toBe(6);
    expect(synchronizeBlockVelocity(10, 2, 4, 8, 8, 0.5)).toBe(18);
  });

  it("detects conserved-vector rotation even when magnitudes match", () => {
    const initial = [0, 0, 10];
    const final = [0, 10, 0];
    expect(vectorDifferenceMagnitude(final, initial)).toBeCloseTo(Math.sqrt(200));
    expect(relativeVectorDrift(final, initial)).toBeCloseTo(Math.sqrt(2));
  });
});
