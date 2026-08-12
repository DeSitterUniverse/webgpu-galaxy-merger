import { describe, expect, it } from "vitest";
import { sanitizeGalaxySettings } from "./galaxyPhysics";
import {
  GALAXY_SOLVERS,
  sanitizeGalaxySettingsForSolver,
} from "./galaxySolver";

describe("galaxy solver selection", () => {
  it("keeps all-pairs bounded while exposing the million-body tree ceiling", () => {
    expect(GALAXY_SOLVERS["all-pairs"].maxTextureWidth).toBe(184);
    expect(GALAXY_SOLVERS["barnes-hut"].maxTextureWidth).toBe(1024);
  });

  it("retains a shared physics ceiling at one million particles", () => {
    expect(
      sanitizeGalaxySettings({ textureWidth: 320, radius: 35, offset: 25 }),
    ).toMatchObject({ textureWidth: 320 });
    expect(
      sanitizeGalaxySettings({ textureWidth: 2048, radius: 35, offset: 25 }),
    ).toMatchObject({ textureWidth: 1024 });
  });

  it("sanitizes invalid settings before solver-specific allocation", () => {
    expect(sanitizeGalaxySettingsForSolver({
      textureWidth: Number.NaN,
      radius: Number.POSITIVE_INFINITY,
      offset: Number.NaN,
    }, "all-pairs")).toMatchObject({
      textureWidth: 72,
      radius: 35,
      offset: 25,
    });
    expect(sanitizeGalaxySettingsForSolver({
      textureWidth: 1024,
      radius: 35,
      offset: 25,
    }, "all-pairs")).toMatchObject({ textureWidth: 184 });
  });
});
