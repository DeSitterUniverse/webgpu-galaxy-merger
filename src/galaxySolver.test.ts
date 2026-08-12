import { describe, expect, it } from "vitest";
import { sanitizeGalaxySettings } from "./galaxyPhysics";
import { GALAXY_SOLVERS } from "./galaxySolver";

describe("galaxy solver selection", () => {
  it("assigns measured interactive limits to each solver", () => {
    expect(GALAXY_SOLVERS["all-pairs"].maxTextureWidth).toBe(160);
    expect(GALAXY_SOLVERS["barnes-hut"].maxTextureWidth).toBe(256);
  });

  it("retains a shared physics ceiling above the UI solver limits", () => {
    expect(
      sanitizeGalaxySettings({ textureWidth: 320, radius: 35, offset: 25 }),
    ).toMatchObject({ textureWidth: 320 });
    expect(
      sanitizeGalaxySettings({ textureWidth: 512, radius: 35, offset: 25 }),
    ).toMatchObject({ textureWidth: 320 });
  });
});
