import { describe, expect, it } from "vitest";
import {
  META_CORE,
  META_HALO,
  createGalaxyInitialState,
} from "../../src/galaxyPhysics";
import {
  componentShape,
  diagnostics,
  evolveReference,
  isolatedFirstGalaxy,
} from "./referenceNbody";

const relativeChange = (before: number, after: number) =>
  Math.abs(after - before) / Math.max(Math.abs(before), Number.EPSILON);

const magnitude = (value: readonly number[]) => Math.hypot(...value);
const difference = (after: readonly number[], before: readonly number[]) =>
  after.map((value, index) => value - before[index]!);

describe("isolated live galaxy validation", () => {
  it("keeps the composite-potential halo stable with disk and core held fixed", () => {
    const initial = createGalaxyInitialState({
      textureWidth: 56,
      radius: 35,
      offset: 25,
    });
    const halo = isolatedFirstGalaxy(
      initial,
      (metadata) => (metadata & META_HALO) !== 0,
    );
    const before = componentShape(halo, (metadata) => (metadata & META_HALO) !== 0);
    evolveReference(halo, initial.parameters.timeStep, 64);
    const after = componentShape(halo, (metadata) => (metadata & META_HALO) !== 0);
    const radiusDrift = relativeChange(before.rmsRadius, after.rmsRadius);

    console.info("galaxy-test isolated halo", { radiusDrift });

    expect(radiusDrift).toBeLessThan(0.08);
  });

  it("tracks conservation and component settling in an isolated live galaxy", () => {
    const initial = createGalaxyInitialState({
      textureWidth: 56,
      radius: 35,
      offset: 25,
    });
    const system = isolatedFirstGalaxy(initial);
    const before = diagnostics(system);
    const diskBefore = componentShape(
      system,
      (metadata) => (metadata & (META_CORE | META_HALO)) === 0,
    );
    const haloBefore = componentShape(
      system,
      (metadata) => (metadata & META_HALO) !== 0,
    );
    evolveReference(system, initial.parameters.timeStep, 64);
    const after = diagnostics(system);
    const diskAfter = componentShape(
      system,
      (metadata) => (metadata & (META_CORE | META_HALO)) === 0,
    );
    const haloAfter = componentShape(
      system,
      (metadata) => (metadata & META_HALO) !== 0,
    );

    console.info("galaxy-test diagnostics", {
      energy: {
        initial: before.totalEnergy,
        final: after.totalEnergy,
        relativeDrift: relativeChange(before.totalEnergy, after.totalEnergy),
      },
      momentum: {
        initial: before.momentum,
        final: after.momentum,
        driftMagnitude: magnitude(difference(after.momentum, before.momentum)),
      },
      centerOfMass: {
        initial: before.centerOfMass,
        final: after.centerOfMass,
        driftMagnitude: magnitude(
          difference(after.centerOfMass, before.centerOfMass),
        ),
      },
      angularMomentum: {
        initial: before.angularMomentum,
        final: after.angularMomentum,
        relativeMagnitudeDrift: relativeChange(
          magnitude(before.angularMomentum),
          magnitude(after.angularMomentum),
        ),
      },
      diskRadiusDrift: relativeChange(
        diskBefore.rmsRadius,
        diskAfter.rmsRadius,
      ),
      haloRadiusDrift: relativeChange(
        haloBefore.rmsRadius,
        haloAfter.rmsRadius,
      ),
      virialRatio: { initial: before.virialRatio, final: after.virialRatio },
    });

    expect(relativeChange(before.totalEnergy, after.totalEnergy)).toBeLessThan(0.005);
    expect(magnitude(after.momentum)).toBeLessThan(1e-6);
    expect(magnitude(after.centerOfMass)).toBeLessThan(1e-6);
    expect(relativeChange(
      magnitude(before.angularMomentum),
      magnitude(after.angularMomentum),
    )).toBeLessThan(1e-10);
    expect(relativeChange(diskBefore.rmsRadius, diskAfter.rmsRadius)).toBeLessThan(0.12);
    expect(relativeChange(haloBefore.rmsRadius, haloAfter.rmsRadius)).toBeLessThan(0.08);
    expect(after.virialRatio).toBeGreaterThan(0.75);
    expect(after.virialRatio).toBeLessThan(1.25);
  });
});
