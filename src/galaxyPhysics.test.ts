import { describe, expect, it } from "vitest";
import {
  CORE_PHASE_BINARY,
  CORE_PHASE_MERGED,
  CORE_PHASE_SEPARATE,
  META_CORE,
  META_HALO,
  advanceCoreBinaryState,
  createGalaxyInitialState,
  sanitizeGalaxySettings,
  symmetricSofteningSquared,
  sweptClosestApproach,
} from "./galaxyPhysics";

const defaultSettings = { textureWidth: 56, radius: 35, offset: 25 };

describe("live galaxy initialization", () => {
  it("is byte-for-byte deterministic", () => {
    const first = createGalaxyInitialState(defaultSettings);
    const second = createGalaxyInitialState(defaultSettings);
    expect(first.state).toEqual(second.state);
    expect(first.metadata).toEqual(second.metadata);
    expect(first.visuals).toEqual(second.visuals);
  });

  it("creates massive live disk, halo, and core components", () => {
    const initial = createGalaxyInitialState(defaultSettings);
    const { parameters } = initial;
    const componentMass = (mask: number, exact = false) => {
      let mass = 0;
      for (let index = 0; index < parameters.particlesPerGalaxy; index++) {
        const matches = exact
          ? initial.metadata[index] === mask
          : (initial.metadata[index]! & mask) !== 0;
        if (matches) mass += initial.state[index * 8 + 3]!;
      }
      return mass;
    };
    expect(componentMass(META_CORE)).toBeCloseTo(parameters.coreMass, 6);
    expect(componentMass(META_HALO)).toBeCloseTo(parameters.haloMass, 5);
    expect(componentMass(0, true)).toBeCloseTo(parameters.diskMass, 5);
    expect(parameters.coreMass + parameters.diskMass + parameters.haloMass).toBe(5);
  });

  it("keeps each galaxy centered with zero internal momentum", () => {
    const initial = createGalaxyInitialState(defaultSettings);
    const { parameters } = initial;
    for (let galaxy = 0; galaxy < 2; galaxy++) {
      const start = galaxy * parameters.particlesPerGalaxy;
      const center = galaxy === 0
        ? [-25, 0, -parameters.orthogonalHalfSeparation]
        : [25, 0, parameters.orthogonalHalfSeparation];
      const coreVelocity = initial.state.slice(start * 8 + 4, start * 8 + 7);
      let mass = 0;
      const weightedPosition = [0, 0, 0];
      const internalMomentum = [0, 0, 0];
      for (let local = 0; local < parameters.particlesPerGalaxy; local++) {
        const offset = (start + local) * 8;
        const particleMass = initial.state[offset + 3]!;
        mass += particleMass;
        for (let axis = 0; axis < 3; axis++) {
          weightedPosition[axis]! += initial.state[offset + axis]! * particleMass;
          internalMomentum[axis]! +=
            (initial.state[offset + 4 + axis]! - coreVelocity[axis]!) * particleMass;
        }
      }
      for (let axis = 0; axis < 3; axis++) {
        expect(weightedPosition[axis]! / mass).toBeCloseTo(center[axis]!, 4);
        expect(internalMomentum[axis]!).toBeCloseTo(0, 4);
      }
    }
  });

  it("maps offset literally to the two core centers", () => {
    const initial = createGalaxyInitialState({ ...defaultSettings, offset: 10 });
    expect(initial.parameters.halfSeparation).toBe(10);
    expect(initial.state[0]).toBe(-10);
    expect(initial.state[2]).toBeCloseTo(
      -initial.parameters.orthogonalHalfSeparation,
      5,
    );
    const second = initial.parameters.core2Index * 8;
    expect(initial.state[second]).toBe(10);
    expect(initial.state[second + 2]).toBeCloseTo(
      initial.parameters.orthogonalHalfSeparation,
      5,
    );
    expect(initial.parameters.centerSeparation).toBeGreaterThanOrEqual(
      initial.parameters.radius * 2.1,
    );
  });

  it("centers disk and halo independently on each compact core", () => {
    const initial = createGalaxyInitialState(defaultSettings);
    const { parameters } = initial;
    for (let galaxy = 0; galaxy < 2; galaxy++) {
      const start = galaxy * parameters.particlesPerGalaxy;
      const coreOffset = start * 8;
      for (const halo of [false, true]) {
        let mass = 0;
        const weighted = [0, 0, 0];
        const momentum = [0, 0, 0];
        for (let local = 1; local < parameters.particlesPerGalaxy; local++) {
          const index = start + local;
          if (((initial.metadata[index]! & META_HALO) !== 0) !== halo) continue;
          const offset = index * 8;
          const particleMass = initial.state[offset + 3]!;
          mass += particleMass;
          for (let axis = 0; axis < 3; axis++) {
            weighted[axis]! +=
              (initial.state[offset + axis]! - initial.state[coreOffset + axis]!) *
              particleMass;
            momentum[axis]! +=
              (initial.state[offset + 4 + axis]! -
                initial.state[coreOffset + 4 + axis]!) * particleMass;
          }
        }
        for (let axis = 0; axis < 3; axis++) {
          expect(weighted[axis]! / mass).toBeCloseTo(0, 5);
          expect(momentum[axis]!).toBeCloseTo(0, 4);
        }
      }
    }
  });

  it("keeps component masses fixed as resolution changes", () => {
    const low = createGalaxyInitialState(defaultSettings);
    const high = createGalaxyInitialState({ ...defaultSettings, textureWidth: 64 });
    const galaxyMass = (initial: typeof low) => {
      let mass = 0;
      for (let index = 0; index < initial.parameters.particlesPerGalaxy; index++) {
        mass += initial.state[index * 8 + 3]!;
      }
      return mass;
    };
    expect(galaxyMass(low)).toBeCloseTo(5, 5);
    expect(galaxyMass(high)).toBeCloseTo(5, 5);
    expect(high.parameters.haloParticlesPerGalaxy).toBeGreaterThan(
      low.parameters.haloParticlesPerGalaxy,
    );
  });

  it("assigns separate symmetric-softening inputs by component", () => {
    const initial = createGalaxyInitialState({
      ...defaultSettings,
      textureWidth: 72,
    });
    const { parameters } = initial;
    const diskIndex = 1;
    const haloIndex = 1 + parameters.diskParticlesPerGalaxy;
    expect(initial.state[7]).toBeCloseTo(parameters.coreSoftening, 6);
    expect(initial.state[diskIndex * 8 + 7]).toBeCloseTo(
      parameters.diskSoftening,
      6,
    );
    expect(initial.state[haloIndex * 8 + 7]).toBeCloseTo(
      parameters.haloSoftening,
      6,
    );
    expect(parameters.coreSoftening).toBeLessThan(parameters.diskSoftening);
    expect(parameters.diskSoftening).toBeLessThan(parameters.haloSoftening);
    expect(
      symmetricSofteningSquared(
        parameters.coreSoftening,
        parameters.haloSoftening,
      ),
    ).toBe(
      symmetricSofteningSquared(
        parameters.haloSoftening,
        parameters.coreSoftening,
      ),
    );
    expect(
      symmetricSofteningSquared(
        parameters.diskSoftening,
        parameters.diskSoftening,
      ),
    ).toBeCloseTo(parameters.diskSoftening ** 2, 10);
  });

  it("reduces live-particle softening as resolution increases", () => {
    const low = createGalaxyInitialState({
      ...defaultSettings,
      textureWidth: 32,
    });
    const reference = createGalaxyInitialState({
      ...defaultSettings,
      textureWidth: 72,
    });
    const high = createGalaxyInitialState({
      ...defaultSettings,
      textureWidth: 160,
    });
    expect(low.parameters.diskSoftening).toBeGreaterThan(
      reference.parameters.diskSoftening,
    );
    expect(low.parameters.haloSoftening).toBeGreaterThan(
      reference.parameters.haloSoftening,
    );
    expect(high.parameters.diskSoftening).toBeLessThan(
      reference.parameters.diskSoftening,
    );
    expect(high.parameters.haloSoftening).toBeLessThan(
      reference.parameters.haloSoftening,
    );
    expect(low.parameters.coreSoftening).toBe(
      high.parameters.coreSoftening,
    );
  });

  it("keeps low-resolution halo bodies below ten percent of core mass", () => {
    const initial = createGalaxyInitialState({
      ...defaultSettings,
      textureWidth: 32,
    });
    expect(initial.settings.textureWidth).toBe(56);
    expect(
      initial.parameters.haloMass /
        initial.parameters.haloParticlesPerGalaxy /
        initial.parameters.coreMass,
    ).toBeLessThan(0.1);
  });

  it("protects the innermost orbit with at least 48 timesteps", () => {
    const initial = createGalaxyInitialState({
      ...defaultSettings,
      textureWidth: 160,
    });
    expect(initial.parameters.timeStep).toBeLessThanOrEqual(0.08);
    expect(
      initial.parameters.innerOrbitalPeriod / initial.parameters.timeStep,
    ).toBeGreaterThanOrEqual(48);
  });

  it("keeps live halo bodies gravitational but visually hidden", () => {
    const initial = createGalaxyInitialState(defaultSettings);
    const haloIndex = 1 + initial.parameters.diskParticlesPerGalaxy;
    expect(initial.state[haloIndex * 8 + 3]).toBeGreaterThan(0);
    expect(initial.visuals[haloIndex * 4 + 3]).toBe(0);
  });

  it("constructs mutually inclined stellar angular-momentum axes", () => {
    const initial = createGalaxyInitialState(defaultSettings);
    const { parameters } = initial;
    const diskAngularMomentum = (galaxy: number) => {
      const start = galaxy * parameters.particlesPerGalaxy;
      const coreOffset = start * 8;
      const corePosition = initial.state.slice(coreOffset, coreOffset + 3);
      const coreVelocity = initial.state.slice(coreOffset + 4, coreOffset + 7);
      const angular = [0, 0, 0];
      for (let local = 1; local <= parameters.diskParticlesPerGalaxy; local++) {
        const offset = (start + local) * 8;
        const position = [
          initial.state[offset]! - corePosition[0]!,
          initial.state[offset + 1]! - corePosition[1]!,
          initial.state[offset + 2]! - corePosition[2]!,
        ];
        const velocity = [
          initial.state[offset + 4]! - coreVelocity[0]!,
          initial.state[offset + 5]! - coreVelocity[1]!,
          initial.state[offset + 6]! - coreVelocity[2]!,
        ];
        const mass = initial.state[offset + 3]!;
        angular[0]! +=
          mass * (position[1]! * velocity[2]! - position[2]! * velocity[1]!);
        angular[1]! +=
          mass * (position[2]! * velocity[0]! - position[0]! * velocity[2]!);
        angular[2]! +=
          mass * (position[0]! * velocity[1]! - position[1]! * velocity[0]!);
      }
      const length = Math.hypot(...angular);
      return angular.map((value) => value / length);
    };
    const first = diskAngularMomentum(0);
    const second = diskAngularMomentum(1);
    const alignment =
      first[0]! * second[0]! +
      first[1]! * second[1]! +
      first[2]! * second[2]!;
    expect(Math.abs(alignment)).toBeLessThan(0.9);
  });

  it("does not reseed morphology when controls change", () => {
    const baseline = createGalaxyInitialState(defaultSettings);
    const shifted = createGalaxyInitialState({ ...defaultSettings, offset: 40 });
    expect(shifted.parameters.seed).toBe(baseline.parameters.seed);
    expect(shifted.metadata).toEqual(baseline.metadata);
    expect(shifted.visuals).toEqual(baseline.visuals);
    const half = baseline.parameters.particlesPerGalaxy;
    for (let index = 0; index < baseline.parameters.particleCount; index++) {
      const offset = index * 8;
      const expectedShift = index < half ? -15 : 15;
      expect(shifted.state[offset]! - baseline.state[offset]!).toBeCloseTo(
        expectedShift,
        4,
      );
      expect(shifted.state[offset + 1]).toBe(baseline.state[offset + 1]);
      expect(shifted.state[offset + 2]).toBe(baseline.state[offset + 2]);
    }
  });

  it("detects swept encounters between missed endpoints", () => {
    const closest = sweptClosestApproach([-5, 1, 0], [5, 1, 0]);
    expect(closest).toEqual({ fraction: 0.5, distance: 1 });
  });

  it("derives core capture distance from the symmetric softening scale", () => {
    const { parameters } = createGalaxyInitialState(defaultSettings);
    const pairSofteningSquared = symmetricSofteningSquared(
      parameters.coreSoftening,
      parameters.coreSoftening,
    );
    expect(
      parameters.captureSofteningFactor * Math.sqrt(pairSofteningSquared),
    ).toBeCloseTo(1.8, 6);
  });

  it("does not capture an unbound swept flyby", () => {
    const result = advanceCoreBinaryState({
      state: { phase: CORE_PHASE_SEPARATE, boundElapsed: 0, mergeDelay: 0 },
      startRelative: [-2, 0, 0],
      endRelative: [2, 0, 0],
      relativeVelocity: [10, 0, 0],
      timeStep: 0.08,
      gravity: 80,
      totalMass: 0.1,
      pairSofteningSquared: 0.81,
      captureSofteningFactor: 2,
    });
    expect(result.state.phase).toBe(CORE_PHASE_SEPARATE);
    expect(result.shouldMerge).toBe(false);
  });

  it("captures a bound close passage before merging it", () => {
    const result = advanceCoreBinaryState({
      state: { phase: CORE_PHASE_SEPARATE, boundElapsed: 0, mergeDelay: 0 },
      startRelative: [-2, 0, 0],
      endRelative: [1.5, 0, 0],
      relativeVelocity: [1, 0, 0],
      timeStep: 0.08,
      gravity: 80,
      totalMass: 0.1,
      pairSofteningSquared: 0.81,
      captureSofteningFactor: 2,
    });
    expect(result.state.phase).toBe(CORE_PHASE_BINARY);
    expect(result.state.mergeDelay).toBeGreaterThan(result.state.boundElapsed);
    expect(result.shouldMerge).toBe(false);
  });

  it("merges a bound binary on a later close passage after one unresolved orbit", () => {
    const captured = advanceCoreBinaryState({
      state: { phase: CORE_PHASE_SEPARATE, boundElapsed: 0, mergeDelay: 0 },
      startRelative: [-2, 0, 0],
      endRelative: [1.5, 0, 0],
      relativeVelocity: [1, 0, 0],
      timeStep: 0.08,
      gravity: 80,
      totalMass: 0.1,
      pairSofteningSquared: 0.81,
      captureSofteningFactor: 2,
    });
    const result = advanceCoreBinaryState({
      state: captured.state,
      startRelative: [2, 0, 0],
      endRelative: [1.5, 0, 0],
      relativeVelocity: [0.5, 0, 0],
      timeStep: captured.state.mergeDelay,
      gravity: 80,
      totalMass: 0.1,
      pairSofteningSquared: 0.81,
      captureSofteningFactor: 2,
    });
    expect(result.state.phase).toBe(CORE_PHASE_MERGED);
    expect(result.shouldMerge).toBe(true);
  });

  it("sanitizes unsafe external values", () => {
    expect(
      sanitizeGalaxySettings({
        textureWidth: Number.POSITIVE_INFINITY,
        radius: -100,
        offset: Number.NaN,
      }),
    ).toMatchObject({ textureWidth: 72, radius: 15, offset: 25 });
    expect(
      sanitizeGalaxySettings({ textureWidth: 32, radius: 35, offset: 25 }),
    ).toMatchObject({ textureWidth: 56 });
  });
});
