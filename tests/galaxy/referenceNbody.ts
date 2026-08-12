import type { GalaxyInitialState } from "../../src/galaxyPhysics";

export type ReferenceSystem = {
  count: number;
  gravity: number;
  positions: Float64Array;
  velocities: Float64Array;
  masses: Float64Array;
  softenings: Float64Array;
  metadata: Uint32Array;
  dynamic: Uint8Array;
};

export type ConservationDiagnostics = {
  kinetic: number;
  potential: number;
  totalEnergy: number;
  momentum: [number, number, number];
  angularMomentum: [number, number, number];
  centerOfMass: [number, number, number];
  virialRatio: number;
};

export const isolatedFirstGalaxy = (
  initial: GalaxyInitialState,
  dynamicPredicate: (metadata: number) => boolean = () => true,
): ReferenceSystem => {
  const count = initial.parameters.particlesPerGalaxy;
  const positions = new Float64Array(count * 3);
  const velocities = new Float64Array(count * 3);
  const masses = new Float64Array(count);
  const softenings = new Float64Array(count);
  const metadata = initial.metadata.slice(0, count);
  const dynamic = new Uint8Array(count);
  const corePosition = initial.state.slice(0, 3);
  const coreVelocity = initial.state.slice(4, 7);
  for (let index = 0; index < count; index++) {
    const source = index * 8;
    const target = index * 3;
    masses[index] = initial.state[source + 3]!;
    softenings[index] = initial.state[source + 7]!;
    dynamic[index] = dynamicPredicate(metadata[index]!) ? 1 : 0;
    for (let axis = 0; axis < 3; axis++) {
      positions[target + axis] =
        initial.state[source + axis]! - corePosition[axis]!;
      velocities[target + axis] =
        initial.state[source + 4 + axis]! - coreVelocity[axis]!;
    }
  }
  return {
    count,
    gravity: initial.parameters.gravity,
    positions,
    velocities,
    masses,
    softenings,
    metadata,
    dynamic,
  };
};

const accelerations = (system: ReferenceSystem) => {
  const output = new Float64Array(system.count * 3);
  for (let left = 0; left < system.count; left++) {
    const leftOffset = left * 3;
    for (let right = left + 1; right < system.count; right++) {
      if (!system.dynamic[left] && !system.dynamic[right]) continue;
      const rightOffset = right * 3;
      const dx = system.positions[rightOffset]! - system.positions[leftOffset]!;
      const dy = system.positions[rightOffset + 1]! - system.positions[leftOffset + 1]!;
      const dz = system.positions[rightOffset + 2]! - system.positions[leftOffset + 2]!;
      const pairSofteningSquared = 0.5 * (
        system.softenings[left]! ** 2 + system.softenings[right]! ** 2
      );
      const distanceSquared = dx * dx + dy * dy + dz * dz + pairSofteningSquared;
      const inverseDistance = 1 / Math.sqrt(distanceSquared);
      const inverseDistanceCubed = inverseDistance ** 3;
      if (system.dynamic[left]) {
        const factor = system.gravity * system.masses[right]! * inverseDistanceCubed;
        output[leftOffset] = output[leftOffset]! + factor * dx;
        output[leftOffset + 1] = output[leftOffset + 1]! + factor * dy;
        output[leftOffset + 2] = output[leftOffset + 2]! + factor * dz;
      }
      if (system.dynamic[right]) {
        const factor = system.gravity * system.masses[left]! * inverseDistanceCubed;
        output[rightOffset] = output[rightOffset]! - factor * dx;
        output[rightOffset + 1] = output[rightOffset + 1]! - factor * dy;
        output[rightOffset + 2] = output[rightOffset + 2]! - factor * dz;
      }
    }
  }
  return output;
};

export const evolveReference = (
  system: ReferenceSystem,
  timeStep: number,
  steps: number,
) => {
  let acceleration = accelerations(system);
  for (let step = 0; step < steps; step++) {
    for (let index = 0; index < system.count; index++) {
      if (!system.dynamic[index]) continue;
      const offset = index * 3;
      for (let axis = 0; axis < 3; axis++) {
        system.velocities[offset + axis] =
          system.velocities[offset + axis]! +
          0.5 * acceleration[offset + axis]! * timeStep;
        system.positions[offset + axis] =
          system.positions[offset + axis]! +
          system.velocities[offset + axis]! * timeStep;
      }
    }
    acceleration = accelerations(system);
    for (let index = 0; index < system.count; index++) {
      if (!system.dynamic[index]) continue;
      const offset = index * 3;
      for (let axis = 0; axis < 3; axis++) {
        system.velocities[offset + axis] =
          system.velocities[offset + axis]! +
          0.5 * acceleration[offset + axis]! * timeStep;
      }
    }
  }
};

export const diagnostics = (
  system: ReferenceSystem,
): ConservationDiagnostics => {
  let totalMass = 0;
  let kinetic = 0;
  let potential = 0;
  const momentum: [number, number, number] = [0, 0, 0];
  const angularMomentum: [number, number, number] = [0, 0, 0];
  const centerOfMass: [number, number, number] = [0, 0, 0];
  for (let index = 0; index < system.count; index++) {
    const offset = index * 3;
    const mass = system.masses[index]!;
    const x = system.positions[offset]!;
    const y = system.positions[offset + 1]!;
    const z = system.positions[offset + 2]!;
    const vx = system.velocities[offset]!;
    const vy = system.velocities[offset + 1]!;
    const vz = system.velocities[offset + 2]!;
    totalMass += mass;
    kinetic += 0.5 * mass * (vx * vx + vy * vy + vz * vz);
    momentum[0] += mass * vx;
    momentum[1] += mass * vy;
    momentum[2] += mass * vz;
    angularMomentum[0] += mass * (y * vz - z * vy);
    angularMomentum[1] += mass * (z * vx - x * vz);
    angularMomentum[2] += mass * (x * vy - y * vx);
    centerOfMass[0] += mass * x;
    centerOfMass[1] += mass * y;
    centerOfMass[2] += mass * z;
    for (let right = index + 1; right < system.count; right++) {
      const rightOffset = right * 3;
      const dx = system.positions[rightOffset]! - x;
      const dy = system.positions[rightOffset + 1]! - y;
      const dz = system.positions[rightOffset + 2]! - z;
      const pairSofteningSquared = 0.5 * (
        system.softenings[index]! ** 2 + system.softenings[right]! ** 2
      );
      potential -= system.gravity * mass * system.masses[right]! /
        Math.sqrt(dx * dx + dy * dy + dz * dz + pairSofteningSquared);
    }
  }
  centerOfMass[0] /= totalMass;
  centerOfMass[1] /= totalMass;
  centerOfMass[2] /= totalMass;
  return {
    kinetic,
    potential,
    totalEnergy: kinetic + potential,
    momentum,
    angularMomentum,
    centerOfMass,
    virialRatio: 2 * kinetic / Math.abs(potential),
  };
};

export const componentShape = (
  system: ReferenceSystem,
  predicate: (metadata: number) => boolean,
) => {
  let count = 0;
  let radiusSquared = 0;
  let heightSquared = 0;
  for (let index = 0; index < system.count; index++) {
    if (!predicate(system.metadata[index]!)) continue;
    const offset = index * 3;
    const x = system.positions[offset]!;
    const y = system.positions[offset + 1]!;
    const z = system.positions[offset + 2]!;
    count++;
    radiusSquared += x * x + y * y + z * z;
    heightSquared += y * y;
  }
  return {
    rmsRadius: Math.sqrt(radiusSquared / count),
    rmsHeight: Math.sqrt(heightSquared / count),
  };
};

// Measure a disk in its own instantaneous angular-momentum frame. This keeps
// the diagnostic valid after initialization tilts the galaxy in world space.
export const diskHeatingDiagnostics = (
  system: ReferenceSystem,
  predicate: (metadata: number) => boolean,
) => {
  let totalMass = 0;
  const center = [0, 0, 0];
  const bulk = [0, 0, 0];
  for (let index = 0; index < system.count; index++) {
    if (!predicate(system.metadata[index]!)) continue;
    const mass = system.masses[index]!;
    const offset = index * 3;
    totalMass += mass;
    for (let axis = 0; axis < 3; axis++) {
      center[axis] = center[axis]! + mass * system.positions[offset + axis]!;
      bulk[axis] = bulk[axis]! + mass * system.velocities[offset + axis]!;
    }
  }
  for (let axis = 0; axis < 3; axis++) {
    center[axis] = center[axis]! / totalMass;
    bulk[axis] = bulk[axis]! / totalMass;
  }
  const angularMomentum = [0, 0, 0];
  for (let index = 0; index < system.count; index++) {
    if (!predicate(system.metadata[index]!)) continue;
    const offset = index * 3;
    const mass = system.masses[index]!;
    const x = system.positions[offset]! - center[0]!;
    const y = system.positions[offset + 1]! - center[1]!;
    const z = system.positions[offset + 2]! - center[2]!;
    const vx = system.velocities[offset]! - bulk[0]!;
    const vy = system.velocities[offset + 1]! - bulk[1]!;
    const vz = system.velocities[offset + 2]! - bulk[2]!;
    angularMomentum[0] = angularMomentum[0]! + mass * (y * vz - z * vy);
    angularMomentum[1] = angularMomentum[1]! + mass * (z * vx - x * vz);
    angularMomentum[2] = angularMomentum[2]! + mass * (x * vy - y * vx);
  }
  const angularMagnitude = Math.max(Math.hypot(...angularMomentum), 1e-20);
  const normal = angularMomentum.map((value) => value / angularMagnitude);
  let heightSquared = 0;
  let radialVelocitySquared = 0;
  let verticalVelocitySquared = 0;
  let tangentialSpeed = 0;
  let measuredMass = 0;
  for (let index = 0; index < system.count; index++) {
    if (!predicate(system.metadata[index]!)) continue;
    const offset = index * 3;
    const mass = system.masses[index]!;
    const position = [
      system.positions[offset]! - center[0]!,
      system.positions[offset + 1]! - center[1]!,
      system.positions[offset + 2]! - center[2]!,
    ];
    const velocity = [
      system.velocities[offset]! - bulk[0]!,
      system.velocities[offset + 1]! - bulk[1]!,
      system.velocities[offset + 2]! - bulk[2]!,
    ];
    const height = position.reduce(
      (sum, value, axis) => sum + value * normal[axis]!,
      0,
    );
    const planar = position.map((value, axis) => value - height * normal[axis]!);
    const planarRadius = Math.max(Math.hypot(...planar), 1e-20);
    const radial = planar.map((value) => value / planarRadius);
    const tangential = [
      normal[1]! * radial[2]! - normal[2]! * radial[1]!,
      normal[2]! * radial[0]! - normal[0]! * radial[2]!,
      normal[0]! * radial[1]! - normal[1]! * radial[0]!,
    ];
    const radialVelocity = velocity.reduce(
      (sum, value, axis) => sum + value * radial[axis]!,
      0,
    );
    const verticalVelocity = velocity.reduce(
      (sum, value, axis) => sum + value * normal[axis]!,
      0,
    );
    const localTangentialSpeed = velocity.reduce(
      (sum, value, axis) => sum + value * tangential[axis]!,
      0,
    );
    measuredMass += mass;
    heightSquared += mass * height * height;
    radialVelocitySquared += mass * radialVelocity * radialVelocity;
    verticalVelocitySquared += mass * verticalVelocity * verticalVelocity;
    tangentialSpeed += mass * localTangentialSpeed;
  }
  return {
    rmsHeight: Math.sqrt(heightSquared / measuredMass),
    radialVelocityDispersion: Math.sqrt(radialVelocitySquared / measuredMass),
    verticalVelocityDispersion: Math.sqrt(verticalVelocitySquared / measuredMass),
    meanTangentialSpeed: tangentialSpeed / measuredMass,
  };
};
