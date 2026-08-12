import { createGalaxyInitialState } from "./galaxyPhysics";
import { createAllPairsSolver } from "./galaxyAllPairsSolver";
import { createBarnesHutSolver } from "./galaxyBarnesHutSolver";
import {
  createBufferWithData,
  createStorageBuffer,
  type GalaxySolverFactory,
  type GalaxySolverKind,
} from "./galaxySolver";
import {
  createInitialActiveIndices,
  createInitialAdaptiveControl,
  createInitialIndirectDispatch,
} from "./galaxyAdaptiveTimesteps";

const SAMPLE_TARGETS = 192;

type Vec3 = [number, number, number];

export type GalaxySolverAccuracy = {
  solver: GalaxySolverKind;
  textureWidth: number;
  particleCount: number;
  sampledTargets: number;
  normalizedRmsError: number;
  medianRelativeError: number;
  p95RelativeError: number;
  p99RelativeError: number;
  maximumRelativeError: number;
  momentumResidual: number;
  torqueResidual: number;
};

const factories: Record<GalaxySolverKind, GalaxySolverFactory> = {
  "all-pairs": createAllPairsSolver,
  "barnes-hut": createBarnesHutSolver,
};

const magnitude = (value: Vec3) => Math.hypot(value[0], value[1], value[2]);

const exactAcceleration = (
  state: Float32Array,
  particleCount: number,
  gravity: number,
  targetIndex: number,
): Vec3 => {
  const targetOffset = targetIndex * 8;
  const targetX = state[targetOffset]!;
  const targetY = state[targetOffset + 1]!;
  const targetZ = state[targetOffset + 2]!;
  const targetSoftening = state[targetOffset + 7]!;
  let accelerationX = 0;
  let accelerationY = 0;
  let accelerationZ = 0;
  for (let sourceIndex = 0; sourceIndex < particleCount; sourceIndex++) {
    if (sourceIndex === targetIndex) continue;
    const sourceOffset = sourceIndex * 8;
    const mass = state[sourceOffset + 3]!;
    if (mass <= 0) continue;
    const dx = state[sourceOffset]! - targetX;
    const dy = state[sourceOffset + 1]! - targetY;
    const dz = state[sourceOffset + 2]! - targetZ;
    const sourceSoftening = state[sourceOffset + 7]!;
    const softeningSquared = 0.5 * (
      targetSoftening * targetSoftening + sourceSoftening * sourceSoftening
    );
    const distanceSquared = dx * dx + dy * dy + dz * dz + softeningSquared;
    const scale = gravity * mass / (distanceSquared * Math.sqrt(distanceSquared));
    accelerationX += dx * scale;
    accelerationY += dy * scale;
    accelerationZ += dz * scale;
  }
  return [accelerationX, accelerationY, accelerationZ];
};

const percentile = (sorted: number[], fraction: number) =>
  sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))] ?? 0;

const sampleIndices = (particleCount: number, core2Index: number) => {
  const indices = new Set<number>([0, core2Index]);
  const stride = particleCount / Math.max(1, SAMPLE_TARGETS - indices.size);
  for (let sample = 0; sample < SAMPLE_TARGETS; sample++) {
    indices.add(Math.min(particleCount - 1, Math.floor(sample * stride)));
  }
  return [...indices].sort((left, right) => left - right).slice(0, SAMPLE_TARGETS);
};

export const runGalaxySolverAccuracy = async (
  device: GPUDevice,
  solverKind: GalaxySolverKind,
  textureWidth = 56,
): Promise<GalaxySolverAccuracy> => {
  const initial = createGalaxyInitialState({
    textureWidth,
    radius: 35,
    offset: 25,
  });
  const stateBuffers = [0, 1].map((index) => createBufferWithData(
    device,
    `Accuracy state ${index}`,
    initial.state,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  )) as [GPUBuffer, GPUBuffer];
  const accelerationBuffer = createStorageBuffer(
    device,
    "Accuracy acceleration",
    initial.parameters.particleCount * 16,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const readback = device.createBuffer({
    label: "Accuracy acceleration readback",
    size: initial.parameters.particleCount * 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const activeIndicesBuffer = createBufferWithData(
    device,
    "Accuracy active indices",
    createInitialActiveIndices(initial.parameters.particleCount),
    GPUBufferUsage.STORAGE,
  );
  const adaptiveControlBuffer = createBufferWithData(
    device,
    "Accuracy adaptive control",
    createInitialAdaptiveControl(initial.parameters.particleCount),
    GPUBufferUsage.STORAGE,
  );
  const indirectDispatchBuffer = createBufferWithData(
    device,
    "Accuracy indirect dispatch",
    createInitialIndirectDispatch(initial.parameters.particleCount),
    GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
  );
  const solver = await factories[solverKind]({
    device,
    initial,
    stateBuffers,
    accelerationBuffer,
    activeIndicesBuffer,
    adaptiveControlBuffer,
    indirectDispatchBuffer,
  });
  const encoder = device.createCommandEncoder({ label: "Solver accuracy encoder" });
  solver.encode(encoder, 0);
  encoder.copyBufferToBuffer(
    accelerationBuffer,
    0,
    readback,
    0,
    initial.parameters.particleCount * 16,
  );
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const approximate = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  const indices = sampleIndices(
    initial.parameters.particleCount,
    initial.parameters.core2Index,
  );
  const relativeErrors: number[] = [];
  let squaredError = 0;
  let squaredReference = 0;
  let weightedForce: Vec3 = [0, 0, 0];
  let weightedForceMagnitude = 0;
  let weightedTorque: Vec3 = [0, 0, 0];
  let weightedTorqueMagnitude = 0;
  for (const index of indices) {
    const exact = exactAcceleration(
      initial.state,
      initial.parameters.particleCount,
      initial.parameters.gravity,
      index,
    );
    const approximateOffset = index * 4;
    const measured: Vec3 = [
      approximate[approximateOffset]!,
      approximate[approximateOffset + 1]!,
      approximate[approximateOffset + 2]!,
    ];
    const error: Vec3 = [
      measured[0] - exact[0],
      measured[1] - exact[1],
      measured[2] - exact[2],
    ];
    const exactMagnitude = magnitude(exact);
    const errorMagnitude = magnitude(error);
    relativeErrors.push(errorMagnitude / Math.max(exactMagnitude, 1e-6));
    squaredError += errorMagnitude * errorMagnitude;
    squaredReference += exactMagnitude * exactMagnitude;

  }
  for (let index = 0; index < initial.parameters.particleCount; index++) {
    const stateOffset = index * 8;
    const approximateOffset = index * 4;
    const mass = initial.state[stateOffset + 3]!;
    const force: Vec3 = [
      approximate[approximateOffset]! * mass,
      approximate[approximateOffset + 1]! * mass,
      approximate[approximateOffset + 2]! * mass,
    ];
    weightedForce = weightedForce.map(
      (component, axis) => component + force[axis]!,
    ) as Vec3;
    weightedForceMagnitude += magnitude(force);
    const position: Vec3 = [
      initial.state[stateOffset]!,
      initial.state[stateOffset + 1]!,
      initial.state[stateOffset + 2]!,
    ];
    const torque: Vec3 = [
      position[1] * force[2] - position[2] * force[1],
      position[2] * force[0] - position[0] * force[2],
      position[0] * force[1] - position[1] * force[0],
    ];
    weightedTorque = weightedTorque.map(
      (component, axis) => component + torque[axis]!,
    ) as Vec3;
    weightedTorqueMagnitude += magnitude(torque);
  }
  relativeErrors.sort((left, right) => left - right);

  solver.destroy();
  activeIndicesBuffer.destroy();
  adaptiveControlBuffer.destroy();
  indirectDispatchBuffer.destroy();
  stateBuffers.forEach((buffer) => buffer.destroy());
  accelerationBuffer.destroy();
  readback.destroy();

  return {
    solver: solverKind,
    textureWidth: initial.settings.textureWidth,
    particleCount: initial.parameters.particleCount,
    sampledTargets: indices.length,
    normalizedRmsError: Math.sqrt(squaredError / Math.max(squaredReference, 1e-20)),
    medianRelativeError: percentile(relativeErrors, 0.5),
    p95RelativeError: percentile(relativeErrors, 0.95),
    p99RelativeError: percentile(relativeErrors, 0.99),
    maximumRelativeError: relativeErrors.at(-1) ?? 0,
    momentumResidual: magnitude(weightedForce) / Math.max(weightedForceMagnitude, 1e-20),
    torqueResidual: magnitude(weightedTorque) / Math.max(weightedTorqueMagnitude, 1e-20),
  };
};
