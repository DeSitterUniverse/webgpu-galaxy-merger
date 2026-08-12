import {
  createInitialActiveIndices,
  createInitialAdaptiveControl,
  createInitialIndirectDispatch,
  galaxyAdaptiveInitializeShader,
  galaxyAdaptiveKickShader,
  galaxyDriftShader,
  galaxyScheduleShader,
  TIMESTEP_ETA,
} from "./galaxyAdaptiveTimesteps";
import { createAllPairsSolver } from "./galaxyAllPairsSolver";
import { createBarnesHutSolver } from "./galaxyBarnesHutSolver";
import type { GalaxyInitialState } from "./galaxyPhysics";
import {
  createBufferWithData,
  type GalaxySolverFactory,
  type GalaxySolverKind,
} from "./galaxySolver";
import { WORKGROUP_SIZE } from "./galaxyShaders";

export type GalaxyGpuEvolutionMode = {
  solver: GalaxySolverKind;
  adaptive: boolean;
};

type Diagnostics = {
  energy: number;
  momentumMagnitude: number;
  angularMomentumMagnitude: number;
  centerOfMassMagnitude: number;
};

export type GalaxyGpuEvolutionResult = GalaxyGpuEvolutionMode & {
  particleCount: number;
  steps: number;
  baseTimeStep: number;
  initial: Diagnostics;
  final: Diagnostics;
  relativeEnergyDrift: number;
  momentumDrift: number;
  relativeAngularMomentumDrift: number;
  centerOfMassDrift: number;
  positionRmsDifferenceFromCpu: number;
  velocityRmsDifferenceFromCpu: number;
  schedulerOrTraversalOverflow: boolean;
  passed: boolean;
};

const PARTICLE_COUNT = 128;
const BASE_TIME_STEP = 0.002;
const GRAVITY = 1;
const SOFTENING = 0.08;

const factoryFor = (solver: GalaxySolverKind): GalaxySolverFactory =>
  solver === "all-pairs" ? createAllPairsSolver : createBarnesHutSolver;

// A deterministic live ring system gives the validation non-trivial mutual
// gravity while remaining small enough for thousands of exact reference steps.
const createValidationState = (): GalaxyInitialState => {
  const state = new Float32Array(PARTICLE_COUNT * 8);
  const metadata = new Uint32Array(PARTICLE_COUNT);
  const visuals = new Float32Array(PARTICLE_COUNT * 4);
  const centralMass = 1;
  const liveMass = 0.3;
  state[3] = centralMass;
  state[7] = SOFTENING;
  for (let index = 1; index < PARTICLE_COUNT; index++) {
    const ordinal = index - 1;
    const angle = 2 * Math.PI * ordinal / (PARTICLE_COUNT - 1);
    const radius = 1.4 + 1.2 * ((ordinal * 37) % 127) / 126;
    const height = 0.025 * Math.sin(angle * 5);
    const mass = liveMass / (PARTICLE_COUNT - 1);
    const speed = Math.sqrt(
      GRAVITY * (centralMass + liveMass * 0.5) * radius * radius /
        (radius * radius + SOFTENING * SOFTENING) ** 1.5,
    );
    const offset = index * 8;
    state.set([
      Math.cos(angle) * radius,
      height,
      Math.sin(angle) * radius,
      mass,
      -Math.sin(angle) * speed,
      0,
      Math.cos(angle) * speed,
      SOFTENING,
    ], offset);
  }
  // Remove finite-N linear momentum from the compact central body.
  for (let axis = 0; axis < 3; axis++) {
    let momentum = 0;
    for (let index = 1; index < PARTICLE_COUNT; index++) {
      momentum += state[index * 8 + 3]! * state[index * 8 + 4 + axis]!;
    }
    state[4 + axis] = -momentum / centralMass;
  }
  return {
    settings: { textureWidth: 56, radius: 15, offset: 10, seed: 0x5eedc0de },
    parameters: {
      particleCount: PARTICLE_COUNT,
      particlesPerGalaxy: PARTICLE_COUNT / 2,
      diskParticlesPerGalaxy: PARTICLE_COUNT / 2 - 1,
      haloParticlesPerGalaxy: 0,
      seed: 0x5eedc0de,
      radius: 3,
      halfSeparation: 0,
      orthogonalHalfSeparation: 0,
      centerSeparation: 0,
      gravity: GRAVITY,
      timeStep: BASE_TIME_STEP,
      innerOrbitalPeriod: 1,
      coreMass: centralMass,
      diskMass: liveMass,
      haloMass: 0,
      haloScale: 1,
      coreSoftening: SOFTENING,
      diskSoftening: SOFTENING,
      haloSoftening: SOFTENING,
      captureSofteningFactor: 2,
      core1Index: 0,
      core2Index: 0,
    },
    state,
    metadata,
    visuals,
  };
};

const acceleration = (state: Float64Array) => {
  const output = new Float64Array(PARTICLE_COUNT * 3);
  for (let left = 0; left < PARTICLE_COUNT; left++) {
    const leftState = left * 8;
    const leftAcceleration = left * 3;
    for (let right = left + 1; right < PARTICLE_COUNT; right++) {
      const rightState = right * 8;
      const rightAcceleration = right * 3;
      const dx = state[rightState]! - state[leftState]!;
      const dy = state[rightState + 1]! - state[leftState + 1]!;
      const dz = state[rightState + 2]! - state[leftState + 2]!;
      const epsilonSquared = 0.5 * (
        state[leftState + 7]! ** 2 + state[rightState + 7]! ** 2
      );
      const distanceSquared = dx * dx + dy * dy + dz * dz + epsilonSquared;
      const inverseDistanceCubed = 1 /
        (distanceSquared * Math.sqrt(distanceSquared));
      const leftFactor = GRAVITY * state[rightState + 3]! * inverseDistanceCubed;
      const rightFactor = GRAVITY * state[leftState + 3]! * inverseDistanceCubed;
      output[leftAcceleration] = output[leftAcceleration]! + dx * leftFactor;
      output[leftAcceleration + 1] = output[leftAcceleration + 1]! + dy * leftFactor;
      output[leftAcceleration + 2] = output[leftAcceleration + 2]! + dz * leftFactor;
      output[rightAcceleration] = output[rightAcceleration]! - dx * rightFactor;
      output[rightAcceleration + 1] = output[rightAcceleration + 1]! - dy * rightFactor;
      output[rightAcceleration + 2] = output[rightAcceleration + 2]! - dz * rightFactor;
    }
  }
  return output;
};

const evolveCpu = (initial: Float32Array, steps: number) => {
  const state = Float64Array.from(initial);
  let forces = acceleration(state);
  for (let step = 0; step < steps; step++) {
    for (let index = 0; index < PARTICLE_COUNT; index++) {
      const stateOffset = index * 8;
      const forceOffset = index * 3;
      for (let axis = 0; axis < 3; axis++) {
        state[stateOffset + 4 + axis] = state[stateOffset + 4 + axis]! +
          0.5 * forces[forceOffset + axis]! * BASE_TIME_STEP;
        state[stateOffset + axis] = state[stateOffset + axis]! +
          state[stateOffset + 4 + axis]! * BASE_TIME_STEP;
      }
    }
    forces = acceleration(state);
    for (let index = 0; index < PARTICLE_COUNT; index++) {
      const stateOffset = index * 8;
      const forceOffset = index * 3;
      for (let axis = 0; axis < 3; axis++) {
        state[stateOffset + 4 + axis] = state[stateOffset + 4 + axis]! +
          0.5 * forces[forceOffset + axis]! * BASE_TIME_STEP;
      }
    }
  }
  return state;
};

const diagnostics = (state: ArrayLike<number>): Diagnostics => {
  let totalMass = 0;
  let kinetic = 0;
  let potential = 0;
  const momentum = [0, 0, 0];
  const angular = [0, 0, 0];
  const center = [0, 0, 0];
  for (let index = 0; index < PARTICLE_COUNT; index++) {
    const offset = index * 8;
    const mass = state[offset + 3]!;
    const x = state[offset]!;
    const y = state[offset + 1]!;
    const z = state[offset + 2]!;
    const vx = state[offset + 4]!;
    const vy = state[offset + 5]!;
    const vz = state[offset + 6]!;
    totalMass += mass;
    kinetic += 0.5 * mass * (vx * vx + vy * vy + vz * vz);
    momentum[0] = momentum[0]! + mass * vx;
    momentum[1] = momentum[1]! + mass * vy;
    momentum[2] = momentum[2]! + mass * vz;
    angular[0] = angular[0]! + mass * (y * vz - z * vy);
    angular[1] = angular[1]! + mass * (z * vx - x * vz);
    angular[2] = angular[2]! + mass * (x * vy - y * vx);
    center[0] = center[0]! + mass * x;
    center[1] = center[1]! + mass * y;
    center[2] = center[2]! + mass * z;
    for (let right = index + 1; right < PARTICLE_COUNT; right++) {
      const rightOffset = right * 8;
      const dx = state[rightOffset]! - x;
      const dy = state[rightOffset + 1]! - y;
      const dz = state[rightOffset + 2]! - z;
      const epsilonSquared = 0.5 * (
        state[offset + 7]! ** 2 + state[rightOffset + 7]! ** 2
      );
      potential -= GRAVITY * mass * state[rightOffset + 3]! /
        Math.sqrt(dx * dx + dy * dy + dz * dz + epsilonSquared);
    }
  }
  for (let axis = 0; axis < 3; axis++) {
    center[axis] = center[axis]! / totalMass;
  }
  return {
    energy: kinetic + potential,
    momentumMagnitude: Math.hypot(...momentum),
    angularMomentumMagnitude: Math.hypot(...angular),
    centerOfMassMagnitude: Math.hypot(...center),
  };
};

const parameterData = (maximumTimeBin: number) => {
  const data = new ArrayBuffer(32);
  const view = new DataView(data);
  view.setFloat32(0, BASE_TIME_STEP, true);
  view.setFloat32(4, GRAVITY, true);
  view.setFloat32(8, TIMESTEP_ETA, true);
  view.setFloat32(12, 2, true);
  view.setUint32(16, PARTICLE_COUNT, true);
  view.setUint32(20, 0, true);
  view.setUint32(24, 0, true);
  view.setUint32(28, maximumTimeBin, true);
  return new Uint8Array(data);
};

const readBuffer = async (
  device: GPUDevice,
  source: GPUBuffer,
  size: number,
) => {
  const readback = device.createBuffer({
    label: "GPU evolution validation readback",
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, size);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const copy = readback.getMappedRange().slice(0);
  readback.unmap();
  readback.destroy();
  return copy;
};

export const runGalaxyGpuEvolutionValidation = async (
  device: GPUDevice,
  mode: GalaxyGpuEvolutionMode,
  steps = 1_000,
): Promise<GalaxyGpuEvolutionResult> => {
  const initial = createValidationState();
  const stateBytes = initial.state.byteLength;
  const stateBuffers = [0, 1].map((index) => createBufferWithData(
    device,
    `GPU evolution state ${index}`,
    initial.state,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  )) as [GPUBuffer, GPUBuffer];
  const accelerationBuffer = device.createBuffer({
    label: "GPU evolution accelerations",
    size: PARTICLE_COUNT * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const timestepBuffer = device.createBuffer({
    label: "GPU evolution timestep state",
    size: PARTICLE_COUNT * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const activeIndicesBuffer = createBufferWithData(
    device,
    "GPU evolution active indices",
    createInitialActiveIndices(PARTICLE_COUNT),
    GPUBufferUsage.STORAGE,
  );
  const adaptiveControlBuffer = createBufferWithData(
    device,
    "GPU evolution adaptive control",
    createInitialAdaptiveControl(PARTICLE_COUNT),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const indirectDispatchBuffer = createBufferWithData(
    device,
    "GPU evolution indirect dispatch",
    createInitialIndirectDispatch(PARTICLE_COUNT),
    GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
  );
  const parameterBuffer = createBufferWithData(
    device,
    "GPU evolution parameters",
    parameterData(mode.adaptive ? 8 : 0),
    GPUBufferUsage.UNIFORM,
  );
  const modules = [
    galaxyAdaptiveInitializeShader,
    galaxyDriftShader,
    galaxyScheduleShader,
    galaxyAdaptiveKickShader,
  ].map((code) => device.createShaderModule({ code }));
  const [initializePipeline, driftPipeline, beginPipeline, collectPipeline,
    finishPipeline, kickPipeline] = await Promise.all([
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: modules[0]!, entryPoint: "initializeAdaptiveLeapfrog" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: modules[1]!, entryPoint: "driftParticles" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: modules[2]!, entryPoint: "beginSchedule" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: modules[2]!, entryPoint: "collectActiveParticles" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: modules[2]!, entryPoint: "finishSchedule" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: modules[3]!, entryPoint: "kickActiveParticles" },
    }),
  ]);
  const solver = await factoryFor(mode.solver)({
    device,
    initial,
    stateBuffers,
    accelerationBuffer,
    activeIndicesBuffer,
    adaptiveControlBuffer,
    indirectDispatchBuffer,
  });
  const initializeGroups = stateBuffers.map((source, index) =>
    device.createBindGroup({
      layout: initializePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: source } },
        { binding: 1, resource: { buffer: stateBuffers[1 - index]! } },
        { binding: 2, resource: { buffer: parameterBuffer } },
        { binding: 3, resource: { buffer: accelerationBuffer } },
        { binding: 4, resource: { buffer: timestepBuffer } },
      ],
    })) as [GPUBindGroup, GPUBindGroup];
  const driftGroups = stateBuffers.map((source, index) => device.createBindGroup({
    layout: driftPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: source } },
      { binding: 1, resource: { buffer: stateBuffers[1 - index]! } },
      { binding: 2, resource: { buffer: parameterBuffer } },
    ],
  })) as [GPUBindGroup, GPUBindGroup];
  const beginGroup = device.createBindGroup({
    layout: beginPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 3, resource: { buffer: adaptiveControlBuffer } },
      { binding: 4, resource: { buffer: indirectDispatchBuffer } },
    ],
  });
  const collectGroups = stateBuffers.map((state) => device.createBindGroup({
    layout: collectPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: state } },
      { binding: 1, resource: { buffer: timestepBuffer } },
      { binding: 2, resource: { buffer: activeIndicesBuffer } },
      { binding: 3, resource: { buffer: adaptiveControlBuffer } },
      { binding: 5, resource: { buffer: parameterBuffer } },
    ],
  })) as [GPUBindGroup, GPUBindGroup];
  const finishGroup = device.createBindGroup({
    layout: finishPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 3, resource: { buffer: adaptiveControlBuffer } },
      { binding: 4, resource: { buffer: indirectDispatchBuffer } },
      { binding: 5, resource: { buffer: parameterBuffer } },
    ],
  });
  const kickGroups = stateBuffers.map((state) => device.createBindGroup({
    layout: kickPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: state } },
      { binding: 1, resource: { buffer: accelerationBuffer } },
      { binding: 2, resource: { buffer: timestepBuffer } },
      { binding: 3, resource: { buffer: activeIndicesBuffer } },
      { binding: 4, resource: { buffer: adaptiveControlBuffer } },
      { binding: 5, resource: { buffer: parameterBuffer } },
    ],
  })) as [GPUBindGroup, GPUBindGroup];

  const particleWorkgroups = Math.ceil(PARTICLE_COUNT / WORKGROUP_SIZE);
  let encoder = device.createCommandEncoder();
  solver.encode(encoder, 0);
  let pass = encoder.beginComputePass();
  pass.setPipeline(initializePipeline);
  pass.setBindGroup(0, initializeGroups[0]);
  pass.dispatchWorkgroups(particleWorkgroups);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  let readIndex: 0 | 1 = 1;
  const batchSize = 100;
  for (let batchStart = 0; batchStart < steps; batchStart += batchSize) {
    encoder = device.createCommandEncoder();
    const batchEnd = Math.min(steps, batchStart + batchSize);
    for (let step = batchStart; step < batchEnd; step++) {
      const source = readIndex;
      const destination = (1 - source) as 0 | 1;
      pass = encoder.beginComputePass();
      pass.setPipeline(driftPipeline);
      pass.setBindGroup(0, driftGroups[source]);
      pass.dispatchWorkgroups(particleWorkgroups);
      pass.setPipeline(beginPipeline);
      pass.setBindGroup(0, beginGroup);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(collectPipeline);
      pass.setBindGroup(0, collectGroups[destination]);
      pass.dispatchWorkgroups(particleWorkgroups);
      pass.setPipeline(finishPipeline);
      pass.setBindGroup(0, finishGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
      solver.encode(encoder, destination);
      pass = encoder.beginComputePass();
      pass.setPipeline(kickPipeline);
      pass.setBindGroup(0, kickGroups[destination]);
      pass.dispatchWorkgroupsIndirect(indirectDispatchBuffer, 0);
      pass.end();
      readIndex = destination;
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  const [stateCopy, accelerationCopy, timestepCopy, controlCopy] = await Promise.all([
    readBuffer(device, stateBuffers[readIndex], stateBytes),
    readBuffer(device, accelerationBuffer, PARTICLE_COUNT * 16),
    readBuffer(device, timestepBuffer, PARTICLE_COUNT * 16),
    readBuffer(device, adaptiveControlBuffer, 16),
  ]);
  const finalState = new Float64Array(Float32Array.from(new Float32Array(stateCopy)));
  const finalAcceleration = new Float32Array(accelerationCopy);
  const finalTimesteps = new Uint32Array(timestepCopy);
  const finalControl = new Uint32Array(controlCopy);
  // Convert each staggered half-step velocity to the common final force time
  // before evaluating conservation and comparing with the CPU DKD reference.
  const finalTick = finalControl[0]!;
  for (let index = 0; index < PARTICLE_COUNT; index++) {
    const stateOffset = index * 8;
    const timestepOffset = index * 4;
    const accelerationOffset = index * 4;
    const nextActiveTick = finalTimesteps[timestepOffset]!;
    const interval = Math.max(finalTimesteps[timestepOffset + 1]!, 1);
    const velocityTick = nextActiveTick - 0.5 * interval;
    const synchronizationOffset = (finalTick - velocityTick) * BASE_TIME_STEP;
    for (let axis = 0; axis < 3; axis++) {
      finalState[stateOffset + 4 + axis] = finalState[stateOffset + 4 + axis]! +
        finalAcceleration[accelerationOffset + axis]! * synchronizationOffset;
    }
  }
  const cpuState = evolveCpu(initial.state, steps);
  let positionError = 0;
  let velocityError = 0;
  let positionReference = 0;
  let velocityReference = 0;
  for (let index = 0; index < PARTICLE_COUNT; index++) {
    const offset = index * 8;
    for (let axis = 0; axis < 3; axis++) {
      const positionDifference = finalState[offset + axis]! - cpuState[offset + axis]!;
      const velocityDifference =
        finalState[offset + 4 + axis]! - cpuState[offset + 4 + axis]!;
      positionError += positionDifference * positionDifference;
      velocityError += velocityDifference * velocityDifference;
      positionReference += cpuState[offset + axis]! ** 2;
      velocityReference += cpuState[offset + 4 + axis]! ** 2;
    }
  }
  const initialDiagnostics = diagnostics(initial.state);
  const finalDiagnostics = diagnostics(finalState);
  const relativeEnergyDrift = Math.abs(
    (finalDiagnostics.energy - initialDiagnostics.energy) /
      initialDiagnostics.energy,
  );
  const momentumDrift = Math.abs(
    finalDiagnostics.momentumMagnitude - initialDiagnostics.momentumMagnitude,
  );
  const relativeAngularMomentumDrift = Math.abs(
    (finalDiagnostics.angularMomentumMagnitude -
      initialDiagnostics.angularMomentumMagnitude) /
      initialDiagnostics.angularMomentumMagnitude,
  );
  const centerOfMassDrift = Math.abs(
    finalDiagnostics.centerOfMassMagnitude - initialDiagnostics.centerOfMassMagnitude,
  );
  const positionRmsDifferenceFromCpu = Math.sqrt(positionError / positionReference);
  const velocityRmsDifferenceFromCpu = Math.sqrt(velocityError / velocityReference);
  const schedulerOrTraversalOverflow = finalControl[2] !== 0 ||
    finalAcceleration.some((value, index) => index % 4 === 3 && value !== 0);
  const result: GalaxyGpuEvolutionResult = {
    ...mode,
    particleCount: PARTICLE_COUNT,
    steps,
    baseTimeStep: BASE_TIME_STEP,
    initial: initialDiagnostics,
    final: finalDiagnostics,
    relativeEnergyDrift,
    momentumDrift,
    relativeAngularMomentumDrift,
    centerOfMassDrift,
    positionRmsDifferenceFromCpu,
    velocityRmsDifferenceFromCpu,
    schedulerOrTraversalOverflow,
    passed: relativeEnergyDrift < 1e-3 &&
      momentumDrift < 1e-4 &&
      relativeAngularMomentumDrift < 1e-3 &&
      centerOfMassDrift < 1e-4 &&
      positionRmsDifferenceFromCpu < 1e-2 &&
      velocityRmsDifferenceFromCpu < 1e-2 &&
      !schedulerOrTraversalOverflow,
  };
  solver.destroy();
  [
    ...stateBuffers,
    accelerationBuffer,
    timestepBuffer,
    activeIndicesBuffer,
    adaptiveControlBuffer,
    indirectDispatchBuffer,
    parameterBuffer,
  ].forEach((buffer) => buffer.destroy());
  return result;
};

export const runGalaxyGpuEvolutionSuite = async (
  device: GPUDevice,
  steps = 1_000,
) => Promise.all([
  runGalaxyGpuEvolutionValidation(device, { solver: "all-pairs", adaptive: false }, steps),
  runGalaxyGpuEvolutionValidation(device, { solver: "all-pairs", adaptive: true }, steps),
  runGalaxyGpuEvolutionValidation(device, { solver: "barnes-hut", adaptive: false }, steps),
  runGalaxyGpuEvolutionValidation(device, { solver: "barnes-hut", adaptive: true }, steps),
]);
