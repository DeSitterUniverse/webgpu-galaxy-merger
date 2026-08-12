import { WORKGROUP_SIZE } from "./galaxyShaders";

// The configuration-derived timestep remains the smallest legal step. Slowly
// varying particles can skip force evaluations for as many as 256 base ticks.
// Active indices are compacted before the indirect force dispatch, so deep
// bins reduce work without leaving mostly idle workgroups in the force pass.
export const MAX_TIME_BIN = 8;
export const MAX_TIME_INTERVAL = 1 << MAX_TIME_BIN;
export const TIMESTEP_ETA = 0.025;
export const TIMESTEP_STATE_BYTES = 16;
export const ADAPTIVE_CONTROL_BYTES = 16;
export const INDIRECT_DISPATCH_BYTES = 12;

export const choosePowerOfTwoInterval = (
  accelerationMagnitude: number,
  softening: number,
  baseTimeStep: number,
  maximumBin = MAX_TIME_BIN,
) => {
  const safeAcceleration = Math.max(accelerationMagnitude, 1e-8);
  const targetStep = Math.sqrt(
    (2 * TIMESTEP_ETA * Math.max(softening, 1e-6)) / safeAcceleration,
  );
  const maximumInterval = 1 << maximumBin;
  const ratio = Math.max(
    1,
    Math.min(maximumInterval, Math.floor(targetStep / baseTimeStep)),
  );
  return 2 ** Math.floor(Math.log2(ratio));
};

const parameterStruct = /* wgsl */ `
struct SimulationParameters {
  baseTimeStep: f32,
  gravity: f32,
  timestepEta: f32,
  captureSofteningFactor: f32,
  particleCount: u32,
  core1Index: u32,
  core2Index: u32,
  maximumTimeBin: u32,
};

struct Particle {
  positionMass: vec4f,
  velocitySoftening: vec4f,
};

struct TimestepState {
  nextActiveTick: u32,
  interval: u32,
  padding0: u32,
  padding1: u32,
};

fn preferredInterval(particle: Particle, acceleration: vec3f) -> u32 {
  // Compact cores always use the smallest bin. Their swept coalescence and
  // rapidly changing binary force should never be temporally downsampled.
  if (particle.positionMass.w > 0.01) {
    return 1u;
  }
  let accelerationMagnitude = max(length(acceleration), 1e-8);
  let targetStep = sqrt(
    2.0 * parameters.timestepEta *
      max(particle.velocitySoftening.w, 1e-6) / accelerationMagnitude,
  );
  let maximumInterval = 1u << parameters.maximumTimeBin;
  let targetRatio = u32(clamp(
    floor(targetStep / parameters.baseTimeStep),
    1.0,
    f32(maximumInterval),
  ));
  var interval = 1u;
  loop {
    let next = interval << 1u;
    if (next > targetRatio || next > maximumInterval) { break; }
    interval = next;
  }
  return interval;
}
`;

export const galaxyAdaptiveInitializeShader = /* wgsl */ `
${parameterStruct}

@group(0) @binding(0) var<storage, read> source: array<Particle>;
@group(0) @binding(1) var<storage, read_write> destination: array<Particle>;
@group(0) @binding(2) var<uniform> parameters: SimulationParameters;
@group(0) @binding(3) var<storage, read> accelerations: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> timesteps: array<TimestepState>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn initializeAdaptiveLeapfrog(
  @builtin(global_invocation_id) globalId: vec3u,
) {
  let index = globalId.x;
  if (index >= parameters.particleCount) { return; }
  let particle = source[index];
  if (particle.positionMass.w <= 0.0) {
    destination[index] = particle;
    timesteps[index] = TimestepState(0xffffffffu, 1u, 0u, 0u);
    return;
  }

  let acceleration = accelerations[index].xyz;
  let interval = preferredInterval(particle, acceleration);
  var initialized = particle;
  // Velocities live halfway between this force time and the next active force
  // time. This staggered representation permits one force evaluation per
  // active update while preserving the ordinary fixed-step leapfrog limit.
  initialized.velocitySoftening = vec4f(
    particle.velocitySoftening.xyz +
      0.5 * acceleration * parameters.baseTimeStep * f32(interval),
    particle.velocitySoftening.w,
  );
  destination[index] = initialized;
  timesteps[index] = TimestepState(interval, interval, 0u, 0u);
}
`;

export const galaxyDriftShader = /* wgsl */ `
${parameterStruct}

@group(0) @binding(0) var<storage, read> source: array<Particle>;
@group(0) @binding(1) var<storage, read_write> destination: array<Particle>;
@group(0) @binding(2) var<uniform> parameters: SimulationParameters;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn driftParticles(@builtin(global_invocation_id) globalId: vec3u) {
  let index = globalId.x;
  if (index >= parameters.particleCount) { return; }
  let particle = source[index];
  var drifted = particle;
  if (particle.positionMass.w > 0.0) {
    drifted.positionMass = vec4f(
      particle.positionMass.xyz +
        particle.velocitySoftening.xyz * parameters.baseTimeStep,
      particle.positionMass.w,
    );
  }
  destination[index] = drifted;
}
`;

export const galaxyScheduleShader = /* wgsl */ `
${parameterStruct}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> timesteps: array<TimestepState>;
@group(0) @binding(2) var<storage, read_write> activeIndices: array<u32>;
@group(0) @binding(3) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> indirectDispatch: array<u32>;
@group(0) @binding(5) var<uniform> parameters: SimulationParameters;

@compute @workgroup_size(1)
fn beginSchedule() {
  atomicAdd(&control[0], 1u);
  atomicStore(&control[1], 0u);
  atomicStore(&control[2], 0u);
  indirectDispatch[0] = 0u;
  indirectDispatch[1] = 1u;
  indirectDispatch[2] = 1u;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn collectActiveParticles(
  @builtin(global_invocation_id) globalId: vec3u,
) {
  let particleIndex = globalId.x;
  if (particleIndex >= parameters.particleCount) { return; }
  if (particles[particleIndex].positionMass.w <= 0.0) { return; }
  let tick = atomicLoad(&control[0]);
  if (timesteps[particleIndex].nextActiveTick > tick) { return; }
  let ordinal = atomicAdd(&control[1], 1u);
  if (ordinal < parameters.particleCount) {
    activeIndices[ordinal] = particleIndex;
  } else {
    atomicStore(&control[2], 1u);
  }
}

@compute @workgroup_size(1)
fn finishSchedule() {
  let activeCount = min(atomicLoad(&control[1]), parameters.particleCount);
  indirectDispatch[0] = (activeCount + ${WORKGROUP_SIZE - 1}u) /
    ${WORKGROUP_SIZE}u;
}
`;

export const galaxyAdaptiveKickShader = /* wgsl */ `
${parameterStruct}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> accelerations: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> timesteps: array<TimestepState>;
@group(0) @binding(3) var<storage, read> activeIndices: array<u32>;
@group(0) @binding(4) var<storage, read> control: array<u32>;
@group(0) @binding(5) var<uniform> parameters: SimulationParameters;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn kickActiveParticles(@builtin(global_invocation_id) globalId: vec3u) {
  let activeOrdinal = globalId.x;
  let activeCount = min(control[1], parameters.particleCount);
  if (activeOrdinal >= activeCount) { return; }
  let particleIndex = activeIndices[activeOrdinal];
  var particle = particles[particleIndex];
  var state = timesteps[particleIndex];
  let oldInterval = max(state.interval, 1u);
  var newInterval = preferredInterval(
    particle,
    accelerations[particleIndex].xyz,
  );
  let tick = control[0];

  // A particle may enter a faster bin immediately. It may move to a slower
  // bin only when the current tick also belongs to that coarser hierarchy.
  // This keeps all block-step boundaries exactly synchronized.
  if (newInterval > oldInterval && tick % newInterval != 0u) {
    newInterval = oldInterval;
  }
  let kickDuration = 0.5 * f32(oldInterval + newInterval) *
    parameters.baseTimeStep;
  particle.velocitySoftening = vec4f(
    particle.velocitySoftening.xyz +
      accelerations[particleIndex].xyz * kickDuration,
    particle.velocitySoftening.w,
  );
  particles[particleIndex] = particle;
  timesteps[particleIndex] = TimestepState(
    tick + newInterval,
    newInterval,
    0u,
    0u,
  );
}
`;

export const createInitialAdaptiveControl = (particleCount: number) =>
  new Uint32Array([0, particleCount, 0, 0]);

export const createInitialIndirectDispatch = (particleCount: number) =>
  new Uint32Array([
    Math.ceil(particleCount / WORKGROUP_SIZE),
    1,
    1,
  ]);

export const createInitialActiveIndices = (particleCount: number) => {
  const indices = new Uint32Array(particleCount);
  for (let index = 0; index < particleCount; index++) indices[index] = index;
  return indices;
};
