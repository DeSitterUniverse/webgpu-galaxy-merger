export const WORKGROUP_SIZE = 128;

export const galaxyComputeShader = /* wgsl */ `
// Keep the storage layout aligned with the CPU's two-vec4 packed state.
// Softening rides in velocity.w, so component-aware forces cost no extra buffer.
struct Particle {
  positionMass: vec4f,
  velocitySoftening: vec4f,
};

struct SimulationParameters {
  timeStep: f32,
  gravity: f32,
  paddingFloat: f32,
  captureSofteningFactor: f32,
  particleCount: u32,
  core1Index: u32,
  core2Index: u32,
  padding: u32,
};

struct CoreTelemetry {
  separation: f32,
  phase: u32,
  boundElapsed: f32,
  mergeDelay: f32,
};

@group(0) @binding(0) var<storage, read> source: array<Particle>;
@group(0) @binding(1) var<storage, read_write> destination: array<Particle>;
@group(0) @binding(2) var<uniform> parameters: SimulationParameters;
@group(0) @binding(3) var<storage, read_write> telemetry: CoreTelemetry;
@group(0) @binding(4) var<storage, read> accelerations: array<vec4f>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn initializeLeapfrog(
  @builtin(global_invocation_id) globalId: vec3u,
) {
  let index = globalId.x;
  if (index >= parameters.particleCount) {
    return;
  }
  let particle = source[index];
  if (particle.positionMass.w <= 0.0) {
    destination[index] = particle;
    return;
  }
  var initialized = particle;
  // Convert full-step input velocities to the half-step representation used by
  // kick-drift leapfrog. Normal frames then need only one force evaluation.
  initialized.velocitySoftening = vec4f(
    particle.velocitySoftening.xyz -
      0.5 * accelerations[index].xyz * parameters.timeStep,
    particle.velocitySoftening.w,
  );
  destination[index] = initialized;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn leapfrogStep(
  @builtin(global_invocation_id) globalId: vec3u,
) {
  let index = globalId.x;
  if (index >= parameters.particleCount) {
    return;
  }
  let particle = source[index];
  if (particle.positionMass.w <= 0.0) {
    destination[index] = particle;
    return;
  }

  // Kick velocity at the half step, then drift position with that velocity.
  let nextHalfVelocity = particle.velocitySoftening.xyz +
    accelerations[index].xyz * parameters.timeStep;
  var nextParticle = particle;
  nextParticle.velocitySoftening = vec4f(
    nextHalfVelocity,
    particle.velocitySoftening.w,
  );
  nextParticle.positionMass = vec4f(
    particle.positionMass.xyz + nextHalfVelocity * parameters.timeStep,
    particle.positionMass.w,
  );
  destination[index] = nextParticle;
}

@compute @workgroup_size(1)
fn mergeCores() {
  let oldCore1 = source[parameters.core1Index];
  let oldCore2 = source[parameters.core2Index];
  let newCore1 = destination[parameters.core1Index];
  let newCore2 = destination[parameters.core2Index];
  if (newCore1.positionMass.w <= 0.0 || newCore2.positionMass.w <= 0.0) {
    telemetry = CoreTelemetry(
      0.0,
      2u,
      telemetry.boundElapsed,
      telemetry.mergeDelay,
    );
    return;
  }

  let oldRelative = oldCore2.positionMass.xyz - oldCore1.positionMass.xyz;
  let newRelative = newCore2.positionMass.xyz - newCore1.positionMass.xyz;
  let relativeMotion = newRelative - oldRelative;
  let denominator = dot(relativeMotion, relativeMotion);
  var closestFraction = 0.0;
  // Find closest approach along this timestep's relative-motion segment. An
  // endpoint-only check would let sufficiently fast cores tunnel through.
  if (denominator > 1e-12) {
    closestFraction = clamp(
      -dot(oldRelative, relativeMotion) / denominator,
      0.0,
      1.0,
    );
  }
  let closestSeparation = oldRelative + closestFraction * relativeMotion;
  let totalMass = newCore1.positionMass.w + newCore2.positionMass.w;
  let pairSofteningSquared = 0.5 * (
    newCore1.velocitySoftening.w * newCore1.velocitySoftening.w +
    newCore2.velocitySoftening.w * newCore2.velocitySoftening.w
  );
  let captureRadius = parameters.captureSofteningFactor *
    sqrt(pairSofteningSquared);
  let crossesCaptureRadius = length(closestSeparation) <= captureRadius;
  let relativeVelocity = newCore2.velocitySoftening.xyz -
    newCore1.velocitySoftening.xyz;
  let softenedSeparation = max(
    sqrt(dot(newRelative, newRelative) + pairSofteningSquared),
    1e-8,
  );
  let specificEnergy = 0.5 * dot(relativeVelocity, relativeVelocity) -
    parameters.gravity * totalMass / softenedSeparation;
  let isBound = specificEnergy < 0.0;

  // A close passage first becomes an unresolved bound binary. Flybys remain
  // separate even when their swept paths cross the capture scale.
  if (telemetry.phase == 0u) {
    if (!isBound || !crossesCaptureRadius) {
      telemetry = CoreTelemetry(length(newRelative), 0u, 0.0, 0.0);
      return;
    }
    let softenedCaptureRadiusCubed = pow(
      captureRadius * captureRadius + pairSofteningSquared,
      1.5,
    );
    let mergeDelay = 6.28318530718 * sqrt(
      softenedCaptureRadiusCubed /
        max(parameters.gravity * totalMass, 1e-8),
    );
    telemetry = CoreTelemetry(
      length(newRelative),
      1u,
      parameters.timeStep,
      mergeDelay,
    );
    return;
  }

  // If the pair escapes, discard the binary state. Otherwise wait for roughly
  // one unresolved orbit and a qualifying close passage before coalescing.
  if (telemetry.phase == 1u) {
    if (!isBound) {
      telemetry = CoreTelemetry(length(newRelative), 0u, 0.0, 0.0);
      return;
    }
    let boundElapsed = telemetry.boundElapsed + parameters.timeStep;
    if (boundElapsed < telemetry.mergeDelay || !crossesCaptureRadius) {
      telemetry = CoreTelemetry(
        length(newRelative),
        1u,
        boundElapsed,
        telemetry.mergeDelay,
      );
      return;
    }
  }

  // Mass-weighted replacement conserves total mass, momentum, and core COM.
  let mergedPosition =
    (newCore1.positionMass.w * newCore1.positionMass.xyz +
     newCore2.positionMass.w * newCore2.positionMass.xyz) / totalMass;
  let mergedVelocity =
    (newCore1.positionMass.w * newCore1.velocitySoftening.xyz +
     newCore2.positionMass.w * newCore2.velocitySoftening.xyz) / totalMass;
  destination[parameters.core1Index] = Particle(
    vec4f(mergedPosition, totalMass),
    vec4f(mergedVelocity, newCore1.velocitySoftening.w),
  );
  destination[parameters.core2Index] = Particle(
    vec4f(mergedPosition, 0.0),
    vec4f(mergedVelocity, newCore2.velocitySoftening.w),
  );
  telemetry = CoreTelemetry(
    0.0,
    2u,
    telemetry.boundElapsed + parameters.timeStep,
    telemetry.mergeDelay,
  );
}
`;

export const galaxyRenderShader = /* wgsl */ `
struct Particle {
  positionMass: vec4f,
  velocitySoftening: vec4f,
};

struct Visual {
  colorSize: vec4f,
};

struct RenderParameters {
  viewProjection: mat4x4f,
  viewport: vec2f,
  fizzle: f32,
  sizeScale: f32,
};

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
  @location(1) local: vec2f,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> visuals: array<Visual>;
@group(0) @binding(2) var<uniform> parameters: RenderParameters;

fn particleHash(index: u32) -> f32 {
  // A stable per-index hash makes fizzle transitions deterministic and avoids
  // frame-to-frame particle sparkling.
  var value = index + 0x9e3779b9u;
  value = (value ^ (value >> 16u)) * 0x21f0aaadu;
  value = (value ^ (value >> 15u)) * 0x735a2d97u;
  value = value ^ (value >> 15u);
  return f32(value & 0x00ffffffu) / 16777216.0;
}

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> VertexOutput {
  let corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  );
  let particle = particles[instanceIndex];
  let visual = visuals[instanceIndex];
  var output: VertexOutput;
  output.color = visual.colorSize.xyz;
  output.local = corners[vertexIndex];
  if (
    particle.positionMass.w <= 0.0 ||
    visual.colorSize.w <= 0.0 ||
    particleHash(instanceIndex) > parameters.fizzle
  ) {
    output.position = vec4f(2.0, 2.0, 2.0, 1.0);
    return output;
  }

  // Expand each simulated point into a six-vertex camera-facing billboard.
  let center = parameters.viewProjection * vec4f(particle.positionMass.xyz, 1.0);
  let pixelSize = visual.colorSize.w * parameters.sizeScale /
    max(abs(center.w), 1.0) * mix(0.45, 1.0, parameters.fizzle);
  let ndcOffset = corners[vertexIndex] * pixelSize * 2.0 / parameters.viewport;
  output.position = center + vec4f(ndcOffset * center.w, 0.0, 0.0);
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  // Circular falloff plus additive blending produces a compact luminous point
  // without storing a particle texture.
  let radius = length(input.local);
  if (radius > 1.0) {
    discard;
  }
  let alpha = pow(1.0 - radius, 1.2);
  return vec4f(input.color * alpha, alpha);
}
`;
