import {
  createBufferWithData,
  type GalaxySolverFactory,
} from "./galaxySolver";
import { WORKGROUP_SIZE } from "./galaxyShaders";

const allPairsShader = /* wgsl */ `
struct Particle {
  positionMass: vec4f,
  velocitySoftening: vec4f,
};

struct SolverParameters {
  particleCount: u32,
  gravity: f32,
  padding: vec2f,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> accelerations: array<vec4f>;
@group(0) @binding(2) var<uniform> parameters: SolverParameters;

var<workgroup> tilePositions: array<vec4f, ${WORKGROUP_SIZE}>;
var<workgroup> tileSoftenings: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn calculateForces(
  @builtin(global_invocation_id) globalId: vec3u,
  @builtin(local_invocation_id) localId: vec3u,
) {
  let index = globalId.x;
  let valid = index < parameters.particleCount;
  var particle = Particle(vec4f(0.0), vec4f(0.0));
  if (valid) {
    particle = particles[index];
  }
  var acceleration = vec3f(0.0);
  var tileStart = 0u;
  loop {
    if (tileStart >= parameters.particleCount) { break; }
    let otherIndex = tileStart + localId.x;
    if (otherIndex < parameters.particleCount) {
      tilePositions[localId.x] = particles[otherIndex].positionMass;
      tileSoftenings[localId.x] = particles[otherIndex].velocitySoftening.w;
    } else {
      tilePositions[localId.x] = vec4f(0.0);
      tileSoftenings[localId.x] = 0.0;
    }
    workgroupBarrier();
    if (valid && particle.positionMass.w > 0.0) {
      let tileCount = min(${WORKGROUP_SIZE}u, parameters.particleCount - tileStart);
      for (var offset = 0u; offset < tileCount; offset++) {
        let otherIndexInTile = tileStart + offset;
        let other = tilePositions[offset];
        if (otherIndexInTile != index && other.w > 0.0) {
          let displacement = other.xyz - particle.positionMass.xyz;
          let softeningSquared = 0.5 * (
            particle.velocitySoftening.w * particle.velocitySoftening.w +
            tileSoftenings[offset] * tileSoftenings[offset]
          );
          let distanceSquared = dot(displacement, displacement) + softeningSquared;
          let inverseDistance = inverseSqrt(distanceSquared);
          acceleration += parameters.gravity * other.w * displacement *
            inverseDistance * inverseDistance * inverseDistance;
        }
      }
    }
    workgroupBarrier();
    tileStart += ${WORKGROUP_SIZE}u;
  }
  if (valid) {
    accelerations[index] = vec4f(acceleration, 0.0);
  }
}
`;

export const createAllPairsSolver: GalaxySolverFactory = async ({
  device,
  initial,
  stateBuffers,
  accelerationBuffer,
}) => {
  const module = device.createShaderModule({
    label: "Galaxy all-pairs solver module",
    code: allPairsShader,
  });
  const pipeline = await device.createComputePipelineAsync({
    label: "Galaxy tiled all-pairs solver",
    layout: "auto",
    compute: { module, entryPoint: "calculateForces" },
  });
  const parameterData = new ArrayBuffer(16);
  const parameterView = new DataView(parameterData);
  parameterView.setUint32(0, initial.parameters.particleCount, true);
  parameterView.setFloat32(4, initial.parameters.gravity, true);
  const parameterBuffer = createBufferWithData(
    device,
    "All-pairs solver parameters",
    new Uint8Array(parameterData),
    GPUBufferUsage.UNIFORM,
  );
  const bindGroups = stateBuffers.map((stateBuffer, index) =>
    device.createBindGroup({
      label: `All-pairs solver state ${index}`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: stateBuffer } },
        { binding: 1, resource: { buffer: accelerationBuffer } },
        { binding: 2, resource: { buffer: parameterBuffer } },
      ],
    }),
  ) as [GPUBindGroup, GPUBindGroup];

  return {
    kind: "all-pairs",
    encode: (encoder, sourceIndex) => {
      const pass = encoder.beginComputePass({ label: "All-pairs force pass" });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroups[sourceIndex]);
      pass.dispatchWorkgroups(
        Math.ceil(initial.parameters.particleCount / WORKGROUP_SIZE),
      );
      pass.end();
    },
    destroy: () => parameterBuffer.destroy(),
  };
};
