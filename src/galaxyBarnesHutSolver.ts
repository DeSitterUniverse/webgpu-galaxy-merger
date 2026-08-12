import {
  createBufferWithData,
  createStorageBuffer,
  type GalaxySolverFactory,
} from "./galaxySolver";
import { WORKGROUP_SIZE } from "./galaxyShaders";

const TREE_NODE_STRIDE = 32;
const TREE_CHILD_STRIDE = 32;
const MINIMUM_TREE_HALF_EXTENT = 128;

// Depth seven keeps leaf occupancy low through the interactive ceiling while
// avoiding an unnecessary traversal level for every target particle.
export const chooseTreeDepth = (particleCount: number) =>
  particleCount > 256 ** 2 ? 8 : 7;

export const maximumTreeNodeCount = (particleCount: number, depth: number) =>
  1 + particleCount * depth;

const createBuildShader = (treeDepth: number) => /* wgsl */ `
struct Particle {
  positionMass: vec4f,
  velocitySoftening: vec4f,
};

// The node payload stays compact. Child topology and leaf heads live in
// separate buffers so accepted monopoles do not fetch unused pointers.
struct TreeNode {
  mass: f32,
  momentX: f32,
  momentY: f32,
  momentZ: f32,
  softeningMoment: f32,
  packedCell: u32,
  packedParent: u32,
  owner: u32,
};

struct TreeParameters {
  particleCount: u32,
  maximumDepth: u32,
  maximumNodes: u32,
  currentLevel: u32,
  gravity: f32,
  minimumHalfExtent: f32,
  theta: f32,
  padding: u32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> nodes: array<TreeNode>;
@group(0) @binding(2) var<storage, read_write> nodeChildren: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> particleNodes: array<u32>;
@group(0) @binding(4) var<storage, read_write> leafHeads: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> particleNext: array<u32>;
@group(0) @binding(6) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> parameters: TreeParameters;

const TREE_DEPTH = ${treeDepth}u;
const EMPTY = 0xffffffffu;
const BOUNDS_SCALE = 1024.0;

fn nextPowerOfTwo(value: u32) -> u32 {
  var result = max(value, 1u) - 1u;
  result = result | (result >> 1u);
  result = result | (result >> 2u);
  result = result | (result >> 4u);
  result = result | (result >> 8u);
  result = result | (result >> 16u);
  return result + 1u;
}

fn treeHalfExtent() -> f32 {
  let measured = f32(atomicLoad(&control[0])) / BOUNDS_SCALE;
  let padded = u32(ceil(measured + 8.0));
  return max(parameters.minimumHalfExtent, f32(nextPowerOfTwo(padded)));
}

fn targetLeaf(position: vec3f) -> vec3u {
  let halfExtent = treeHalfExtent();
  let normalized = clamp(
    (position + vec3f(halfExtent)) / (2.0 * halfExtent),
    vec3f(0.0),
    vec3f(0.999999),
  );
  return vec3u(normalized * f32(1u << TREE_DEPTH));
}

fn levelBase(level: u32) -> u32 {
  if (level == 0u) { return 0u; }
  return 1u + (level - 1u) * parameters.particleCount;
}

fn packCell(depth: u32, cell: vec3u) -> u32 {
  return (depth << 27u) | (cell.z << 18u) | (cell.y << 9u) | cell.x;
}

fn initializeNode(
  index: u32,
  parent: u32,
  depth: u32,
  cell: vec3u,
  childSlot: u32,
  owner: u32,
) {
  nodes[index].mass = 0.0;
  nodes[index].momentX = 0.0;
  nodes[index].momentY = 0.0;
  nodes[index].momentZ = 0.0;
  nodes[index].softeningMoment = 0.0;
  nodes[index].packedCell = packCell(depth, cell);
  nodes[index].packedParent = select(
    parent | (childSlot << 20u),
    EMPTY,
    parent == EMPTY,
  );
  nodes[index].owner = owner;
  for (var child = 0u; child < 8u; child++) {
    atomicStore(&nodeChildren[index * 8u + child], EMPTY);
  }
  atomicStore(&leafHeads[index], EMPTY);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn resetBuild(@builtin(global_invocation_id) globalId: vec3u) {
  if (globalId.x != 0u) { return; }
  atomicStore(&control[0], 0u);
  initializeNode(0u, EMPTY, 0u, vec3u(0u), 0u, 0u);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn measureBounds(@builtin(global_invocation_id) globalId: vec3u) {
  let index = globalId.x;
  if (index >= parameters.particleCount) { return; }
  let particle = particles[index];
  if (particle.positionMass.w <= 0.0) { return; }
  let maximum = max(
    abs(particle.positionMass.x),
    max(abs(particle.positionMass.y), abs(particle.positionMass.z)),
  );
  atomicMax(&control[0], u32(ceil(maximum * BOUNDS_SCALE)));
}

// Each child slot elects the lowest particle index. A separate dispatch turns
// that stable winner into a node, avoiding scheduler-dependent spin locks.
@compute @workgroup_size(${WORKGROUP_SIZE})
fn claimTreeLevel(@builtin(global_invocation_id) globalId: vec3u) {
  let particleIndex = globalId.x;
  if (particleIndex >= parameters.particleCount) { return; }
  let level = parameters.currentLevel;
  let leaf = targetLeaf(particles[particleIndex].positionMass.xyz);
  let shift = TREE_DEPTH - level;
  let cell = leaf >> vec3u(shift);
  var parent = 0u;
  if (level > 1u) {
    parent = particleNodes[(level - 2u) * parameters.particleCount + particleIndex];
  }
  let childSlot = (cell.x & 1u) | ((cell.y & 1u) << 1u) |
    ((cell.z & 1u) << 2u);
  let childAddress = parent * 8u + childSlot;
  atomicMin(&nodeChildren[childAddress], particleIndex);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn finalizeTreeLevel(@builtin(global_invocation_id) globalId: vec3u) {
  let particleIndex = globalId.x;
  if (particleIndex >= parameters.particleCount) { return; }
  let level = parameters.currentLevel;
  let leaf = targetLeaf(particles[particleIndex].positionMass.xyz);
  let shift = TREE_DEPTH - level;
  let cell = leaf >> vec3u(shift);
  var parent = 0u;
  if (level > 1u) {
    parent = particleNodes[(level - 2u) * parameters.particleCount + particleIndex];
  }
  let childSlot = (cell.x & 1u) | ((cell.y & 1u) << 1u) |
    ((cell.z & 1u) << 2u);
  let childAddress = parent * 8u + childSlot;
  let owner = atomicLoad(&nodeChildren[childAddress]);
  let childNode = levelBase(level) + owner;
  if (owner == particleIndex) {
    initializeNode(childNode, parent, level, cell, childSlot, owner);
  }
  particleNodes[(level - 1u) * parameters.particleCount + particleIndex] = childNode;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn publishTreeTopology(@builtin(global_invocation_id) globalId: vec3u) {
  let particleIndex = globalId.x;
  if (particleIndex >= parameters.particleCount) { return; }
  for (var level = 1u; level <= TREE_DEPTH; level++) {
    let node = particleNodes[(level - 1u) * parameters.particleCount + particleIndex];
    if (nodes[node].owner != particleIndex) { continue; }
    let packedParent = nodes[node].packedParent;
    let parent = packedParent & 0x000fffffu;
    let childSlot = (packedParent >> 20u) & 7u;
    atomicStore(&nodeChildren[parent * 8u + childSlot], node);
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn linkLeafParticles(@builtin(global_invocation_id) globalId: vec3u) {
  let particleIndex = globalId.x;
  if (particleIndex >= parameters.particleCount) { return; }
  let leaf = particleNodes[(TREE_DEPTH - 1u) * parameters.particleCount + particleIndex];
  particleNext[particleIndex] = atomicExchange(&leafHeads[leaf], particleIndex);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn aggregateTreeLevel(@builtin(global_invocation_id) globalId: vec3u) {
  let particleIndex = globalId.x;
  let level = parameters.currentLevel;
  if (particleIndex >= parameters.particleCount) { return; }
  if (level == 0u && particleIndex > 0u) { return; }
  let nodeIndex = select(0u, levelBase(level) + particleIndex, level > 0u);
  if (nodes[nodeIndex].owner != particleIndex) { return; }
  var mass = 0.0;
  var moment = vec3f(0.0);
  var softeningMoment = 0.0;

  if (level == TREE_DEPTH) {
    var sourceIndex = atomicLoad(&leafHeads[nodeIndex]);
    loop {
      if (sourceIndex == EMPTY) { break; }
      let source = particles[sourceIndex];
      if (source.positionMass.w > 0.0) {
        mass += source.positionMass.w;
        moment += source.positionMass.w * source.positionMass.xyz;
        softeningMoment += source.positionMass.w * source.velocitySoftening.w *
          source.velocitySoftening.w;
      }
      sourceIndex = particleNext[sourceIndex];
    }
  } else {
    for (var slot = 0u; slot < 8u; slot++) {
      let child = atomicLoad(&nodeChildren[nodeIndex * 8u + slot]);
      if (child == EMPTY) { continue; }
      let childNode = nodes[child];
      mass += childNode.mass;
      moment += vec3f(childNode.momentX, childNode.momentY, childNode.momentZ);
      softeningMoment += childNode.softeningMoment;
    }
  }
  nodes[nodeIndex].mass = mass;
  nodes[nodeIndex].momentX = moment.x;
  nodes[nodeIndex].momentY = moment.y;
  nodes[nodeIndex].momentZ = moment.z;
  nodes[nodeIndex].softeningMoment = softeningMoment;
}

`;

const createForceShader = (treeDepth: number) => /* wgsl */ `
struct Particle {
  positionMass: vec4f,
  velocitySoftening: vec4f,
};

struct TreeNode {
  mass: f32,
  momentX: f32,
  momentY: f32,
  momentZ: f32,
  softeningMoment: f32,
  packedCell: u32,
  packedParent: u32,
  owner: u32,
};

struct TreeParameters {
  particleCount: u32,
  maximumDepth: u32,
  maximumNodes: u32,
  currentLevel: u32,
  gravity: f32,
  minimumHalfExtent: f32,
  theta: f32,
  padding: u32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> accelerations: array<vec4f>;
@group(0) @binding(2) var<storage, read> nodes: array<TreeNode>;
@group(0) @binding(3) var<storage, read_write> nodeChildren: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> leafHeads: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> particleNext: array<u32>;
@group(0) @binding(6) var<storage, read> control: array<u32>;
@group(0) @binding(7) var<uniform> parameters: TreeParameters;

const TREE_DEPTH = ${treeDepth}u;
const EMPTY = 0xffffffffu;
const BOUNDS_SCALE = 1024.0;

fn nextPowerOfTwo(value: u32) -> u32 {
  var result = max(value, 1u) - 1u;
  result = result | (result >> 1u);
  result = result | (result >> 2u);
  result = result | (result >> 4u);
  result = result | (result >> 8u);
  result = result | (result >> 16u);
  return result + 1u;
}

fn treeHalfExtent() -> f32 {
  let measured = f32(control[0]) / BOUNDS_SCALE;
  let padded = u32(ceil(measured + 8.0));
  return max(parameters.minimumHalfExtent, f32(nextPowerOfTwo(padded)));
}

fn targetLeaf(position: vec3f, halfExtent: f32) -> vec3u {
  let normalized = clamp(
    (position + vec3f(halfExtent)) / (2.0 * halfExtent),
    vec3f(0.0),
    vec3f(0.999999),
  );
  return vec3u(normalized * f32(1u << TREE_DEPTH));
}

fn particleAcceleration(targetParticle: Particle, source: Particle) -> vec3f {
  let displacement = source.positionMass.xyz - targetParticle.positionMass.xyz;
  let softeningSquared = 0.5 * (
    targetParticle.velocitySoftening.w * targetParticle.velocitySoftening.w +
    source.velocitySoftening.w * source.velocitySoftening.w
  );
  let distanceSquared = dot(displacement, displacement) + softeningSquared;
  let inverseDistance = inverseSqrt(distanceSquared);
  return parameters.gravity * source.positionMass.w * displacement *
    inverseDistance * inverseDistance * inverseDistance;
}

fn exactLeafAcceleration(targetIndex: u32, targetParticle: Particle, nodeIndex: u32) -> vec3f {
  var acceleration = vec3f(0.0);
  var sourceIndex = atomicLoad(&leafHeads[nodeIndex]);
  loop {
    if (sourceIndex == EMPTY) { break; }
    if (sourceIndex != targetIndex) {
      let source = particles[sourceIndex];
      if (source.positionMass.w > 0.0) {
        acceleration += particleAcceleration(targetParticle, source);
      }
    }
    sourceIndex = particleNext[sourceIndex];
  }
  return acceleration;
}

fn monopoleAcceleration(targetParticle: Particle, node: TreeNode) -> vec3f {
  let centerOfMass = vec3f(node.momentX, node.momentY, node.momentZ) / node.mass;
  let displacement = centerOfMass - targetParticle.positionMass.xyz;
  let averageSofteningSquared = node.softeningMoment / node.mass;
  let softeningSquared = 0.5 * (
    targetParticle.velocitySoftening.w * targetParticle.velocitySoftening.w +
    averageSofteningSquared
  );
  let distanceSquared = dot(displacement, displacement) + softeningSquared;
  let inverseDistance = inverseSqrt(distanceSquared);
  return parameters.gravity * node.mass * displacement *
    inverseDistance * inverseDistance * inverseDistance;
}

fn unpackCell(packedCell: u32) -> vec3u {
  return vec3u(
    packedCell & 0x1ffu,
    (packedCell >> 9u) & 0x1ffu,
    (packedCell >> 18u) & 0x1ffu,
  );
}

fn overlapsExactNearField(node: TreeNode, targetCell: vec3u) -> bool {
  let depth = node.packedCell >> 27u;
  let cell = unpackCell(node.packedCell);
  let shift = TREE_DEPTH - depth;
  let minimumLeaf = cell << vec3u(shift);
  let maximumLeaf = ((cell + vec3u(1u)) << vec3u(shift)) - vec3u(1u);
  let targetMinimum = vec3u(max(vec3i(targetCell) - vec3i(1), vec3i(0)));
  let targetMaximum = min(
    targetCell + vec3u(1u),
    vec3u((1u << TREE_DEPTH) - 1u),
  );
  return all(maximumLeaf >= targetMinimum) && all(minimumLeaf <= targetMaximum);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn calculateTreeForces(@builtin(global_invocation_id) globalId: vec3u) {
  let targetIndex = globalId.x;
  if (targetIndex >= parameters.particleCount) { return; }
  let targetParticle = particles[targetIndex];
  if (targetParticle.positionMass.w <= 0.0) {
    accelerations[targetIndex] = vec4f(0.0);
    return;
  }

  let halfExtent = treeHalfExtent();
  let targetCell = targetLeaf(targetParticle.positionMass.xyz, halfExtent);
  var stack: array<u32, 64>;
  var stackSize = 1u;
  stack[0] = 0u;
  var acceleration = vec3f(0.0);

  loop {
    if (stackSize == 0u) { break; }
    stackSize--;
    let current = stack[stackSize];
    let node = nodes[current];
    if (node.mass <= 0.0) { continue; }

    let depth = node.packedCell >> 27u;
    let cell = unpackCell(node.packedCell);
    let dimension = 1u << depth;
    let shift = TREE_DEPTH - depth;
    let containsTarget = all((targetCell >> vec3u(shift)) == cell);
    let nearField = overlapsExactNearField(node, targetCell);
    let nodeSize = 2.0 * halfExtent / f32(dimension);
    let cellCenter = -vec3f(halfExtent) + (vec3f(cell) + vec3f(0.5)) * nodeSize;
    let centerOfMass = vec3f(node.momentX, node.momentY, node.momentZ) / node.mass;
    let centerDistance = length(centerOfMass - targetParticle.positionMass.xyz);
    let centerOffset = length(centerOfMass - cellCenter);
    let openingDistance = centerDistance - centerOffset;
    // Cores are few and orbit-sensitive, so they receive a tighter criterion.
    let openingTheta = select(parameters.theta, 0.5, targetParticle.positionMass.w > 0.01);
    let accept = !containsTarget && !nearField && openingDistance > 0.0 &&
      nodeSize / openingDistance < openingTheta;

    if (depth == TREE_DEPTH) {
      if (accept) {
        acceleration += monopoleAcceleration(targetParticle, node);
      } else {
        acceleration += exactLeafAcceleration(targetIndex, targetParticle, current);
      }
      continue;
    }
    if (accept) {
      acceleration += monopoleAcceleration(targetParticle, node);
      continue;
    }
    for (var reverseSlot = 8u; reverseSlot > 0u; reverseSlot--) {
      let child = atomicLoad(&nodeChildren[current * 8u + reverseSlot - 1u]);
      if (child != EMPTY && stackSize < 64u) {
        stack[stackSize] = child;
        stackSize++;
      }
    }
  }
  accelerations[targetIndex] = vec4f(acceleration, 0.0);
}
`;

export const createBarnesHutSolver: GalaxySolverFactory = async ({
  device,
  initial,
  stateBuffers,
  accelerationBuffer,
}) => {
  const particleCount = initial.parameters.particleCount;
  const treeDepth = chooseTreeDepth(particleCount);
  const maximumNodes = maximumTreeNodeCount(particleCount, treeDepth);
  const buildModule = device.createShaderModule({
    label: "Galaxy compact Barnes-Hut build module",
    code: createBuildShader(treeDepth),
  });
  const forceModule = device.createShaderModule({
    label: "Galaxy compact Barnes-Hut force module",
    code: createForceShader(treeDepth),
  });

  const buildLayout = device.createBindGroupLayout({
    label: "Compact Barnes-Hut build layout",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ...Array.from({ length: 6 }, (_, index) => ({
        binding: index + 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" as const },
      })),
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const forceLayout = device.createBindGroupLayout({
    label: "Compact Barnes-Hut force layout",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const buildPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [buildLayout] });
  const forcePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [forceLayout] });
  const buildEntries = [
    ["Reset", "resetBuild"],
    ["Bounds", "measureBounds"],
    ["Claim level", "claimTreeLevel"],
    ["Finalize level", "finalizeTreeLevel"],
    ["Publish topology", "publishTreeTopology"],
    ["Link leaves", "linkLeafParticles"],
    ["Aggregate level", "aggregateTreeLevel"],
  ] as const;
  const [
    resetPipeline,
    boundsPipeline,
    claimLevelPipeline,
    finalizeLevelPipeline,
    publishTopologyPipeline,
    linkLeavesPipeline,
    aggregatePipeline,
  ] = await Promise.all(buildEntries.map(([label, entryPoint]) =>
    device.createComputePipelineAsync({
      label: `Compact Barnes-Hut ${label}`,
      layout: buildPipelineLayout,
      compute: { module: buildModule, entryPoint },
    })));
  const forcePipeline = await device.createComputePipelineAsync({
    label: "Compact Barnes-Hut occupied-node force traversal",
    layout: forcePipelineLayout,
    compute: { module: forceModule, entryPoint: "calculateTreeForces" },
  });

  const nodeBuffer = createStorageBuffer(
    device,
    "Compact occupied Barnes-Hut nodes",
    maximumNodes * TREE_NODE_STRIDE,
  );
  const childBuffer = createStorageBuffer(
    device,
    "Compact Barnes-Hut child topology",
    maximumNodes * TREE_CHILD_STRIDE,
  );
  const particleNodeBuffer = createStorageBuffer(
    device,
    "Barnes-Hut particle paths",
    particleCount * treeDepth * 4,
  );
  const leafHeadBuffer = createStorageBuffer(
    device,
    "Barnes-Hut terminal bucket heads",
    maximumNodes * 4,
  );
  const particleNextBuffer = createStorageBuffer(
    device,
    "Barnes-Hut terminal particle links",
    particleCount * 4,
  );
  const controlBuffer = createStorageBuffer(
    device,
    "Barnes-Hut bounds and level counts",
    (treeDepth + 1) * 4,
  );
  const parameterBuffers = Array.from({ length: treeDepth + 1 }, (_, level) => {
    const data = new ArrayBuffer(32);
    const view = new DataView(data);
    view.setUint32(0, particleCount, true);
    view.setUint32(4, treeDepth, true);
    view.setUint32(8, maximumNodes, true);
    view.setUint32(12, level, true);
    view.setFloat32(16, initial.parameters.gravity, true);
    view.setFloat32(20, MINIMUM_TREE_HALF_EXTENT, true);
    view.setFloat32(24, 0.7, true);
    return createBufferWithData(
      device,
      `Barnes-Hut parameters level ${level}`,
      new Uint8Array(data),
      GPUBufferUsage.UNIFORM,
    );
  });

  const buildBindGroups = stateBuffers.map((stateBuffer, stateIndex) =>
    parameterBuffers.map((parameterBuffer, level) => device.createBindGroup({
      label: `Barnes-Hut build state ${stateIndex} level ${level}`,
      layout: buildLayout,
      entries: [
        { binding: 0, resource: { buffer: stateBuffer } },
        { binding: 1, resource: { buffer: nodeBuffer } },
        { binding: 2, resource: { buffer: childBuffer } },
        { binding: 3, resource: { buffer: particleNodeBuffer } },
        { binding: 4, resource: { buffer: leafHeadBuffer } },
        { binding: 5, resource: { buffer: particleNextBuffer } },
        { binding: 6, resource: { buffer: controlBuffer } },
        { binding: 7, resource: { buffer: parameterBuffer } },
      ],
    })),
  );
  const forceBindGroups = stateBuffers.map((stateBuffer, stateIndex) =>
    device.createBindGroup({
      label: `Barnes-Hut force state ${stateIndex}`,
      layout: forceLayout,
      entries: [
        { binding: 0, resource: { buffer: stateBuffer } },
        { binding: 1, resource: { buffer: accelerationBuffer } },
        { binding: 2, resource: { buffer: nodeBuffer } },
        { binding: 3, resource: { buffer: childBuffer } },
        { binding: 4, resource: { buffer: leafHeadBuffer } },
        { binding: 5, resource: { buffer: particleNextBuffer } },
        { binding: 6, resource: { buffer: controlBuffer } },
        { binding: 7, resource: { buffer: parameterBuffers[0]! } },
      ],
    }));

  const particleWorkgroups = Math.ceil(particleCount / WORKGROUP_SIZE);

  return {
    kind: "barnes-hut",
    encode: (encoder, sourceIndex) => {
      const pass = encoder.beginComputePass({ label: "Compact Barnes-Hut pass" });
      pass.setBindGroup(0, buildBindGroups[sourceIndex]![0]!);
      pass.setPipeline(resetPipeline!);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(boundsPipeline!);
      pass.dispatchWorkgroups(particleWorkgroups);
      for (let level = 1; level <= treeDepth; level++) {
        pass.setBindGroup(0, buildBindGroups[sourceIndex]![level]!);
        pass.setPipeline(claimLevelPipeline!);
        pass.dispatchWorkgroups(particleWorkgroups);
        pass.setPipeline(finalizeLevelPipeline!);
        pass.dispatchWorkgroups(particleWorkgroups);
      }
      pass.setBindGroup(0, buildBindGroups[sourceIndex]![0]!);
      pass.setPipeline(publishTopologyPipeline!);
      pass.dispatchWorkgroups(particleWorkgroups);
      pass.setPipeline(linkLeavesPipeline!);
      pass.dispatchWorkgroups(particleWorkgroups);
      pass.setPipeline(aggregatePipeline!);
      for (let level = treeDepth; level >= 0; level--) {
        pass.setBindGroup(0, buildBindGroups[sourceIndex]![level]!);
        pass.dispatchWorkgroups(particleWorkgroups);
      }
      pass.setBindGroup(0, forceBindGroups[sourceIndex]!);
      pass.setPipeline(forcePipeline);
      pass.dispatchWorkgroups(particleWorkgroups);
      pass.end();
    },
    destroy: () => {
      [
        nodeBuffer,
        childBuffer,
        particleNodeBuffer,
        leafHeadBuffer,
        particleNextBuffer,
        controlBuffer,
        ...parameterBuffers,
      ].forEach((buffer) => buffer.destroy());
    },
  };
};

export const calculateTreeHalfExtent = (maximumAbsolutePosition: number) => {
  const required = Math.max(1, Math.ceil(maximumAbsolutePosition + 8));
  return Math.max(
    MINIMUM_TREE_HALF_EXTENT,
    2 ** Math.ceil(Math.log2(required)),
  );
};

export const BARNES_HUT_TEST_CONSTANTS = {
  minimumHalfExtent: MINIMUM_TREE_HALF_EXTENT,
  nodeStride: TREE_NODE_STRIDE,
  childStride: TREE_CHILD_STRIDE,
} as const;
