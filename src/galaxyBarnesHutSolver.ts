import {
  createBufferWithData,
  createStorageBuffer,
  type GalaxySolverFactory,
} from "./galaxySolver";
import { WORKGROUP_SIZE } from "./galaxyShaders";

const TREE_NODE_STRIDE = 32;
const TREE_CHILD_STRIDE = 32;
const MINIMUM_TREE_HALF_EXTENT = 128;
const TOPOLOGY_DISPATCH_BYTES = 24;
// A depth-first octree walk retains at most one chosen child plus seven siblings
// per level. Depth ten needs exactly 71 entries, avoiding the occupancy cost
// of the former undersized or overly conservative private allocations.
export const BARNES_HUT_TRAVERSAL_STACK_CAPACITY = 71;
export const maximumTraversalStackEntries = (depth: number) => 1 + 7 * depth;

// Deeper trees keep terminal buckets small as resolution rises. Ten levels
// map directly to a 30-bit Morton-style cell coordinate at million-body scale.
export const chooseTreeDepth = (particleCount: number) =>
  particleCount > 512 ** 2 ? 10 : particleCount > 256 ** 2 ? 9 : 7;

// At each level there can be no more occupied cells than particles or cells.
// This exact upper bound replaces the old N*depth reservation and remains
// safe even for highly clustered or nearly coincident particle layouts.
export const maximumTreeNodeCount = (particleCount: number, depth: number) => {
  let nodes = 1;
  let cells = 1;
  for (let level = 1; level <= depth; level++) {
    cells *= 8;
    nodes += Math.min(particleCount, cells);
  }
  return nodes;
};

export const calculateBarnesHutMemoryLayout = (particleCount: number) => {
  const depth = chooseTreeDepth(particleCount);
  const maximumNodes = maximumTreeNodeCount(particleCount, depth);
  const buffers = {
    nodes: maximumNodes * TREE_NODE_STRIDE,
    children: maximumNodes * TREE_CHILD_STRIDE,
    particlePaths: particleCount * depth * 4,
    leafHeads: maximumNodes * 4,
    particleNext: particleCount * 4,
    topologyDispatch: TOPOLOGY_DISPATCH_BYTES,
  };
  return {
    depth,
    maximumNodes,
    buffers,
    largestBufferBytes: Math.max(...Object.values(buffers)),
    totalBytes: Object.values(buffers).reduce((sum, size) => sum + size, 0),
  };
};

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
  parent: u32,
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

fn packCell(depth: u32, cell: vec3u) -> u32 {
  return (cell.z << 20u) | (cell.y << 10u) | cell.x;
}

fn packOwner(depth: u32, owner: u32) -> u32 {
  return (depth << 20u) | (owner & 0xfffffu);
}

fn nodeOwner(node: TreeNode) -> u32 {
  return node.owner & 0xfffffu;
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
  nodes[index].parent = parent;
  nodes[index].owner = packOwner(depth, owner);
  for (var child = 0u; child < 8u; child++) {
    atomicStore(&nodeChildren[index * 8u + child], EMPTY);
  }
  atomicStore(&leafHeads[index], EMPTY);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn resetBuild(@builtin(global_invocation_id) globalId: vec3u) {
  if (globalId.x != 0u) { return; }
  atomicStore(&control[0], 0u);
  atomicStore(&control[1], 1u);
  atomicStore(&control[2], 0u);
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

@compute @workgroup_size(1)
fn finalizeBounds() {
  nodes[0].parent = bitcast<u32>(treeHalfExtent());
}

// Each child slot elects the lowest particle index. Allocation and path
// resolution use separate dispatches so node indices can be compact.
@compute @workgroup_size(${WORKGROUP_SIZE})
fn claimTreeLevel(@builtin(global_invocation_id) globalId: vec3u) {
  let particleIndex = globalId.x;
  if (particleIndex >= parameters.particleCount) { return; }
  if (particles[particleIndex].positionMass.w <= 0.0) { return; }
  let level = parameters.currentLevel;
  let leaf = targetLeaf(particles[particleIndex].positionMass.xyz);
  let shift = TREE_DEPTH - level;
  let cell = leaf >> vec3u(shift);
  var parent = 0u;
  if (level > 1u) {
    parent = particleNodes[(level - 2u) * parameters.particleCount + particleIndex];
  }
  if (parent == EMPTY) { return; }
  let childSlot = (cell.x & 1u) | ((cell.y & 1u) << 1u) |
    ((cell.z & 1u) << 2u);
  let childAddress = parent * 8u + childSlot;
  atomicMin(&nodeChildren[childAddress], particleIndex);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn finalizeTreeLevel(@builtin(global_invocation_id) globalId: vec3u) {
  let particleIndex = globalId.x;
  if (particleIndex >= parameters.particleCount) { return; }
  if (particles[particleIndex].positionMass.w <= 0.0) { return; }
  let level = parameters.currentLevel;
  let leaf = targetLeaf(particles[particleIndex].positionMass.xyz);
  let shift = TREE_DEPTH - level;
  let cell = leaf >> vec3u(shift);
  var parent = 0u;
  if (level > 1u) {
    parent = particleNodes[(level - 2u) * parameters.particleCount + particleIndex];
  }
  if (parent == EMPTY) { return; }
  let childSlot = (cell.x & 1u) | ((cell.y & 1u) << 1u) |
    ((cell.z & 1u) << 2u);
  let childAddress = parent * 8u + childSlot;
  let owner = atomicLoad(&nodeChildren[childAddress]);
  if (owner == particleIndex) {
    let childNode = atomicAdd(&control[1], 1u);
    if (childNode < parameters.maximumNodes) {
      initializeNode(childNode, parent, level, cell, childSlot, owner);
      atomicStore(&nodeChildren[childAddress], childNode);
    } else {
      atomicStore(&control[2], 1u);
      atomicStore(&nodeChildren[childAddress], EMPTY);
    }
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn resolveTreeLevel(@builtin(global_invocation_id) globalId: vec3u) {
  let particleIndex = globalId.x;
  if (particleIndex >= parameters.particleCount) { return; }
  let level = parameters.currentLevel;
  if (particles[particleIndex].positionMass.w <= 0.0) {
    particleNodes[(level - 1u) * parameters.particleCount + particleIndex] = EMPTY;
    return;
  }
  let leaf = targetLeaf(particles[particleIndex].positionMass.xyz);
  let shift = TREE_DEPTH - level;
  let cell = leaf >> vec3u(shift);
  var parent = 0u;
  if (level > 1u) {
    parent = particleNodes[(level - 2u) * parameters.particleCount + particleIndex];
  }
  if (parent == EMPTY) { return; }
  let childSlot = (cell.x & 1u) | ((cell.y & 1u) << 1u) |
    ((cell.z & 1u) << 2u);
  particleNodes[(level - 1u) * parameters.particleCount + particleIndex] =
    atomicLoad(&nodeChildren[parent * 8u + childSlot]);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn linkLeafParticles(@builtin(global_invocation_id) globalId: vec3u) {
  let particleIndex = globalId.x;
  if (particleIndex >= parameters.particleCount) { return; }
  if (particles[particleIndex].positionMass.w <= 0.0) { return; }
  let leaf = particleNodes[(TREE_DEPTH - 1u) * parameters.particleCount + particleIndex];
  if (leaf == EMPTY) { return; }
  particleNext[particleIndex] = atomicExchange(&leafHeads[leaf], particleIndex);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn aggregateTreeLevel(@builtin(global_invocation_id) globalId: vec3u) {
  let particleIndex = globalId.x;
  let level = parameters.currentLevel;
  if (particleIndex >= parameters.particleCount) { return; }
  var nodeIndex = 0u;
  if (level > 0u) {
    nodeIndex = particleNodes[(level - 1u) * parameters.particleCount + particleIndex];
    if (nodeIndex == EMPTY || nodeOwner(nodes[nodeIndex]) != particleIndex) { return; }
  } else if (particleIndex > 0u) {
    return;
  }
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
  // Keep parent links intact for topology diagnostics and future traversal
  // variants; terminal occupancy is represented by the exact linked list.
}

`;

const createTopologyShader = (treeDepth: number) => /* wgsl */ `
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
  parent: u32,
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
@group(0) @binding(1) var<storage, read> nodes: array<TreeNode>;
@group(0) @binding(2) var<storage, read> particleNodes: array<u32>;
@group(0) @binding(3) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> topologyDispatch: array<u32>;
@group(0) @binding(5) var<uniform> parameters: TreeParameters;

const TREE_DEPTH = ${treeDepth}u;
const EMPTY = 0xffffffffu;

fn targetLeaf(position: vec3f, halfExtent: f32) -> vec3u {
  let normalized = clamp(
    (position + vec3f(halfExtent)) / (2.0 * halfExtent),
    vec3f(0.0),
    vec3f(0.999999),
  );
  return vec3u(normalized * f32(1u << TREE_DEPTH));
}

fn packCell(cell: vec3u) -> u32 {
  return (cell.z << 20u) | (cell.y << 10u) | cell.x;
}

@compute @workgroup_size(1)
fn beginTopologyCheck() {
  atomicStore(&control[3], 0u);
  topologyDispatch[0] = 0u;
  topologyDispatch[1] = 1u;
  topologyDispatch[2] = 1u;
  topologyDispatch[3] = 0u;
  topologyDispatch[4] = 1u;
  topologyDispatch[5] = 1u;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn detectTopologyChanges(@builtin(global_invocation_id) globalId: vec3u) {
  let particleIndex = globalId.x;
  if (particleIndex >= parameters.particleCount) { return; }
  let particle = particles[particleIndex];
  if (particle.positionMass.w <= 0.0) { return; }
  let nodeIndex = particleNodes[(TREE_DEPTH - 1u) * parameters.particleCount + particleIndex];
  let halfExtent = bitcast<f32>(nodes[0].parent);
  if (nodeIndex == EMPTY || any(abs(particle.positionMass.xyz) >= vec3f(halfExtent))) {
    atomicStore(&control[3], 1u);
    return;
  }
  if (nodes[nodeIndex].packedCell != packCell(targetLeaf(particle.positionMass.xyz, halfExtent))) {
    atomicStore(&control[3], 1u);
  }
}

@compute @workgroup_size(1)
fn finalizeTopologyCheck() {
  let rebuild = atomicLoad(&control[3]);
  topologyDispatch[0] = rebuild;
  topologyDispatch[3] = rebuild *
    ((parameters.particleCount + ${WORKGROUP_SIZE - 1}u) / ${WORKGROUP_SIZE}u);
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
  parent: u32,
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
@group(0) @binding(3) var<storage, read> nodeChildren: array<u32>;
@group(0) @binding(4) var<storage, read> leafHeads: array<u32>;
@group(0) @binding(5) var<storage, read> particleNext: array<u32>;
@group(0) @binding(6) var<storage, read> activeIndices: array<u32>;
@group(0) @binding(7) var<storage, read> adaptiveControl: array<u32>;
@group(0) @binding(8) var<uniform> parameters: TreeParameters;

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
  return bitcast<f32>(nodes[0].parent);
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
  var sourceIndex = leafHeads[nodeIndex];
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
    packedCell & 0x3ffu,
    (packedCell >> 10u) & 0x3ffu,
    (packedCell >> 20u) & 0x3ffu,
  );
}

fn nodeDepth(nodeIndex: u32, node: TreeNode) -> u32 {
  return select(node.owner >> 20u, 0u, nodeIndex == 0u);
}

fn overlapsExactNearField(nodeIndex: u32, node: TreeNode, targetCell: vec3u) -> bool {
  let depth = nodeDepth(nodeIndex, node);
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
  let activeOrdinal = globalId.x;
  let activeCount = min(adaptiveControl[1], parameters.particleCount);
  if (activeOrdinal >= activeCount) { return; }
  let targetIndex = activeIndices[activeOrdinal];
  let targetParticle = particles[targetIndex];
  if (targetParticle.positionMass.w <= 0.0) {
    accelerations[targetIndex] = vec4f(0.0);
    return;
  }

  let halfExtent = treeHalfExtent();
  let targetCell = targetLeaf(targetParticle.positionMass.xyz, halfExtent);
  var stack: array<u32, ${BARNES_HUT_TRAVERSAL_STACK_CAPACITY}>;
  var stackSize = 1u;
  stack[0] = 0u;
  var acceleration = vec3f(0.0);
  var traversalOverflow = false;

  loop {
    if (stackSize == 0u) { break; }
    stackSize--;
    let current = stack[stackSize];
    let node = nodes[current];
    if (node.mass <= 0.0) { continue; }

    let depth = nodeDepth(current, node);
    let cell = unpackCell(node.packedCell);
    let dimension = 1u << depth;
    let shift = TREE_DEPTH - depth;
    let containsTarget = all((targetCell >> vec3u(shift)) == cell);
    let nearField = overlapsExactNearField(current, node, targetCell);
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
      let child = nodeChildren[current * 8u + reverseSlot - 1u];
      if (child != EMPTY) {
        if (stackSize < ${BARNES_HUT_TRAVERSAL_STACK_CAPACITY}u) {
          stack[stackSize] = child;
          stackSize++;
        } else {
          traversalOverflow = true;
        }
      }
    }
  }
  // The fourth acceleration lane is otherwise unused. Test readbacks scan it
  // so any future depth/capacity mismatch is explicit without another storage
  // binding or an atomic in the production force loop.
  accelerations[targetIndex] = vec4f(
    acceleration,
    select(0.0, 1.0, traversalOverflow),
  );
}
`;

export const createBarnesHutSolver: GalaxySolverFactory = async ({
  device,
  initial,
  stateBuffers,
  accelerationBuffer,
  activeIndicesBuffer,
  adaptiveControlBuffer,
  indirectDispatchBuffer,
}) => {
  const particleCount = initial.parameters.particleCount;
  const treeDepth = chooseTreeDepth(particleCount);
  const maximumNodes = maximumTreeNodeCount(particleCount, treeDepth);
  const buildModule = device.createShaderModule({
    label: "Galaxy compact Barnes-Hut build module",
    code: createBuildShader(treeDepth),
  });
  const topologyModule = device.createShaderModule({
    label: "Galaxy Barnes-Hut topology check module",
    code: createTopologyShader(treeDepth),
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
  const topologyLayout = device.createBindGroupLayout({
    label: "Barnes-Hut topology check layout",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const forceLayout = device.createBindGroupLayout({
    label: "Compact Barnes-Hut force layout",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const buildPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [buildLayout] });
  const topologyPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [topologyLayout],
  });
  const forcePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [forceLayout] });
  const buildEntries = [
    ["Reset", "resetBuild"],
    ["Bounds", "measureBounds"],
    ["Finalize bounds", "finalizeBounds"],
    ["Claim level", "claimTreeLevel"],
    ["Allocate level", "finalizeTreeLevel"],
    ["Resolve level", "resolveTreeLevel"],
    ["Link leaves", "linkLeafParticles"],
    ["Aggregate level", "aggregateTreeLevel"],
  ] as const;
  const [
    resetPipeline,
    boundsPipeline,
    finalizeBoundsPipeline,
    claimLevelPipeline,
    allocateLevelPipeline,
    resolveLevelPipeline,
    linkLeavesPipeline,
    aggregatePipeline,
  ] = await Promise.all(buildEntries.map(([label, entryPoint]) =>
    device.createComputePipelineAsync({
      label: `Compact Barnes-Hut ${label}`,
      layout: buildPipelineLayout,
      compute: { module: buildModule, entryPoint },
    })));
  const [
    beginTopologyCheckPipeline,
    detectTopologyChangesPipeline,
    finalizeTopologyCheckPipeline,
  ] = await Promise.all(([
    ["Begin", "beginTopologyCheck"],
    ["Detect", "detectTopologyChanges"],
    ["Finalize", "finalizeTopologyCheck"],
  ] as const).map(([label, entryPoint]) => device.createComputePipelineAsync({
    label: `Barnes-Hut ${label} topology check`,
    layout: topologyPipelineLayout,
    compute: { module: topologyModule, entryPoint },
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
  const topologyDispatchBuffer = createStorageBuffer(
    device,
    "Barnes-Hut conditional topology dispatch",
    TOPOLOGY_DISPATCH_BYTES,
    GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
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
        { binding: 6, resource: { buffer: activeIndicesBuffer } },
        { binding: 7, resource: { buffer: adaptiveControlBuffer } },
        { binding: 8, resource: { buffer: parameterBuffers[0]! } },
      ],
    }));
  const topologyBindGroups = stateBuffers.map((stateBuffer, stateIndex) =>
    device.createBindGroup({
      label: `Barnes-Hut topology check state ${stateIndex}`,
      layout: topologyLayout,
      entries: [
        { binding: 0, resource: { buffer: stateBuffer } },
        { binding: 1, resource: { buffer: nodeBuffer } },
        { binding: 2, resource: { buffer: particleNodeBuffer } },
        { binding: 3, resource: { buffer: controlBuffer } },
        { binding: 4, resource: { buffer: topologyDispatchBuffer } },
        { binding: 5, resource: { buffer: parameterBuffers[0]! } },
      ],
    }));

  const particleWorkgroups = Math.ceil(particleCount / WORKGROUP_SIZE);
  let topologyInitialized = false;

  return {
    kind: "barnes-hut",
    encode: (encoder, sourceIndex) => {
      if (topologyInitialized) {
        // WebGPU does not allow one buffer to be writable storage and an
        // indirect argument in the same pass. Finish the check pass before
        // consuming its conditional dispatch commands in the tree pass.
        const checkPass = encoder.beginComputePass({
          label: "Barnes-Hut topology validity check",
        });
        checkPass.setBindGroup(0, topologyBindGroups[sourceIndex]!);
        checkPass.setPipeline(beginTopologyCheckPipeline!);
        checkPass.dispatchWorkgroups(1);
        checkPass.setPipeline(detectTopologyChangesPipeline!);
        checkPass.dispatchWorkgroups(particleWorkgroups);
        checkPass.setPipeline(finalizeTopologyCheckPipeline!);
        checkPass.dispatchWorkgroups(1);
        checkPass.end();
      }
      const pass = encoder.beginComputePass({ label: "Compact Barnes-Hut pass" });
      pass.setBindGroup(0, buildBindGroups[sourceIndex]![0]!);
      if (!topologyInitialized) {
        pass.setPipeline(resetPipeline!);
        pass.dispatchWorkgroups(1);
        pass.setPipeline(boundsPipeline!);
        pass.dispatchWorkgroups(particleWorkgroups);
        pass.setPipeline(finalizeBoundsPipeline!);
        pass.dispatchWorkgroups(1);
        for (let level = 1; level <= treeDepth; level++) {
          pass.setBindGroup(0, buildBindGroups[sourceIndex]![level]!);
          pass.setPipeline(claimLevelPipeline!);
          pass.dispatchWorkgroups(particleWorkgroups);
          pass.setPipeline(allocateLevelPipeline!);
          pass.dispatchWorkgroups(particleWorkgroups);
          pass.setPipeline(resolveLevelPipeline!);
          pass.dispatchWorkgroups(particleWorkgroups);
        }
        pass.setBindGroup(0, buildBindGroups[sourceIndex]![0]!);
        pass.setPipeline(linkLeavesPipeline!);
        pass.dispatchWorkgroups(particleWorkgroups);
        topologyInitialized = true;
      } else {
        pass.setPipeline(resetPipeline!);
        pass.dispatchWorkgroupsIndirect(topologyDispatchBuffer, 0);
        pass.setPipeline(boundsPipeline!);
        pass.dispatchWorkgroupsIndirect(topologyDispatchBuffer, 12);
        pass.setPipeline(finalizeBoundsPipeline!);
        pass.dispatchWorkgroupsIndirect(topologyDispatchBuffer, 0);
        for (let level = 1; level <= treeDepth; level++) {
          pass.setBindGroup(0, buildBindGroups[sourceIndex]![level]!);
          pass.setPipeline(claimLevelPipeline!);
          pass.dispatchWorkgroupsIndirect(topologyDispatchBuffer, 12);
          pass.setPipeline(allocateLevelPipeline!);
          pass.dispatchWorkgroupsIndirect(topologyDispatchBuffer, 12);
          pass.setPipeline(resolveLevelPipeline!);
          pass.dispatchWorkgroupsIndirect(topologyDispatchBuffer, 12);
        }
        pass.setBindGroup(0, buildBindGroups[sourceIndex]![0]!);
        pass.setPipeline(linkLeavesPipeline!);
        pass.dispatchWorkgroupsIndirect(topologyDispatchBuffer, 12);
      }
      pass.setBindGroup(0, buildBindGroups[sourceIndex]![0]!);
      pass.setPipeline(aggregatePipeline!);
      for (let level = treeDepth; level >= 0; level--) {
        pass.setBindGroup(0, buildBindGroups[sourceIndex]![level]!);
        pass.dispatchWorkgroups(particleWorkgroups);
      }
      pass.setBindGroup(0, forceBindGroups[sourceIndex]!);
      pass.setPipeline(forcePipeline);
      pass.dispatchWorkgroupsIndirect(indirectDispatchBuffer, 0);
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
        topologyDispatchBuffer,
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
  traversalStackCapacity: BARNES_HUT_TRAVERSAL_STACK_CAPACITY,
} as const;
