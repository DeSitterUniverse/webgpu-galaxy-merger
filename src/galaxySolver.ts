import {
  sanitizeGalaxySettings,
  type GalaxyInitialState,
  type GalaxySettings,
} from "./galaxyPhysics";

export type GalaxySolverKind = "all-pairs" | "barnes-hut";

export type GalaxySolverDefinition = {
  kind: GalaxySolverKind;
  label: string;
  description: string;
  maxTextureWidth: number;
};

export const GALAXY_SOLVERS: Record<GalaxySolverKind, GalaxySolverDefinition> = {
  "all-pairs": {
    kind: "all-pairs",
    label: "All-Pairs",
    description: "Exact tiled O(n²) reference solver",
    maxTextureWidth: 184,
  },
  "barnes-hut": {
    kind: "barnes-hut",
    label: "Barnes-Hut",
    description: "Compact occupied-node linear octree",
    maxTextureWidth: 1024,
  },
};

// Sanitize once at the engine boundary, then apply the selected solver's
// capacity. This prevents NaN or injected custom-event values from reaching
// memory preflight, allocation, camera framing, or initialization.
export const sanitizeGalaxySettingsForSolver = (
  settings: GalaxySettings,
  solver: GalaxySolverKind,
) => {
  const sanitized = sanitizeGalaxySettings(settings);
  return {
    ...sanitized,
    textureWidth: Math.min(
      sanitized.textureWidth,
      GALAXY_SOLVERS[solver].maxTextureWidth,
    ),
  };
};

export type GalaxySolverInstance = {
  kind: GalaxySolverKind;
  encode: (
    encoder: GPUCommandEncoder,
    sourceIndex: 0 | 1,
  ) => void;
  destroy: () => void;
};

export type GalaxySolverCreateOptions = {
  device: GPUDevice;
  initial: GalaxyInitialState;
  stateBuffers: [GPUBuffer, GPUBuffer];
  accelerationBuffer: GPUBuffer;
  activeIndicesBuffer: GPUBuffer;
  adaptiveControlBuffer: GPUBuffer;
  indirectDispatchBuffer: GPUBuffer;
};

export type GalaxySolverFactory = (
  options: GalaxySolverCreateOptions,
) => Promise<GalaxySolverInstance>;

export const createSolverParameterData = (
  particleCount: number,
  gravity: number,
  values: readonly number[] = [],
) => {
  const data = new ArrayBuffer(32);
  const view = new DataView(data);
  view.setUint32(0, particleCount, true);
  view.setFloat32(4, gravity, true);
  values.forEach((value, index) => view.setFloat32(8 + index * 4, value, true));
  return new Uint8Array(data);
};

export const createStorageBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags = GPUBufferUsage.STORAGE,
) => device.createBuffer({
  label,
  size: Math.max(4, Math.ceil(size / 4) * 4),
  usage,
});

export const createBufferWithData = (
  device: GPUDevice,
  label: string,
  data: ArrayBufferView<ArrayBufferLike>,
  usage: GPUBufferUsageFlags,
) => {
  const buffer = device.createBuffer({
    label,
    size: Math.max(4, Math.ceil(data.byteLength / 4) * 4),
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  );
  buffer.unmap();
  return buffer;
};
