import {
  createGalaxyInitialState,
  MIN_LIVE_HALO_TEXTURE_WIDTH,
  type GalaxyInitialState,
} from "./galaxyPhysics";
import {
  galaxyMergeShader,
  galaxyRenderShader,
  WORKGROUP_SIZE,
} from "./galaxyShaders";
import {
  createInitialActiveIndices,
  createInitialAdaptiveControl,
  createInitialIndirectDispatch,
  galaxyAdaptiveInitializeShader,
  galaxyAdaptiveKickShader,
  galaxyDriftShader,
  galaxyScheduleShader,
  MAX_TIME_BIN,
  TIMESTEP_ETA,
  TIMESTEP_STATE_BYTES,
} from "./galaxyAdaptiveTimesteps";
import { createAllPairsSolver } from "./galaxyAllPairsSolver";
import {
  calculateBarnesHutMemoryLayout,
  createBarnesHutSolver,
} from "./galaxyBarnesHutSolver";
import {
  GALAXY_SOLVERS,
  sanitizeGalaxySettingsForSolver,
  type GalaxySolverFactory,
  type GalaxySolverInstance,
  type GalaxySolverKind,
} from "./galaxySolver";

declare global {
  interface Window {
    appStats?: { begin: () => void; end: () => void };
  }
}

type PingPongIndex = 0 | 1;

// CPU readbacks are deliberately ring-buffered: mapping one buffer must never
// stall command encoding for the next simulation frame.
type TelemetryReadbackSlot = {
  buffer: GPUBuffer;
  pending: boolean;
  sequence: number;
};

type WebGPUBackend = {
  adapter: GPUAdapter;
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  initializePipeline: GPUComputePipeline;
  driftPipeline: GPUComputePipeline;
  scheduleBeginPipeline: GPUComputePipeline;
  scheduleCollectPipeline: GPUComputePipeline;
  scheduleFinishPipeline: GPUComputePipeline;
  kickPipeline: GPUComputePipeline;
  mergePipeline: GPUComputePipeline;
  renderPipeline: GPURenderPipeline;
};

// Resources rebuilt by particle-count/radius/offset controls live separately
// from the device-level pipelines, which are compiled only once.
type SimulationResources = {
  initial: GalaxyInitialState;
  stateBuffers: [GPUBuffer, GPUBuffer];
  visualBuffer: GPUBuffer;
  parameterBuffer: GPUBuffer;
  accelerationBuffer: GPUBuffer;
  timestepBuffer: GPUBuffer;
  activeIndicesBuffer: GPUBuffer;
  adaptiveControlBuffer: GPUBuffer;
  indirectDispatchBuffer: GPUBuffer;
  solver: GalaxySolverInstance;
  telemetryBuffer: GPUBuffer;
  telemetryReadbacks: TelemetryReadbackSlot[];
  renderParameterBuffer: GPUBuffer;
  renderIndexBuffer: GPUBuffer;
  renderCount: number;
  initializeBindGroups: [GPUBindGroup, GPUBindGroup];
  driftBindGroups: [GPUBindGroup, GPUBindGroup];
  scheduleBeginBindGroup: GPUBindGroup;
  scheduleCollectBindGroups: [GPUBindGroup, GPUBindGroup];
  scheduleFinishBindGroup: GPUBindGroup;
  kickBindGroups: [GPUBindGroup, GPUBindGroup];
  mergeBindGroups: [GPUBindGroup, GPUBindGroup];
  renderBindGroups: [GPUBindGroup, GPUBindGroup];
  readIndex: PingPongIndex;
  simulationTime: number;
  telemetrySequence: number;
  publishedTelemetrySequence: number;
};

type Vector3 = [number, number, number];

const DEFAULT_DESKTOP_WIDTH = 72;
const DEFAULT_MOBILE_WIDTH = MIN_LIVE_HALO_TEXTURE_WIDTH;
const TARGET_FPS = 60;
const TELEMETRY_READBACK_SLOTS = 3;
const TELEMETRY_BYTES = 16;
const STATE_BYTES_PER_PARTICLE = 8 * Float32Array.BYTES_PER_ELEMENT;
const ACCELERATION_BYTES_PER_PARTICLE = 4 * Float32Array.BYTES_PER_ELEMENT;
const MAX_RENDER_PARTICLES = 262_144;
const SOLVER_FACTORIES: Record<GalaxySolverKind, GalaxySolverFactory> = {
  "all-pairs": createAllPairsSolver,
  "barnes-hut": createBarnesHutSolver,
};

let currentTextureWidth =
  typeof window !== "undefined" && window.innerWidth < 768
    ? DEFAULT_MOBILE_WIDTH
    : DEFAULT_DESKTOP_WIDTH;
let currentGalaxyRadius = 35;
let currentOffset = 25;
let currentSolverKind: GalaxySolverKind = "all-pairs";
let isPaused = false;

const applySanitizedSettings = (requested: {
  textureWidth: number;
  radius: number;
  offset: number;
}) => {
  const settings = sanitizeGalaxySettingsForSolver(
    requested,
    currentSolverKind,
  );
  currentTextureWidth = settings.textureWidth;
  currentGalaxyRadius = settings.radius;
  currentOffset = settings.offset;
  return settings;
};

const clamp = (value: number, minimum: number, maximum: number) =>
  Math.min(maximum, Math.max(minimum, value));

const vectorSubtract = (left: Vector3, right: Vector3): Vector3 => [
  left[0] - right[0],
  left[1] - right[1],
  left[2] - right[2],
];

const vectorCross = (left: Vector3, right: Vector3): Vector3 => [
  left[1] * right[2] - left[2] * right[1],
  left[2] * right[0] - left[0] * right[2],
  left[0] * right[1] - left[1] * right[0],
];

const vectorDot = (left: Vector3, right: Vector3) =>
  left[0] * right[0] + left[1] * right[1] + left[2] * right[2];

const vectorNormalize = (value: Vector3): Vector3 => {
  const magnitude = Math.max(Math.hypot(...value), Number.EPSILON);
  return [value[0] / magnitude, value[1] / magnitude, value[2] / magnitude];
};

const matrixMultiply = (left: Float32Array, right: Float32Array) => {
  const output = new Float32Array(16);
  for (let column = 0; column < 4; column++) {
    for (let row = 0; row < 4; row++) {
      let value = 0;
      for (let index = 0; index < 4; index++) {
        value += left[index * 4 + row]! * right[column * 4 + index]!;
      }
      output[column * 4 + row] = value;
    }
  }
  return output;
};

const perspectiveMatrix = (
  fieldOfView: number,
  aspect: number,
  near: number,
  far: number,
) => {
  const focalLength = 1 / Math.tan(fieldOfView / 2);
  const output = new Float32Array(16);
  output[0] = focalLength / aspect;
  output[5] = focalLength;
  output[10] = far / (near - far);
  output[11] = -1;
  output[14] = (near * far) / (near - far);
  return output;
};

const lookAtMatrix = (
  eye: Vector3,
  target: Vector3,
  up: Vector3,
) => {
  const forward = vectorNormalize(vectorSubtract(target, eye));
  const side = vectorNormalize(vectorCross(forward, up));
  const correctedUp = vectorCross(side, forward);
  return new Float32Array([
    side[0], correctedUp[0], -forward[0], 0,
    side[1], correctedUp[1], -forward[1], 0,
    side[2], correctedUp[2], -forward[2], 0,
    -vectorDot(side, eye),
    -vectorDot(correctedUp, eye),
    vectorDot(forward, eye),
    1,
  ]);
};

const compilationErrors = async (module: GPUShaderModule) => {
  const information = await module.getCompilationInfo();
  return information.messages
    .filter((message) => message.type === "error")
    .map(
      (message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`,
    );
};

const createBufferWithData = (
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

export const createRenderSample = (
  visuals: Float32Array,
  maximumParticles = MAX_RENDER_PARTICLES,
) => {
  const visibleCount = visuals.length / 4;
  const luminous: number[] = [];
  for (let index = 0; index < visibleCount; index++) {
    if (visuals[index * 4 + 3]! > 0) luminous.push(index);
  }
  const sampleCount = Math.min(luminous.length, maximumParticles);
  const indices = new Uint32Array(sampleCount);
  const sampledVisuals = new Float32Array(sampleCount * 4);
  for (let ordinal = 0; ordinal < sampleCount; ordinal++) {
    const sourceOrdinal = Math.min(
      luminous.length - 1,
      Math.floor((ordinal * luminous.length) / sampleCount),
    );
    const particleIndex = luminous[sourceOrdinal]!;
    indices[ordinal] = particleIndex;
    sampledVisuals.set(
      visuals.subarray(particleIndex * 4, particleIndex * 4 + 4),
      ordinal * 4,
    );
  }
  return { indices, visuals: sampledVisuals };
};

const destroySimulation = (simulation: SimulationResources | null) => {
  if (!simulation) return;
  simulation.stateBuffers.forEach((buffer) => buffer.destroy());
  simulation.visualBuffer.destroy();
  simulation.parameterBuffer.destroy();
  simulation.accelerationBuffer.destroy();
  simulation.timestepBuffer.destroy();
  simulation.activeIndicesBuffer.destroy();
  simulation.adaptiveControlBuffer.destroy();
  simulation.indirectDispatchBuffer.destroy();
  simulation.solver.destroy();
  simulation.telemetryBuffer.destroy();
  simulation.telemetryReadbacks.forEach(({ buffer }) => buffer.destroy());
  simulation.renderParameterBuffer.destroy();
  simulation.renderIndexBuffer.destroy();
};

export const initGalaxyEngine = () => {
  const canvas = document.getElementById(
    "galaxy-canvas",
  ) as HTMLCanvasElement | null;
  if (!canvas) return;

  let backend: WebGPUBackend | null = null;
  let backendPromise: Promise<WebGPUBackend> | null = null;
  let simulation: SimulationResources | null = null;
  let disposed = false;
  let themeActive = document.documentElement.dataset.theme !== "dark";
  let currentFizzle = 0;
  let fizzleAnimation = 0;
  let animationFrame = 0;
  let previousFrame = performance.now();
  let buildGeneration = 0;
  let canvasConfigured = false;
  let validationError: string | null = null;
  let physicsSubmissionPending = false;

  const dispatchStatus = (
    state: "initializing" | "ready" | "unsupported" | "error",
    message: string,
  ) => {
    const output = document.getElementById("galaxy-backend-status");
    if (output) {
      output.textContent = message;
      output.hidden = state === "ready";
    }
    window.dispatchEvent(
      new CustomEvent("galaxy:status", { detail: { state, message } }),
    );
  };

  const dispatchFizzleComplete = (visible: boolean) =>
    window.dispatchEvent(
      new CustomEvent("galaxy:fizzle-complete", { detail: { visible } }),
    );

  const configureCanvas = () => {
    if (!backend) return;
    const pixelRatio = Math.min(window.devicePixelRatio, 2);
    const width = Math.max(1, Math.floor(window.innerWidth * pixelRatio));
    const height = Math.max(1, Math.floor(window.innerHeight * pixelRatio));
    const resized = canvas.width !== width || canvas.height !== height;
    if (resized) {
      canvas.width = width;
      canvas.height = height;
    }
    canvas.style.width = `${window.innerWidth}px`;
    canvas.style.height = `${window.innerHeight}px`;
    if (!canvasConfigured || resized) {
      backend.context.configure({
        device: backend.device,
        format: backend.format,
        alphaMode: "premultiplied",
      });
      canvasConfigured = true;
    }
  };

  // Device and pipeline setup is cached across simulation rebuilds. Controls
  // only replace buffers/bind groups, avoiding shader compilation hitches.
  const initializeBackend = async () => {
    if (backend) return backend;
    if (backendPromise) return backendPromise;
    backendPromise = (async () => {
      dispatchStatus("initializing", "Requesting WebGPU…");
      if (!navigator.gpu) {
        throw new Error(
          "WebGPU is unavailable in this browser. Use a current browser with WebGPU enabled.",
        );
      }
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
      });
      if (!adapter) {
        throw new Error("No compatible WebGPU adapter was found.");
      }
      if (adapter.limits.maxStorageBuffersPerShaderStage < 8) {
        throw new Error(
          "This GPU exposes too few storage buffers for the galaxy solver.",
        );
      }
      const requiredWorkgroupStorage = WORKGROUP_SIZE * 20;
      // The tiled shader stores one vec4 position/mass plus one f32 softening
      // per thread. Validate the hardcoded workgroup against the chosen adapter.
      if (
        WORKGROUP_SIZE > adapter.limits.maxComputeInvocationsPerWorkgroup ||
        WORKGROUP_SIZE > adapter.limits.maxComputeWorkgroupSizeX ||
        requiredWorkgroupStorage > adapter.limits.maxComputeWorkgroupStorageSize
      ) {
        throw new Error(
          `The ${WORKGROUP_SIZE}-thread galaxy solver exceeds this GPU's WebGPU limits.`,
        );
      }
      const requiredLimits: Record<string, GPUSize64> = {};
      // Million-particle trees need buffers above WebGPU's conservative
      // defaults. Request only limits already exposed by this adapter.
      requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
      requiredLimits.maxStorageBufferBindingSize =
        adapter.limits.maxStorageBufferBindingSize;
      if (WORKGROUP_SIZE > 256) {
        requiredLimits.maxComputeInvocationsPerWorkgroup = WORKGROUP_SIZE;
        requiredLimits.maxComputeWorkgroupSizeX = WORKGROUP_SIZE;
      }
      if (requiredWorkgroupStorage > 16_384) {
        requiredLimits.maxComputeWorkgroupStorageSize = requiredWorkgroupStorage;
      }
      const device = await adapter.requestDevice({
        requiredLimits,
      });
      const context = canvas.getContext("webgpu");
      if (!context) {
        device.destroy();
        throw new Error("The canvas could not create a WebGPU context.");
      }
      const format = navigator.gpu.getPreferredCanvasFormat();

      const initializeModule = device.createShaderModule({
        label: "Galaxy adaptive leapfrog initializer",
        code: galaxyAdaptiveInitializeShader,
      });
      const driftModule = device.createShaderModule({
        label: "Galaxy adaptive drift module",
        code: galaxyDriftShader,
      });
      const scheduleModule = device.createShaderModule({
        label: "Galaxy block timestep scheduler",
        code: galaxyScheduleShader,
      });
      const kickModule = device.createShaderModule({
        label: "Galaxy adaptive kick module",
        code: galaxyAdaptiveKickShader,
      });
      const mergeModule = device.createShaderModule({
        label: "Galaxy swept core merger module",
        code: galaxyMergeShader,
      });
      const renderModule = device.createShaderModule({
        label: "Galaxy billboard render module",
        code: galaxyRenderShader,
      });
      const shaderErrors = (await Promise.all([
        compilationErrors(initializeModule),
        compilationErrors(driftModule),
        compilationErrors(scheduleModule),
        compilationErrors(kickModule),
        compilationErrors(mergeModule),
        compilationErrors(renderModule),
      ])).flat();
      if (shaderErrors.length) {
        device.destroy();
        throw new Error(shaderErrors.join("\n"));
      }

      const [
        initializePipeline,
        driftPipeline,
        scheduleBeginPipeline,
        scheduleCollectPipeline,
        scheduleFinishPipeline,
        kickPipeline,
        mergePipeline,
        renderPipeline,
      ] =
        await Promise.all([
          device.createComputePipelineAsync({
            label: "Galaxy adaptive leapfrog initializer",
            layout: "auto",
            compute: { module: initializeModule, entryPoint: "initializeAdaptiveLeapfrog" },
          }),
          device.createComputePipelineAsync({
            label: "Galaxy block-step drift",
            layout: "auto",
            compute: { module: driftModule, entryPoint: "driftParticles" },
          }),
          device.createComputePipelineAsync({
            label: "Galaxy begin active schedule",
            layout: "auto",
            compute: { module: scheduleModule, entryPoint: "beginSchedule" },
          }),
          device.createComputePipelineAsync({
            label: "Galaxy collect active particles",
            layout: "auto",
            compute: { module: scheduleModule, entryPoint: "collectActiveParticles" },
          }),
          device.createComputePipelineAsync({
            label: "Galaxy finish active schedule",
            layout: "auto",
            compute: { module: scheduleModule, entryPoint: "finishSchedule" },
          }),
          device.createComputePipelineAsync({
            label: "Galaxy adaptive block kick",
            layout: "auto",
            compute: { module: kickModule, entryPoint: "kickActiveParticles" },
          }),
          device.createComputePipelineAsync({
            label: "Galaxy swept core merger",
            layout: "auto",
            compute: { module: mergeModule, entryPoint: "mergeCores" },
          }),
          device.createRenderPipelineAsync({
            label: "Galaxy WebGPU particle renderer",
            layout: "auto",
            vertex: { module: renderModule, entryPoint: "vertexMain" },
            fragment: {
              module: renderModule,
              entryPoint: "fragmentMain",
              targets: [
                {
                  format,
                  blend: {
                    color: {
                      srcFactor: "src-alpha",
                      dstFactor: "one",
                      operation: "add",
                    },
                    alpha: {
                      srcFactor: "one",
                      dstFactor: "one",
                      operation: "add",
                    },
                  },
                  writeMask: GPUColorWrite.ALL,
                },
              ],
            },
            primitive: { topology: "triangle-list" },
          }),
        ]);

      const created: WebGPUBackend = {
        adapter,
        device,
        context,
        format,
        initializePipeline,
        driftPipeline,
        scheduleBeginPipeline,
        scheduleCollectPipeline,
        scheduleFinishPipeline,
        kickPipeline,
        mergePipeline,
        renderPipeline,
      };
      backend = created;
      configureCanvas();
      device.addEventListener("uncapturederror", (event) => {
        if (validationError) return;
        validationError = event.error.message;
        dispatchStatus("error", `WebGPU validation error: ${validationError}`);
      });
      void device.lost.then((information) => {
        if (!disposed) {
          dispatchStatus(
            "error",
            `WebGPU device lost: ${information.message || information.reason}`,
          );
        }
      });
      const adapterName = adapter.info.description || adapter.info.device;
      dispatchStatus(
        "ready",
        adapterName ? `WebGPU · ${adapterName}` : "WebGPU ready",
      );
      // Accuracy readback is opt-in through the dedicated test-server query.
      // Normal development never imports or executes the O(n²) reference path.
      if (import.meta.env.PUBLIC_GALAXY_TEST_MODE === "true") {
        const testParameters = new URLSearchParams(location.search);
        const solver = testParameters.get("solverAccuracy");
        if (solver === "all-pairs" || solver === "barnes-hut") {
          void import("./galaxySolverAccuracy").then(async ({ runGalaxySolverAccuracy }) => {
            const requestedWidth = Number(
              new URLSearchParams(location.search).get("accuracyWidth") ?? 56,
            );
            const requestedSamples = Number(
              new URLSearchParams(location.search).get("accuracySamples") ?? 192,
            );
            const result = await runGalaxySolverAccuracy(
              device,
              solver,
              Math.min(1024, Math.max(56, requestedWidth)),
              Math.min(512, Math.max(2, requestedSamples)),
            );
            const output = document.createElement("pre");
            output.id = "galaxy-accuracy-results";
            output.textContent = JSON.stringify(result, null, 2);
            document.body.append(output);
          }).catch((error: unknown) => {
            dispatchStatus(
              "error",
              `Solver accuracy test failed: ${
                error instanceof Error ? error.message : String(error)
              }`,
            );
          });
        }
        if (testParameters.get("gpuEvolution") === "true") {
          void import("./galaxyGpuEvolutionValidation").then(async ({
            runGalaxyGpuEvolutionSuite,
          }) => {
            const requestedSteps = Number(testParameters.get("steps") ?? 1_000);
            const result = await runGalaxyGpuEvolutionSuite(
              device,
              Math.min(10_000, Math.max(1, requestedSteps)),
            );
            const output = document.createElement("pre");
            output.id = "galaxy-evolution-results";
            output.textContent = JSON.stringify(result, null, 2);
            document.body.append(output);
          }).catch((error: unknown) => {
            dispatchStatus(
              "error",
              `GPU evolution test failed: ${
                error instanceof Error ? error.message : String(error)
              }`,
            );
          });
        }
      }
      return created;
    })().catch((error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      dispatchStatus(
        navigator.gpu ? "error" : "unsupported",
        message,
      );
      throw error;
    });
    return backendPromise;
  };

  const createParameterData = (initial: GalaxyInitialState) => {
    // WGSL uniforms follow 16-byte alignment; keep the block at two vec4 lanes.
    const data = new ArrayBuffer(32);
    const view = new DataView(data);
    const { parameters } = initial;
    view.setFloat32(0, parameters.timeStep, true);
    view.setFloat32(4, parameters.gravity, true);
    view.setFloat32(8, TIMESTEP_ETA, true);
    view.setFloat32(12, parameters.captureSofteningFactor, true);
    view.setUint32(16, parameters.particleCount, true);
    view.setUint32(20, parameters.core1Index, true);
    view.setUint32(24, parameters.core2Index, true);
    view.setUint32(28, MAX_TIME_BIN, true);
    return new Uint8Array(data);
  };

  const buildGalaxy = async (generation: number) => {
    const activeBackend = await initializeBackend();
    if (disposed || generation !== buildGeneration) return;
    const settings = applySanitizedSettings({
      textureWidth: currentTextureWidth,
      radius: currentGalaxyRadius,
      offset: currentOffset,
    });
    const requestedParticleCount = settings.textureWidth ** 2;
    if (currentSolverKind === "barnes-hut") {
      const memory = calculateBarnesHutMemoryLayout(requestedParticleCount);
      const bufferLimit = Math.min(
        Number(activeBackend.device.limits.maxBufferSize),
        Number(activeBackend.device.limits.maxStorageBufferBindingSize),
      );
      if (memory.largestBufferBytes > bufferLimit) {
        throw new Error(
          `${settings.textureWidth}x${settings.textureWidth} needs a ${
            (memory.largestBufferBytes / 2 ** 20).toFixed(0)
          } MiB tree buffer, above this GPU's ${
            (bufferLimit / 2 ** 20).toFixed(0)
          } MiB WebGPU limit.`,
        );
      }
    }
    dispatchStatus("initializing", "Building deterministic galaxy state…");
    const initial = createGalaxyInitialState(settings);
    if (disposed || generation !== buildGeneration) return;
    const { device } = activeBackend;
    // State A/B alternate source and destination every step. Rendering reads
    // whichever buffer the latest compute pass just produced.
    const stateBuffers: [GPUBuffer, GPUBuffer] = [
      createBufferWithData(
        device,
        "Galaxy state A",
        initial.state,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      ),
      device.createBuffer({
        label: "Galaxy state B",
        size: initial.parameters.particleCount * STATE_BYTES_PER_PARTICLE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      }),
    ];
    const accelerationBuffer = device.createBuffer({
      label: "Galaxy accelerations",
      size:
        initial.parameters.particleCount * ACCELERATION_BYTES_PER_PARTICLE,
      usage: GPUBufferUsage.STORAGE,
    });
    const timestepBuffer = device.createBuffer({
      label: "Galaxy per-particle timestep state",
      size: initial.parameters.particleCount * TIMESTEP_STATE_BYTES,
      usage: GPUBufferUsage.STORAGE,
    });
    const activeIndicesBuffer = createBufferWithData(
      device,
      "Galaxy active particle indices",
      createInitialActiveIndices(initial.parameters.particleCount),
      GPUBufferUsage.STORAGE,
    );
    const adaptiveControlBuffer = createBufferWithData(
      device,
      "Galaxy adaptive timestep control",
      createInitialAdaptiveControl(initial.parameters.particleCount),
      GPUBufferUsage.STORAGE,
    );
    const indirectDispatchBuffer = createBufferWithData(
      device,
      "Galaxy active indirect dispatch",
      createInitialIndirectDispatch(initial.parameters.particleCount),
      GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    );
    const solver = await SOLVER_FACTORIES[currentSolverKind]({
      device,
      initial,
      stateBuffers,
      accelerationBuffer,
      activeIndicesBuffer,
      adaptiveControlBuffer,
      indirectDispatchBuffer,
    });
    if (disposed || generation !== buildGeneration) {
      solver.destroy();
      accelerationBuffer.destroy();
      timestepBuffer.destroy();
      activeIndicesBuffer.destroy();
      adaptiveControlBuffer.destroy();
      indirectDispatchBuffer.destroy();
      stateBuffers.forEach((buffer) => buffer.destroy());
      return;
    }
    const renderSample = createRenderSample(initial.visuals);
    const visualBuffer = createBufferWithData(
      device,
      "Galaxy particle visuals",
      renderSample.visuals,
      GPUBufferUsage.STORAGE,
    );
    const renderIndexBuffer = createBufferWithData(
      device,
      "Galaxy luminous render indices",
      renderSample.indices,
      GPUBufferUsage.STORAGE,
    );
    const parameterBuffer = createBufferWithData(
      device,
      "Galaxy simulation parameters",
      createParameterData(initial),
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    const core1Offset = initial.parameters.core1Index * 8;
    const core2Offset = initial.parameters.core2Index * 8;
    const initialCoreSeparation = Math.hypot(
      initial.state[core2Offset]! - initial.state[core1Offset]!,
      initial.state[core2Offset + 1]! - initial.state[core1Offset + 1]!,
      initial.state[core2Offset + 2]! - initial.state[core1Offset + 2]!,
    );
    // The GPU owns the packed state from here. Release large CPU initialization
    // arrays so million-body configurations do not retain a duplicate copy.
    initial.state = new Float32Array(0);
    initial.metadata = new Uint32Array(0);
    initial.visuals = new Float32Array(0);
    const telemetryData = new ArrayBuffer(TELEMETRY_BYTES);
    new DataView(telemetryData).setFloat32(0, initialCoreSeparation, true);
    const telemetryBuffer = createBufferWithData(
      device,
      "Galaxy core telemetry",
      new Uint8Array(telemetryData),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );
    const telemetryReadbacks = Array.from(
      { length: TELEMETRY_READBACK_SLOTS },
      (_, index): TelemetryReadbackSlot => ({
        buffer: device.createBuffer({
          label: `Galaxy telemetry readback ${index}`,
          size: TELEMETRY_BYTES,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        pending: false,
        sequence: 0,
      }),
    );
    const renderParameterBuffer = device.createBuffer({
      label: "Galaxy render parameters",
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const initializeBindGroups = stateBuffers.map((sourceBuffer, index) =>
      device.createBindGroup({
        label: `Galaxy adaptive initialization ${index}`,
        layout: activeBackend.initializePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: sourceBuffer } },
          { binding: 1, resource: { buffer: stateBuffers[1 - index]! } },
          { binding: 2, resource: { buffer: parameterBuffer } },
          { binding: 3, resource: { buffer: accelerationBuffer } },
          { binding: 4, resource: { buffer: timestepBuffer } },
        ],
      }),
    ) as [GPUBindGroup, GPUBindGroup];
    const driftBindGroups = stateBuffers.map((sourceBuffer, index) =>
      device.createBindGroup({
        label: `Galaxy drift ${index}`,
        layout: activeBackend.driftPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: sourceBuffer } },
          { binding: 1, resource: { buffer: stateBuffers[1 - index]! } },
          { binding: 2, resource: { buffer: parameterBuffer } },
        ],
      }),
    ) as [GPUBindGroup, GPUBindGroup];
    const scheduleBeginBindGroup = device.createBindGroup({
      label: "Galaxy begin active schedule",
      layout: activeBackend.scheduleBeginPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 3, resource: { buffer: adaptiveControlBuffer } },
        { binding: 4, resource: { buffer: indirectDispatchBuffer } },
      ],
    });
    const scheduleCollectBindGroups = stateBuffers.map((stateBuffer, index) =>
      device.createBindGroup({
        label: `Galaxy collect active state ${index}`,
        layout: activeBackend.scheduleCollectPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: stateBuffer } },
          { binding: 1, resource: { buffer: timestepBuffer } },
          { binding: 2, resource: { buffer: activeIndicesBuffer } },
          { binding: 3, resource: { buffer: adaptiveControlBuffer } },
          { binding: 5, resource: { buffer: parameterBuffer } },
        ],
      }),
    ) as [GPUBindGroup, GPUBindGroup];
    const scheduleFinishBindGroup = device.createBindGroup({
      label: "Galaxy finish active schedule",
      layout: activeBackend.scheduleFinishPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 3, resource: { buffer: adaptiveControlBuffer } },
        { binding: 4, resource: { buffer: indirectDispatchBuffer } },
        { binding: 5, resource: { buffer: parameterBuffer } },
      ],
    });
    const kickBindGroups = stateBuffers.map((stateBuffer, index) =>
      device.createBindGroup({
        label: `Galaxy adaptive kick state ${index}`,
        layout: activeBackend.kickPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: stateBuffer } },
          { binding: 1, resource: { buffer: accelerationBuffer } },
          { binding: 2, resource: { buffer: timestepBuffer } },
          { binding: 3, resource: { buffer: activeIndicesBuffer } },
          { binding: 4, resource: { buffer: adaptiveControlBuffer } },
          { binding: 5, resource: { buffer: parameterBuffer } },
        ],
      }),
    ) as [GPUBindGroup, GPUBindGroup];
    const mergeBindGroups = stateBuffers.map((sourceBuffer, index) =>
      device.createBindGroup({
        label: `Galaxy merge state ${index}`,
        layout: activeBackend.mergePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: sourceBuffer } },
          { binding: 1, resource: { buffer: stateBuffers[1 - index]! } },
          { binding: 2, resource: { buffer: parameterBuffer } },
          { binding: 3, resource: { buffer: telemetryBuffer } },
        ],
      }),
    ) as [GPUBindGroup, GPUBindGroup];
    const renderBindGroups = stateBuffers.map((stateBuffer, index) =>
      device.createBindGroup({
        label: `Galaxy render ${index}`,
        layout: activeBackend.renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: stateBuffer } },
          { binding: 1, resource: { buffer: visualBuffer } },
          { binding: 2, resource: { buffer: renderParameterBuffer } },
          { binding: 3, resource: { buffer: renderIndexBuffer } },
        ],
      }),
    ) as [GPUBindGroup, GPUBindGroup];

    const initializationEncoder = device.createCommandEncoder({
      label: "Galaxy leapfrog initialization encoder",
    });
    solver.encode(initializationEncoder, 0);
    const initializationPass = initializationEncoder.beginComputePass({
      label: "Galaxy leapfrog half-step initialization pass",
    });
    initializationPass.setPipeline(activeBackend.initializePipeline);
    initializationPass.setBindGroup(0, initializeBindGroups[0]);
    initializationPass.dispatchWorkgroups(
      Math.ceil(initial.parameters.particleCount / WORKGROUP_SIZE),
    );
    initializationPass.end();
    // Seeded CPU velocities are full-step values. Each particle receives its
    // own forward half-kick based on its first synchronized timestep bin.
    device.queue.submit([initializationEncoder.finish()]);

    const nextSimulation: SimulationResources = {
      initial,
      stateBuffers,
      visualBuffer,
      parameterBuffer,
      accelerationBuffer,
      timestepBuffer,
      activeIndicesBuffer,
      adaptiveControlBuffer,
      indirectDispatchBuffer,
      solver,
      telemetryBuffer,
      telemetryReadbacks,
      renderParameterBuffer,
      renderIndexBuffer,
      renderCount: renderSample.indices.length,
      initializeBindGroups,
      driftBindGroups,
      scheduleBeginBindGroup,
      scheduleCollectBindGroups,
      scheduleFinishBindGroup,
      kickBindGroups,
      mergeBindGroups,
      renderBindGroups,
      readIndex: 1,
      simulationTime: 0,
      telemetrySequence: 0,
      publishedTelemetrySequence: 0,
    };
    const previousSimulation = simulation;
    simulation = nextSimulation;
    // A rebuild can finish while the previous simulation still has a queued
    // submission. Its completion callback intentionally cannot mutate the new
    // simulation, so explicitly release the new instance here.
    physicsSubmissionPending = false;
    if (previousSimulation) {
      void device.queue.onSubmittedWorkDone().then(() =>
        destroySimulation(previousSimulation),
      );
    }
    window.dispatchEvent(
      new CustomEvent("galaxy:initialized", {
        detail: { initialCoreSeparation },
      }),
    );
    dispatchStatus(
      "ready",
      `WebGPU · ${GALAXY_SOLVERS[currentSolverKind].label} · live disk + halo`,
    );
  };

  const updateRenderParameters = (activeSimulation: SimulationResources) => {
    if (!backend) return;
    const { parameters } = activeSimulation.initial;
    const visibleExtent =
      parameters.centerSeparation * 0.5 + parameters.radius * 1.15;
    const cameraDistance = Math.max(130, visibleExtent * 2.4);
    const cameraDirection = vectorNormalize([0, 0.49, 0.87]);
    const eye = [
      cameraDirection[0] * cameraDistance,
      cameraDirection[1] * cameraDistance,
      cameraDirection[2] * cameraDistance,
    ] satisfies Vector3;
    const view = lookAtMatrix(eye, [0, 0, 0], [0, 1, 0]);
    const projection = perspectiveMatrix(
      Math.PI / 3,
      canvas.width / Math.max(canvas.height, 1),
      0.1,
      cameraDistance + visibleExtent * 5 + 200,
    );
    const data = new Float32Array(20);
    data.set(matrixMultiply(projection, view), 0);
    data[16] = canvas.width;
    data[17] = canvas.height;
    data[18] = currentFizzle;
    data[19] = Math.min(window.devicePixelRatio, 2) * 150;
    backend.device.queue.writeBuffer(
      activeSimulation.renderParameterBuffer,
      0,
      data,
    );
  };

  const scheduleTelemetryReadback = (
    activeSimulation: SimulationResources,
    slot: TelemetryReadbackSlot,
  ) => {
    // Mapping is asynchronous; sequence numbers stop an older slot from
    // overwriting newer core telemetry when callbacks resolve out of order.
    const target = activeSimulation;
    void slot.buffer.mapAsync(GPUMapMode.READ).then(() => {
      try {
        const copy = slot.buffer.getMappedRange().slice(0);
        slot.buffer.unmap();
        if (
          simulation === target &&
          !disposed &&
          slot.sequence >= target.publishedTelemetrySequence
        ) {
          target.publishedTelemetrySequence = slot.sequence;
          const view = new DataView(copy);
          const corePhase = view.getUint32(4, true);
          const binary = corePhase === 1;
          const merged = corePhase === 2;
          const separation = view.getFloat32(0, true);
          window.dispatchEvent(
            new CustomEvent("galaxy:live-stats", {
              detail: {
                coreSeparation: merged ? 0 : separation,
                binary,
                merged,
              },
            }),
          );
        }
      } catch (error) {
        if (!disposed && simulation === target) {
          dispatchStatus(
            "error",
            `Core telemetry readback failed: ${
              error instanceof Error ? error.message : String(error)
            }`,
          );
        }
      } finally {
        slot.pending = false;
      }
    }).catch((error: unknown) => {
      slot.pending = false;
      if (!disposed && simulation === target) {
        dispatchStatus(
          "error",
          `Core telemetry mapping failed: ${
            error instanceof Error ? error.message : String(error)
          }`,
        );
      }
    });
  };

  const renderFrame = (_now: number, runPhysics: boolean) => {
    if (!backend || !simulation || validationError) return;
    configureCanvas();
    const activeSimulation = simulation;
    const telemetrySlot = runPhysics
      ? activeSimulation.telemetryReadbacks.find((slot) => !slot.pending)
      : undefined;
    if (telemetrySlot) {
      telemetrySlot.pending = true;
      telemetrySlot.sequence = ++activeSimulation.telemetrySequence;
    }
    const encoder = backend.device.createCommandEncoder({
      label: "Galaxy frame encoder",
    });
    let renderIndex = activeSimulation.readIndex;

    const advancePhysics = runPhysics && !physicsSubmissionPending;
    if (advancePhysics) {
      physicsSubmissionPending = true;
      const sourceIndex = activeSimulation.readIndex;
      const destinationIndex = (1 - sourceIndex) as PingPongIndex;
      const particleWorkgroups = Math.ceil(
        activeSimulation.initial.parameters.particleCount / WORKGROUP_SIZE,
      );

      // Drift every particle to the next synchronization point, then compact
      // only particles whose power-of-two timestep expires at this tick.
      const schedulePass = encoder.beginComputePass({
        label: "Galaxy adaptive drift and active schedule",
      });
      schedulePass.setPipeline(backend.driftPipeline);
      schedulePass.setBindGroup(0, activeSimulation.driftBindGroups[sourceIndex]);
      schedulePass.dispatchWorkgroups(particleWorkgroups);
      schedulePass.setPipeline(backend.scheduleBeginPipeline);
      schedulePass.setBindGroup(0, activeSimulation.scheduleBeginBindGroup);
      schedulePass.dispatchWorkgroups(1);
      schedulePass.setPipeline(backend.scheduleCollectPipeline);
      schedulePass.setBindGroup(
        0,
        activeSimulation.scheduleCollectBindGroups[destinationIndex],
      );
      schedulePass.dispatchWorkgroups(particleWorkgroups);
      schedulePass.setPipeline(backend.scheduleFinishPipeline);
      schedulePass.setBindGroup(0, activeSimulation.scheduleFinishBindGroup);
      schedulePass.dispatchWorkgroups(1);
      schedulePass.end();

      // Tree construction still sees the complete current state. The expensive
      // force traversal is dispatched only for the compact active target list.
      activeSimulation.solver.encode(
        encoder,
        destinationIndex,
      );
      const kickPass = encoder.beginComputePass({
        label: "Galaxy adaptive kick and merger pass",
      });
      kickPass.setPipeline(backend.kickPipeline);
      kickPass.setBindGroup(0, activeSimulation.kickBindGroups[destinationIndex]);
      kickPass.dispatchWorkgroupsIndirect(activeSimulation.indirectDispatchBuffer, 0);
      kickPass.setPipeline(backend.mergePipeline);
      kickPass.setBindGroup(0, activeSimulation.mergeBindGroups[sourceIndex]);
      kickPass.dispatchWorkgroups(1);
      kickPass.end();
      renderIndex = destinationIndex;
      activeSimulation.readIndex = renderIndex;
      activeSimulation.simulationTime +=
        activeSimulation.initial.parameters.timeStep;
      window.dispatchEvent(
        new CustomEvent("galaxy:live-stats", {
          detail: { simulationTime: activeSimulation.simulationTime },
        }),
      );
    }

    if (telemetrySlot) {
      encoder.copyBufferToBuffer(
        activeSimulation.telemetryBuffer,
        0,
        telemetrySlot.buffer,
        0,
        TELEMETRY_BYTES,
      );
    }

    updateRenderParameters(activeSimulation);
    const texture = backend.context.getCurrentTexture();
    const renderPass = encoder.beginRenderPass({
      label: "Galaxy WebGPU render pass",
      colorAttachments: [
        {
          view: texture.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(backend.renderPipeline);
    renderPass.setBindGroup(0, activeSimulation.renderBindGroups[renderIndex]);
    renderPass.draw(
      6,
      activeSimulation.renderCount,
    );
    renderPass.end();

    backend.device.queue.submit([encoder.finish()]);
    if (advancePhysics) {
      const target = activeSimulation;
      void backend.device.queue.onSubmittedWorkDone().then(() => {
        if (simulation === target && !disposed) {
          physicsSubmissionPending = false;
        }
      }).catch((error: unknown) => {
        physicsSubmissionPending = false;
        if (!disposed && simulation === target) {
          dispatchStatus(
            "error",
            `WebGPU submission failed: ${
              error instanceof Error ? error.message : String(error)
            }`,
          );
        }
      });
    }
    if (telemetrySlot) {
      scheduleTelemetryReadback(activeSimulation, telemetrySlot);
    }
  };

  const setFizzle = (value: number) => {
    currentFizzle = clamp(value, 0, 1);
  };

  const animateFizzle = (target: number, onComplete?: () => void) => {
    window.cancelAnimationFrame(fizzleAnimation);
    const start = currentFizzle;
    const startedAt = performance.now();
    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    const duration = reduceMotion ? 1 : 360;
    const frame = (now: number) => {
      const progress = Math.min((now - startedAt) / duration, 1);
      const eased = progress * progress * (3 - 2 * progress);
      setFizzle(start + (target - start) * eased);
      if (progress < 1) {
        fizzleAnimation = window.requestAnimationFrame(frame);
      } else {
        onComplete?.();
      }
    };
    fizzleAnimation = window.requestAnimationFrame(frame);
  };

  const rebuildWithFizzle = () => {
    // Generation tokens discard stale async builds if a user moves a control
    // again before the previous rebuild finishes.
    const generation = ++buildGeneration;
    physicsSubmissionPending = false;
    animateFizzle(0, () => {
      void buildGalaxy(generation)
        .then(() => {
          if (disposed || generation !== buildGeneration) return;
          setFizzle(0);
          if (themeActive) animateFizzle(1);
        })
        .catch(() => undefined);
    });
  };

  const tick = (now: number) => {
    const elapsed = now - previousFrame;
    const frameInterval = 1_000 / TARGET_FPS;
    if (elapsed >= frameInterval) {
      previousFrame = now - (elapsed % frameInterval);
      if ((themeActive || currentFizzle > 0.001) && backend && simulation) {
        window.appStats?.begin();
        renderFrame(now, themeActive && !isPaused);
        window.appStats?.end();
      }
    }
    animationFrame = window.requestAnimationFrame(tick);
  };
  animationFrame = window.requestAnimationFrame(tick);

  const onResize = () => configureCanvas();
  const onPause = (event: Event) => {
    isPaused = Boolean(
      (event as CustomEvent<{ paused: boolean }>).detail.paused,
    );
  };
  const onVisibility = (event: Event) => {
    const visible = Boolean(
      (event as CustomEvent<{ visible: boolean }>).detail.visible,
    );
    themeActive = visible;
    if (visible) {
      isPaused = false;
      const generation = ++buildGeneration;
      void initializeBackend()
        .then(async () => {
          if (!simulation) await buildGalaxy(generation);
          if (disposed || generation !== buildGeneration) return;
          setFizzle(0);
          animateFizzle(1, () => dispatchFizzleComplete(true));
        })
        .catch(() => dispatchFizzleComplete(true));
    } else {
      animateFizzle(0, () => {
      if (simulation && backend) {
          const previousSimulation = simulation;
          simulation = null;
          physicsSubmissionPending = false;
          void backend.device.queue.onSubmittedWorkDone().then(() =>
            destroySimulation(previousSimulation),
          );
        }
        dispatchFizzleComplete(false);
      });
    }
  };
  const onGalaxyResize = (event: Event) => {
    applySanitizedSettings({
      textureWidth: (event as CustomEvent<{ textureWidth: number }>).detail
        .textureWidth,
      radius: currentGalaxyRadius,
      offset: currentOffset,
    });
    rebuildWithFizzle();
  };
  const onGalaxyRadius = (event: Event) => {
    applySanitizedSettings({
      textureWidth: currentTextureWidth,
      radius: (event as CustomEvent<{ radius: number }>).detail.radius,
      offset: currentOffset,
    });
    rebuildWithFizzle();
  };
  const onGalaxyOffset = (event: Event) => {
    applySanitizedSettings({
      textureWidth: currentTextureWidth,
      radius: currentGalaxyRadius,
      offset: (event as CustomEvent<{ offset: number }>).detail.offset,
    });
    rebuildWithFizzle();
  };
  const onGalaxySolver = (event: Event) => {
    const requested = (
      event as CustomEvent<{ solver: GalaxySolverKind }>
    ).detail.solver;
    if (!Object.hasOwn(GALAXY_SOLVERS, requested) || requested === currentSolverKind) return;
    currentSolverKind = requested;
    applySanitizedSettings({
      textureWidth: currentTextureWidth,
      radius: currentGalaxyRadius,
      offset: currentOffset,
    });
    rebuildWithFizzle();
  };

  window.addEventListener("resize", onResize);
  window.addEventListener("galaxy:pause", onPause);
  window.addEventListener("galaxy:visibility", onVisibility);
  window.addEventListener("galaxy:resize", onGalaxyResize);
  window.addEventListener("galaxy:radius", onGalaxyRadius);
  window.addEventListener("galaxy:offset", onGalaxyOffset);
  window.addEventListener("galaxy:solver", onGalaxySolver);

  return () => {
    disposed = true;
    buildGeneration++;
    window.removeEventListener("resize", onResize);
    window.removeEventListener("galaxy:pause", onPause);
    window.removeEventListener("galaxy:visibility", onVisibility);
    window.removeEventListener("galaxy:resize", onGalaxyResize);
    window.removeEventListener("galaxy:radius", onGalaxyRadius);
    window.removeEventListener("galaxy:offset", onGalaxyOffset);
    window.removeEventListener("galaxy:solver", onGalaxySolver);
    window.cancelAnimationFrame(animationFrame);
    window.cancelAnimationFrame(fizzleAnimation);
    destroySimulation(simulation);
    simulation = null;
    backend?.context.unconfigure();
    canvasConfigured = false;
    backend?.device.destroy();
    backend = null;
  };
};
