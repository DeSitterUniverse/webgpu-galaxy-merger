export const GALAXY_SEED = 0x5eedc0de;

export const META_CORE = 1 << 0;
export const META_GALAXY_TWO = 1 << 1;
export const META_HALO = 1 << 2;

// CPU-side initialization owns morphology, masses, and the initial encounter.
// Once packed into `state`, WebGPU advances it without CPU-side force work.

export type GalaxySettings = {
  textureWidth: number;
  radius: number;
  offset: number;
  seed?: number;
};

export type GalaxyParameters = {
  particleCount: number;
  particlesPerGalaxy: number;
  diskParticlesPerGalaxy: number;
  haloParticlesPerGalaxy: number;
  seed: number;
  radius: number;
  halfSeparation: number;
  orthogonalHalfSeparation: number;
  centerSeparation: number;
  gravity: number;
  timeStep: number;
  innerOrbitalPeriod: number;
  coreMass: number;
  diskMass: number;
  haloMass: number;
  haloScale: number;
  coreSoftening: number;
  diskSoftening: number;
  haloSoftening: number;
  captureSofteningFactor: number;
  core1Index: number;
  core2Index: number;
};

export type GalaxyInitialState = {
  settings: Required<GalaxySettings>;
  parameters: GalaxyParameters;
  state: Float32Array;
  metadata: Uint32Array;
  visuals: Float32Array;
};

type Vec3 = [number, number, number];
type Particle = {
  position: Vec3;
  velocity: Vec3;
  mass: number;
  softening: number;
  metadata: number;
  color: Vec3;
  size: number;
};

// Simulation units are intentionally normalized: one galaxy has total mass 5,
// which keeps the f32 GPU state well-conditioned while preserving mass ratios.
const TIME_STEP = 0.08;
const GRAVITY = 80;
const CORE_MASS = 0.05;
const DISK_MASS = 0.95;
const HALO_MASS = 4;
const HALO_SCALE = 28;
const HALO_MAX_RADIUS = 80;
const REFERENCE_TEXTURE_WIDTH = 72;
export const MIN_LIVE_HALO_TEXTURE_WIDTH = 56;
const CORE_SOFTENING = 0.9;
const DISK_SOFTENING = 1.4;
const HALO_SOFTENING = 2.2;
const CAPTURE_SOFTENING_FACTOR = 2;
const SYSTEM_Z_OFFSET = 10;
const DISK_CLEARANCE_FACTOR = 2.1;
const MIN_CENTER_OFFSET = 10;
const HALO_PARTICLE_FRACTION = 0.6;
const TANGENTIAL_ORBIT_FRACTION = 0.45;
const INWARD_ORBIT_FRACTION = 0.35;

// [tilt around X, roll around Z]. Rotate positions and velocities together so
// each disk keeps its internal angular momentum after being placed in 3D.
const DISK_ORIENTATIONS: ReadonlyArray<readonly [number, number]> = [
  [20 * (Math.PI / 180), -12 * (Math.PI / 180)],
  [-12 * (Math.PI / 180), 32 * (Math.PI / 180)],
];

const GALAXY_COLORS: ReadonlyArray<ReadonlyArray<readonly [number, number, number]>> = [
  [
    [251 / 255, 146 / 255, 60 / 255],
    [250 / 255, 204 / 255, 21 / 255],
    [239 / 255, 68 / 255, 68 / 255],
  ],
  [
    [56 / 255, 189 / 255, 248 / 255],
    [129 / 255, 140 / 255, 248 / 255],
    [232 / 255, 121 / 255, 249 / 255],
  ],
];

const clamp = (value: number, minimum: number, maximum: number) =>
  Math.min(maximum, Math.max(minimum, value));

const finiteOr = (value: number, fallback: number) =>
  Number.isFinite(value) ? value : fallback;

const add = (left: Vec3, right: Vec3): Vec3 => [
  left[0] + right[0],
  left[1] + right[1],
  left[2] + right[2],
];

const scale = (value: Vec3, factor: number): Vec3 => [
  value[0] * factor,
  value[1] * factor,
  value[2] * factor,
];

const magnitude = (value: Vec3) => Math.hypot(...value);

const normalize = (value: Vec3): Vec3 => {
  const length = Math.max(magnitude(value), Number.EPSILON);
  return scale(value, 1 / length);
};

const rotateDiskVector = (value: Vec3, galaxy: number): Vec3 => {
  const [tiltX, tiltZ] = DISK_ORIENTATIONS[galaxy]!;
  const cosineX = Math.cos(tiltX);
  const sineX = Math.sin(tiltX);
  const rotatedY = value[1] * cosineX - value[2] * sineX;
  const rotatedZ = value[1] * sineX + value[2] * cosineX;
  const cosineZ = Math.cos(tiltZ);
  const sineZ = Math.sin(tiltZ);
  return [
    value[0] * cosineZ - rotatedY * sineZ,
    value[0] * sineZ + rotatedY * cosineZ,
    rotatedZ,
  ];
};

export const sanitizeGalaxySettings = (
  settings: GalaxySettings,
): Required<GalaxySettings> => {
  const requestedWidth =
    Math.round(finiteOr(settings.textureWidth, 72) / 8) * 8;
  return {
    textureWidth: clamp(
      requestedWidth,
      MIN_LIVE_HALO_TEXTURE_WIDTH,
      320,
    ),
    radius: clamp(finiteOr(settings.radius, 35), 15, 80),
    offset: clamp(finiteOr(settings.offset, 25), MIN_CENTER_OFFSET, 60),
    seed: finiteOr(settings.seed ?? GALAXY_SEED, GALAXY_SEED) >>> 0,
  };
};

// Stateless integer hashing makes every particle reproducible from
// (seed, galaxy, ordinal, channel), even when another UI control changes.
const randomSample = (
  seed: number,
  galaxy: number,
  ordinal: number,
  channel: number,
) => {
  let value = (
    seed ^
    Math.imul(galaxy + 1, 0x9e3779b9) ^
    Math.imul(ordinal + 1, 0x85ebca6b) ^
    Math.imul(channel + 1, 0xc2b2ae35)
  ) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d);
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b);
  value = (value ^ (value >>> 16)) >>> 0;
  return value / 4_294_967_296;
};

const gaussianSample = (
  seed: number,
  galaxy: number,
  ordinal: number,
  channel: number,
) => {
  const first = Math.max(
    randomSample(seed, galaxy, ordinal, channel),
    Number.EPSILON,
  );
  const second = randomSample(seed, galaxy, ordinal, channel + 1);
  return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
};

export const symmetricSofteningSquared = (left: number, right: number) =>
  0.5 * (left * left + right * right);

export const deriveGalaxyParameters = (
  requestedSettings: GalaxySettings,
): { settings: Required<GalaxySettings>; parameters: GalaxyParameters } => {
  const settings = sanitizeGalaxySettings(requestedSettings);
  const particleCount = settings.textureWidth ** 2;
  const particlesPerGalaxy = particleCount / 2;
  const liveParticlesPerGalaxy = particlesPerGalaxy - 1;
  const haloParticlesPerGalaxy = Math.floor(
    liveParticlesPerGalaxy * HALO_PARTICLE_FRACTION,
  );
  const diskParticlesPerGalaxy =
    liveParticlesPerGalaxy - haloParticlesPerGalaxy;
  const referenceLiveParticlesPerGalaxy =
    REFERENCE_TEXTURE_WIDTH ** 2 / 2 - 1;
  const referenceHaloParticlesPerGalaxy = Math.floor(
    referenceLiveParticlesPerGalaxy * HALO_PARTICLE_FRACTION,
  );
  const referenceDiskParticlesPerGalaxy =
    referenceLiveParticlesPerGalaxy - referenceHaloParticlesPerGalaxy;

  // Collisionless N-body softening follows the component sampling density.
  // N^(-1/3) exposes finer forces at higher counts without changing total mass.
  const diskResolutionScale = Math.cbrt(
    referenceDiskParticlesPerGalaxy / diskParticlesPerGalaxy,
  );
  const haloResolutionScale = Math.cbrt(
    referenceHaloParticlesPerGalaxy / haloParticlesPerGalaxy,
  );
  // Radius determines the orthogonal clearance. The offset control remains a
  // literal X displacement and never silently changes another control value.
  const minimumCenterSeparation = settings.radius * DISK_CLEARANCE_FACTOR;
  const minimumXSeparation = MIN_CENTER_OFFSET * 2;
  const orthogonalHalfSeparation = Math.max(
    SYSTEM_Z_OFFSET,
    0.5 * Math.sqrt(
      Math.max(
        minimumCenterSeparation * minimumCenterSeparation -
          minimumXSeparation * minimumXSeparation,
        0,
      ),
    ),
  );
  const centerSeparation = Math.hypot(
    settings.offset * 2,
    orthogonalHalfSeparation * 2,
  );
  return {
    settings,
    parameters: {
      particleCount,
      particlesPerGalaxy,
      diskParticlesPerGalaxy,
      haloParticlesPerGalaxy,
      seed: settings.seed,
      radius: settings.radius,
      halfSeparation: settings.offset,
      orthogonalHalfSeparation,
      centerSeparation,
      gravity: GRAVITY,
      timeStep: TIME_STEP,
      innerOrbitalPeriod: Number.POSITIVE_INFINITY,
      coreMass: CORE_MASS,
      diskMass: DISK_MASS,
      haloMass: HALO_MASS,
      haloScale: HALO_SCALE,
      coreSoftening: CORE_SOFTENING,
      diskSoftening: DISK_SOFTENING * diskResolutionScale,
      haloSoftening: HALO_SOFTENING * haloResolutionScale,
      captureSofteningFactor: CAPTURE_SOFTENING_FACTOR,
      core1Index: 0,
      core2Index: particlesPerGalaxy,
    },
  };
};

export const sweptClosestApproach = (
  startRelative: readonly [number, number, number],
  endRelative: readonly [number, number, number],
) => {
  // Test the full segment travelled this step, not only its endpoints. This
  // prevents fast cores from crossing the merge radius without being noticed.
  const motion: Vec3 = [
    endRelative[0] - startRelative[0],
    endRelative[1] - startRelative[1],
    endRelative[2] - startRelative[2],
  ];
  const denominator = motion[0] ** 2 + motion[1] ** 2 + motion[2] ** 2;
  const fraction =
    denominator > Number.EPSILON
      ? clamp(
          -(
            startRelative[0] * motion[0] +
            startRelative[1] * motion[1] +
            startRelative[2] * motion[2]
          ) / denominator,
          0,
          1,
        )
      : 0;
  const separation: Vec3 = [
    startRelative[0] + fraction * motion[0],
    startRelative[1] + fraction * motion[1],
    startRelative[2] + fraction * motion[2],
  ];
  return { fraction, distance: magnitude(separation) };
};

export const CORE_PHASE_SEPARATE = 0 as const;
export const CORE_PHASE_BINARY = 1 as const;
export const CORE_PHASE_MERGED = 2 as const;

export type CoreBinaryPhase =
  | typeof CORE_PHASE_SEPARATE
  | typeof CORE_PHASE_BINARY
  | typeof CORE_PHASE_MERGED;

export type CoreBinaryState = {
  phase: CoreBinaryPhase;
  boundElapsed: number;
  mergeDelay: number;
};

export const advanceCoreBinaryState = ({
  state,
  startRelative,
  endRelative,
  relativeVelocity,
  timeStep,
  gravity,
  totalMass,
  pairSofteningSquared,
  captureSofteningFactor,
}: {
  state: CoreBinaryState;
  startRelative: Vec3;
  endRelative: Vec3;
  relativeVelocity: Vec3;
  timeStep: number;
  gravity: number;
  totalMass: number;
  pairSofteningSquared: number;
  captureSofteningFactor: number;
}) => {
  if (state.phase === CORE_PHASE_MERGED) {
    return { state, shouldMerge: true };
  }

  const closest = sweptClosestApproach(startRelative, endRelative);
  const captureRadius =
    captureSofteningFactor * Math.sqrt(pairSofteningSquared);
  const softenedSeparation = Math.sqrt(
    endRelative[0] ** 2 +
      endRelative[1] ** 2 +
      endRelative[2] ** 2 +
      pairSofteningSquared,
  );
  const speedSquared =
    relativeVelocity[0] ** 2 +
    relativeVelocity[1] ** 2 +
    relativeVelocity[2] ** 2;
  const specificEnergy =
    0.5 * speedSquared -
    (gravity * totalMass) / Math.max(softenedSeparation, Number.EPSILON);
  const isBound = specificEnergy < 0;
  const crossesCaptureRadius = closest.distance <= captureRadius;

  if (state.phase === CORE_PHASE_SEPARATE) {
    if (!isBound || !crossesCaptureRadius) {
      return {
        state: {
          phase: CORE_PHASE_SEPARATE,
          boundElapsed: 0,
          mergeDelay: 0,
        },
        shouldMerge: false,
      };
    }

    const softenedCaptureRadiusCubed =
      (captureRadius * captureRadius + pairSofteningSquared) ** 1.5;
    const mergeDelay =
      2 *
      Math.PI *
      Math.sqrt(
        softenedCaptureRadiusCubed /
          Math.max(gravity * totalMass, Number.EPSILON),
      );
    return {
      state: {
        phase: CORE_PHASE_BINARY,
        boundElapsed: timeStep,
        mergeDelay,
      },
      shouldMerge: false,
    };
  }

  if (!isBound) {
    return {
      state: {
        phase: CORE_PHASE_SEPARATE,
        boundElapsed: 0,
        mergeDelay: 0,
      },
      shouldMerge: false,
    };
  }

  const boundElapsed = state.boundElapsed + timeStep;
  const shouldMerge =
    boundElapsed >= state.mergeDelay && crossesCaptureRadius;
  return {
    state: {
      phase: shouldMerge ? CORE_PHASE_MERGED : CORE_PHASE_BINARY,
      boundElapsed,
      mergeDelay: state.mergeDelay,
    },
    shouldMerge,
  };
};

type EquilibriumModel = {
  radii: Float64Array;
  diskCumulative: Float64Array;
  haloCumulative: Float64Array;
  diskAcceleration: Float64Array;
  haloDispersionSquared: Float64Array;
  escapeSpeed: Float64Array;
  innerOrbitalPeriod: number;
};

const RADIAL_TABLE_SIZE = 257;
const DISK_RING_COUNT = 160;
const DISK_AZIMUTH_SAMPLES = 64;
const equilibriumModelCache = new Map<string, EquilibriumModel>();

const cosineTaper = (radius: number, start: number, end: number) => {
  if (radius <= start) return 1;
  if (radius >= end) return 0;
  return 0.5 * (1 + Math.cos(Math.PI * (radius - start) / (end - start)));
};

const interpolateTable = (
  radii: Float64Array,
  values: Float64Array,
  radius: number,
) => {
  const coordinate = clamp(radius / HALO_MAX_RADIUS, 0, 1) * (radii.length - 1);
  const lower = Math.floor(coordinate);
  const upper = Math.min(lower + 1, radii.length - 1);
  const fraction = coordinate - lower;
  return values[lower]! * (1 - fraction) + values[upper]! * fraction;
};

const invertCumulative = (
  radii: Float64Array,
  cumulative: Float64Array,
  sample: number,
) => {
  let lower = 0;
  let upper = cumulative.length - 1;
  while (upper - lower > 1) {
    const middle = (lower + upper) >>> 1;
    if (cumulative[middle]! < sample) lower = middle;
    else upper = middle;
  }
  const range = Math.max(cumulative[upper]! - cumulative[lower]!, Number.EPSILON);
  const fraction = (sample - cumulative[lower]!) / range;
  return radii[lower]! + fraction * (radii[upper]! - radii[lower]!);
};

const buildEquilibriumModel = (parameters: GalaxyParameters) => {
  const cacheKey = [
    parameters.radius,
    parameters.diskSoftening.toFixed(8),
    parameters.haloSoftening.toFixed(8),
  ].join(":");
  const cached = equilibriumModelCache.get(cacheKey);
  if (cached) return cached;

  const radii = new Float64Array(RADIAL_TABLE_SIZE);
  const diskDensity = new Float64Array(RADIAL_TABLE_SIZE);
  const haloDensity = new Float64Array(RADIAL_TABLE_SIZE);
  const diskCumulative = new Float64Array(RADIAL_TABLE_SIZE);
  const haloCumulative = new Float64Array(RADIAL_TABLE_SIZE);
  const spacing = HALO_MAX_RADIUS / (RADIAL_TABLE_SIZE - 1);
  const diskScale = parameters.radius / 3.2;
  const diskMinimum = Math.max(1.5, parameters.diskSoftening * 0.9);
  for (let index = 0; index < RADIAL_TABLE_SIZE; index++) {
    const radius = index * spacing;
    radii[index] = radius;
    const diskTaper = cosineTaper(
      radius,
      parameters.radius * 0.82,
      parameters.radius,
    );
    diskDensity[index] = radius >= diskMinimum
      ? radius * Math.exp(-radius / diskScale) * diskTaper
      : 0;
    const haloTaper = cosineTaper(
      radius,
      HALO_MAX_RADIUS * 0.85,
      HALO_MAX_RADIUS,
    );
    haloDensity[index] =
      (1 + radius * radius / (HALO_SCALE * HALO_SCALE)) ** -2.5 *
      haloTaper;
    if (index > 0) {
      diskCumulative[index] = diskCumulative[index - 1]! +
        0.5 * (diskDensity[index - 1]! + diskDensity[index]!) * spacing;
      const previousHaloWeight = radii[index - 1]! ** 2 * haloDensity[index - 1]!;
      const haloWeight = radius * radius * haloDensity[index]!;
      haloCumulative[index] = haloCumulative[index - 1]! +
        0.5 * (previousHaloWeight + haloWeight) * spacing;
    }
  }
  const diskNormalization = diskCumulative[RADIAL_TABLE_SIZE - 1]!;
  const haloNormalization = haloCumulative[RADIAL_TABLE_SIZE - 1]!;
  for (let index = 0; index < RADIAL_TABLE_SIZE; index++) {
    diskCumulative[index] = diskCumulative[index]! / diskNormalization;
    haloCumulative[index] = haloCumulative[index]! / haloNormalization;
  }

  // Numerically integrate the actual tapered, flattened disk in its midplane.
  // The same radial density above drives both sampling and this force table.
  const diskAcceleration = new Float64Array(RADIAL_TABLE_SIZE);
  const ringWeights = new Float64Array(DISK_RING_COUNT);
  const ringRadii = new Float64Array(DISK_RING_COUNT);
  let ringNormalization = 0;
  for (let ring = 0; ring < DISK_RING_COUNT; ring++) {
    const radius = parameters.radius * (ring + 0.5) / DISK_RING_COUNT;
    const taper = cosineTaper(
      radius,
      parameters.radius * 0.82,
      parameters.radius,
    );
    const weight = radius >= diskMinimum
      ? radius * Math.exp(-radius / diskScale) * taper
      : 0;
    ringRadii[ring] = radius;
    ringWeights[ring] = weight;
    ringNormalization += weight;
  }
  for (let target = 1; target < RADIAL_TABLE_SIZE; target++) {
    const radius = radii[target]!;
    let acceleration = 0;
    for (let ring = 0; ring < DISK_RING_COUNT; ring++) {
      const ringMass = DISK_MASS * ringWeights[ring]! / ringNormalization;
      if (ringMass <= 0) continue;
      const ringRadius = ringRadii[ring]!;
      for (let azimuth = 0; azimuth < DISK_AZIMUTH_SAMPLES; azimuth++) {
        const cosine = Math.cos(2 * Math.PI * (azimuth + 0.5) / DISK_AZIMUTH_SAMPLES);
        const radialDisplacement = radius - ringRadius * cosine;
        const distanceSquared =
          radius * radius + ringRadius * ringRadius -
          2 * radius * ringRadius * cosine +
          parameters.diskSoftening * parameters.diskSoftening;
        acceleration += GRAVITY * ringMass * radialDisplacement /
          (DISK_AZIMUTH_SAMPLES * distanceSquared ** 1.5);
      }
    }
    diskAcceleration[target] = Math.max(0, acceleration);
  }

  // The halo uses the spherical monopole of the full composite potential.
  // Solving the isotropic Jeans equation against this acceleration makes its
  // velocity moments consistent with the truncated halo, disk, and core.
  const totalMonopoleAcceleration = new Float64Array(RADIAL_TABLE_SIZE);
  const relativePotential = new Float64Array(RADIAL_TABLE_SIZE);
  const haloDispersionSquared = new Float64Array(RADIAL_TABLE_SIZE);
  const escapeSpeed = new Float64Array(RADIAL_TABLE_SIZE);
  for (let index = 1; index < RADIAL_TABLE_SIZE; index++) {
    const radius = radii[index]!;
    const corePairSoftening = symmetricSofteningSquared(
      parameters.haloSoftening,
      parameters.coreSoftening,
    );
    const diskPairSoftening = symmetricSofteningSquared(
      parameters.haloSoftening,
      parameters.diskSoftening,
    );
    const coreAcceleration = GRAVITY * CORE_MASS * radius /
      (radius * radius + corePairSoftening) ** 1.5;
    const diskAccelerationMonopole = GRAVITY * DISK_MASS *
      diskCumulative[index]! * radius /
      (radius * radius + diskPairSoftening) ** 1.5;
    const haloAcceleration = GRAVITY * HALO_MASS *
      haloCumulative[index]! * radius /
      (radius * radius + parameters.haloSoftening ** 2) ** 1.5;
    totalMonopoleAcceleration[index] =
      coreAcceleration + diskAccelerationMonopole + haloAcceleration;
  }
  relativePotential[RADIAL_TABLE_SIZE - 1] =
    GRAVITY * (CORE_MASS + DISK_MASS + HALO_MASS) / HALO_MAX_RADIUS;
  let pressureIntegral = 0;
  for (let index = RADIAL_TABLE_SIZE - 2; index >= 0; index--) {
    relativePotential[index] = relativePotential[index + 1]! +
      0.5 * (
        totalMonopoleAcceleration[index + 1]! +
        totalMonopoleAcceleration[index]!
      ) * spacing;
    pressureIntegral += 0.5 * (
      haloDensity[index + 1]! * totalMonopoleAcceleration[index + 1]! +
      haloDensity[index]! * totalMonopoleAcceleration[index]!
    ) * spacing;
    haloDispersionSquared[index] = haloDensity[index]! > 1e-12
      ? pressureIntegral / haloDensity[index]!
      : 0;
  }
  for (let index = 0; index < RADIAL_TABLE_SIZE; index++) {
    escapeSpeed[index] = Math.sqrt(2 * relativePotential[index]!);
  }

  const innerCoreAcceleration = GRAVITY * CORE_MASS * diskMinimum /
    (diskMinimum * diskMinimum + symmetricSofteningSquared(
      parameters.diskSoftening,
      parameters.coreSoftening,
    )) ** 1.5;
  const innerHaloAcceleration = GRAVITY * HALO_MASS *
    interpolateTable(radii, haloCumulative, diskMinimum) * diskMinimum /
    (diskMinimum * diskMinimum + symmetricSofteningSquared(
      parameters.diskSoftening,
      parameters.haloSoftening,
    )) ** 1.5;
  const innerAcceleration =
    interpolateTable(radii, diskAcceleration, diskMinimum) +
    innerCoreAcceleration + innerHaloAcceleration;
  const innerOrbitalPeriod = 2 * Math.PI * Math.sqrt(
    diskMinimum / Math.max(innerAcceleration, Number.EPSILON),
  );
  const model = {
    radii,
    diskCumulative,
    haloCumulative,
    diskAcceleration,
    haloDispersionSquared,
    escapeSpeed,
    innerOrbitalPeriod,
  };
  equilibriumModelCache.set(cacheKey, model);
  return model;
};

const createDiskParticle = (
  seed: number,
  galaxy: number,
  ordinal: number,
  parameters: GalaxyParameters,
  model: EquilibriumModel,
): Particle => {
  const radius = invertCumulative(
    model.radii,
    model.diskCumulative,
    randomSample(seed, galaxy, ordinal, 20),
  );
  const angle = randomSample(seed, galaxy, ordinal, 1) * Math.PI * 2;
  const verticalCoordinate = clamp(
    2 * randomSample(seed, galaxy, ordinal, 2) - 1,
    -0.995,
    0.995,
  );
  const palette = GALAXY_COLORS[galaxy]!;
  const color = palette[
    Math.floor(randomSample(seed, galaxy, ordinal, 10) * palette.length)
  ]!;
  return {
    position: [
      Math.cos(angle) * radius,
      parameters.radius * 0.035 * Math.atanh(verticalCoordinate),
      Math.sin(angle) * radius,
    ],
    velocity: [0, 0, 0],
    mass: DISK_MASS / parameters.diskParticlesPerGalaxy,
    softening: parameters.diskSoftening,
    metadata: galaxy === 1 ? META_GALAXY_TWO : 0,
    color: [color[0], color[1], color[2]],
    size: 2 + randomSample(seed, galaxy, ordinal, 11) * 5,
  };
};

const isotropicDirection = (
  seed: number,
  galaxy: number,
  ordinal: number,
  channel: number,
): Vec3 => {
  const cosine = 2 * randomSample(seed, galaxy, ordinal, channel) - 1;
  const angle =
    randomSample(seed, galaxy, ordinal, channel + 1) * Math.PI * 2;
  const sine = Math.sqrt(Math.max(0, 1 - cosine * cosine));
  return [sine * Math.cos(angle), cosine, sine * Math.sin(angle)];
};

const createHaloParticle = (
  seed: number,
  galaxy: number,
  ordinal: number,
  parameters: GalaxyParameters,
  model: EquilibriumModel,
): Particle => {
  const radius = invertCumulative(
    model.radii,
    model.haloCumulative,
    randomSample(seed, galaxy, ordinal, 220),
  );
  const positionDirection = isotropicDirection(seed, galaxy, ordinal, 210);
  return {
    position: scale(positionDirection, radius),
    velocity: [0, 0, 0],
    mass: HALO_MASS / parameters.haloParticlesPerGalaxy,
    softening: parameters.haloSoftening,
    metadata: META_HALO | (galaxy === 1 ? META_GALAXY_TWO : 0),
    // Halo bodies remain gravitationally live but are not luminous tracers.
    color: [0, 0, 0],
    size: 0,
  };
};

const recenterComponentPositions = (particles: Particle[]) => {
  let totalMass = 0;
  let weightedPosition: Vec3 = [0, 0, 0];
  for (const particle of particles) {
    totalMass += particle.mass;
    weightedPosition = add(
      weightedPosition,
      scale(particle.position, particle.mass),
    );
  }
  const center = scale(weightedPosition, 1 / totalMass);
  for (const particle of particles) {
    particle.position = add(particle.position, scale(center, -1));
  }
};

const removeComponentBulkVelocity = (particles: Particle[]) => {
  let totalMass = 0;
  let momentum: Vec3 = [0, 0, 0];
  for (const particle of particles) {
    totalMass += particle.mass;
    momentum = add(momentum, scale(particle.velocity, particle.mass));
  }
  const bulkVelocity = scale(momentum, 1 / totalMass);
  for (const particle of particles) {
    particle.velocity = add(particle.velocity, scale(bulkVelocity, -1));
  }
};

const diskRadialAcceleration = (
  radius: number,
  parameters: GalaxyParameters,
  model: EquilibriumModel,
) => {
  const coreAcceleration = GRAVITY * CORE_MASS * radius /
    (radius * radius + symmetricSofteningSquared(
      parameters.diskSoftening,
      parameters.coreSoftening,
    )) ** 1.5;
  const haloAcceleration = GRAVITY * HALO_MASS *
    interpolateTable(model.radii, model.haloCumulative, radius) * radius /
    (radius * radius + symmetricSofteningSquared(
      parameters.diskSoftening,
      parameters.haloSoftening,
    )) ** 1.5;
  return coreAcceleration + haloAcceleration +
    interpolateTable(model.radii, model.diskAcceleration, radius);
};

const initializeDiskVelocities = (
  particles: Particle[],
  seed: number,
  galaxy: number,
  parameters: GalaxyParameters,
  model: EquilibriumModel,
) => {
  particles.forEach((particle, ordinal) => {
    const [x, , z] = particle.position;
    const radius = Math.max(Math.hypot(x, z), Number.EPSILON);
    const angle = Math.atan2(z, x);
    const circularSpeed = Math.sqrt(
      Math.max(0, diskRadialAcceleration(radius, parameters, model) * radius),
    );
    const radialDispersion =
      gaussianSample(seed, galaxy, ordinal, 4) * circularSpeed * 0.035;
    const tangentialDispersion =
      gaussianSample(seed, galaxy, ordinal, 6) * circularSpeed * 0.02;
    const verticalDispersion =
      gaussianSample(seed, galaxy, ordinal, 8) * circularSpeed * 0.025;
    const radial: Vec3 = [Math.cos(angle), 0, Math.sin(angle)];
    const tangent: Vec3 = [-Math.sin(angle), 0, Math.cos(angle)];
    particle.velocity = add(
      add(
        scale(radial, radialDispersion),
        scale(tangent, circularSpeed + tangentialDispersion),
      ),
      [0, verticalDispersion, 0],
    );
  });
};

const initializeHaloVelocities = (
  particles: Particle[],
  seed: number,
  galaxy: number,
  model: EquilibriumModel,
) => {
  particles.forEach((particle, ordinal) => {
    const radius = magnitude(particle.position);
    const dispersion = Math.sqrt(Math.max(
      0,
      interpolateTable(model.radii, model.haloDispersionSquared, radius),
    ));
    const velocity: Vec3 = [
      gaussianSample(seed, galaxy, ordinal, 300) * dispersion,
      gaussianSample(seed, galaxy, ordinal, 302) * dispersion,
      gaussianSample(seed, galaxy, ordinal, 304) * dispersion,
    ];
    const escapeSpeed = interpolateTable(model.radii, model.escapeSpeed, radius);
    const speed = magnitude(velocity);
    particle.velocity = speed > escapeSpeed * 0.95
      ? scale(velocity, escapeSpeed * 0.95 / speed)
      : velocity;
  });
};

export const createGalaxyInitialState = (
  requestedSettings: GalaxySettings,
): GalaxyInitialState => {
  const { settings, parameters } = deriveGalaxyParameters(requestedSettings);
  const galaxies: [Particle[], Particle[]] = [[], []];
  const model = buildEquilibriumModel(parameters);
  parameters.innerOrbitalPeriod = model.innerOrbitalPeriod;
  parameters.timeStep = Math.min(TIME_STEP, model.innerOrbitalPeriod / 48);

  // Particle zero in each half is the compact core, followed by disk then halo.
  // Keeping that ordering deterministic gives shaders stable core indices.
  for (let galaxy = 0; galaxy < 2; galaxy++) {
    const particles = galaxies[galaxy]!;
    const core: Particle = {
      position: [0, 0, 0],
      velocity: [0, 0, 0],
      mass: CORE_MASS,
      softening: parameters.coreSoftening,
      metadata: META_CORE | (galaxy === 1 ? META_GALAXY_TWO : 0),
      color: [1, 1, 1],
      size: 10,
    };
    const disk: Particle[] = [];
    const halo: Particle[] = [];
    for (
      let ordinal = 0;
      ordinal < parameters.diskParticlesPerGalaxy;
      ordinal++
    ) {
      disk.push(
        createDiskParticle(
          settings.seed,
          galaxy,
          ordinal,
          parameters,
          model,
        ),
      );
    }
    for (
      let ordinal = 0;
      ordinal < parameters.haloParticlesPerGalaxy;
      ordinal++
    ) {
      halo.push(
        createHaloParticle(
          settings.seed,
          galaxy,
          ordinal,
          parameters,
          model,
        ),
      );
    }

    // Center each independently before velocities are calculated. This keeps
    // finite-N halo noise from translating the disk away from its core.
    recenterComponentPositions(disk);
    recenterComponentPositions(halo);
    initializeDiskVelocities(
      disk,
      settings.seed,
      galaxy,
      parameters,
      model,
    );
    initializeHaloVelocities(halo, settings.seed, galaxy, model);
    removeComponentBulkVelocity(disk);
    removeComponentBulkVelocity(halo);
    for (const particle of disk) {
      particle.position = rotateDiskVector(particle.position, galaxy);
      particle.velocity = rotateDiskVector(particle.velocity, galaxy);
    }
    particles.push(core, ...disk, ...halo);
  }

  const centers: [Vec3, Vec3] = [
    [-settings.offset, 0, -parameters.orthogonalHalfSeparation],
    [settings.offset, 0, parameters.orthogonalHalfSeparation],
  ];
  const relativePosition = add(centers[1], scale(centers[0], -1));
  const separation = magnitude(relativePosition);
  const separationDirection = normalize(relativePosition);
  const tangentDirection = normalize([
    -separationDirection[2],
    0,
    separationDirection[0],
  ]);
  const totalGalaxyMass = CORE_MASS + DISK_MASS + HALO_MASS;

  // Start with a bound, mildly plunging two-body orbit. Equal and opposite
  // center velocities make total system momentum exactly zero by construction.
  const relativeCircularSpeed = Math.sqrt(
    (GRAVITY * totalGalaxyMass * 2) / separation,
  );
  const centerVelocityOne = add(
    scale(
      tangentDirection,
      0.5 * relativeCircularSpeed * TANGENTIAL_ORBIT_FRACTION,
    ),
    scale(
      separationDirection,
      0.5 * relativeCircularSpeed * INWARD_ORBIT_FRACTION,
    ),
  );
  const centerVelocities: [Vec3, Vec3] = [
    centerVelocityOne,
    scale(centerVelocityOne, -1),
  ];

  const particles = [...galaxies[0], ...galaxies[1]];
  for (let galaxy = 0; galaxy < 2; galaxy++) {
    for (const particle of galaxies[galaxy]!) {
      particle.position = add(particle.position, centers[galaxy]!);
      particle.velocity = add(particle.velocity, centerVelocities[galaxy]!);
    }
  }

  const state = new Float32Array(parameters.particleCount * 8);
  const metadata = new Uint32Array(parameters.particleCount);
  const visuals = new Float32Array(parameters.particleCount * 4);
  particles.forEach((particle, index) => {
    // Two vec4 values match the WGSL Particle layout exactly:
    // position+mass, then velocity+per-particle softening.
    const stateOffset = index * 8;
    state[stateOffset] = particle.position[0];
    state[stateOffset + 1] = particle.position[1];
    state[stateOffset + 2] = particle.position[2];
    state[stateOffset + 3] = particle.mass;
    state[stateOffset + 4] = particle.velocity[0];
    state[stateOffset + 5] = particle.velocity[1];
    state[stateOffset + 6] = particle.velocity[2];
    state[stateOffset + 7] = particle.softening;
    metadata[index] = particle.metadata;
    const visualOffset = index * 4;
    visuals[visualOffset] = particle.color[0];
    visuals[visualOffset + 1] = particle.color[1];
    visuals[visualOffset + 2] = particle.color[2];
    visuals[visualOffset + 3] = particle.size;
  });

  return { settings, parameters, state, metadata, visuals };
};
