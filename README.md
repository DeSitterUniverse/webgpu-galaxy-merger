# WebGPU Galaxy Merger

A real-time collisionless galaxy merger running entirely in the browser. Each galaxy contains a live stellar disk, dark-matter halo, and compact core. The model follows a bound, inclined encounter through tidal deformation, core coalescence, and remnant formation.

## Simulation model

- WebGPU compute and rendering pipelines written in WGSL
- Second-order leapfrog integration with half-step velocities
- Exact workgroup-tiled all-pairs gravity up to 160x160 particles
- Compact occupied-node Barnes-Hut gravity up to 256x256 particles
- Equal-mass stellar and halo super-particles with fixed component masses
- Separate, resolution-aware disk, halo, and core softening
- Truncated exponential stellar disks with smooth outer tapers
- Live, smoothly truncated dark-matter halos
- Warm disk velocity dispersions and composite rotation curves
- Seeded and independently recentered component initialization
- Swept, bound-state core coalescence with mass and momentum conservation

The live model supports disk-halo coupling, halo wakes, dynamical friction, tidal bridges, extended tails, and collective energy transfer during the encounter. Halo particles remain invisible in the luminous rendering layer.

## Gravity solvers

The all-pairs solver is the exact reference implementation. It tiles particle data through workgroup memory and evaluates every pair with symmetric component softening.

The Barnes-Hut solver constructs a compact seven-level octree on the GPU. Lock-free child elections create only occupied paths, terminal buckets use exact direct sums, and a center-of-mass-offset opening criterion controls monopole acceptance. Compact cores use a tighter opening angle to protect their orbital forces.

Measured on an AMD Radeon RX 6700 XT:

| Resolution | Bodies | Barnes-Hut physics steps/s |
| ---: | ---: | ---: |
| 160x160 | 25,600 | 60.0 |
| 192x192 | 36,864 | 46.2 |
| 256x256 | 65,536 | 27.2 |

The 160x160 result reaches the demo's 60-step frame-loop ceiling.

Force comparison against exact all-pairs gravity:

| Resolution | Normalized RMS error | p95 relative error | Momentum residual |
| ---: | ---: | ---: | ---: |
| 56x56 | 0.89% | 1.71% | 0.046% |
| 160x160 | 0.64% | 1.12% | 0.018% |

## Run locally

WebGPU must be enabled in a compatible browser.

```sh
npm install
npm run dev
```

Open `http://localhost:4321`.

## Validation

```sh
npm run typecheck
npm test
npm run test:galaxy
npm run build
```

The longer galaxy suite checks isolated stability and conservation against a CPU reference model. GPU force comparison is available only through the opt-in `npm run dev:galaxy-test` environment.

## Current limits

This is a low-particle collisionless model. It does not include gas, hydrodynamics, star formation, feedback, or a separate bulge component. The live halo improves collective behavior but is not intended as a converged research-scale calculation.
