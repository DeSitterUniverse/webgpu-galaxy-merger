# WebGPU Galaxy Merger

A real-time collisionless galaxy merger running entirely in the browser. Each galaxy contains a live stellar disk, dark-matter halo, and compact core. The model follows a bound, inclined encounter through tidal deformation, core coalescence, and remnant formation.

## Simulation model

- WebGPU compute and rendering pipelines written in WGSL
- Hierarchical leapfrog integration with per-particle power-of-two block timesteps
- Exact workgroup-tiled all-pairs gravity up to 184x184 particles
- Compact occupied-node Barnes-Hut gravity up to 1024x1024 particles
- Equal-mass super-particles within each component, with fixed component masses
- Separate, resolution-aware disk, halo, and core softening
- Truncated exponential stellar disks with smooth outer tapers
- Live, smoothly truncated Plummer-like dark-matter halos
- Toomre-Q disk dispersions, epicyclic anisotropy, asymmetric drift, and composite rotation curves
- Seeded and independently recentered component initialization
- Swept, bound-state core coalescence with mass and momentum conservation

The live model supports disk-halo coupling, halo wakes, dynamical friction, tidal bridges, extended tails, and collective energy transfer during the encounter. Halo particles remain invisible in the luminous rendering layer.

## Gravity solvers

The all-pairs solver is the exact reference implementation. It tiles particle data through workgroup memory and evaluates every pair with symmetric component softening.

The Barnes-Hut solver constructs a compact seven-to-ten-level octree on the GPU. Lock-free child elections create only occupied paths, every opened terminal bucket uses an exact softened direct sum, and a center-of-mass-offset opening criterion controls distant monopole acceptance. Compact cores use a tighter opening angle to protect their orbital forces. The depth-first traversal stack is sized from the octree's exact worst-case pending-node bound, and its otherwise-unused acceleration lane reports any overflow to the isolated GPU test environment.

Every particle drifts on the base clock, while force traversal and kicks run only for particles whose individual timestep bin is active. Timestep intervals are quantized from 1 to 256 base ticks using the local acceleration and component softening. Faster bins can be entered immediately; slower bins are entered only at synchronized power-of-two boundaries. Rendering is sampled independently from the simulation state, so a million live bodies do not require a million visible instances.

## GPU architecture

Particle position, mass, velocity, softening, component metadata, acceleration, and timestep state remain in storage buffers for the lifetime of a run. Physics uses two particle-state buffers for drift output and frame-to-frame ownership, while acceleration and timestep buffers persist across swaps. The renderer reads a bounded luminous index sample from the same current state, keeping halo particles live in gravity without paying to draw them.

The frame command stream is assembled as drift, active-list compaction, tree maintenance, active force calculation, kick, swept core coalescence, and rendering. The scheduler evaluates the local acceleration criterion, maps each particle to a synchronized power-of-two interval, atomically compacts due particle indices, and writes an indirect dispatch command. Barnes-Hut traversal and kick shaders therefore launch dense workgroups containing only active particles.

The Barnes-Hut tree uses flat structure-of-arrays GPU buffers for node moments, eight child indices, particle paths, and terminal particle links. A lightweight pass compares every particle's current 30-bit terminal-cell coordinate with its recorded leaf. Any crossing or bounds escape writes a rebuild flag and produces conditional indirect dispatches, reconstructing topology before the same tick's force calculation. Frames without crossings retain their topology and refit all node moments from current positions. This removes fixed-age topology caching without forcing a full rebuild every frame.

Above 512x512, ten spatial levels provide 1024 cells per axis inside the dynamic root. Terminal cells always use exact softened particle interactions when opened. There is no occupancy-triggered monopole replacement. Distant cells use a center-of-mass-offset opening test, with a tighter threshold for compact cores.

CPU initialization writes directly into packed typed arrays, uploads once, and releases duplicate host arrays. Before allocating a large tree, the engine checks the adapter's storage-binding and maximum-buffer limits against the calculated layout. GPU submissions are bounded with `queue.onSubmittedWorkDone()` so animation cannot accumulate an unbounded command backlog.

Measured on an AMD Radeon RX 6700 XT:

| Resolution | Bodies | Barnes-Hut base ticks/s |
| ---: | ---: | ---: |
| 512x512 | 262,144 | 21.5 |
| 768x768 | 589,824 | 12.6 |
| 1024x1024 | 1,048,576 | 6.1 |

These are completed base integration ticks, measured after initialization. Individual force evaluations occur according to each particle's active timestep bin. Million-body mode requires a WebGPU adapter with sufficiently large storage-buffer limits and several hundred MiB of available GPU memory; the engine checks those limits before allocation.

Force comparison against exact all-pairs gravity:

| Resolution | Normalized RMS error | p95 relative error | Momentum residual |
| ---: | ---: | ---: | ---: |
| 56x56 | 0.89% | 1.71% | 0.046% |
| 160x160 | 0.64% | 1.12% | 0.018% |
| 520x520, depth 10 | 0.76% | 1.29% | 0.012% |

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

The longer galaxy suite checks isolated stability, conservation, and live-halo disk heating at multiple particle resolutions against a Float64 CPU reference model. GPU-only validation is available through the opt-in `npm run dev:galaxy-test` environment, keeping readbacks and exact reference work out of normal development and production builds.

The GPU test environment provides two browser-driven checks:

- `/?solverAccuracy=barnes-hut&accuracyWidth=520&accuracySamples=32` compares a depth-ten force evaluation with exact CPU summation and reports traversal overflow.
- `/?gpuEvolution=true&steps=1024` advances fixed and adaptive all-pairs and Barnes-Hut modes through the production WGSL integration passes. The default checkpoint aligns every power-of-two timestep bin. An all-particle diagnostic force pass also supports arbitrary checkpoints, reconstructing synchronized velocities from both the last-active and current accelerations. Conservation thresholds use full momentum, angular-momentum, and center-of-mass vector differences alongside energy and CPU trajectory comparisons.
