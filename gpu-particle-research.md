# GPU-Accelerated Falling Sand / Particle Simulation in the Browser

## Research Document — April 2026

---

## Table of Contents

1. [WebGPU Compute Shaders — The Modern Approach](#1-webgpu-compute-shaders)
2. [WebGL Fallback — Fragment Shader Techniques](#2-webgl-fallback)
3. [Hybrid CPU/GPU Architecture](#3-hybrid-cpugpu-architecture)
4. [Key Architectural Decisions](#4-key-architectural-decisions)
5. [Existing Projects and References](#5-existing-projects-and-references)
6. [VFX Considerations](#6-vfx-considerations)
7. [Practical Recommendations](#7-practical-recommendations)

---

## 1. WebGPU Compute Shaders

### Browser Support (as of April 2026)

WebGPU now ships by default in all major browsers:
- **Chrome/Edge**: Supported since Chrome 113 (2023), including Android 12+ devices
- **Firefox**: Windows support since Firefox 141, macOS Apple Silicon since Firefox 145, Linux expected 2026
- **Safari**: Safari 26.0 on macOS Tahoe 26, iOS 26, iPadOS 26, visionOS 26

Production-ready adoption is at approximately **70% of browsers**. This is now a viable primary target with WebGL as fallback.

Sources: [Can I Use WebGPU](https://caniuse.com/webgpu), [WebGPU Hits Critical Mass](https://www.webgpu.com/news/webgpu-hits-critical-mass-all-major-browsers/)

### Core Architecture: Double-Buffer Texture Ping-Pong

The fundamental pattern for running cellular automata on the GPU is **ping-pong double-buffering**:

1. Create two storage textures (or storage buffers) of identical size
2. Each frame, the compute shader **reads from texture A** and **writes to texture B**
3. Next frame, swap: read from B, write to A
4. This eliminates read-write hazards — every thread reads consistent state from the previous frame

In WebGPU, this is implemented via two bind groups:

```
Bind Group 0: read from Buffer 0, write to Buffer 1
Bind Group 1: read from Buffer 1, write to Buffer 0
```

Each frame, toggle which bind group is active. The GPU handles the rest.

**Key advantage over WebGL**: WebGPU storage textures allow compute shaders to write to **any arbitrary pixel**, not just the pixel being "rendered" as in a fragment shader. This is critical for falling sand where a particle at position (x, y) needs to write itself to (x, y+1).

### Compute Shader Dispatch Pattern

A typical WGSL compute shader for cellular automata:

```wgsl
@group(0) @binding(0) var<storage, read> inputGrid : array<u32>;
@group(0) @binding(1) var<storage, read_write> outputGrid : array<u32>;
@group(0) @binding(2) var<uniform> params : SimParams;

@compute @workgroup_size(8, 8, 1)
fn simulate(@builtin(global_invocation_id) id : vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let idx = y * params.width + x;

    let cell = inputGrid[idx];
    let below = inputGrid[(y + 1) * params.width + x];
    let belowLeft = inputGrid[(y + 1) * params.width + (x - 1)];
    let belowRight = inputGrid[(y + 1) * params.width + (x + 1)];

    // Apply cellular automata rules based on material type
    // Write result to outputGrid[idx]
}
```

**Workgroup size**: The standard recommendation is a workgroup size of 64 threads. For 2D grids, `@workgroup_size(8, 8, 1)` (= 64 threads) is the typical choice, dispatching over tiles. Grid dimensions should be multiples of 8.

**Dispatch call**:
```javascript
passEncoder.dispatchWorkgroups(
    Math.ceil(width / 8),
    Math.ceil(height / 8),
    1
);
```

### Reading Neighbor Cells

The shader reads neighbors by computing offsets from the current cell's global invocation ID:

```wgsl
for (var yOffset : i32 = -1; yOffset <= 1; yOffset += 1) {
    for (var xOffset : i32 = -1; xOffset <= 1; xOffset += 1) {
        if (xOffset == 0 && yOffset == 0) { continue; }
        let nx = (i32(x) + xOffset + i32(params.width)) % i32(params.width);
        let ny = (i32(y) + yOffset + i32(params.height)) % i32(params.height);
        let neighborIdx = u32(ny) * params.width + u32(nx);
        let neighbor = inputGrid[neighborIdx];
        // process neighbor
    }
}
```

The modulo wrapping creates toroidal (wrap-around) topology. For falling sand you would typically clamp at boundaries instead of wrapping.

Sources: [Parallelizing Cellular Automata with WebGPU](https://vectrx.substack.com/p/webgpu-cellular-automata), [WebGPU Fundamentals — Compute Shaders](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html), [WebGPU from Ping Pong WebGL to Compute Shader](https://medium.com/phishchiang/webgpu-from-ping-pong-webgl-to-compute-shader-%EF%B8%8F-1ab3d8a461e2)

---

## 2. WebGL Fallback

### Fragment Shader Approach (Render-to-Texture)

Without compute shaders, WebGL simulates GPGPU via **fragment shaders rendering to framebuffer objects (FBOs)**:

1. Encode simulation state as texel data in a texture (e.g., RGBA channels encode material type, velocity, age, flags)
2. Draw a full-screen quad
3. The fragment shader runs once per pixel, reads neighbors from the input texture, computes the new state, and writes to the output FBO
4. Swap input/output textures each frame (classic "double-FBO-swap" a la Shadertoy)

This is the same ping-pong concept, but constrained by the fragment shader model:

**Critical limitation**: A fragment shader can only write to the pixel it is responsible for rendering. A sand grain at (x, y) cannot directly write itself to (x, y+1). Instead, the fragment shader at (x, y+1) must look UP at (x, y) and ask "is there a particle above me that wants to fall here?" This inverted logic makes complex material interactions significantly harder to express.

### WebGL 2 Transform Feedback

WebGL 2 introduced **transform feedback**, which captures vertex shader output varyings into GPU buffers without going through rasterization. This is useful for particle systems:

- Two GPU buffers alternate as read/write targets
- A vertex shader updates particle state (position, velocity, age)
- Updated data stays entirely on GPU memory — no CPU-GPU transfers after initialization
- Demonstrated handling ~10,000 particles interactively in older tutorials; modern hardware handles significantly more

**Limitation**: Particles cannot interact with each other through transform feedback. Each vertex is processed independently with no access to other vertices' data. This makes it unsuitable for cellular automata but fine for independent particle effects (sparks, rain, debris).

### WebGL Limits vs WebGPU

| Feature | WebGL 2 | WebGPU |
|---------|---------|--------|
| Compute shaders | No | Yes |
| Arbitrary write location | No (fragment writes own pixel only) | Yes (storage buffers/textures) |
| Shared memory between threads | No | Yes (workgroup shared memory) |
| Atomic operations | No | Yes |
| Read-write storage buffers | No | Yes |
| Max texture size | 4096-16384 (device-dependent) | Similar hardware limits |
| GPGPU pattern | Encode as textures, full-screen quad | Native compute pipeline |
| Particle data in vertex shader | Only via texture fetch | Storage buffer access |
| Performance at 2048x2048+ sim | Baseline | 3-8x faster |

**Bottom line**: WebGL can run cellular automata via fragment shaders but the "inverted logic" constraint, lack of shared memory, lack of atomics, and texture-encoding overhead make complex falling sand rules much harder and slower. For simple Game-of-Life style automata, WebGL is adequate. For multi-material falling sand with water/fire/gas, WebGPU is dramatically better.

Sources: [GPU-Accelerated Particles with WebGL 2](https://gpfault.net/posts/webgl2-particles.txt.html), [m4ym4y/falling-sand-shader](https://github.com/m4ym4y/falling-sand-shader), [WebGL vs WebGPU Performance](https://www.sitepoint.com/webgpu-vs-webgl-inference-benchmarks/), [From WebGL to WebGPU (Chrome)](https://developer.chrome.com/docs/web-platform/webgpu/from-webgl-to-webgpu)

---

## 3. Hybrid CPU/GPU Architecture

### When Hybrid Makes Sense

**Full GPU simulation** (all logic in compute shaders):
- Best for: massive particle counts, simple per-cell rules, visual-first simulations
- Achievable: 1M+ particles at 60fps
- Drawback: complex game logic is hard to express in WGSL; GPU-to-CPU readback is expensive

**Full CPU simulation** (GPU only renders):
- Best for: complex game logic, extensive inter-system interactions, debugging ease
- Practical limit: ~500K-2M cells at 30fps with aggressive optimization (chunking, dirty rects, multithreading via Web Workers or WASM)
- Drawback: CPU-to-GPU data upload every frame is a bottleneck

**Hybrid approaches**:
- **CPU for game logic, GPU for simulation**: CPU handles triggers (explosions, spawning, player interaction), writes commands into a buffer, GPU runs the cellular automata step. Good when you need complex game events but want fast simulation.
- **CPU for complex materials, GPU for simple ones**: Run water/fire/gas on GPU (simple local rules), run structural integrity / electricity propagation on CPU (requires global state). Upload only changed regions.
- **GPU simulation with selective CPU readback**: Run everything on GPU, but `mapAsync` specific small regions back to CPU when game logic needs to query the world state (e.g., "is there lava under the player?").

### Noita's Approach (Instructive Even Though CPU-Only)

Noita runs its entire falling sand simulation on the CPU. Key architectural lessons:

- **64x64 chunk system**: World divided into chunks, each with a dirty rect tracking which pixels need updating. Only dirty chunks are simulated.
- **Checkerboard multithreading**: Updated in 4 passes using a checker pattern. Each pass processes every other chunk, with each chunk allowed to move pixels within its 64x64 area plus 32 pixels into adjacent chunks. This prevents two threads from modifying the same pixel.
- **Frame counter per pixel**: Each pixel tracks whether it has been updated this frame, preventing double-updates as particles move between chunks.
- **Rigid body integration**: Each pixel knows which rigid body it belongs to and its offset within that body. When a rigid body pixel is destroyed, the rigid body shape is recalculated.

This checkerboard-with-overlap pattern is directly analogous to the Margolus block pattern used for GPU parallelization.

Sources: [GDC Vault — Exploring the Tech and Design of Noita](https://www.gdcvault.com/play/1025695/Exploring-the-Tech-and-Design), [80.lv — Noita: A Game Based on Falling Sand Simulation](https://80.lv/articles/noita-a-game-based-on-falling-sand-simulation), [Noita GDC YouTube](https://www.youtube.com/watch?v=prXuyMCgbTc)

---

## 4. Key Architectural Decisions

### 4a. Handling Cellular Automata Rules in Compute Shaders

**The Race Condition Problem**: In a naive parallel implementation, two sand particles above the same empty cell would both try to move into it simultaneously, causing data races and lost particles.

**Solution: Block Cellular Automata (Margolus Neighborhood)**

The standard GPU-friendly approach divides the grid into non-overlapping 2x2 blocks. Within each block, cells can only swap with each other. The blocks shift diagonally on alternating timesteps:

```
Frame 0 blocks:        Frame 1 blocks (offset by 1,1):
[AB][CD][EF]           A[BC][DE][F
[GH][IJ][KL]           G[HI][JK][L
[MN][OP][QR]           M[NO][PQ][R
```

Because blocks don't overlap, every block can be processed in parallel with zero race conditions. The alternating offset ensures particles can eventually move in any direction.

**4-step variant** (used by GelamiSalami/GPU-Falling-Sand-CA): Instead of 2-step diagonal shifts, use a 4-step z-shaped pattern to eliminate directional bias in particle movement. This produces more natural-looking results.

**Material rules within blocks**: Each 2x2 block is evaluated as a unit. The shader checks what materials are present and applies swap rules:

```
// Pseudocode for a 2x2 block evaluation
if (topLeft == SAND && bottomLeft == EMPTY) {
    swap(topLeft, bottomLeft);  // sand falls down
} else if (topLeft == WATER && bottomRight == EMPTY) {
    swap(topLeft, bottomRight); // water flows diagonally
} else if (topRight == FIRE && topLeft == WOOD) {
    topLeft = FIRE;             // fire spreads
    topRight = SMOKE;           // fire produces smoke
}
```

**Encoding materials**: Each cell is typically a u32 encoding material type (8 bits), variant/color (8 bits), and metadata like velocity, temperature, or lifetime (remaining 16 bits). More complex sims use RGBA textures where each channel carries different data.

### 4b. Long-Range Interactions (Structural Integrity, Pressure, Electricity)

This is the hardest problem for GPU falling sand. Local 2x2 block rules cannot propagate information across the grid in a single frame.

**Approaches**:

1. **Multi-pass propagation**: Run the compute shader multiple times per frame, each pass propagating information one cell further. For pressure or structural integrity, 4-8 passes per frame can propagate across 4-8 cells per frame, which feels instant for small structures. Cost: linear in propagation distance.

2. **Jump Flooding Algorithm (JFA)**: Used by GelamiSalami's project for lighting. Propagates distance information logarithmically — O(log N) passes cover the entire grid. Applicable to pressure fields, distance-to-surface calculations, and similar flood-fill operations.

3. **Hierarchical/mipmap approach**: Downsample the grid to progressively smaller textures, compute global properties (center of mass, total pressure) at coarse levels, then propagate back up. Used in fluid simulations for pressure solving.

4. **Hybrid CPU readback**: For complex structural integrity (e.g., "does this bridge still have a path to a support column?"), it may be simpler to read the relevant chunk back to CPU, run a flood fill or union-find on CPU, then upload the result. This only needs to happen when a structural change occurs, not every frame.

5. **Sparse updates**: Maintain a separate GPU buffer of "active propagation events." When a block is destroyed, add its neighbors to the active list. A separate compute pass processes only active cells. This avoids full-grid scans for rare events.

### 4c. Practical Particle Count Differences

| Approach | Practical Limit (interactive, 30-60fps) | Notes |
|----------|----------------------------------------|-------|
| **CPU single-threaded JS** | ~50K-100K cells | Naive iteration, no optimization |
| **CPU optimized JS** (dirty rects, chunking) | ~200K-500K cells | What most browser falling sand games do today |
| **CPU WASM + multithreading** | ~1M-2M cells (1920x1080) | Noita-style; ~30fps at 2M on high-end PC |
| **WebGL fragment shader** | ~4M+ pixels at 60fps | Confirmed by multiple demos; limited material complexity |
| **WebGPU compute shader** | ~1M-16M+ cells at 60fps | Depends on rule complexity; simple rules scale to enormous grids |
| **WebGPU particles (non-grid)** | ~1M at 60fps confirmed | With spatial hashing; sub-2ms compute per frame |

The jump from CPU to GPU is roughly **10-50x** for cellular automata workloads, depending on rule complexity.

### 4d. GPU-to-CPU Data Readback

When game logic on the CPU needs to query the GPU simulation state (collision detection, player queries, event triggers):

**WebGPU `mapAsync` pattern**:

```javascript
// Create a staging buffer with MAP_READ usage
const stagingBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

// Copy from GPU buffer to staging buffer
encoder.copyBufferToBuffer(gpuSimBuffer, 0, stagingBuffer, 0, bufferSize);
device.queue.submit([encoder.finish()]);

// Asynchronously map and read
await stagingBuffer.mapAsync(GPUMapMode.READ);
const data = new Uint32Array(stagingBuffer.getMappedRange());
// ... use data ...
stagingBuffer.unmap();
```

**Performance considerations**:

- **Never** `await mapAsync()` between every dispatch — this idles the GPU ~60% of the time
- **Double-buffer staging buffers**: CPU reads frame N-1 results while GPU processes frame N. This provides ~2.1x speedup over serial readback.
- **Minimize readback size**: Don't read the entire grid. Copy only the region around the player, or a downsampled version. A 64x64 region is 16KB — trivial. A 1024x1024 grid is 4MB — painful every frame.
- **Event-driven readback**: Only read back when the game logic actually needs the data (player collision, explosion trigger), not every frame.
- **GPU-side collision**: For simple checks like "is the pixel below the player solid?", consider encoding the answer into a tiny 1-pixel buffer via a dedicated compute pass, then reading just that one value back.

Sources: [WebGPU Concurrency Guide](https://www.sitepoint.com/the-webgpu-concurrency-guide-mastering-async-compute-shaders/), [GPUBuffer MDN](https://developer.mozilla.org/en-US/docs/Web/API/GPUBuffer), [gpuweb issue #1972](https://github.com/gpuweb/gpuweb/issues/1972)

---

## 5. Existing Projects and References

### WebGPU Falling Sand / Cellular Automata

| Project | Description | Link |
|---------|-------------|------|
| **GelamiSalami/GPU-Falling-Sand-CA** | Block cellular automata falling sand on GPU with 4-step Margolus neighborhood and JFA lighting. JS + WebGL. Live demo available. | [GitHub](https://github.com/GelamiSalami/GPU-Falling-Sand-CA) / [Live Demo](https://gelamisalami.github.io/GPU-Falling-Sand-CA/) |
| **ShaderVine** | WebGPU shader editor with 16 built-in compute simulations including falling sand, all using ping-pong buffer architecture | [Blog Post](https://meditations.metavert.io/p/shadervine-a-webgpu-shader-editor) |
| **scttfrdmn/webgpu-compute-exploration** | 10 interactive WebGPU compute demos including Game of Life, SPH fluids, boids, molecular dynamics. Rust+WASM+WGSL. | [GitHub](https://github.com/scttfrdmn/webgpu-compute-exploration) |
| **m4ym4y/falling-sand-shader** | WebGL fragment shader falling sand with 9 material types (dust, water, fire, metal, lightning, etc.) | [GitHub](https://github.com/m4ym4y/falling-sand-shader) |
| **ARez2/sandengine** | Falling sand engine in Rust + GLSL | [GitHub](https://github.com/ARez2/sandengine) |
| **MattyD1/Sandfall** | Falling sand physics simulator using compute shaders | [GitHub](https://github.com/MattyD1/Sandfall) |
| **NivMiz0/GPU-Sand-Sim-Unity** | Unity GPU falling sand with compute shaders | [GitHub](https://github.com/NivMiz0/GPU-Sand-Sim-Unity) |

### WebGPU Particle Systems

| Project | Description | Link |
|---------|-------------|------|
| **piellardj/particles-webgpu** | Full-GPU particle sim with gravitational attraction. Simple, clean reference. | [GitHub](https://github.com/piellardj/particles-webgpu) / [Live Demo](https://piellardj.github.io/particles-webgpu/) |
| **MankyDanky/particle-system** | WebGPU particle emitter with compute physics + GPU instancing | [GitHub](https://github.com/MankyDanky/particle-system) |
| **WebGPU Official Samples — Particles** | HDR particle rendering with compute shaders | [Live Demo](https://webgpu.github.io/webgpu-samples/samples/particles/) |
| **WebGPU Official Samples — Compute Boids** | Flocking simulation with ping-pong compute | [Live Demo](https://webgpu.github.io/webgpu-samples/samples/computeBoids/) |
| **James-Leste/WebGPU_ParticlesSimulation** | WebGPU particle simulation project | [GitHub](https://github.com/James-Leste/WebGPU_ParticlesSimulation) |
| **paulrobello/par-particle-life** | Rust + wgpu particle life with emergent behaviors | [GitHub](https://github.com/paulrobello/par-particle-life) |
| **jeantimex/fluid** | WebGPU fluid simulation using SPH and PIC/FLIP | [GitHub](https://github.com/jeantimex/fluid) |
| **njuptkid/WebGPU-Particles** | Smoke-like particle simulation in WebGPU | [GitHub](https://github.com/njuptkid/WebGPU-Particles) |

### Noita (Reference Architecture)

Noita is the gold standard for falling sand game design, despite being CPU-based:

- **Engine**: Custom C++ CPU engine, ~7 years of development
- **World**: Continuous, chunked (64x64), dirty-rect optimized
- **Parallelism**: 4-pass checkerboard with 32-pixel overlap for inter-chunk movement
- **Rigid bodies**: Pixels know their parent rigid body; shape recalculated on destruction
- **GDC Talk**: [Exploring the Tech and Design of Noita](https://www.gdcvault.com/play/1025695/Exploring-the-Tech-and-Design) (free on YouTube)

### Tutorials and Learning Resources

| Resource | Description | Link |
|----------|-------------|------|
| **Parallelizing Cellular Automata with WebGPU** | Step-by-step tutorial on ping-pong buffers, neighbor reading, WGSL patterns | [Substack](https://vectrx.substack.com/p/webgpu-cellular-automata) |
| **WebGPU Fundamentals — Compute Shaders** | Official-quality tutorial on compute shader basics | [webgpufundamentals.org](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html) |
| **The Compute Shader Tutorial** | Beginner-friendly WebGPU compute walkthrough | [barthpaleologue.github.io](https://barthpaleologue.github.io/Blog/posts/the-compute-shader-tutorial-1/) |
| **Codrops — Reaction-Diffusion in WebGPU** | Ping-pong storage texture tutorial for reaction-diffusion | [Codrops](https://tympanus.net/codrops/2024/05/01/reaction-diffusion-compute-shader-in-webgpu/) |
| **Recreating Noita's Falling Sand in C + OpenGL** | Detailed CPU implementation tutorial | [GameDev.net](https://gamedev.net/tutorials/programming/graphics/recreating-noitas-falling-sand-simulation-in-c-and-opengl-r5419) |
| **A Million Pixels of Falling Sand** | CPU optimization deep-dive (chunking, dirty rects, sleep states) | [hdyar.com](https://hdyar.com/blog/posts/falling-sand/) |
| **lisyarus — Particle Life in WebGPU** | Detailed WebGPU particle sim with spatial binning, atomics, prefix sums | [Blog](https://lisyarus.github.io/blog/posts/particle-life-simulation-in-browser-using-webgpu.html) |

---

## 6. VFX Considerations

### Particle Effects (Explosions, Sparks, Debris)

For visual-only particles (not part of the simulation grid), use a separate **instanced particle system**:

**Compute + Instanced Rendering Pipeline**:
1. **Compute shader** updates particle positions, velocities, lifetimes each frame (stays on GPU)
2. **Vertex shader** reads particle data directly from a storage buffer using vertex ID — no data duplication needed
3. **Render as camera-facing quads** (billboards) or point sprites
4. Each particle is a quad instance; GPU instancing draws thousands in a single draw call

**Key pattern**: The vertex shader can read the particle storage buffer directly:
```wgsl
@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32,
           @builtin(instance_index) instanceIndex : u32) -> VertexOutput {
    let particle = particles[instanceIndex];
    // Billboard quad corners from vertexIndex
    // Transform by particle position, size, rotation
}
```

### Smoke and Fire Rendering

**Additive blending**: Fire and explosions use additive blending to create glowing effects. Render particles to an HDR (rgba16float) render target, then composite.

**Layered approach for a falling sand game**:
1. **Base layer**: The cellular automata grid rendered as a texture (nearest-neighbor sampling for pixel art, or bilinear for smooth look)
2. **Particle overlay**: Instanced quads for sparks, embers, debris — these are purely visual and don't interact with the simulation
3. **Post-processing**: Bloom/glow pass on the composited result

### Post-Processing Effects

**Bloom pipeline** (works in both WebGL and WebGPU):
1. Render scene to HDR framebuffer
2. Extract bright pixels (threshold > 1.0) to a separate texture
3. Apply separable Gaussian blur (two passes: horizontal then vertical) — can be done at half resolution for performance
4. Composite blurred bright texture back onto the scene with additive blending
5. Tone-map from HDR to SDR (ACES tone mapping is the current standard)

**Screen-space effects relevant to falling sand**:
- **Heat distortion**: Render a distortion map from fire/lava cells, apply as UV offset in a post-process pass
- **Color grading**: LUT-based color grading in a final post-process pass
- **Dithering**: Apply dithering in the final pass to reduce banding (especially important for dark gradients in caves)
- **CRT/pixel-art filters**: Scanlines, chromatic aberration, barrel distortion for retro aesthetic

### Lighting in 2D Falling Sand

GelamiSalami's GPU-Falling-Sand-CA demonstrates an effective approach:
- **Jump Flooding Algorithm (JFA)** computes distance fields from light sources using Manhattan distance
- Applied separately to R, G, B channels with per-channel attenuation for colored lighting
- Runs entirely on GPU as additional compute passes
- Cost is O(log N) passes for full-grid coverage — very efficient

**Simpler alternative**: Render light sources as additive radial gradients in a separate "light map" texture, multiply with the scene. This is fast but produces only circular lights without occlusion.

Sources: [WebGPU Samples — Particles HDR](https://webgpu.github.io/webgpu-samples/samples/particles/), [GPGPU Particles with TSL + WebGPU](https://wawasensei.dev/courses/react-three-fiber/lessons/tsl-gpgpu), [LearnOpenGL — Bloom](https://learnopengl.com/Advanced-Lighting/Bloom), [Interactive Galaxy with WebGPU Compute](https://threejsroadmap.com/blog/galaxy-simulation-webgpu-compute-shaders)

---

## 7. Practical Recommendations

### Recommended Architecture for a Browser Falling Sand Game

**Primary target: WebGPU compute shaders**

```
Frame Loop:
  1. Process user input (CPU) → write spawn/destroy commands to a small GPU buffer
  2. Run cellular automata compute shader (GPU)
     - Margolus block CA with 4-step offset pattern
     - Double-buffer ping-pong
     - Material rules encoded as switch/if chains in WGSL
  3. [Optional] Run propagation passes (GPU) — pressure, temperature, lighting
  4. [Optional] Run VFX particle compute shader (GPU) — sparks, smoke, debris
  5. Render grid as textured quad (GPU)
  6. Render VFX particles as instanced billboards (GPU)
  7. Post-process: bloom, heat distortion, tone mapping (GPU)
  8. [If needed] Async readback of small region for game logic (GPU → CPU)
```

**Fallback: WebGL 2 fragment shader**

- Same ping-pong architecture but with FBO swap instead of storage textures
- Fragment shader reads neighbors and applies rules in "pull" mode (each pixel asks its neighbors what should be here now?)
- Simpler material rules due to the inverted-logic constraint
- Skip compute-based VFX; use CPU-updated instanced particles instead

### Data Encoding

Pack each cell into a single `u32`:

```
Bits 0-7:   Material type (256 types)
Bits 8-15:  Variant / color index
Bits 16-23: Metadata (velocity, temperature, lifetime, etc.)
Bits 24-31: Flags (updated-this-frame, is-falling, is-source, etc.)
```

Or use `rgba8unorm` textures where:
- R = material type
- G = variant
- B = velocity/temperature
- A = flags

### Performance Budget (Targeting 60fps = 16.6ms per frame)

For a 1024x768 grid (~786K cells):
- Cellular automata compute: ~1-3ms
- 4 propagation passes: ~1-2ms
- VFX particle update (10K particles): ~0.2ms
- Grid render + VFX render: ~1-2ms
- Post-processing: ~1-2ms
- **Total: ~4-9ms** — well within budget on modern hardware

For a 1920x1080 grid (~2M cells):
- Everything roughly 2-3x the above
- **Total: ~8-18ms** — tight at 60fps, comfortable at 30fps
- Consider half-resolution simulation rendered at 2x with nearest-neighbor scaling

### Feature Detection and Fallback

```javascript
async function initRenderer() {
    if (navigator.gpu) {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
            const device = await adapter.requestDevice();
            return new WebGPURenderer(device);
        }
    }
    // Fallback to WebGL 2
    const canvas = document.getElementById('canvas');
    const gl = canvas.getContext('webgl2');
    if (gl) {
        return new WebGLRenderer(gl);
    }
    throw new Error('Neither WebGPU nor WebGL 2 supported');
}
```

---

## Summary of Key Takeaways

1. **WebGPU is ready for production** — 70% browser support as of April 2026, all major browsers shipping it. Use it as the primary path with WebGL fallback.

2. **Block cellular automata (Margolus neighborhood)** is the proven technique for parallelizing falling sand on the GPU. It eliminates race conditions by design and maps perfectly to compute shader workgroups.

3. **Ping-pong double-buffering** is non-negotiable for any GPU simulation — two buffers, alternate read/write each frame.

4. **Performance gains are dramatic**: CPU tops out around 500K-2M cells at 30fps. GPU can handle 4M-16M+ cells at 60fps for simple rules, or 1M+ cells with complex multi-material rules.

5. **Long-range interactions** (structural integrity, pressure) are the hardest problem. Use multi-pass propagation, JFA, or selective CPU readback for these.

6. **GPU-to-CPU readback** should be minimized and double-buffered. Never await mapAsync synchronously between dispatches. Read small regions, not the whole grid.

7. **VFX layering** — separate the simulation grid from visual particle effects. Use instanced rendering for sparks/smoke/fire overlays, with bloom post-processing for glow.

8. **Study the references**: GelamiSalami's GPU-Falling-Sand-CA for the Margolus block technique, the WebGPU official samples for compute patterns, and Noita's GDC talk for game design around falling sand.
