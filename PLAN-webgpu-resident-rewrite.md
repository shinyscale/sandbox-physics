# Sandpipe-body WebGPU: port to powerhouse + GPU-resident rewrite

## Why

Current `sandpipe-body-webgpu.html` is a hybrid that stutters on strixhalo's iGPU despite hardware acceleration being enabled. Diagnosis confirmed 2026-04-20: dropping physics iters to 1 and grain to 8 smoothed it out, which points at CPU↔GPU bandwidth + submit overhead, not raw compute. Strixhalo's unified memory doesn't rescue us here because WebGPU still pays mapping/copy costs across address spaces.

Powerhouse + discrete GPU is the right deployment target. PCIe bandwidth is ample, state can stay GPU-resident, and we only pay the round-trip when explicitly saving or reading. This is also the moment to do the rewrite we've been deferring.

## Current bottlenecks (file: sandpipe-body-webgpu.html, 1599 lines)

1. **Per-frame GPU→CPU readback** — line 1402, `readbackGridsFromGPU()` pulls the entire grid back every frame so CPU can run body-mask logic, emitters, and overlays.
2. **Per-frame CPU→GPU upload** — line 1244, `uploadGridsToGPU()` re-pushes the grid whenever CPU mutated it (which is ~every frame).
3. **~10 separate `device.queue.submit()` calls per frame** — lines 1215, 1268, 1282, 1296, 1308, 1320, 1332, 1344, 1361, plus readback. Each one is a driver round-trip. On an iGPU this is death.
4. **All video frames preloaded into memory** — line 379, `videoFrames.push(img)`. Long clips balloon memory and choke chromium.

## Target architecture

- Body mask lives as a GPU texture (R8 or R32F), updated by a compute pass that consumes the frame's body-detection output.
- All grid mutations (emitter spawns, body interactions, gravity, collision, displacement) happen in compute shaders operating on GPU-resident buffers.
- A single command encoder per frame; all passes batched into one `submit()`.
- Video frames stream in as needed, not preloaded — decode-to-texture on demand, keep a small ring buffer.
- Readback only on explicit user action (save snapshot, export).

## Concrete work

1. **Audit which CPU-side logic actually needs CPU.** Emitters, body-mask intake, user-drawn modifications — each one either moves into a shader or becomes a small uniform/buffer update that doesn't require full readback.
2. **Consolidate submits.** Single encoder, single submit per frame. Profile before/after; this alone may fix most of the stutter on decent hardware.
3. **Port body mask to texture-resident.** Body detection writes straight to a GPU texture; physics shaders sample it.
4. **Rewrite grid mutation paths as compute shaders.** No `uploadGridsToGPU` in the hot path.
5. **Stream video frames.** Replace preload-all with on-demand decode; small LRU or ring buffer.
6. **Delete readback from the render loop.** Keep a utility for save/export only.

## Port-to-powerhouse checklist

- Copy `sandbox-physics/` to powerhouse.
- Confirm chromium on powerhouse has WebGPU + hardware acceleration (same flags alias pattern we set up on strixhalo).
- Baseline the current hybrid on powerhouse *before* rewriting — gives a free perf comparison and tells us how much of the stutter was strixhalo-specific vs. architectural.
- Then do the rewrite. Keep strixhalo around as the "does it still run on a weak iGPU" regression check.

## Open questions

- Is there any CPU-side logic that genuinely can't move to a shader? (User drawing input is the obvious candidate — but even that can be a small upload of stroke points, not a full grid re-push.)
- Do we want to keep the current hybrid around as a "low-end mode" or retire it once the resident version works?

## Status

Planned 2026-04-20. Not started. Pick this up when work moves to powerhouse.
