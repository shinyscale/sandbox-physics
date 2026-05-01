# Sandpipe-body WebGPU port — checkpoint 2026-04-21

Task: f235-task #1619 (journal #412). Port `sandpipe-body-webgpu.html` to powerhouse + GPU-resident rewrite + SAM2 wiring.
Session: f235-Claude, 2026-04-21 ~07:00–08:15 MDT.

## Status summary

| Phase | Status | Notes |
|---|---|---|
| Port files | ✅ done | 15 files + engine/ dir copied to `F:\sandbox-physics\` |
| Chrome present | ✅ done | Chrome 147.0.7727.56 at `C:\Program Files\Google\Chrome\Application\chrome.exe` |
| WebGPU API | ✅ partial | `navigator.gpu` present in headless. **Discrete-GPU adapter path not verified** — headless SSH can't reach discrete GPU on Windows. |
| SAM2 install | ✅ done | In ComfyUI venv (see below). Checkpoint is `sam2_hiera_base_plus.pt`, not the `sam2.1_hiera_large` strixhalo uses. |
| Baseline hybrid | ⚠️ blocked on interactive | Runner `run-baseline.bat` uploaded. Needs Zach or an interactive session. |
| GPU-resident rewrite | ⛔ not started | Depends on baseline numbers to target + interactive runtime to validate. |
| SAM2 wiring | ⛔ not started | Depends on rewrite. |
| Perf report | ⛔ not started | Depends on rewrite + baseline. |

## Environment facts

- Machine: `POWERHOUSE-4090-96GB` (hostname misleading — actual GPU is **RTX PRO 6000 Blackwell Max-Q 96GB**, confirmed via torch)
- SSH user (same as logged-in desktop user): `zacharymandrews`
- Python options:
  - System: `C:\Python313\python.exe` (no SAM2)
  - ComfyUI venv: `F:\ComfyUI\ComfyUI\.venv\Scripts\python.exe` — **use this for SAM2**
- Torch: `2.11.0.dev20260115+cu130`, `cuda.is_available() = True`
- SAM2 checkpoint on disk: `F:\GVHMR\GVHMR\inputs\checkpoints\sam2\sam2_hiera_base_plus.pt` (323 MB)
- F: drive: `AI_4TB`, 2 TB free
- bodypipe/GVHMR live at `F:\gvhmr\` (per MEMORY.md)

## What was ported

`F:\sandbox-physics\` contains:
```
bodypipe-prototype-fluid.html
bodypipe-prototype.html
engine\
falling-sand-phase1.5.html
falling-sand-phase1.html
falling-sand-phase2.html
falling-sand-webgpu.html
falling-sand.html
gpu-particle-research.md
PLAN-webgpu-resident-rewrite.md
sandpipe-body-webgpu.html     ← TARGET
sandpipe-body.html
server.py
sound-sandbox-fx.html
sound-sandbox-plinko.html
sound-sandbox.html
run-baseline.bat              ← NEW: double-click to run baseline
webgpu-probe.html             ← NEW: navigator.gpu probe page
```

## Baseline instructions (for Zach)

1. Double-click `F:\sandbox-physics\run-baseline.bat`.
2. Chrome opens with `--enable-unsafe-webgpu` on the hybrid page.
3. Load a clip (try `lieberman`). Let it run ~60 seconds.
4. Open DevTools (F12), note FPS in Performance tab, note any console errors.
5. Report: FPS (min/avg/max), stutter events, any WebGPU errors.
6. Close the cmd window to stop the server.

## Why the rewrite wasn't started this session

1. **No baseline number** — "make it faster than X" without X is guesswork.
2. **No interactive runtime** — headless Chrome in a Windows SSH session doesn't access the discrete GPU, so I couldn't validate any change.
3. **Scope** — per `PLAN-webgpu-resident-rewrite.md`, the rewrite touches:
   - 10 per-frame `submit()` calls → one encoder
   - Per-frame readback (`readbackGridsFromGPU`, line 1402) → eliminated
   - Per-frame upload (`uploadGridsToGPU`, line 1244) → eliminated from hot path
   - Body mask → GPU texture (R8), consumed by compute shader
   - Grid mutation logic → WGSL compute shaders
   - Video frames → streaming decode, not preload
   
   This is a ~1–2 day rewrite that needs incremental validation — not a single-session shot in the dark.

## Next session pickup

1. **Read this file first**: `F:\sandbox-physics\port-checkpoint-20260421.md` (also on strixhalo as `~/claude/sandbox-physics/port-checkpoint-20260421.md`).
2. **Get baseline numbers** from Zach (or run interactively via RDP/console).
3. **Start the rewrite on a copy**: `sandpipe-body-webgpu-v2.html`. Keep the hybrid running as regression ground truth.
4. **Order of attack** (lowest risk → highest reward first):
   1. Consolidate `device.queue.submit()` calls into one per frame (single encoder). Expected: 2–5× FPS on iGPU, modest gain on discrete.
   2. Remove per-frame readback. Move CPU-only logic (emitters, body-mask intake) into upload-small / compute paths.
   3. Port body mask to R8 GPU texture. Physics shader samples it.
   4. Rewrite grid mutations as WGSL compute shaders.
   5. Stream video frames (LRU decode-to-texture), retire `videoFrames.push(img)` at line 379.
   6. Delete readback from render loop; keep as utility for save/export only.
5. **Wire SAM2**: `F:\ComfyUI\ComfyUI\.venv\Scripts\python.exe` drives `tools/process_video_sam2.py`. Hot-path is: video frame → SAM2 body mask → upload to R8 texture → sandpipe compute shaders sample it. Server.py (Flask) already has the scaffold — may just need adjusted paths for Windows.
6. **SAM2 large checkpoint**: consider downloading `sam2.1_hiera_large.pt` for parity with strixhalo, if quality matters more than speed.

## Open questions for Zach

1. OK to create `sandpipe-body-webgpu-v2.html` as a working copy, preserving the hybrid as regression reference? (Plan says "keep strixhalo as weak-iGPU regression check" — this preserves powerhouse's copy too.)
2. Download `sam2.1_hiera_large.pt` to match strixhalo, or stay on `base_plus` on powerhouse for speed?
3. Any specific perf target? ("60 FPS at default params on lieberman clip"? "No visible stutter"?)

## Tags

`sandpipe,powerhouse,webgpu,sam2,f235-task,port,checkpoint`

Memory refs: strixhalo memory id 1619, journal #412.

---

## Decisions (Zach, 2026-04-21 ~08:20 MDT)

1. **Rewrite in parallel file** — `sandpipe-body-webgpu-v2.html`. Hybrid stays as regression reference.
2. **SAM2 large downloaded** — `F:\models\sam2\sam2.1_hiera_large.pt` (898,083,611 bytes, SHA256 `2647878d…47dd318`, matches strixhalo byte-for-byte).
3. **Perf target** — 60 FPS on the **soul-dance** clip, no visible stutter.
