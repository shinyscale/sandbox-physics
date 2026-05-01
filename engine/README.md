# Sand Physics Engine

Standalone 2D pixel-physics engine extracted from falling-sand-phase2.html.
No DOM dependencies — games bring their own renderer and input handling.

## Quick Start

```html
<script type="module">
import { SandEngine, Materials } from './sand-engine.js';

const engine = new SandEngine(300, 200);
engine.generateCity();           // populate with a destructible city
engine.step();                   // advance one simulation tick
const pixels = engine.render();  // Uint32Array (ABGR) for blitting to canvas
</script>
```

See `demo.html` for a complete working example.

## API Reference

### Constructor

```js
const engine = new SandEngine(width, height, opts?)
```

- `width`, `height` — grid dimensions in pixels
- `opts.roomTemp` — ambient temperature (default: 22)

### Grid Access

| Method | Description |
|--------|-------------|
| `engine.get(x, y)` | Returns material ID at (x, y). Out-of-bounds returns STONE. |
| `engine.getTemp(x, y)` | Returns temperature at (x, y). |
| `engine.set(x, y, mat, temp?)` | Place material. Auto-sets sensible temps (fire=500, lava=800, ice=-10). Triggers integrity check if structural material removed. |
| `engine.erase(x, y)` | Clear cell to empty. Triggers integrity check if structural. |
| `engine.paintCircle(cx, cy, r, mat, density?)` | Paint a circular brush. `density` (0-1, default 0.7) controls fill randomness. |
| `engine.clear()` | Reset entire grid to empty. |
| `engine.particleCount()` | Count non-empty cells. |

### Simulation

| Method | Description |
|--------|-------------|
| `engine.step()` | Advance simulation by one tick. Runs thermal, transitions, material behaviors, chunks, and structural integrity. |

### Destruction

| Method | Description |
|--------|-------------|
| `engine.explode(cx, cy, radius)` | Radial explosion. Flings debris, spawns fire/embers, heats area. Structural materials resist based on STRENGTH. |
| `engine.napalmSplash(cx, cy, radius)` | Small explosion + sticky burning napalm gel splash. |
| `engine.fireRailgun(targetX)` | Vertical beam trace through entire grid column. Shockwave damages nearby structure. |

### Rendering

```js
const pixels = engine.render({ showTemp?, showStress?, time? });
```

Returns a `Uint32Array` (ABGR packed, length W×H). Blit to an ImageData buffer:

```js
new Uint32Array(imageData.data.buffer).set(pixels);
```

Options:
- `showTemp` — thermal heatmap overlay
- `showStress` — structural stress heatmap
- `time` — `performance.now()` for animated stress pulses

### City Generator

```js
engine.generateCity();
```

Clears grid and generates a procedural destructible city with buildings (brick/concrete walls, metal supports, glass windows, peaked/flat roofs, antennas, skybridges, doors).

### Direct Buffer Access

For advanced use, the engine exposes its typed array buffers directly:

| Buffer | Type | Description |
|--------|------|-------------|
| `engine.grid` | `Uint8Array(W*H)` | Material IDs |
| `engine.temp` | `Float32Array(W*H)` | Temperature per cell |
| `engine.life` | `Uint8Array(W*H)` | Tick counter per cell (material-dependent) |
| `engine.weightBuf` | `Float32Array(W*H)` | Structural weight (valid after step with structures) |
| `engine.chunks` | `Array` | Active falling chunk objects |

Index: `y * engine.W + x`

## Materials

```js
import { Materials } from './sand-engine.js';
```

| ID | Name | Category |
|----|------|----------|
| 0 | EMPTY | — |
| 1 | SAND | powder |
| 2 | WATER | liquid |
| 3 | STONE | solid |
| 4 | FIRE | gas-like |
| 5 | WOOD | solid (structural, flammable) |
| 6 | OIL | liquid (flammable) |
| 7 | GUNPOWDER | powder (explosive) |
| 8 | STEAM | gas |
| 9 | ACID | liquid (corrosive) |
| 10 | LAVA | liquid (hot) |
| 11 | ICE | solid |
| 12 | SMOKE | gas |
| 13 | PLANT | solid (grows) |
| 14 | SALT | powder (dissolves in water) |
| 15 | METAL | solid (structural, meltable) |
| 16 | GAS | gas (flammable) |
| 17 | CLONE | solid (replicates neighbor) |
| 18 | VOID | solid (destroys neighbor) |
| 19 | EMBER | gas-like (hot, short-lived) |
| 20 | CONCRETE | structural |
| 21 | GLASS | structural (fragile, shatters to sand) |
| 22 | BRICK | structural |
| 23 | NAPALM | liquid (sticky, long-burning) |

### Classification helpers

```js
import { isPowder, isLiquid, isGas, isSolid, isStructural } from './sand-engine.js';
```

### Material properties

```js
import { DENSITY, CONDUCT, STRENGTH } from './sand-engine.js';
```

- `DENSITY[mat]` — displacement priority (heavier sinks)
- `CONDUCT[mat]` — thermal conductivity
- `STRENGTH[mat]` — structural load capacity (0 = non-structural)

## Building a Game Against This Engine

Typical game loop:

```js
function gameLoop() {
  handleInput();       // your input code
  applyGameLogic();    // spawn projectiles, check win conditions, etc.
  engine.step();       // physics
  const px = engine.render();
  blitToCanvas(px);    // your renderer
  drawUI();            // your HUD overlay
  requestAnimationFrame(gameLoop);
}
```

The engine doesn't know about warheads, projectiles, scores, or turns — games implement those on top. Use `engine.explode()`, `engine.napalmSplash()`, `engine.fireRailgun()`, and direct grid writes to create game-specific destruction effects.

## Architecture Notes

- Simulation alternates left-to-right and right-to-left scan each tick (prevents directional bias)
- Structural integrity uses BFS from bottom row to find unsupported components, then drops them as rigid chunks
- Chunks fall with gravity, topple based on remaining support asymmetry, shatter on impact
- Thermal model: 4-neighbor conduction + ambient decay, drives phase transitions
- All state is in flat typed arrays for cache-friendly iteration
