/**
 * Sand Physics Engine — extracted from falling-sand-phase2.html
 *
 * Pure simulation module. No DOM dependencies.
 * Games bring their own renderer and input handling.
 *
 * Usage:
 *   import { SandEngine, Materials } from './sand-engine.js';
 *   const engine = new SandEngine(300, 200);
 *   engine.step();                         // advance one tick
 *   engine.set(50, 10, Materials.SAND);    // place material
 *   engine.explode(100, 100, 10);          // boom
 *   const pixels = engine.render();        // Uint32Array ABGR for blitting
 */

// ==================== MATERIALS ====================

export const Materials = Object.freeze({
  EMPTY: 0, SAND: 1, WATER: 2, STONE: 3, FIRE: 4, WOOD: 5, OIL: 6, GUNPOWDER: 7,
  STEAM: 8, ACID: 9, LAVA: 10, ICE: 11, SMOKE: 12, PLANT: 13, SALT: 14, METAL: 15,
  GAS: 16, CLONE: 17, VOID: 18, EMBER: 19,
  CONCRETE: 20, GLASS: 21, BRICK: 22, NAPALM: 23,
  COUNT: 24,
});

const M = Materials; // shorthand for internal use

const DENSITY  = [0, 4, 3, 8, 0, 5, 2, 4, 0, 3, 7, 3, 0, 5, 4, 9, 0, 10, 10, 0, 8, 5, 7, 3.5];
const CONDUCT  = [0.02, 0.3, 0.6, 0.5, 0.1, 0.08, 0.12, 0.2, 0.04, 0.5, 0.8, 0.8, 0.02, 0.08, 0.3, 0.95, 0.05, 0, 0, 0.3, 0.4, 0.3, 0.35, 0.15];
const STRENGTH = new Float32Array(M.COUNT);
STRENGTH[M.STONE] = 4; STRENGTH[M.WOOD] = 2; STRENGTH[M.METAL] = 5;
STRENGTH[M.CONCRETE] = 3; STRENGTH[M.GLASS] = 0.5; STRENGTH[M.BRICK] = 2.5;

export { DENSITY, CONDUCT, STRENGTH };

const ROOM_TEMP = 22;
const NEIGHBOR4 = [[0, -1], [0, 1], [-1, 0], [1, 0]];
const NEIGHBOR8 = [[0,-1],[0,1],[-1,0],[1,0],[-1,-1],[1,-1],[-1,1],[1,1]];

// ==================== MATERIAL CLASSIFICATION ====================

export function isPowder(m)     { return m === M.SAND || m === M.GUNPOWDER || m === M.SALT; }
export function isLiquid(m)     { return m === M.WATER || m === M.OIL || m === M.ACID || m === M.LAVA; }
export function isGas(m)        { return m === M.STEAM || m === M.SMOKE || m === M.GAS || m === M.EMBER; }
export function isSolid(m)      { return m === M.STONE || m === M.WOOD || m === M.ICE || m === M.PLANT || m === M.METAL || m === M.CLONE || m === M.VOID || m === M.CONCRETE || m === M.GLASS || m === M.BRICK; }
export function isStructural(m) { return STRENGTH[m] > 0; }

function canDisplace(mover, target) {
  if (target === M.EMPTY) return true;
  if (isSolid(target)) return false;
  if (isGas(mover)) return false;
  return DENSITY[mover] > DENSITY[target];
}

// ==================== ENGINE ====================

export class SandEngine {
  constructor(width, height, opts = {}) {
    this.W = width;
    this.H = height;
    this.ROOM_TEMP = opts.roomTemp ?? ROOM_TEMP;

    // Core state buffers
    this.grid    = new Uint8Array(width * height);
    this.temp    = new Float32Array(width * height);
    this.life    = new Uint8Array(width * height);
    this.updated = new Uint8Array(width * height);

    // Structural integrity buffers
    this.supportBuf = new Uint8Array(width * height);
    this.groupBuf   = new Uint8Array(width * height);
    this.weightBuf  = new Float32Array(width * height);

    // Falling chunks and integrity state
    this.chunks = [];
    this.integrityDirty = false;
    this.integrityTimer = 0;

    // Rendering buffer (ABGR packed Uint32)
    this._pixels = new Uint32Array(width * height);

    this.temp.fill(this.ROOM_TEMP);
  }

  // ---- Coordinate helpers ----

  idx(x, y) { return y * this.W + x; }
  inBounds(x, y) { return x >= 0 && x < this.W && y >= 0 && y < this.H; }

  get(x, y) {
    return this.inBounds(x, y) ? this.grid[this.idx(x, y)] : M.STONE;
  }

  getTemp(x, y) {
    return this.inBounds(x, y) ? this.temp[this.idx(x, y)] : this.ROOM_TEMP;
  }

  // ---- Public mutation API ----

  set(x, y, mat, t) {
    if (!this.inBounds(x, y)) return;
    const i = this.idx(x, y);
    const oldMat = this.grid[i];
    this.grid[i] = mat;
    if (t !== undefined) {
      this.temp[i] = t;
    } else if (mat === M.FIRE) {
      this.temp[i] = 500;
    } else if (mat === M.LAVA) {
      this.temp[i] = 800;
    } else if (mat === M.ICE) {
      this.temp[i] = -10;
    } else if (mat === M.EMBER) {
      this.temp[i] = 350;
    } else {
      this.temp[i] = this.ROOM_TEMP;
    }
    this.life[i] = 0;
    // If we removed structural material, flag integrity check
    if (isStructural(oldMat) && !isStructural(mat)) {
      this.integrityDirty = true;
      this.integrityTimer = Math.max(this.integrityTimer, 2);
    }
  }

  /** Erase a cell back to empty */
  erase(x, y) {
    if (!this.inBounds(x, y)) return;
    const i = this.idx(x, y);
    if (isStructural(this.grid[i])) {
      this.integrityDirty = true;
      this.integrityTimer = Math.max(this.integrityTimer, 2);
    }
    this.grid[i] = M.EMPTY;
    this.life[i] = 0;
    this.temp[i] = this.ROOM_TEMP;
  }

  /** Paint a circular brush of material */
  paintCircle(cx, cy, radius, mat, density = 0.7) {
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx * dx + dy * dy > radius * radius) continue;
        const px = cx + dx, py = cy + dy;
        if (!this.inBounds(px, py)) continue;
        if (mat === M.EMPTY) {
          this.erase(px, py);
        } else if (this.grid[this.idx(px, py)] === M.EMPTY) {
          if (Math.random() < density) this.set(px, py, mat);
        }
      }
    }
  }

  /** Clear the entire grid */
  clear() {
    this.grid.fill(0);
    this.life.fill(0);
    this.temp.fill(this.ROOM_TEMP);
    this.chunks.length = 0;
    this.integrityDirty = false;
    this.integrityTimer = 0;
    this.weightBuf.fill(0);
  }

  /** Count non-empty particles */
  particleCount() {
    let count = 0;
    for (let i = 0; i < this.W * this.H; i++) {
      if (this.grid[i] !== M.EMPTY) count++;
    }
    return count;
  }

  // ---- Internal swap ----

  _swap(x1, y1, x2, y2) {
    const i1 = this.idx(x1, y1), i2 = this.idx(x2, y2);
    let t;
    t = this.grid[i1]; this.grid[i1] = this.grid[i2]; this.grid[i2] = t;
    t = this.life[i1]; this.life[i1] = this.life[i2]; this.life[i2] = t;
    t = this.temp[i1]; this.temp[i1] = this.temp[i2]; this.temp[i2] = t;
    this.updated[i2] = 1;
  }

  // ==================== THERMAL ====================

  _simulateThermal() {
    const { W, H, grid, temp, ROOM_TEMP: RT } = this;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const mat = grid[i];

        if (mat === M.EMPTY) {
          if (Math.abs(temp[i] - RT) < 0.5) { temp[i] = RT; continue; }
          temp[i] += (RT - temp[i]) * 0.1;
          continue;
        }

        if (mat === M.CLONE || mat === M.VOID) continue;

        if (mat === M.FIRE) temp[i] = Math.max(temp[i], 400 + Math.random() * 200);
        else if (mat === M.LAVA) temp[i] = Math.max(temp[i], 700 + Math.random() * 150);
        else if (mat === M.ICE) temp[i] = Math.min(temp[i], -5);
        else if (mat === M.EMBER) temp[i] = Math.max(temp[i], 300 + Math.random() * 100);

        const myCond = CONDUCT[mat];
        let heatFlow = 0;
        let neighbors = 0;

        for (const [dx, dy] of NEIGHBOR4) {
          const nx = x + dx, ny = y + dy;
          if (!this.inBounds(nx, ny)) continue;
          const ni = ny * W + nx;
          const avgCond = (myCond + CONDUCT[grid[ni]]) * 0.5;
          heatFlow += (temp[ni] - temp[i]) * avgCond;
          neighbors++;
        }

        if (neighbors > 0) temp[i] += heatFlow / neighbors * 0.15;
        temp[i] += (RT - temp[i]) * 0.001;
        if (temp[i] < -50) temp[i] = -50;
        if (temp[i] > 1200) temp[i] = 1200;
      }
    }
  }

  // ==================== EXPLOSIONS ====================

  explode(cx, cy, radius) {
    const { W, H, grid, temp, life, ROOM_TEMP: RT } = this;
    const displaced = [];

    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const dist2 = dx * dx + dy * dy;
        if (dist2 > radius * radius) continue;
        const nx = cx + dx, ny = cy + dy;
        if (!this.inBounds(nx, ny)) continue;
        const ni = ny * W + nx;
        const m = grid[ni];

        const heatAmt = Math.max(0, 500 * (1 - Math.sqrt(dist2) / radius));
        temp[ni] = Math.max(temp[ni], heatAmt);

        if (m === M.EMPTY || m === M.CLONE || m === M.VOID || isGas(m) || m === M.FIRE) continue;

        if (m === M.GUNPOWDER) {
          grid[ni] = M.FIRE; life[ni] = 0; temp[ni] = 600;
          continue;
        }

        const dist = Math.sqrt(dist2) || 0.5;
        const force = 1 - dist / radius;

        if (isStructural(m)) {
          const threshold = 0.15 * STRENGTH[m];
          if (force < threshold) continue;
          if (force < threshold * 2 && Math.random() > (force - threshold) / threshold) continue;
          if (m === M.GLASS) {
            grid[ni] = Math.random() < 0.6 ? M.SAND : M.EMPTY;
            temp[ni] = heatAmt;
            life[ni] = 0;
            continue;
          }
        } else {
          if ((m === M.STONE || m === M.METAL) && force < 0.5) continue;
        }

        const flingDist = Math.max(1, Math.round(force * radius * 2));
        const dirX = dx / dist;
        const dirY = dy / dist;

        displaced.push({ x: nx, y: ny, mat: m, temp: temp[ni], life: life[ni], dirX, dirY, flingDist, dist });
        grid[ni] = M.EMPTY; temp[ni] = RT; life[ni] = 0;
      }
    }

    displaced.sort((a, b) => b.dist - a.dist);

    for (const p of displaced) {
      let placed = false;
      for (let s = p.flingDist; s >= 1; s--) {
        const tx = Math.round(p.x + p.dirX * s);
        const ty = Math.round(p.y + p.dirY * s);
        if (!this.inBounds(tx, ty)) continue;
        const ti = ty * W + tx;
        if (grid[ti] === M.EMPTY) {
          grid[ti] = p.mat; temp[ti] = p.temp; life[ti] = p.life;
          placed = true;
          break;
        }
      }
      if (!placed && grid[p.y * W + p.x] === M.EMPTY) {
        const oi = p.y * W + p.x;
        grid[oi] = p.mat; temp[oi] = p.temp; life[oi] = p.life;
      }
    }

    // Post-explosion fire
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const dist2 = dx * dx + dy * dy;
        if (dist2 > radius * radius) continue;
        const nx = cx + dx, ny = cy + dy;
        if (!this.inBounds(nx, ny)) continue;
        const ni = ny * W + nx;
        if (grid[ni] === M.EMPTY) {
          const chance = 0.3 * (1 - Math.sqrt(dist2) / radius);
          if (Math.random() < chance) {
            grid[ni] = M.FIRE; life[ni] = Math.random() * 10 | 0;
          }
        }
      }
    }

    // Embers
    for (let e = 0; e < 5 + radius; e++) {
      const angle = Math.random() * Math.PI * 2;
      const edist = 1 + Math.random() * radius * 0.6;
      const ex = Math.round(cx + Math.cos(angle) * edist);
      const ey = Math.round(cy + Math.sin(angle) * edist);
      if (this.inBounds(ex, ey) && grid[ey * W + ex] === M.EMPTY) {
        const ei = ey * W + ex;
        grid[ei] = M.EMBER; life[ei] = 0; temp[ei] = 400;
      }
    }

    this.integrityDirty = true;
    this.integrityTimer = 3;
  }

  /** Spawn napalm gel in a splash pattern */
  napalmSplash(cx, cy, radius) {
    const { W, grid, life, temp } = this;
    this.explode(cx, cy, 5);

    const napalmCount = 30 + Math.floor(Math.random() * 20);
    for (let n = 0; n < napalmCount; n++) {
      const angle = Math.random() * Math.PI * 2;
      const dist = 1 + Math.random() * radius;
      const nx = Math.round(cx + Math.cos(angle) * dist);
      const ny = Math.round(cy + Math.sin(angle) * dist);
      if (!this.inBounds(nx, ny)) continue;
      const ni = ny * W + nx;
      if (grid[ni] === M.EMPTY || grid[ni] === M.FIRE || isGas(grid[ni])) {
        grid[ni] = M.NAPALM;
        life[ni] = Math.floor(Math.random() * 30);
        temp[ni] = 600;
      }
    }

    for (let d = 0; d < 8; d++) {
      const dx = Math.round((Math.random() - 0.5) * radius * 2);
      const dy = -Math.floor(Math.random() * 6) - 1;
      const nx = cx + dx, ny = cy + dy;
      if (this.inBounds(nx, ny) && grid[ny * W + nx] === M.EMPTY) {
        const ni = ny * W + nx;
        grid[ni] = M.NAPALM;
        life[ni] = Math.floor(Math.random() * 20);
        temp[ni] = 550;
      }
    }
  }

  /** Fire a railgun trace through the grid at column x */
  fireRailgun(targetX) {
    const { W, H, grid, temp, life } = this;
    const dx = targetX;

    for (let y = 0; y < H; y++) {
      if (!this.inBounds(dx, y)) continue;
      const pi = y * W + dx;
      const mat = grid[pi];
      if (mat !== M.EMPTY) {
        grid[pi] = (mat === M.GLASS && Math.random() < 0.5) ? M.SAND : M.EMPTY;
        temp[pi] = 300 + Math.random() * 200;
        life[pi] = 0;
      }
    }

    const shockRadius = 4;
    for (let y = 0; y < H; y++) {
      for (let sx = -shockRadius; sx <= shockRadius; sx++) {
        if (sx === 0) continue;
        const nx = dx + sx;
        if (!this.inBounds(nx, y)) continue;
        const ni = y * W + nx;
        const absDist = Math.abs(sx);
        const forceFrac = 1 - absDist / shockRadius;

        temp[ni] += forceFrac * 150;
        const mat = grid[ni];
        if (mat === M.EMPTY) continue;

        if (mat === M.GLASS && absDist <= 2) {
          grid[ni] = Math.random() < 0.6 ? M.SAND : M.EMPTY;
          continue;
        }

        if (isStructural(mat)) {
          if (Math.random() < forceFrac * 0.15) {
            grid[ni] = mat === M.GLASS ? M.EMPTY : (Math.random() < 0.4 ? M.SAND : M.EMPTY);
          }
          continue;
        }

        if (!isSolid(mat) && !isGas(mat)) {
          const flingDir = sx > 0 ? 1 : -1;
          const flingDist = Math.ceil(forceFrac * 3);
          for (let f = flingDist; f >= 1; f--) {
            const fx = nx + flingDir * f;
            if (this.inBounds(fx, y) && grid[y * W + fx] === M.EMPTY) {
              this._swap(nx, y, fx, y);
              break;
            }
          }
        }
      }
    }

    for (let y = 0; y < H; y += 2 + Math.floor(Math.random() * 3)) {
      if (this.inBounds(dx, y) && grid[y * W + dx] === M.EMPTY) {
        grid[y * W + dx] = M.EMBER; life[y * W + dx] = 0; temp[y * W + dx] = 400;
      }
      for (const side of [-1, 1]) {
        if (Math.random() < 0.2 && this.inBounds(dx + side, y) && grid[y * W + dx + side] === M.EMPTY) {
          grid[y * W + dx + side] = M.EMBER; life[y * W + dx + side] = 0; temp[y * W + dx + side] = 300;
        }
      }
    }

    this.integrityDirty = true;
    this.integrityTimer = 2;
  }

  // ==================== STRUCTURAL INTEGRITY ====================

  _checkIntegrity() {
    const { W, H, grid, temp, life, supportBuf, groupBuf, ROOM_TEMP: RT } = this;
    supportBuf.fill(0);

    const queue = [];
    for (let x = 0; x < W; x++) {
      const i = (H - 1) * W + x;
      if (isStructural(grid[i])) {
        supportBuf[i] = 1;
        queue.push(i);
      }
    }

    let head = 0;
    while (head < queue.length) {
      const ci = queue[head++];
      const cx = ci % W, cy = (ci / W) | 0;
      for (const [dx, dy] of NEIGHBOR8) {
        const nx = cx + dx, ny = cy + dy;
        if (!this.inBounds(nx, ny)) continue;
        const ni = ny * W + nx;
        if (!supportBuf[ni] && isStructural(grid[ni])) {
          supportBuf[ni] = 1;
          queue.push(ni);
        }
      }
    }

    groupBuf.fill(0);
    const unsupported = [];
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        if (isStructural(grid[i]) && !supportBuf[i]) {
          groupBuf[i] = 1;
          unsupported.push(i);
        }
      }
    }

    if (unsupported.length > 0) {
      for (const startI of unsupported) {
        if (groupBuf[startI] !== 1) continue;
        const component = [];
        const cq = [startI];
        groupBuf[startI] = 2;
        let ch = 0;
        while (ch < cq.length) {
          const ci = cq[ch++];
          component.push(ci);
          const cx = ci % W, cy = (ci / W) | 0;
          for (const [dx, dy] of NEIGHBOR8) {
            const nx = cx + dx, ny = cy + dy;
            if (!this.inBounds(nx, ny)) continue;
            const ni = ny * W + nx;
            if (groupBuf[ni] === 1) {
              groupBuf[ni] = 2;
              cq.push(ni);
            }
          }
        }
        if (component.length > 0) this._createChunk(component);
      }
    }

    this._checkStress();
  }

  _checkStress() {
    const { W, H, grid, temp, life, supportBuf, weightBuf, ROOM_TEMP: RT } = this;
    const STRESS_LIMIT = 18;
    weightBuf.fill(0);

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        if (!isStructural(grid[i]) || !supportBuf[i]) continue;
        weightBuf[i] += 1;

        if (y > 0) {
          const above = grid[(y - 1) * W + x];
          if (above !== M.EMPTY && !isStructural(above) && !isGas(above) && above !== M.FIRE) {
            weightBuf[i] += 0.5;
          }
        }

        const below = [];
        for (const [dx, dy] of [[-1, 1], [0, 1], [1, 1]]) {
          const nx = x + dx, ny = y + dy;
          if (this.inBounds(nx, ny)) {
            const ni = ny * W + nx;
            if (isStructural(grid[ni]) && supportBuf[ni]) below.push(ni);
          }
        }

        if (below.length > 0) {
          const share = weightBuf[i] / below.length;
          for (const bi of below) weightBuf[bi] += share;
        }
      }
    }

    let broken = false;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        if (!isStructural(grid[i]) || !supportBuf[i]) continue;
        const stress = weightBuf[i] / STRENGTH[grid[i]];
        if (stress > STRESS_LIMIT) {
          const excessRatio = (stress - STRESS_LIMIT) / STRESS_LIMIT;
          if (Math.random() < Math.min(0.5, excessRatio * 0.12)) {
            const mat = grid[i];
            grid[i] = mat === M.GLASS ? M.SAND : (Math.random() < 0.3 ? M.SAND : M.EMPTY);
            temp[i] = RT; life[i] = 0;
            broken = true;
          }
        }
      }
    }

    if (broken) {
      this.integrityDirty = true;
      this.integrityTimer = Math.max(this.integrityTimer, 2);
    }
  }

  // ==================== FALLING CHUNKS ====================

  _createChunk(pixelIndices) {
    const { W, H, grid, temp, life, supportBuf, ROOM_TEMP: RT } = this;
    let sumX = 0, sumY = 0, minY = H, maxY = 0;

    for (const i of pixelIndices) {
      const x = i % W, y = (i / W) | 0;
      sumX += x; sumY += y;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const anchorX = Math.round(sumX / pixelIndices.length);
    const anchorY = Math.round(sumY / pixelIndices.length);

    let leftSupport = 0, rightSupport = 0;
    for (const i of pixelIndices) {
      const x = i % W, y = (i / W) | 0;
      if (y !== maxY) continue;
      if (this.inBounds(x - 1, y + 1) && isStructural(grid[(y+1)*W+x-1]) && supportBuf[(y+1)*W+x-1]) leftSupport++;
      if (this.inBounds(x + 1, y + 1) && isStructural(grid[(y+1)*W+x+1]) && supportBuf[(y+1)*W+x+1]) rightSupport++;
    }

    let vx = 0;
    if (leftSupport !== rightSupport) {
      vx = leftSupport < rightSupport ? -0.3 : 0.3;
    } else {
      vx = (Math.random() - 0.5) * 0.2;
    }
    const height = maxY - minY + 1;
    if (height > 8) vx *= 1 + (height - 8) * 0.05;

    const chunkPixels = [];
    for (const i of pixelIndices) {
      const x = i % W, y = (i / W) | 0;
      chunkPixels.push({ rx: x - anchorX, ry: y - anchorY, mat: grid[i], temp: temp[i], life: life[i] });
      grid[i] = M.EMPTY; temp[i] = RT; life[i] = 0;
    }

    this.chunks.push({ x: anchorX, y: anchorY, vx, vy: 0, fracX: 0, fracY: 0, pixels: chunkPixels });
  }

  _updateChunks() {
    const { W, H, grid, temp, life } = this;
    const GRAVITY = 0.15;
    const settled = [];

    for (let ci = 0; ci < this.chunks.length; ci++) {
      const chunk = this.chunks[ci];
      chunk.vy += GRAVITY;
      if (chunk.vy > 6) chunk.vy = 6;
      chunk.fracY += chunk.vy;
      chunk.fracX += chunk.vx;

      // Horizontal movement
      const hSteps = Math.floor(Math.abs(chunk.fracX));
      const hDir = chunk.vx > 0 ? 1 : -1;
      chunk.fracX -= hSteps * hDir;
      for (let s = 0; s < hSteps; s++) {
        let hBlocked = false;
        for (const p of chunk.pixels) {
          const wx = chunk.x + p.rx + hDir, wy = chunk.y + p.ry;
          if (wx < 0 || wx >= W) { hBlocked = true; break; }
          if (wy < 0 || wy >= H) continue;
          const m = grid[wy * W + wx];
          if (isSolid(m) || isPowder(m)) { hBlocked = true; break; }
        }
        if (hBlocked) { chunk.vx *= -0.3; break; }
        chunk.x += hDir;
      }

      // Vertical movement
      const steps = Math.floor(chunk.fracY);
      chunk.fracY -= steps;
      let collided = false;

      for (let s = 0; s < steps; s++) {
        let blocked = false;
        for (const p of chunk.pixels) {
          const wy = chunk.y + p.ry + 1, wx = chunk.x + p.rx;
          if (wy >= H) { blocked = true; break; }
          if (wx < 0 || wx >= W) continue;
          const gi = wy * W + wx;
          const m = grid[gi];
          if (m !== M.EMPTY) {
            if (isSolid(m) || isPowder(m)) { blocked = true; break; }
            else { grid[gi] = M.EMPTY; temp[gi] = this.ROOM_TEMP; life[gi] = 0; }
          }
        }
        if (blocked) { collided = true; break; }
        chunk.y += 1;
      }

      if (collided) {
        this._shatterChunk(chunk);
        settled.push(ci);
      }
    }

    for (let i = settled.length - 1; i >= 0; i--) {
      this.chunks.splice(settled[i], 1);
    }
  }

  _shatterChunk(chunk) {
    const { W, H, grid, temp, life, ROOM_TEMP: RT } = this;
    const impactSpeed = chunk.vy;
    let maxY = -Infinity, minY = Infinity;

    for (const p of chunk.pixels) {
      const wy = chunk.y + p.ry;
      if (wy > maxY) maxY = wy;
      if (wy < minY) minY = wy;
    }

    for (const p of chunk.pixels) {
      const wx = chunk.x + p.rx, wy = chunk.y + p.ry;
      if (!this.inBounds(wx, wy)) continue;
      const gi = wy * W + wx;

      let placeMat = p.mat, placeTemp = p.temp;
      if (p.mat === M.GLASS && impactSpeed > 2) {
        placeMat = Math.random() < 0.35 ? M.SAND : M.EMPTY;
        placeTemp = p.temp + impactSpeed * 5;
      }
      if (placeMat === M.EMPTY) continue;

      if (grid[gi] === M.EMPTY) {
        grid[gi] = placeMat; temp[gi] = placeTemp; life[gi] = p.life;
      } else {
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, -1], [1, -1], [-1, -1], [0, 1]]) {
          const ax = wx + dx, ay = wy + dy;
          if (this.inBounds(ax, ay) && grid[ay * W + ax] === M.EMPTY) {
            const ai = ay * W + ax;
            grid[ai] = placeMat; temp[ai] = placeTemp; life[ai] = p.life;
            break;
          }
        }
      }
    }

    if (impactSpeed > 1.5) {
      const impactX = chunk.x, impactY = maxY + 1;
      const effectRadius = Math.min(6, Math.floor(impactSpeed));

      for (let dy = -effectRadius; dy <= effectRadius; dy++) {
        for (let dx = -effectRadius; dx <= effectRadius; dx++) {
          const d2 = dx * dx + dy * dy;
          if (d2 > effectRadius * effectRadius) continue;
          const fx = impactX + dx, fy = impactY + dy;
          if (!this.inBounds(fx, fy)) continue;
          const fi = fy * W + fx;
          const forceFrac = 1 - Math.sqrt(d2) / effectRadius;

          temp[fi] += forceFrac * impactSpeed * 8;

          if (grid[fi] === M.GLASS && forceFrac > 0.3) {
            grid[fi] = Math.random() < 0.5 ? M.SAND : M.EMPTY;
          }
          if (grid[fi] === M.EMPTY && Math.random() < forceFrac * 0.15) {
            grid[fi] = Math.random() < 0.4 ? M.FIRE : M.EMBER;
            life[fi] = 0; temp[fi] = 300;
          }
        }
      }

      if (impactSpeed > 3 && chunk.pixels.length > 20) {
        for (let dx = -3; dx <= 3; dx++) {
          const bx = impactX + dx, by = impactY;
          if (!this.inBounds(bx, by)) continue;
          const bi = by * W + bx;
          if (isStructural(grid[bi]) && STRENGTH[grid[bi]] < impactSpeed * 0.8) {
            if (Math.random() < 0.3) {
              grid[bi] = grid[bi] === M.GLASS ? M.EMPTY : M.SAND;
              temp[bi] = 100;
            }
          }
        }
      }
    }

    this.integrityDirty = true;
    this.integrityTimer = Math.max(this.integrityTimer, 4);
  }

  // ==================== NAPALM ====================

  _simNapalm(x, y) {
    const { W, grid, temp, life, ROOM_TEMP: RT } = this;
    const i = y * W + x;
    life[i]++;

    temp[i] = Math.max(temp[i], 500 + Math.random() * 200);

    if (life[i] > 900 + Math.random() * 600) {
      grid[i] = Math.random() < 0.3 ? M.SMOKE : M.EMPTY;
      life[i] = 0;
      temp[i] = Math.random() < 0.3 ? 200 : RT;
      return;
    }

    for (const [dx, dy] of NEIGHBOR4) {
      const nx = x + dx, ny = y + dy;
      if (!this.inBounds(nx, ny)) continue;
      const ni = ny * W + nx;
      const neighbor = grid[ni];
      if (neighbor === M.WOOD || neighbor === M.OIL || neighbor === M.PLANT) temp[ni] = Math.max(temp[ni], 350);
      if (neighbor === M.GUNPOWDER) temp[ni] = Math.max(temp[ni], 120);
      if (neighbor === M.GAS) { grid[ni] = M.FIRE; life[ni] = 0; temp[ni] = 400; }
      if (isStructural(neighbor)) temp[ni] += 3;
      if (neighbor === M.WATER) { grid[ni] = M.STEAM; life[ni] = 0; temp[ni] = 110; life[i] += 20; }
    }

    if (Math.random() < 0.06 && this.inBounds(x, y - 1) && this.get(x, y - 1) === M.EMPTY) {
      this.grid[(y-1)*W+x] = M.SMOKE; this.life[(y-1)*W+x] = 0; this.temp[(y-1)*W+x] = 200;
    }

    if (Math.random() < 0.08) {
      const fx = x + (Math.random() > 0.5 ? 1 : -1);
      if (this.inBounds(fx, y - 1) && this.get(fx, y - 1) === M.EMPTY) {
        const fi = (y-1) * W + fx;
        grid[fi] = M.FIRE; life[fi] = 10; temp[fi] = 450;
      }
    }

    if (Math.random() > 0.3) return;

    let touchingSolid = false;
    for (const [dx, dy] of NEIGHBOR4) {
      const neighbor = this.get(x + dx, y + dy);
      if (isSolid(neighbor) || isPowder(neighbor)) { touchingSolid = true; break; }
    }

    const below = this.get(x, y + 1);
    if (below === M.EMPTY) {
      if (!touchingSolid || Math.random() < 0.4) this._swap(x, y, x, y + 1);
    } else if (below === M.WATER) {
      this._swap(x, y, x, y + 1);
    } else {
      const dir = Math.random() > 0.5 ? 1 : -1;
      if (this.get(x + dir, y + 1) === M.EMPTY) this._swap(x, y, x + dir, y + 1);
      else if (this.get(x - dir, y + 1) === M.EMPTY) this._swap(x, y, x - dir, y + 1);
      else if (this.get(x + dir, y) === M.EMPTY && Math.random() < 0.15) this._swap(x, y, x + dir, y);
    }
  }

  // ==================== TRANSITIONS ====================

  _checkTransitions() {
    const { W, H, grid, temp, life } = this;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const mat = grid[i];
        const t = temp[i];

        switch (mat) {
          case M.WATER:
            if (t >= 100) { grid[i] = M.STEAM; life[i] = 0; }
            else if (t <= -2) { grid[i] = M.ICE; }
            break;
          case M.ICE:
            if (t > 5) { grid[i] = M.WATER; }
            break;
          case M.STEAM:
            if (t < 85) { grid[i] = M.WATER; life[i] = 0; }
            break;
          case M.LAVA:
            if (t < 400) { grid[i] = M.STONE; temp[i] = 350; }
            break;
          case M.STONE:
            if (t > 900 && Math.random() < 0.01) { grid[i] = M.LAVA; }
            break;
          case M.METAL:
            if (t > 650 && Math.random() < 0.01) {
              grid[i] = M.LAVA; temp[i] = 700;
              this.integrityDirty = true; this.integrityTimer = Math.max(this.integrityTimer, 3);
            }
            break;
          case M.WOOD:
            if (t > 300) {
              grid[i] = M.FIRE; life[i] = 0;
              this.integrityDirty = true; this.integrityTimer = Math.max(this.integrityTimer, 3);
            }
            break;
          case M.OIL:
            if (t > 150) { grid[i] = M.FIRE; life[i] = 0; }
            break;
          case M.GUNPOWDER:
            if (t > 100) {
              grid[i] = M.FIRE; life[i] = 0; temp[i] = 600;
              this.explode(x, y, 4);
            }
            break;
          case M.PLANT:
            if (t > 250) { grid[i] = M.FIRE; life[i] = 0; }
            break;
          case M.GAS:
            if (t > 50) {
              grid[i] = M.FIRE; life[i] = 0; temp[i] = Math.max(t, 400);
              for (let dy2 = -2; dy2 <= 2; dy2++) {
                for (let dx2 = -2; dx2 <= 2; dx2++) {
                  if (dx2 === 0 && dy2 === 0) continue;
                  const gx = x + dx2, gy = y + dy2;
                  if (this.inBounds(gx, gy) && grid[gy * W + gx] === M.GAS) {
                    temp[gy * W + gx] = Math.max(temp[gy * W + gx], 60);
                  }
                }
              }
              for (const [dx, dy] of NEIGHBOR4) {
                const px = x + dx, py = y + dy;
                if (!this.inBounds(px, py)) continue;
                const pm = this.get(px, py);
                if (pm !== M.EMPTY && !isGas(pm) && pm !== M.CLONE && pm !== M.VOID && !isSolid(pm) && Math.random() < 0.3) {
                  const fx = px + dx, fy = py + dy;
                  if (this.inBounds(fx, fy) && this.get(fx, fy) === M.EMPTY) this._swap(px, py, fx, fy);
                }
              }
            }
            break;
          case M.GLASS:
            if (t > 600 && Math.random() < 0.02) {
              grid[i] = M.LAVA; temp[i] = 650;
              this.integrityDirty = true; this.integrityTimer = Math.max(this.integrityTimer, 3);
            }
            break;
          case M.BRICK:
            if (t > 800 && Math.random() < 0.005) {
              grid[i] = M.SAND;
              this.integrityDirty = true; this.integrityTimer = Math.max(this.integrityTimer, 3);
            }
            break;
          case M.CONCRETE:
            if (t > 1000 && Math.random() < 0.003) {
              grid[i] = M.SAND;
              this.integrityDirty = true; this.integrityTimer = Math.max(this.integrityTimer, 3);
            }
            break;
        }
      }
    }
  }

  // ==================== MATERIAL BEHAVIORS ====================

  _simPowder(x, y, mat) {
    if (canDisplace(mat, this.get(x, y + 1))) {
      this._swap(x, y, x, y + 1);
    } else {
      const dir = Math.random() > 0.5 ? 1 : -1;
      if (canDisplace(mat, this.get(x + dir, y + 1))) this._swap(x, y, x + dir, y + 1);
      else if (canDisplace(mat, this.get(x - dir, y + 1))) this._swap(x, y, x - dir, y + 1);
    }

    if (mat === M.SALT) {
      for (const [dx, dy] of NEIGHBOR4) {
        if (this.get(x + dx, y + dy) === M.WATER && Math.random() < 0.05) {
          this.grid[y * this.W + x] = M.EMPTY;
          this.temp[y * this.W + x] = this.ROOM_TEMP;
          return;
        }
      }
    }
  }

  _simLiquid(x, y, mat) {
    if (canDisplace(mat, this.get(x, y + 1))) { this._swap(x, y, x, y + 1); return; }
    const dir = Math.random() > 0.5 ? 1 : -1;
    if (canDisplace(mat, this.get(x + dir, y + 1))) this._swap(x, y, x + dir, y + 1);
    else if (canDisplace(mat, this.get(x - dir, y + 1))) this._swap(x, y, x - dir, y + 1);
    else if (this.get(x + dir, y) === M.EMPTY) this._swap(x, y, x + dir, y);
    else if (this.get(x - dir, y) === M.EMPTY) this._swap(x, y, x - dir, y);
  }

  _simAcid(x, y) {
    const { W, grid, temp, life, ROOM_TEMP: RT } = this;
    const i = y * W + x;

    for (const [dx, dy] of NEIGHBOR4) {
      const nx = x + dx, ny = y + dy;
      if (!this.inBounds(nx, ny)) continue;
      const ni = ny * W + nx;
      const neighbor = grid[ni];
      if (neighbor === M.EMPTY || neighbor === M.ACID || neighbor === M.STONE || neighbor === M.LAVA || neighbor === M.CLONE || neighbor === M.VOID) continue;

      if (neighbor === M.WATER) {
        if (Math.random() < 0.08) { grid[ni] = M.EMPTY; temp[ni] = RT; grid[i] = M.EMPTY; temp[i] = RT; return; }
        continue;
      }
      if (neighbor === M.METAL) {
        if (Math.random() < 0.02) { grid[ni] = M.GAS; life[ni] = 0; temp[ni] = 30; grid[i] = M.SMOKE; life[i] = 0; return; }
        continue;
      }
      if (neighbor === M.GLASS) {
        if (Math.random() < 0.05) {
          grid[ni] = M.EMPTY; temp[ni] = RT;
          this.integrityDirty = true; this.integrityTimer = Math.max(this.integrityTimer, 5);
          if (Math.random() < 0.3) { grid[i] = M.SMOKE; life[i] = 0; return; }
        }
        continue;
      }
      if (neighbor === M.BRICK) {
        if (Math.random() < 0.015) {
          grid[ni] = M.SAND; temp[ni] = RT;
          this.integrityDirty = true; this.integrityTimer = Math.max(this.integrityTimer, 5);
          if (Math.random() < 0.4) { grid[i] = M.SMOKE; life[i] = 0; return; }
        }
        continue;
      }
      if (neighbor === M.CONCRETE) {
        if (Math.random() < 0.008) {
          grid[ni] = M.SAND; temp[ni] = RT;
          this.integrityDirty = true; this.integrityTimer = Math.max(this.integrityTimer, 5);
          if (Math.random() < 0.5) { grid[i] = M.SMOKE; life[i] = 0; return; }
        }
        continue;
      }

      if (Math.random() < 0.03) {
        grid[ni] = M.EMPTY; temp[ni] = RT;
        if (Math.random() < 0.4) {
          grid[i] = Math.random() < 0.5 ? M.SMOKE : M.EMPTY;
          life[i] = 0; return;
        }
      }
    }

    if (canDisplace(M.ACID, this.get(x, y + 1))) { this._swap(x, y, x, y + 1); return; }
    const dir = Math.random() > 0.5 ? 1 : -1;
    if (this.get(x + dir, y + 1) === M.EMPTY) this._swap(x, y, x + dir, y + 1);
    else if (this.get(x + dir, y) === M.EMPTY) this._swap(x, y, x + dir, y);
    else if (this.get(x - dir, y) === M.EMPTY) this._swap(x, y, x - dir, y);
  }

  _simLava(x, y) {
    const { W, grid, temp, life } = this;
    const i = y * W + x;

    for (const [dx, dy] of NEIGHBOR4) {
      const nx = x + dx, ny = y + dy;
      if (!this.inBounds(nx, ny)) continue;
      const ni = ny * W + nx;
      const neighbor = grid[ni];
      if (neighbor === M.WATER) {
        grid[ni] = M.STEAM; life[ni] = 0; temp[ni] = 110; temp[i] -= 100;
        if (temp[i] < 400) { grid[i] = M.STONE; temp[i] = 350; return; }
      } else if (neighbor === M.ICE) {
        grid[ni] = M.WATER; temp[ni] = 20; temp[i] -= 50;
      } else if (neighbor === M.ACID) {
        grid[ni] = M.STEAM; life[ni] = 0; temp[ni] = 110; temp[i] -= 30;
      }
    }

    if (Math.random() < 0.3) return;
    if (canDisplace(M.LAVA, this.get(x, y + 1))) { this._swap(x, y, x, y + 1); return; }
    if (Math.random() < 0.3) {
      const dir = Math.random() > 0.5 ? 1 : -1;
      if (this.get(x + dir, y + 1) === M.EMPTY) this._swap(x, y, x + dir, y + 1);
      else if (this.get(x + dir, y) === M.EMPTY) this._swap(x, y, x + dir, y);
      else if (this.get(x - dir, y) === M.EMPTY) this._swap(x, y, x - dir, y);
    }
  }

  _simFire(x, y) {
    const { W, grid, temp, life } = this;
    const i = y * W + x;
    life[i]++;

    if (life[i] > 20 + Math.random() * 30) {
      const r = Math.random();
      if (r < 0.35) { grid[i] = M.SMOKE; life[i] = 0; }
      else if (r < 0.45) { grid[i] = M.EMBER; life[i] = 0; temp[i] = 350; }
      else { grid[i] = M.EMPTY; life[i] = 0; temp[i] = this.ROOM_TEMP; }
      return;
    }

    const dirs6 = [[0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [1, -1]];
    for (const [dx, dy] of dirs6) {
      const nx = x + dx, ny = y + dy;
      if (!this.inBounds(nx, ny)) continue;
      const ni = ny * W + nx;
      const neighbor = grid[ni];
      if (neighbor === M.WATER) { grid[i] = M.STEAM; life[i] = 0; temp[i] = 100; grid[ni] = M.STEAM; life[ni] = 0; temp[ni] = 100; return; }
      if (neighbor === M.GAS) { grid[ni] = M.FIRE; life[ni] = 0; temp[ni] = 400; }
      if (neighbor === M.WOOD || neighbor === M.OIL || neighbor === M.PLANT || neighbor === M.GUNPOWDER) temp[ni] += 2;
      if (dy === -1 && neighbor === M.EMPTY && Math.random() < 0.06) { grid[ni] = M.SMOKE; life[ni] = 0; }
    }

    if (Math.random() < 0.02) {
      const edx = Math.random() > 0.5 ? 1 : -1;
      if (this.inBounds(x + edx, y - 1) && this.get(x + edx, y - 1) === M.EMPTY) {
        const ei = (y-1) * W + (x + edx);
        grid[ei] = M.EMBER; life[ei] = 0; temp[ei] = 350;
      }
    }

    if (Math.random() < 0.35 && this.get(x, y - 1) === M.EMPTY) {
      this._swap(x, y, x, y - 1);
    } else if (Math.random() < 0.15) {
      const dir = Math.random() > 0.5 ? 1 : -1;
      if (this.get(x + dir, y - 1) === M.EMPTY) this._swap(x, y, x + dir, y - 1);
    }
  }

  _simGas(x, y, mat, maxLife) {
    const { W, grid, life } = this;
    const i = y * W + x;
    life[i]++;

    if (life[i] > maxLife + Math.random() * maxLife * 0.5) {
      grid[i] = M.EMPTY; life[i] = 0; return;
    }

    if (mat === M.STEAM && isLiquid(this.get(x, y - 1)) && Math.random() < 0.3) {
      this._swap(x, y, x, y - 1); return;
    }

    if (mat === M.SMOKE) {
      for (const [dx, dy] of NEIGHBOR4) {
        if (this.get(x + dx, y + dy) === M.WATER) { grid[i] = M.EMPTY; life[i] = 0; return; }
      }
    }

    if (this.get(x, y - 1) === M.EMPTY) {
      if (Math.random() < 0.7) this._swap(x, y, x, y - 1);
    } else {
      const dir = Math.random() > 0.5 ? 1 : -1;
      if (this.get(x + dir, y - 1) === M.EMPTY) this._swap(x, y, x + dir, y - 1);
      else if (this.get(x + dir, y) === M.EMPTY && Math.random() < 0.4) this._swap(x, y, x + dir, y);
      else if (this.get(x - dir, y) === M.EMPTY && Math.random() < 0.4) this._swap(x, y, x - dir, y);
    }
  }

  _simGasFlammable(x, y) {
    const { W, grid, temp, life } = this;
    const i = y * W + x;
    life[i]++;

    if (life[i] > 150 + Math.random() * 80) { grid[i] = M.EMPTY; life[i] = 0; return; }

    for (const [dx, dy] of NEIGHBOR4) {
      const nx = x + dx, ny = y + dy;
      if (!this.inBounds(nx, ny)) continue;
      const neighbor = this.get(nx, ny);
      if (neighbor === M.FIRE || neighbor === M.EMBER || neighbor === M.LAVA) {
        grid[i] = M.FIRE; life[i] = 0; temp[i] = 400; return;
      }
    }

    if (this.get(x, y - 1) === M.EMPTY) {
      if (Math.random() < 0.5) this._swap(x, y, x, y - 1);
    } else {
      const dir = Math.random() > 0.5 ? 1 : -1;
      if (this.get(x + dir, y) === M.EMPTY && Math.random() < 0.5) this._swap(x, y, x + dir, y);
      else if (this.get(x - dir, y) === M.EMPTY && Math.random() < 0.5) this._swap(x, y, x - dir, y);
      else if (this.get(x + dir, y - 1) === M.EMPTY && Math.random() < 0.3) this._swap(x, y, x + dir, y - 1);
    }
  }

  _simEmber(x, y) {
    const { W, grid, temp, life } = this;
    const i = y * W + x;
    life[i]++;

    if (life[i] > 10 + Math.random() * 15) {
      grid[i] = M.EMPTY; life[i] = 0; temp[i] = this.ROOM_TEMP; return;
    }

    for (const [dx, dy] of NEIGHBOR4) {
      const nx = x + dx, ny = y + dy;
      if (!this.inBounds(nx, ny)) continue;
      const ni = ny * W + nx;
      const neighbor = grid[ni];
      if (neighbor === M.WATER) { grid[i] = M.STEAM; life[i] = 0; temp[i] = 100; return; }
      if (neighbor === M.WOOD || neighbor === M.OIL || neighbor === M.PLANT) temp[ni] = Math.max(temp[ni], 250);
      if (neighbor === M.GUNPOWDER) temp[ni] = Math.max(temp[ni], 110);
      if (neighbor === M.GAS) { grid[ni] = M.FIRE; life[ni] = 0; temp[ni] = 400; }
    }

    const r = Math.random();
    if (r < 0.4) {
      const dir = Math.random() > 0.5 ? 1 : -1;
      if (this.get(x, y - 1) === M.EMPTY) this._swap(x, y, x, y - 1);
      else if (this.get(x + dir, y - 1) === M.EMPTY) this._swap(x, y, x + dir, y - 1);
    } else if (r < 0.65) {
      const dir = Math.random() > 0.5 ? 1 : -1;
      if (this.get(x + dir, y) === M.EMPTY) this._swap(x, y, x + dir, y);
    } else {
      if (this.get(x, y + 1) === M.EMPTY) this._swap(x, y, x, y + 1);
    }
  }

  _simPlant(x, y) {
    const { W, grid, temp } = this;
    let waterNear = false, waterCount = 0;

    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        if (this.get(x + dx, y + dy) === M.WATER) { waterNear = true; waterCount++; }
      }
    }
    if (!waterNear) {
      for (let dx = -1; dx <= 1; dx++) {
        if (this.get(x + dx, y - 2) === M.WATER || this.get(x + dx, y + 2) === M.WATER) waterNear = true;
      }
    }

    const growRate = waterNear ? 0.008 + waterCount * 0.004 : 0.001;
    if (Math.random() < growRate) {
      const growDirs = waterNear ? [[0, -1], [-1, -1], [1, -1], [-1, 0], [1, 0], [0, 1]] : [[0, 1], [-1, 0], [1, 0]];
      for (const [dx, dy] of growDirs) {
        const nx = x + dx, ny = y + dy;
        if (this.inBounds(nx, ny) && this.get(nx, ny) === M.EMPTY) {
          grid[ny * W + nx] = M.PLANT; this.life[ny * W + nx] = 0; temp[ny * W + nx] = temp[y * W + x];
          for (const [wx, wy] of NEIGHBOR4) {
            if (this.get(x + wx, y + wy) === M.WATER && Math.random() < 0.3) {
              const wi = (y + wy) * W + (x + wx);
              grid[wi] = M.EMPTY; temp[wi] = this.ROOM_TEMP;
              break;
            }
          }
          return;
        }
      }
    }
  }

  _simClone(x, y) {
    const { W, grid, temp, life } = this;
    let sourceMat = M.EMPTY, sourceTemp = this.ROOM_TEMP;

    for (const [dx, dy] of [[0, -1], [-1, 0], [1, 0], [0, 1]]) {
      const nx = x + dx, ny = y + dy;
      if (!this.inBounds(nx, ny)) continue;
      const m = this.get(nx, ny);
      if (m !== M.EMPTY && m !== M.CLONE && m !== M.VOID) {
        sourceMat = m; sourceTemp = temp[(ny) * W + nx]; break;
      }
    }

    if (sourceMat === M.EMPTY) return;

    for (const [dx, dy] of [[0, 1], [-1, 0], [1, 0], [0, -1]]) {
      const nx = x + dx, ny = y + dy;
      if (!this.inBounds(nx, ny) || this.get(nx, ny) !== M.EMPTY) continue;
      if (Math.random() < 0.1) {
        const ni = ny * W + nx;
        grid[ni] = sourceMat; life[ni] = 0;
        if (sourceMat === M.FIRE) temp[ni] = 500;
        else if (sourceMat === M.LAVA) temp[ni] = 800;
        else if (sourceMat === M.ICE) temp[ni] = -10;
        else if (sourceMat === M.EMBER) temp[ni] = 350;
        else temp[ni] = sourceTemp;
        return;
      }
    }
  }

  _simVoid(x, y) {
    const { W, grid, temp, life, ROOM_TEMP: RT } = this;
    for (const [dx, dy] of NEIGHBOR4) {
      const nx = x + dx, ny = y + dy;
      if (!this.inBounds(nx, ny)) continue;
      const neighbor = this.get(nx, ny);
      if (neighbor !== M.EMPTY && neighbor !== M.VOID && neighbor !== M.CLONE) {
        const ni = ny * W + nx;
        grid[ni] = M.EMPTY; temp[ni] = RT; life[ni] = 0;
      }
    }
  }

  // ==================== MAIN SIMULATION STEP ====================

  step() {
    const { W, H, grid, updated } = this;
    updated.fill(0);
    this._simulateThermal();
    this._checkTransitions();

    const leftToRight = Math.random() > 0.5;
    for (let y = H - 1; y >= 0; y--) {
      const startX = leftToRight ? 0 : W - 1;
      const endX = leftToRight ? W : -1;
      const stepX = leftToRight ? 1 : -1;

      for (let x = startX; x !== endX; x += stepX) {
        const i = y * W + x;
        if (updated[i]) continue;
        const mat = grid[i];
        if (mat === M.EMPTY) continue;

        if (isPowder(mat)) this._simPowder(x, y, mat);
        else if (mat === M.ACID) this._simAcid(x, y);
        else if (mat === M.LAVA) this._simLava(x, y);
        else if (isLiquid(mat)) this._simLiquid(x, y, mat);
        else if (mat === M.FIRE) this._simFire(x, y);
        else if (mat === M.STEAM) this._simGas(x, y, M.STEAM, 80);
        else if (mat === M.SMOKE) this._simGas(x, y, M.SMOKE, 60);
        else if (mat === M.GAS) this._simGasFlammable(x, y);
        else if (mat === M.EMBER) this._simEmber(x, y);
        else if (mat === M.PLANT) this._simPlant(x, y);
        else if (mat === M.NAPALM) this._simNapalm(x, y);
        else if (mat === M.CLONE) this._simClone(x, y);
        else if (mat === M.VOID) this._simVoid(x, y);
      }
    }

    this._updateChunks();

    if (this.integrityDirty) {
      if (this.integrityTimer > 0) {
        this.integrityTimer--;
      } else {
        this.integrityDirty = false;
        this._checkIntegrity();
      }
    }
  }

  // ==================== CITY GENERATOR ====================

  generateCity() {
    this.clear();
    const { W, H, grid, temp, ROOM_TEMP: RT } = this;
    const gnd = H - 3;

    // Ground layer
    for (let x = 0; x < W; x++) {
      for (let y = gnd; y < H; y++) {
        grid[y * W + x] = M.STONE; temp[y * W + x] = RT;
      }
    }

    const buildings = [];
    let bx = 4 + (Math.random() * 8 | 0);

    while (bx < W - 10) {
      const bw = 14 + (Math.random() * 22 | 0);
      let bh = 25 + (Math.random() * 70 | 0);
      if (bh > gnd - 8) bh = gnd - 8;
      const top = gnd - bh;
      const wallMat = Math.random() < 0.45 ? M.BRICK : M.CONCRETE;
      const floorSpace = 7 + (Math.random() * 4 | 0);

      const _set = (x, y, m) => {
        if (this.inBounds(x, y)) { grid[y * W + x] = m; temp[y * W + x] = RT; }
      };

      // Outer walls (2px thick)
      for (let y = top; y < gnd; y++) {
        _set(bx, y, wallMat); _set(bx + 1, y, wallMat);
        _set(bx + bw - 1, y, wallMat); _set(bx + bw - 2, y, wallMat);
      }

      // Roof
      const roofStyle = Math.random();
      if (roofStyle < 0.3 && bw > 16) {
        const midX = bx + (bw / 2 | 0);
        const peakH = 4 + (Math.random() * 4 | 0);
        for (let rx = bx; rx < bx + bw; rx++) {
          const distFromCenter = Math.abs(rx - midX);
          const roofY = top - Math.round(peakH * (1 - distFromCenter / (bw / 2)));
          if (roofY >= 0) {
            for (let ry = roofY; ry <= top; ry++) _set(rx, ry, M.CONCRETE);
          }
        }
      } else {
        for (let x = bx; x < bx + bw; x++) { _set(x, top, M.CONCRETE); _set(x, top + 1, M.CONCRETE); }
      }

      // Antenna
      if (bh > 50 && Math.random() < 0.6) {
        const antX = bx + (bw / 2 | 0);
        const antH = 6 + (Math.random() * 10 | 0);
        for (let ay = top - antH; ay < top; ay++) { if (ay >= 0) _set(antX, ay, M.METAL); }
        if (antH > 8 && top - antH + 2 >= 0) {
          const crossY = top - antH + 2;
          for (let cx = -2; cx <= 2; cx++) { if (this.inBounds(antX + cx, crossY)) _set(antX + cx, crossY, M.METAL); }
        }
      }

      // Floors
      const floors = [];
      for (let fy = gnd - floorSpace; fy > top + 2; fy -= floorSpace) {
        floors.push(fy);
        for (let x = bx; x < bx + bw; x++) _set(x, fy, M.CONCRETE);
      }

      // Internal supports
      const supSpacing = 5 + (Math.random() * 4 | 0);
      for (let sx = bx + 3; sx < bx + bw - 3; sx += supSpacing) {
        for (let y = top + 2; y < gnd; y++) {
          if (this.get(sx, y) === M.EMPTY) _set(sx, y, M.METAL);
        }
      }

      // Windows
      for (let fi = 0; fi < floors.length; fi++) {
        const floorY = floors[fi];
        const ceilY = fi === 0 ? top + 2 : floors[fi - 1] + 1;
        const winTop = ceilY + 1, winBot = floorY - 1;
        if (winBot <= winTop) continue;

        for (let wy = winTop; wy <= winBot; wy++) {
          if ((wy - winTop) % 2 === 0) continue;
          _set(bx + 1, wy, M.GLASS);
        }
        for (let wy = winTop; wy <= winBot; wy++) {
          if ((wy - winTop) % 2 === 0) continue;
          _set(bx + bw - 2, wy, M.GLASS);
        }
        for (let wx = bx + 4; wx < bx + bw - 4; wx += 3) {
          if (this.get(wx, winTop) !== M.EMPTY) continue;
          for (let wy = winTop; wy <= Math.min(winBot, winTop + 2); wy++) {
            if (this.get(wx, wy) === M.EMPTY) _set(wx, wy, M.GLASS);
            if (wx + 1 < bx + bw - 2 && this.get(wx + 1, wy) === M.EMPTY) _set(wx + 1, wy, M.GLASS);
          }
        }
      }

      // Door
      const doorX = bx + (bw / 2 | 0) - 1;
      for (let dx = 0; dx < 3; dx++) {
        for (let dy = 1; dy <= 5; dy++) {
          const gx = doorX + dx, gy = gnd - dy;
          if (this.inBounds(gx, gy) && isStructural(this.get(gx, gy)) && this.get(gx, gy) !== M.METAL) {
            grid[gy * W + gx] = M.EMPTY; temp[gy * W + gx] = RT;
          }
        }
      }

      buildings.push({ x: bx, w: bw, top, right: bx + bw });
      bx += bw + 2 + (Math.random() * 6 | 0);
    }

    // Skybridges
    for (let i = 0; i < buildings.length - 1; i++) {
      const a = buildings[i], b = buildings[i + 1];
      const gap = b.x - a.right;
      if (gap > 12 || gap < 2 || Math.random() > 0.45) continue;

      const bridgeY = Math.max(a.top, b.top) + 15 + (Math.random() * 20 | 0);
      if (bridgeY >= gnd - 5) continue;

      for (let bx2 = a.right - 1; bx2 <= b.x + 1; bx2++) {
        if (this.inBounds(bx2, bridgeY)) { grid[bridgeY * W + bx2] = M.CONCRETE; temp[bridgeY * W + bx2] = RT; }
        if (this.inBounds(bx2, bridgeY + 1)) { grid[(bridgeY+1) * W + bx2] = M.CONCRETE; temp[(bridgeY+1) * W + bx2] = RT; }
      }
      for (let bx2 = a.right; bx2 <= b.x; bx2++) {
        if (this.inBounds(bx2, bridgeY - 1)) { grid[(bridgeY-1) * W + bx2] = M.GLASS; temp[(bridgeY-1) * W + bx2] = RT; }
        if (this.inBounds(bx2, bridgeY - 2)) { grid[(bridgeY-2) * W + bx2] = M.GLASS; temp[(bridgeY-2) * W + bx2] = RT; }
      }
    }
  }

  // ==================== RENDERING ====================

  /** Render the current state to a Uint32Array of ABGR-packed pixels.
   *  opts.showTemp: thermal heatmap overlay
   *  opts.showStress: structural stress overlay
   *  opts.time: performance.now() for pulsing effects (stress view)
   *  Returns the engine's internal pixel buffer (not a copy — read before next render).
   */
  render(opts = {}) {
    const { W, H, grid, temp, weightBuf, chunks } = this;
    const pixels = this._pixels;
    const showTemp = opts.showTemp || false;
    const showStress = opts.showStress || false;
    const time = opts.time || 0;

    for (let i = 0; i < W * H; i++) {
      let c;
      if (showStress) {
        c = _stressHeatmap(weightBuf[i], grid[i], time);
      } else if (showTemp) {
        c = _tempHeatmap(temp[i]);
      } else {
        c = _matColor(grid[i]);
        const mat = grid[i];
        if (mat !== M.EMPTY && mat !== M.FIRE && mat !== M.LAVA && mat !== M.CLONE && mat !== M.VOID && mat !== M.EMBER) {
          c = _thermalTint(c[0], c[1], c[2], temp[i]);
        }
      }
      pixels[i] = (255 << 24) | (c[2] << 16) | (c[1] << 8) | c[0];
    }

    // Overlay falling chunks
    if (!showTemp) {
      for (const chunk of chunks) {
        for (const p of chunk.pixels) {
          const wx = chunk.x + p.rx, wy = chunk.y + p.ry;
          if (wx < 0 || wx >= W || wy < 0 || wy >= H) continue;
          const c = _matColor(p.mat);
          pixels[wy * W + wx] = (255 << 24) | (c[2] << 16) | (c[1] << 8) | c[0];
        }
      }
    }

    return pixels;
  }
}

// ==================== RENDERING HELPERS (module-private) ====================

function _varyColor(base, amount) {
  const v = (Math.random() - 0.5) * amount * 2;
  return [
    Math.max(0, Math.min(255, base[0] + v)),
    Math.max(0, Math.min(255, base[1] + v)),
    Math.max(0, Math.min(255, base[2] + v)),
  ];
}

function _matColor(mat) {
  switch (mat) {
    case M.SAND: return _varyColor([210, 180, 100], 15);
    case M.WATER: return _varyColor([40, 80, 200], 20);
    case M.STONE: return _varyColor([120, 120, 130], 10);
    case M.FIRE: return [255, 80 + Math.random() * 120 | 0, Math.random() * 40 | 0];
    case M.WOOD: return _varyColor([100, 65, 30], 10);
    case M.OIL: return _varyColor([60, 40, 70], 8);
    case M.GUNPOWDER: return _varyColor([80, 80, 80], 8);
    case M.STEAM: return [180 + Math.random() * 40 | 0, 180 + Math.random() * 40 | 0, 210 + Math.random() * 30 | 0];
    case M.ACID: return _varyColor([30, 220, 50], 25);
    case M.LAVA:
      if (Math.random() > 0.7) return [255, 180 + Math.random() * 75 | 0, 30 + Math.random() * 60 | 0];
      return _varyColor([200, 60, 10], 20);
    case M.ICE: return _varyColor([180, 215, 250], 8);
    case M.SMOKE: return [70 + Math.random() * 30 | 0, 70 + Math.random() * 30 | 0, 75 + Math.random() * 30 | 0];
    case M.PLANT: return _varyColor([30, 140 + Math.random() * 40 | 0, 35], 12);
    case M.SALT: return _varyColor([230, 230, 240], 8);
    case M.METAL: return _varyColor([170, 175, 182], 6);
    case M.GAS: return _varyColor([140, 180, 40], 20);
    case M.CLONE: return [50 + Math.random() * 40 | 0, 190 + Math.random() * 30 | 0, 200 + Math.random() * 30 | 0];
    case M.VOID: {
      if (Math.random() < 0.12) return [120, 30, 180];
      return [20 + Math.random() * 15 | 0, 0, 30 + Math.random() * 15 | 0];
    }
    case M.EMBER: return [255, 150 + Math.random() * 105 | 0, Math.random() * 50 | 0];
    case M.CONCRETE: return _varyColor([160, 160, 155], 8);
    case M.GLASS: {
      const shimmer = Math.random() * 20 | 0;
      return [170 + shimmer, 200 + shimmer, 225 + (Math.random() * 15 | 0)];
    }
    case M.BRICK: return _varyColor([155, 70, 45], 12);
    case M.NAPALM: {
      const flicker = Math.random();
      if (flicker < 0.3) return [255, 120 + Math.random() * 60 | 0, 20];
      if (flicker < 0.6) return [240, 80 + Math.random() * 40 | 0, 10];
      return [200, 60 + Math.random() * 30 | 0, 5];
    }
    default: return [26, 26, 46];
  }
}

// Also export matColor for games that want custom renderers
export { _matColor as matColor };

function _thermalTint(r, g, b, t) {
  if (t > 60) {
    const f = Math.min(1, (t - 60) / 400);
    r = Math.min(255, r + f * 120);
    g = Math.min(255, g + f * 25);
    b = Math.max(0, b - f * 60);
  } else if (t < 5) {
    const f = Math.min(1, (5 - t) / 30);
    r = Math.max(0, r - f * 40);
    g = Math.min(255, g + f * 15);
    b = Math.min(255, b + f * 80);
  }
  return [r | 0, g | 0, b | 0];
}

function _stressHeatmap(w, mat, time) {
  if (!isStructural(mat) || STRENGTH[mat] === 0) return [26, 26, 46];
  const stress = w / STRENGTH[mat];
  if (stress < 2) return [20, 80, 20];
  if (stress < 8) { const f = (stress - 2) / 6; return [20 + f * 200 | 0, 80 + f * 100 | 0, 20]; }
  if (stress < 15) { const f = (stress - 8) / 7; return [220, 180 - f * 130 | 0, 20]; }
  if (stress < 18) { const f = (stress - 15) / 3; return [220 + f * 35 | 0, 50 - f * 30 | 0, 20]; }
  const pulse = Math.sin(time * 0.01) * 0.3 + 0.7;
  return [255 * pulse | 0, 30 * pulse | 0, 30 * pulse | 0];
}

function _tempHeatmap(t) {
  if (t < 0) {
    const f = Math.min(1, -t / 30);
    return [20, 40 + f * 80 | 0, 100 + f * 155 | 0];
  }
  if (t < 25) return [30, 30, 50];
  const hot = t - 25;
  if (hot < 75) { const f = hot / 75; return [f * 200 | 0, f * 150 | 0, 20]; }
  if (hot < 275) { const f = (hot - 75) / 200; return [200 + f * 55 | 0, 150 - f * 100 | 0, 0]; }
  if (hot < 675) { const f = (hot - 275) / 400; return [255, 50 + f * 130 | 0, f * 60 | 0]; }
  return [255, 200, 100];
}
