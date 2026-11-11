// src/core/observation/ObservationBase.ts
var ObservationBase = class {
  constructor(runner, config) {
    this.runner = runner;
    this.config = config;
  }
};

// ../../../../../../ts/TraceAlignedAnchorOri.ts
function mul(a, b) {
  const [w1, x1, y1, z1] = a;
  const [w2, x2, y2, z2] = b;
  return [
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
  ];
}
function conj(q) {
  return [q[0], -q[1], -q[2], -q[3]];
}
function normalize(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}
function mat(q) {
  const [w, x, y, z] = normalize(q);
  return [
    1 - 2 * (y * y + z * z),
    2 * (x * y - w * z),
    2 * (x * z + w * y),
    2 * (x * y + w * z),
    1 - 2 * (x * x + z * z),
    2 * (y * z - w * x),
    2 * (x * z - w * y),
    2 * (y * z + w * x),
    1 - 2 * (x * x + y * y)
  ];
}
function headAngle(q) {
  const m = mat(q);
  return Math.atan2(m[5], m[2]);
}
function chestUp(q) {
  return mat(q)[6] > 0;
}
var TraceAlignedAnchorOri = class extends ObservationBase {
  offset = 0;
  still = 0;
  sinceLock = 1e9;
  lastTime = -1;
  frames = [];
  needsPrime = true;
  /** Number of headings locked since the last reset (for debugging in the console). */
  locks = 0;
  get size() {
    return 6 * (this.config.history ?? 4);
  }
  reset() {
    this.offset = 0;
    this.still = 0;
    this.sinceLock = 1e9;
    this.lastTime = -1;
    this.needsPrime = true;
    this.locks = 0;
  }
  term() {
    const manager = this.runner.getContext()?.commandManager;
    return manager?.getTerm(this.config.command_name ?? "motion");
  }
  /** Advance the lying-still detector once per control step (keyed on simulation time). */
  updateAlignment(state, robotQuat) {
    const data = this.runner.getContext()?.mjData;
    const time = data?.time ?? 0;
    if (time < this.lastTime - 1e-9) this.reset();
    if (this.lastTime < 0) {
      this.lastTime = time;
      return;
    }
    if (time - this.lastTime < 1e-6) return;
    this.lastTime = time;
    const pos = state.rootPos;
    const lin = state.rootLinVel;
    const ang = state.rootAngVel;
    if (!pos || !lin || !ang) return;
    const lyingStill = pos[2] < (this.config.lying_pelvis ?? 0.3) && Math.hypot(lin[0], lin[1], lin[2]) < (this.config.still_linear ?? 0.5) && Math.hypot(ang[0], ang[1], ang[2]) < (this.config.still_angular ?? 1.5);
    this.still = lyingStill ? this.still + 1 : 0;
    this.sinceLock += 1;
    if (this.still >= (this.config.settle_steps ?? 15) && this.sinceLock >= (this.config.relock_steps ?? 150)) {
      const reference = chestUp(robotQuat) ? this.config.supine_head : this.config.prone_head;
      const d = headAngle(robotQuat) - reference;
      this.offset = Math.atan2(Math.sin(d), Math.cos(d));
      this.sinceLock = 0;
      this.locks += 1;
    }
  }
  compute(state) {
    const term = this.term();
    const ref = term?.getStateField("anchor_quat_w");
    const robot = term?.getStateField("robot_anchor_quat_w");
    let frame = new Float32Array(6);
    if (ref && robot) {
      const robotQuat = normalize([robot[0], robot[1], robot[2], robot[3]]);
      this.updateAlignment(state, robotQuat);
      const yaw = [Math.cos(this.offset / 2), 0, 0, Math.sin(this.offset / 2)];
      const aligned = mul(yaw, normalize([ref[0], ref[1], ref[2], ref[3]]));
      const m = mat(mul(conj(robotQuat), aligned));
      frame = new Float32Array([m[0], m[1], m[3], m[4], m[6], m[7]]);
    }
    const history = this.config.history ?? 4;
    if (this.needsPrime || this.frames.length !== history) {
      this.frames = Array.from({ length: history }, () => frame.slice());
      this.needsPrime = false;
    } else {
      this.frames.shift();
      this.frames.push(frame);
    }
    const out = new Float32Array(6 * history);
    this.frames.forEach((f, i) => out.set(f, 6 * i));
    return out;
  }
};

// src/.plugins-entry.ts
var observations = { TraceAlignedAnchorOri };
export {
  observations
};
