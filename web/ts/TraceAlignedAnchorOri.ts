/**
 * TRACE: dance-anchor orientation observation with deployment-time heading alignment.
 *
 * Same value as mjlab's `motion_anchor_ori_b` (first two columns of R(robot torso)^T R(reference torso),
 * row-major, 6 numbers), except that the reference is first turned about world z by `offset`.
 *
 * Recovery was trained lying in one heading relative to the dance. Once the robot has lain still for
 * `settle_steps` control steps, `offset` is set so that the robot-to-reference relation equals the
 * trained one: (robot head yaw) - (head yaw of frame 0 of the Supine or Prone get-up reference,
 * chosen by whether the chest faces up). Still lying `relock_steps` steps later -> read again. Reset clears it.
 * Port of scripts/reinforcement_learning/rsl_rl/trace_heading_alignment.py (exhibition script) and
 * web/trace_mujoco.py:HeadingAligner, which push_test.py checks.
 *
 * The term keeps its own 4-frame history (oldest first, filled with the first frame after a reset),
 * because mjswan's per-term history wrapper only applies to traced/native terms. So `size` is 24.
 */
import { ObservationBase } from 'mjswan/observation';

type Quat = [number, number, number, number]; // w, x, y, z (MuJoCo order)

type Config = {
  name: string;
  supine_head: number;
  prone_head: number;
  settle_steps?: number;
  relock_steps?: number;
  lying_pelvis?: number;
  still_linear?: number;
  still_angular?: number;
  history?: number;
  command_name?: string;
};

function mul(a: Quat, b: Quat): Quat {
  const [w1, x1, y1, z1] = a;
  const [w2, x2, y2, z2] = b;
  return [
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ];
}

function conj(q: Quat): Quat {
  return [q[0], -q[1], -q[2], -q[3]];
}

function normalize(q: Quat): Quat {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

/** Row-major 3x3 rotation matrix. */
function mat(q: Quat): number[] {
  const [w, x, y, z] = normalize(q);
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
}

/** Yaw of the body's z axis (head direction) on the floor plane. */
function headAngle(q: Quat): number {
  const m = mat(q);
  return Math.atan2(m[5], m[2]);
}

function chestUp(q: Quat): boolean {
  return mat(q)[6] > 0; // world z of the body's x axis
}

export class TraceAlignedAnchorOri extends ObservationBase<Config> {
  private offset = 0;
  private still = 0;
  private sinceLock = 1e9;
  private lastTime = -1;
  private frames: Float32Array[] = [];
  private needsPrime = true;
  /** Number of headings locked since the last reset (for debugging in the console). */
  locks = 0;

  get size(): number {
    return 6 * (this.config.history ?? 4);
  }

  reset(): void {
    this.offset = 0;
    this.still = 0;
    this.sinceLock = 1e9;
    this.lastTime = -1;
    this.needsPrime = true;
    this.locks = 0;
  }

  private term(): { getStateField(field: string): Float32Array | null } | undefined {
    const manager = this.runner.getContext()?.commandManager as
      | { getTerm(name: string): unknown }
      | undefined;
    return manager?.getTerm(this.config.command_name ?? 'motion') as
      | { getStateField(field: string): Float32Array | null }
      | undefined;
  }

  /** Advance the lying-still detector once per control step (keyed on simulation time). */
  private updateAlignment(state: Record<string, unknown>, robotQuat: Quat): void {
    const data = this.runner.getContext()?.mjData as { time: number } | null | undefined;
    const time = data?.time ?? 0;
    if (time < this.lastTime - 1e-9) this.reset(); // the simulation was reset under us
    if (this.lastTime < 0) {
      // First observation of an episode: nothing has been simulated yet (as in the Python port).
      this.lastTime = time;
      return;
    }
    if (time - this.lastTime < 1e-6) return; // same control step, already counted
    this.lastTime = time;

    const pos = state.rootPos as Float32Array | undefined;
    const lin = state.rootLinVel as Float32Array | undefined;
    const ang = state.rootAngVel as Float32Array | undefined;
    if (!pos || !lin || !ang) return;
    const lyingStill =
      pos[2] < (this.config.lying_pelvis ?? 0.3) &&
      Math.hypot(lin[0], lin[1], lin[2]) < (this.config.still_linear ?? 0.5) &&
      Math.hypot(ang[0], ang[1], ang[2]) < (this.config.still_angular ?? 1.5);
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

  compute(state: Record<string, unknown>): Float32Array {
    const term = this.term();
    const ref = term?.getStateField('anchor_quat_w');
    const robot = term?.getStateField('robot_anchor_quat_w');
    let frame = new Float32Array(6);
    if (ref && robot) {
      const robotQuat = normalize([robot[0], robot[1], robot[2], robot[3]]);
      this.updateAlignment(state, robotQuat);
      const yaw: Quat = [Math.cos(this.offset / 2), 0, 0, Math.sin(this.offset / 2)];
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
}
