// Replays a Python rollout through the bundled TS term and compares the newest frame with Python's.
// usage: node ts_parity.mjs <bundled TraceAlignedAnchorOri.mjs> <trace.json>
import { readFileSync } from 'node:fs';
const [modPath, tracePath] = process.argv.slice(2);
const { TraceAlignedAnchorOri } = await import(modPath);
const { rows } = JSON.parse(readFileSync(tracePath, 'utf8'));
let current = rows[0];
const data = { get time() { return current.time; } };
const term = { getStateField: (f) => f === 'anchor_quat_w' ? Float32Array.from(current.ref)
                                   : f === 'robot_anchor_quat_w' ? Float32Array.from(current.robot) : null };
const runner = { getContext: () => ({ mjData: data, commandManager: { getTerm: () => term } }) };
const cfg = { name: 'TraceAlignedAnchorOri', supine_head: Number(process.env.SUPINE), prone_head: Number(process.env.PRONE), history: 4 };
const obs = new TraceAlignedAnchorOri(runner, cfg);
obs.reset();
let maxErr = 0, worst = -1;
rows.forEach((r, i) => {
  current = r;
  const out = obs.compute({ rootPos: Float32Array.from(r.rootPos), rootLinVel: Float32Array.from(r.rootLinVel), rootAngVel: Float32Array.from(r.rootAngVel) });
  const frame = out.slice(18, 24);
  const err = Math.max(...frame.map((v, j) => Math.abs(v - r.expected[j])));
  if (err > maxErr) { maxErr = err; worst = i; }
});
console.log(`steps ${rows.length}, TS locks ${obs.locks}, max |TS - Python| ${maxErr.toExponential(2)} at step ${worst}`);
process.exit(maxErr < 1e-4 ? 0 : 1);
