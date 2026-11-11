"""Log one pushed rollout with heading alignment ON, for the TS parity test (tests/ts_parity.mjs)."""
import json, math, sys
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__) + "/..")
from trace_mujoco import ANCHOR_ID, CONTROL_DT, K1, Controller, TracePolicy, load_motions, mat6, quat_conj, quat_mul

robot = K1(); ctrl = Controller(robot, TracePolicy(), load_motions(), align_heading=True); ctrl.reset("stand")
rows, angle = [], math.radians(135)
for k in range(int(20 / CONTROL_DT)):
    if k == 100:
        robot.data.qvel[0] += 3.0 * math.cos(angle); robot.data.qvel[1] += 3.0 * math.sin(angle)
    ref = ctrl.dance.body_quat[ctrl.phase, ANCHOR_ID]
    expected = mat6(quat_mul(quat_conj(robot.anchor_quat()), ctrl.aligner.rotate(ref)))
    rows.append(dict(time=robot.data.time, rootPos=robot.data.qpos[:3].tolist(), rootLinVel=robot.data.qvel[:3].tolist(),
                     rootAngVel=robot.data.qvel[3:6].tolist(), ref=ref.tolist(), robot=robot.anchor_quat().tolist(),
                     expected=expected.tolist(), offset=ctrl.aligner.offset))
    ctrl.step()
json.dump(dict(rows=rows, locks=ctrl.aligner.locks), open(sys.argv[1], "w"))
print("steps", len(rows), "locks", ctrl.aligner.locks, "final offset deg", round(math.degrees(ctrl.aligner.offset), 1))
