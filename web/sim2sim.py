"""Sim-to-sim check: run the run-23 TRACE policy in CPU MuJoCo from the three frame-0 starts.

usage: .venv/bin/python sim2sim.py [--seconds 20] [--video out_dir]
Success = standing (pelvis >= 0.6 m), gate on Tracking (w_T >= 0.9) and dance joint error small at the end.
"""
import argparse
import json

import mujoco
import numpy as np

from trace_mujoco import CONTROL_DT, K1, Controller, TracePolicy, load_motions

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=20.)
parser.add_argument("--starts", default="stand,supine,prone")
parser.add_argument("--video", default=None, help="directory for mp4 videos (needs EGL/OSMesa)")
args = parser.parse_args()

policy, motions = TracePolicy(), load_motions()
summary = {}
for start in args.starts.split(","):
    robot = K1()
    ctrl = Controller(robot, policy, motions)
    ctrl.reset(start)
    steps = int(args.seconds / CONTROL_DT)
    log, frames = [], []
    renderer = None
    if args.video:
        renderer = mujoco.Renderer(robot.model, 360, 640)
        cam = mujoco.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 3.0, -15, 90
    first_tracking = None
    for k in range(steps):
        w_t = ctrl.step()
        err = float(np.linalg.norm(robot.joint_pos() - motions["tracking"].joint_pos[ctrl.phase]))
        h = robot.pelvis_height()
        log.append((k, h, w_t, err))
        if first_tracking is None and h >= .6 and w_t >= .9:
            first_tracking = k
        if renderer is not None and k % 2 == 0:
            cam.lookat[:] = robot.data.xpos[robot.pelvis]
            renderer.update_scene(robot.data, cam)
            frames.append(renderer.render())
        if not np.isfinite(robot.data.qpos).all():
            break
    a = np.array(log)
    last = a[-int(2 / CONTROL_DT):]
    ok = bool((last[:, 1] >= .6).all() and (last[:, 2] >= .9).all())
    summary[start] = dict(success=ok, first_tracking_step=first_tracking,
                          final_pelvis=round(float(a[-1, 1]), 3), min_pelvis_last2s=round(float(last[:, 1].min()), 3),
                          mean_wT_last2s=round(float(last[:, 2].mean()), 3),
                          mean_joint_err_last2s=round(float(last[:, 3].mean()), 3))
    print(start, summary[start], flush=True)
    for t in range(0, len(a), int(1 / CONTROL_DT)):
        print(f"   t={a[t,0]*CONTROL_DT:5.1f}s pelvis={a[t,1]:.2f} wT={a[t,2]:.2f} jerr={a[t,3]:.2f}")
    if frames:
        import cv2, os
        os.makedirs(args.video, exist_ok=True)
        out = cv2.VideoWriter(f"{args.video}/{start}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
        for f in frames:
            out.write(f[:, :, ::-1])
        out.release()
print(json.dumps(summary, indent=1))
