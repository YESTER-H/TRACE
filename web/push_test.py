"""Push test in CPU MuJoCo, with and without heading alignment (cf. diagnose_trace_push_recovery.py).

The robot dances from frame 0. One pelvis velocity push (8 directions x speeds x push times) is applied,
and the robot runs on with no reset for --seconds. A run counts as a fall if the pelvis drops below
0.45 m. A fall counts as recovered if the robot is upright at the end (pelvis >= 0.6 m, tilt <= 30 deg,
for the last 2 s).

usage: .venv/bin/python push_test.py [--speeds 1.8,2.4,3.0] [--seconds 25] [--workers 8]
"""
import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--speeds", default="1.8,2.4,3.0")
parser.add_argument("--directions", type=int, default=8)
parser.add_argument("--push-times", default="2.0,3.7")
parser.add_argument("--seconds", type=float, default=25.)
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()


def run(case):
    import torch
    torch.set_num_threads(1)
    from trace_mujoco import CONTROL_DT, K1, Controller, TracePolicy, load_motions
    direction, speed, push_t, align = case
    robot = K1()
    ctrl = Controller(robot, TracePolicy(), load_motions(), align_heading=align)
    ctrl.reset("stand")
    steps, push_step = int(args.seconds / CONTROL_DT), int(push_t / CONTROL_DT)
    angle = 2 * math.pi * direction / args.directions
    heights, tilts, fell = [], [], False
    for k in range(steps):
        if k == push_step:
            robot.data.qvel[0] += speed * math.cos(angle)
            robot.data.qvel[1] += speed * math.sin(angle)
        ctrl.step()
        h = robot.pelvis_height()
        g = robot.data.xmat[robot.pelvis].reshape(3, 3)[2, 2]
        fell |= h < .45
        heights.append(h); tilts.append(g)
    last = int(2 / CONTROL_DT)
    recovered = bool(fell and min(heights[-last:]) >= .6 and min(tilts[-last:]) >= math.cos(math.radians(30)))
    return dict(direction_deg=round(math.degrees(angle)), speed=speed, push_t=push_t, align=align,
                fell=bool(fell), recovered=recovered, locks=ctrl.aligner.locks if ctrl.aligner else 0,
                final_pelvis=round(heights[-1], 2))


if __name__ == "__main__":
    speeds = [float(s) for s in args.speeds.split(",")]
    times = [float(t) for t in args.push_times.split(",")]
    cases = [(d, s, t, a) for a in (False, True) for d in range(args.directions) for s in speeds for t in times]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, cases))
    for align in (False, True):
        r = [x for x in rows if x["align"] == align]
        falls = [x for x in r if x["fell"]]
        print(f"heading alignment {'ON ' if align else 'OFF'}: runs {len(r)}, falls {len(falls)}, "
              f"recovered {sum(x['recovered'] for x in falls)}/{len(falls)}")
        for x in falls:
            if not x["recovered"]:
                print("   not recovered:", json.dumps(x))
