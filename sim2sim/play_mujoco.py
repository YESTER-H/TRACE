"""Run a TRACE policy in the native MuJoCo viewer (sim-to-sim, CPU, real time). Policy: TRACE_POLICY=run24 (default) or run23.

usage (from the TRACE repo root):
    web/.venv/bin/python sim2sim/play_mujoco.py [--start stand|supine|prone] [--condition delay_40ms]
                                                [--no-align] [--speed 1.0] [--seed 0]

--condition takes any name from strict_sweep.py (nominal, delay_40ms, friction_0.4, armature_x0, random_hard,
limit_kp_kd_x0.5, ...). "train" (default) is the exact training physics without observation noise.

Keys in the viewer window (chosen to avoid MuJoCo's own letter/digit shortcuts):
    Home / End / Insert   reset: Stand / Supine / Prone (frame 0 of the dance or get-up reference)
    Arrow keys            push the pelvis (Up = away from the camera, Left/Right = sideways)
    Delete                push in a random direction
    Page Up / Page Down   push strength up / down (0.5 .. 4.0 m/s)
    Ctrl + right drag     MuJoCo's own perturbation (drag a body with a spring force)
Terminal commands (type and press Enter):
    stand | supine | prone      reset to that start
    cond <name>                 switch condition (keeps the current start, resets)
    list                        list condition names
    push <deg> | random         push in a world direction
    quit
"""
import argparse
import math
import queue
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "web"))
sys.path.insert(0, str(HERE))
from trace_mujoco import CONTROL_DT, K1, Controller, TracePolicy, load_motions  # noqa: E402
from strict_sweep import CONDITIONS, configure, snapshot  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--start", default="stand", choices=["stand", "supine", "prone"])
parser.add_argument("--condition", default="train", help="'train' or a strict_sweep.py condition name")
parser.add_argument("--no-align", action="store_true", help="disable deployment-time heading alignment (run 24 does not need it)")
parser.add_argument("--speed", type=float, default=1.0, help="playback speed relative to real time")
parser.add_argument("--push", type=float, default=1.8, help="initial push strength in m/s")
parser.add_argument("--seed", type=int, default=0, help="seed for noise / randomized conditions")
args = parser.parse_args()
if args.condition != "train" and args.condition not in CONDITIONS:
    sys.exit(f"unknown condition {args.condition!r}; choose from: train, " + ", ".join(CONDITIONS))

torch.set_num_threads(1)
KEY_HOME, KEY_END, KEY_INSERT, KEY_DELETE = 268, 269, 260, 261
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_PAGE_UP, KEY_PAGE_DOWN = 266, 267

robot = K1()
base = snapshot(robot)
ctrl = Controller(robot, TracePolicy(), load_motions(), align_heading=not args.no_align)
commands: "queue.Queue[tuple]" = queue.Queue()
state = dict(push=args.push, start=args.start, last_event="", w_t=1.0, t0=0.0, condition=args.condition,
             setup=None, detail="")
rng = np.random.default_rng(args.seed)


def describe(cond):
    keys = {k: v for k, v in cond.items() if k not in ("group", "random")}
    return ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in keys.items()) or "-"


def set_condition(name):
    """(Re)apply a condition. 'train' = training physics, no observation noise."""
    if hasattr(robot, "step") and "step" in vars(robot):
        del robot.step
    if name == "train":
        for f, v in base["fields"].items():
            getattr(robot.model, f)[:] = v
        for o, v in base["opts"].items():
            setattr(robot.model.opt, o, v)
        state.update(setup=None, detail="training physics, no obs noise")
    else:
        setup = configure(robot, base, name, rng)
        robot.step = setup["physics"]
        state.update(setup=setup, detail=describe(setup["cond"]))
    state["condition"] = name
    print(f"condition: {name}  ({state['detail']})", flush=True)


def reset(start):
    ctrl.reset(start)
    setup = state["setup"]
    if setup is not None:
        ctrl.observer = setup["make_observer"](ctrl.aligner)
        setup["after_reset"]()
    state.update(start=start, t0=robot.data.time, last_event=f"reset {start}")


def key_callback(key):
    if key == KEY_HOME:
        commands.put(("reset", "stand"))
    elif key == KEY_END:
        commands.put(("reset", "supine"))
    elif key == KEY_INSERT:
        commands.put(("reset", "prone"))
    elif key in (KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT):
        commands.put(("push_cam", {KEY_UP: 0, KEY_LEFT: 90, KEY_DOWN: 180, KEY_RIGHT: -90}[key]))
    elif key == KEY_DELETE:
        commands.put(("push_world", float(np.random.uniform(0, 360))))
    elif key == KEY_PAGE_UP:
        state["push"] = min(4.0, state["push"] + 0.3)
    elif key == KEY_PAGE_DOWN:
        state["push"] = max(0.5, state["push"] - 0.3)


def stdin_reader():
    for line in sys.stdin:
        words = line.strip().split()
        if not words:
            continue
        w0 = words[0].lower()
        if w0 in ("stand", "supine", "prone"):
            commands.put(("reset", w0))
        elif w0 == "cond" and len(words) > 1:
            if words[1] == "train" or words[1] in CONDITIONS:
                commands.put(("cond", words[1]))
            else:
                print(f"unknown condition {words[1]!r} (type 'list')", flush=True)
        elif w0 == "list":
            groups = {}
            for n, c in CONDITIONS.items():
                groups.setdefault(c["group"], []).append(n)
            print("train\n" + "\n".join(f"  {g}: {', '.join(ns)}" for g, ns in groups.items()), flush=True)
        elif w0 == "push" and len(words) > 1:
            commands.put(("push_world", float(words[1])))
        elif w0 == "random":
            commands.put(("push_world", float(np.random.uniform(0, 360))))
        elif w0 in ("quit", "exit", "q"):
            commands.put(("quit", None))


def apply(cmd, viewer):
    kind, value = cmd
    if kind == "reset":
        reset(value)
    elif kind == "cond":
        set_condition(value)
        reset(state["start"])
    elif kind in ("push_cam", "push_world"):
        deg = value + (viewer.cam.azimuth if kind == "push_cam" else 0.0)
        a = math.radians(deg)
        robot.data.qvel[0] += state["push"] * math.cos(a)
        robot.data.qvel[1] += state["push"] * math.sin(a)
        state["last_event"] = f"push {state['push']:.1f} m/s @ {deg % 360:.0f} deg (t={robot.data.time:.1f}s)"
    elif kind == "quit":
        viewer.close()


set_condition(args.condition)
reset(args.start)
threading.Thread(target=stdin_reader, daemon=True).start()
print(__doc__)
with mujoco.viewer.launch_passive(robot.model, robot.data, key_callback=key_callback) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = robot.pelvis
    viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 3.0, -15.0, 90.0
    wall_next = time.perf_counter()
    while viewer.is_running():
        with viewer.lock():
            while not commands.empty():
                apply(commands.get(), viewer)
            state["w_t"] = ctrl.step()   # policy at 50 Hz + physics steps; Ctrl+drag forces stay in xfrc_applied
        h = robot.pelvis_height()
        mode = "Tracking" if state["w_t"] >= 0.5 else "Recovery"
        align = ctrl.aligner
        viewer.set_texts([
            (None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
             "condition\nstart\ntime\npelvis z\ngate w_T\nexpert\ndance frame\nheading lock\npush strength\nlast",
             f"{state['condition']}\n{state['start']}\n{robot.data.time - state['t0']:.1f} s\n{h:.2f} m\n"
             f"{state['w_t']:.2f}\n{mode}\n{ctrl.phase}\n"
             f"{'off' if align is None else f'{math.degrees(align.offset):+.0f} deg ({align.locks}x)'}\n"
             f"{state['push']:.1f} m/s\n{state['last_event']}"),
            (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, state["detail"], None),
        ])
        if not np.isfinite(robot.data.qpos).all():
            print("NaN state -> reset", flush=True)
            commands.put(("reset", state["start"]))
        viewer.sync()
        wall_next += CONTROL_DT / args.speed
        delay = wall_next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            wall_next = time.perf_counter()   # slower than real time: don't try to catch up
