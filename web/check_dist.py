"""Check the built web bundle in Python: dist scene.mjz + dist policy ONNX (onnxruntime), entity order.

This mirrors what the browser runs (mjlab observation definitions, H4 term-major history repeated at
reset, sim-time reference frame, clip -> scale -> default offset -> position actuators), so a pass here
means the exported policy, its joint-order conversion and the scene physics are right. The browser
runtime itself (mjswan's TypeScript) is not covered.

usage: .venv/bin/python check_dist.py [--seconds 15]
"""
import argparse
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--dist", default=str(HERE / "dist"))
parser.add_argument("--seconds", type=float, default=15.)
args = parser.parse_args()

dist = Path(args.dist)
manifest = json.load(open(dist / "manifest.json"))
scene = manifest["projects"][0]["scenes"][0]
scene_dir = dist / manifest["projects"][0]["id"] / scene["id"]
policy_cfg = scene["policies"][0]
mdp = next(m for m in scene["mdps"] if m["id"] == policy_cfg["mdp"])

tmp = tempfile.mkdtemp()
zipfile.ZipFile(scene_dir / "scene.mjz").extractall(tmp)
model = mujoco.MjModel.from_xml_path(os.path.join(tmp, next(f for f in os.listdir(tmp) if f.endswith(".xml"))))
print("solver iterations", model.opt.iterations, model.opt.ls_iterations, "dt", model.opt.timestep)
session = ort.InferenceSession(str(scene_dir / policy_cfg["onnx"]))
joints = policy_cfg["policy_joint_names"]
default = np.array(policy_cfg["default_joint_pos"], np.float32)
scale = np.array([mdp["actions"]["joint_pos"]["scale"][j] for j in joints], np.float32)
qadr = np.array([model.jnt_qposadr[model.joint("robot/" + j).id] for j in joints])
vadr = np.array([model.jnt_dofadr[model.joint("robot/" + j).id] for j in joints])
act = np.array([model.actuator("robot/" + j).id for j in joints])
root_q = model.jnt_qposadr[model.joint("robot/floating_base_joint").id]
root_v = model.jnt_dofadr[model.joint("robot/floating_base_joint").id]
torso = model.body("robot/torso_link").id
pelvis = model.body("robot/pelvis").id
motion = np.load(scene_dir / policy_cfg["motions"][0]["path"])
anchor_i = policy_cfg["motions"][0]["body_names"].index("torso_link")
decimation = round(scene["control_dt"] / model.opt.timestep)
events = {e["name"]: e["params"] for e in mdp["events"]}


def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def rot(q):
    m = np.zeros(9); mujoco.mju_quat2Mat(m, q); return m.reshape(3, 3)


def run(start):
    d = mujoco.MjData(model)
    # Reset = the TrackingCommand writes reference frame 0 (pose ranges are empty).
    d.qpos[root_q:root_q+3] = motion["body_pos_w"][0, 0]
    d.qpos[root_q+3:root_q+7] = motion["body_quat_w"][0, 0]
    d.qpos[qadr] = motion["joint_pos"][0]
    if start != "stand":
        p = events["lie_face_up" if start == "supine" else "lie_face_down"]
        d.qpos[root_q+2] = p["root_pos_z"]; d.qpos[root_q+3:root_q+7] = p["root_quat"]; d.qpos[qadr] = p["joint_pos"]
    d.ctrl[act] = d.qpos[qadr]
    mujoco.mj_forward(model, d)
    hist, last, log = None, np.zeros(23, np.float32), []
    for k in range(int(args.seconds / scene["control_dt"])):
        f = round(d.time * 50) % len(motion["joint_pos"])
        q_root = d.qpos[root_q+3:root_q+7]
        terms = [np.concatenate([motion["joint_pos"][f], motion["joint_vel"][f]]),
                 rot(quat_mul(d.xquat[torso] * [1, -1, -1, -1], motion["body_quat_w"][f, anchor_i]))[:, :2].reshape(6),
                 d.qvel[root_v+3:root_v+6].copy(), d.qpos[qadr] - default, d.qvel[vadr].copy(), last.copy(),
                 rot(q_root).T @ np.array([0, 0, -1.])]
        hist = [[t] * 4 for t in terms] if hist is None else [h[1:] + [t] for h, t in zip(hist, terms)]
        obs = np.concatenate([np.concatenate(h) for h in hist]).astype(np.float32)[None]
        a = np.clip(session.run(None, {session.get_inputs()[0].name: obs})[0][0], -10, 10)
        d.ctrl[act] = default + scale * a
        for _ in range(decimation):
            mujoco.mj_step(model, d)
        last = a.astype(np.float32)
        err = np.linalg.norm(d.qpos[qadr] - motion["joint_pos"][round(d.time * 50) % len(motion["joint_pos"])])
        log.append((d.xpos[pelvis, 2], err))
    a = np.array(log)
    up = np.argmax(a[:, 0] >= .6) if (a[:, 0] >= .6).any() else None
    print(f"{start:7s} first pelvis>=0.6 at step {up}; last 3 s: min pelvis {a[-150:,0].min():.2f}, "
          f"mean joint err {a[-150:,1].mean():.2f}")


for s in ("stand", "supine", "prone"):
    run(s)
