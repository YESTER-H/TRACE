"""Render the README teaser: three K1 robots in one CPU-MuJoCo scene, one TRACE policy each (run 24 by default).

Left: lying face down (Prone). Middle: lying face up (Supine). Right: standing. The camera looks at the
dancing robot's front. Both lying robots have their heads pointing away from the camera:
  * Supine sits up already facing the camera and goes straight into the dance.
  * Prone gets up facing away from the camera and turns around to rejoin the dance.
Run 23 was trained with one lying heading and mostly fails the Prone start; run 24 (heading-randomized
Recovery) handles it.

usage: TRACE_POLICY=run24 MUJOCO_GL=egl .venv/bin/python make_teaser.py [--seconds 12] [--out ../assets/teaser_run24.gif]
"""
import argparse
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

import trace_mujoco as tm

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--fps", type=int, default=10, help="GIF frame rate")
parser.add_argument("--width", type=int, default=720)
parser.add_argument("--height", type=int, default=360)
parser.add_argument("--spacing", type=float, default=1.3, help="distance between the robots (m)")
parser.add_argument("--out", default=str(tm.REPO / "assets" / f"teaser_{tm.POLICY}.gif"))
parser.add_argument("--mp4", default=None, help="also write an mp4 at the control rate (50 fps)")
args = parser.parse_args()

HEAD_AWAY = math.pi / 2  # head direction +y; the camera sits on the -y side, facing the dance


def robot_only_spec():
    spec = mujoco.MjSpec.from_file(str(tm.K1_MJCF))
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for key in list(spec.keys):
        spec.delete(key)
    for i, name in enumerate(tm.JOINT_NAMES):
        a = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = tm.KP[i]
        a.biasprm[:3] = [0, -tm.KP[i], -tm.KD[i]]
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-tm.EFFORT[i], tm.EFFORT[i]]
    return spec


def build_scene(prefixes):
    """Floor, light and the training solver settings (as in trace_mujoco.build_spec), then one K1 per prefix."""
    ref = tm.build_spec()
    scene = mujoco.MjSpec()
    for field in ("timestep", "integrator", "solver", "cone", "iterations", "ls_iterations", "impratio", "tolerance"):
        setattr(scene.option, field, getattr(ref.option, field))
    scene.compiler.degree = False
    world = scene.worldbody
    world.add_light(pos=[0, -2, 4], dir=[0, 0.4, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    tex = scene.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                            rgb1=[.2, .3, .4], rgb2=[.1, .2, .3], width=300, height=300, mark=mujoco.mjtMark.mjMARK_EDGE,
                            markrgb=[.8, .8, .8])
    mat = scene.add_material(name="grid", texrepeat=[5, 5], texuniform=True, reflectance=.2)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, .05], material="grid")
    for prefix in prefixes:
        frame = scene.worldbody.add_frame()
        frame.attach_body(robot_only_spec().body("pelvis"), prefix, "")
    scene.visual.global_.offwidth = max(args.width, 640)
    scene.visual.global_.offheight = max(args.height, 480)
    return scene


class Robot:
    """The K1 interface trace_mujoco's Observer needs, for one prefixed robot in a shared scene."""

    def __init__(self, model, data, prefix):
        self.model, self.data, self.prefix = model, data, prefix
        j = lambda n: model.joint(prefix + n).id
        self.qadr = np.array([model.jnt_qposadr[j(n)] for n in tm.JOINT_NAMES])
        self.vadr = np.array([model.jnt_dofadr[j(n)] for n in tm.JOINT_NAMES])
        self.act = np.array([model.actuator(prefix + n).id for n in tm.JOINT_NAMES])
        root = model.body(prefix + "pelvis").jntadr[0]
        self.root_q, self.root_v = model.jnt_qposadr[root], model.jnt_dofadr[root]
        self.pelvis = model.body(prefix + "pelvis").id
        self.anchor = model.body(prefix + tm.ANCHOR_BODY).id

    def set_state(self, pos, quat, joint_pos):
        self.data.qpos[self.root_q:self.root_q + 3] = pos
        self.data.qpos[self.root_q + 3:self.root_q + 7] = quat
        self.data.qpos[self.qadr] = joint_pos
        self.data.qvel[self.root_v:self.root_v + 6] = 0
        self.data.qvel[self.vadr] = 0
        self.data.ctrl[self.act] = joint_pos

    def joint_pos(self): return self.data.qpos[self.qadr]
    def joint_vel(self): return self.data.qvel[self.vadr]
    def root_quat(self): return self.data.qpos[self.root_q + 3:self.root_q + 7].copy()
    def root_ang_vel_b(self): return self.data.qvel[self.root_v + 3:self.root_v + 6].copy()
    def anchor_quat(self): return self.data.xquat[self.anchor].copy()
    def pelvis_height(self): return float(self.data.xpos[self.pelvis, 2])


def yaw_quat(angle):
    return np.array([math.cos(angle / 2), 0, 0, math.sin(angle / 2)])


def start_pose(motions, kind, x):
    """Frame 0 of the dance (stand) or of a get-up reference, turned so that the head points away."""
    ref = motions["tracking" if kind == "stand" else kind]
    pos, quat = ref.body_pos[0, 0].copy(), ref.body_quat[0, 0].copy()
    if kind != "stand":
        turn = HEAD_AWAY - tm.head_angle(ref.body_quat[0, tm.ANCHOR_ID])
        quat = tm.quat_mul(yaw_quat(turn), quat)
    pos[0], pos[1] = x, 0.0
    return pos, quat, ref.joint_pos[0]


def main():
    motions, policy = tm.load_motions(), tm.TracePolicy()
    dance = motions["tracking"]
    layout = [("prone", -args.spacing), ("supine", 0.0), ("stand", args.spacing)]  # left -> right on screen
    model = build_scene([f"{k}/" for k, _ in layout]).compile()
    data = mujoco.MjData(model)
    robots = []
    for kind, x in layout:
        r = Robot(model, data, f"{kind}/")
        r.set_state(*start_pose(motions, kind, x))
        robots.append((kind, r, tm.Observer(), np.zeros(23, np.float32)))
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, -0.2, 0.5]
    cam.distance, cam.elevation, cam.azimuth = 3.7, -10.0, 90.0  # from -y, looking at the dancers' front
    opt = mujoco.MjvOption()

    steps, every = int(args.seconds / tm.CONTROL_DT), round(1 / (tm.CONTROL_DT * args.fps))
    frames, video, log = [], [], {k: [] for k, _ in layout}
    for k in range(steps):
        phase = k % dance.num_frames
        targets = []
        for i, (kind, r, obs, last) in enumerate(robots):
            p, g = obs(r, dance, phase, last)
            with torch.no_grad():
                a, w = policy(torch.from_numpy(p)[None], torch.from_numpy(g)[None])
            a = a[0].numpy()
            robots[i] = (kind, r, obs, a)
            targets.append((r, tm.DEFAULT_JOINT_POS + tm.ACTION_SCALE * a))
            log[kind].append((r.pelvis_height(), float(w[0, 0]), math.degrees(
                math.atan2(*(lambda m: (m[1, 0], m[0, 0]))(r.data.xmat[r.anchor].reshape(3, 3))))))
        for r, t in targets:
            data.ctrl[r.act] = t
        for _ in range(tm.DECIMATION):
            mujoco.mj_step(model, data)
        if k % every == 0 or args.mp4:
            renderer.update_scene(data, cam, opt)
            img = renderer.render()
            if k % every == 0:
                frames.append(Image.fromarray(img))
            if args.mp4:
                video.append(img)

    for kind, rows in log.items():
        a = np.array(rows)
        up = np.argmax((a[:, 0] >= .6) & (a[:, 1] >= .9)) if ((a[:, 0] >= .6) & (a[:, 1] >= .9)).any() else None
        print(f"{kind:7s} start facing {a[0, 2]:+.0f} deg, back in the dance (pelvis>=.6, w_T>=.9) at "
              f"{'never' if up is None else f'{up * tm.CONTROL_DT:.1f} s'}, final facing {a[-1, 2]:+.0f} deg "
              f"(dance faces -90), final pelvis {a[-1, 0]:.2f} m")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")
    if args.mp4:
        import cv2
        w = cv2.VideoWriter(args.mp4, cv2.VideoWriter_fourcc(*"mp4v"), round(1 / tm.CONTROL_DT), (args.width, args.height))
        for img in video:
            w.write(img[:, :, ::-1])
        w.release()
        print(f"wrote {args.mp4}")


if __name__ == "__main__":
    main()
