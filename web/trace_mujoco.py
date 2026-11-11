"""TRACE policy in plain MuJoCo (CPU): model building, policy, observations, rollouts.

Shared by sim2sim.py (validation), export_onnx.py and the web build. Mirrors the Isaac Lab / Newton
training setup of run 23 (Cyclo-TRACE-K1-Rev1-Integrated-Gated-ProneTrim):
  sim dt 0.005, decimation 4 (50 Hz policy), MuJoCo-Warp settings (Newton solver, implicitfast,
  pyramidal cone, 100 iterations, 50 line-search iterations, impratio 1), implicit PD actuators.
"""
from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CHECKPOINT = REPO / "checkpoints/run23_gated_prone_trim/model_40600.pt"
MOTIONS = REPO / "overlay/source/cyclo_lab/data/motions/K1_rev1"
MOTION_FILES = {
    "tracking": MOTIONS / "trace_inputs/tracking.npz",
    "supine": MOTIONS / "get_up/supine/supine_getup_trim100.npz",
    "prone": MOTIONS / "get_up/prone/prone_getup_trim80.npz",
}
K1_MJCF = Path(os.environ.get(
    "K1_MJCF", Path.home() / "humanoid/cyclo_lab_newton/third_party/ai_sapiens/ai_sapiens_description/mujoco/k1/k1.xml"))

SIM_DT, DECIMATION = 0.005, 4
CONTROL_DT = SIM_DT * DECIMATION
CLIP_ACTIONS = 10.0

# Canonical (policy) joint order, identical to the action term's joint_names.
JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_joint", "right_elbow_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
]
LEG_SCALE, ARM_SCALE = 0.31674936343787685, 0.5299864397817197
BIG = dict(kp=76.45161378437756, kd=4.867060896454468, effort=96.864)     # hips, knees, ankle pitch, waist
SMALL = dict(kp=22.30104227736068, kd=1.4197284458173163, effort=47.277)  # arms, ankle roll


def _is_small(name):
    return "ankle_roll" in name or any(k in name for k in ("shoulder", "elbow", "wrist"))


ACTION_SCALE = np.array([ARM_SCALE if _is_small(n) else LEG_SCALE for n in JOINT_NAMES], np.float32)
KP = np.array([(SMALL if _is_small(n) else BIG)["kp"] for n in JOINT_NAMES])
KD = np.array([(SMALL if _is_small(n) else BIG)["kd"] for n in JOINT_NAMES])
EFFORT = np.array([(SMALL if _is_small(n) else BIG)["effort"] for n in JOINT_NAMES])
_DEFAULTS = {"hip_pitch": -0.3, "knee": 0.63, "ankle_pitch": -0.33, "elbow": 0.6}
DEFAULT_JOINT_POS = np.array([
    next((v for k, v in _DEFAULTS.items() if k in n), 0.0) for n in JOINT_NAMES], np.float32)
for n, v in (("left_shoulder_roll_joint", .2), ("left_shoulder_pitch_joint", .2),
             ("right_shoulder_roll_joint", -.2), ("right_shoulder_pitch_joint", .2)):
    DEFAULT_JOINT_POS[JOINT_NAMES.index(n)] = v

ANCHOR_BODY = "torso_link"
LEGACY_BODY_NAMES = [
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "torso_link", "left_hip_roll_link", "right_hip_roll_link",
    "head_link", "left_shoulder_pitch_link", "right_shoulder_pitch_link", "left_hip_yaw_link", "right_hip_yaw_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link", "left_knee_link", "right_knee_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link", "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_elbow_link", "right_elbow_link", "left_ankle_roll_link", "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand",
]
ANCHOR_ID = LEGACY_BODY_NAMES.index(ANCHOR_BODY)


# ----------------------------------------------------------------------------------------------- model
def build_spec(mjcf=K1_MJCF) -> mujoco.MjSpec:
    """K1 MJCF + floor, motors replaced by implicit PD position actuators, MuJoCo-Warp solver options."""
    spec = mujoco.MjSpec.from_file(str(mjcf))
    spec.option.timestep = SIM_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.iterations = 100
    spec.option.ls_iterations = 50
    spec.option.impratio = 1.0
    spec.option.tolerance = 1e-6
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for i, name in enumerate(JOINT_NAMES):
        a = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = KP[i]
        a.biasprm[:3] = [0, -KP[i], -KD[i]]
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-EFFORT[i], EFFORT[i]]
    world = spec.worldbody
    world.add_light(pos=[0, 0, 3], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[.2, .3, .4], rgb2=[.1, .2, .3], width=300, height=300, mark=mujoco.mjtMark.mjMARK_EDGE,
                           markrgb=[.8, .8, .8])
    mat = spec.add_material(name="grid", texrepeat=[5, 5], texuniform=True, reflectance=.2)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, .05], material="grid")
    return spec


class K1:
    def __init__(self, spec=None):
        self.spec = spec or build_spec()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in JOINT_NAMES])
        self.vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in JOINT_NAMES])
        self.act = np.array([m.actuator(n).id for n in JOINT_NAMES])
        self.pelvis = m.body("pelvis").id
        self.anchor = m.body(ANCHOR_BODY).id

    def set_state(self, root_pos, root_quat_wxyz, joint_pos):
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[:3], d.qpos[3:7] = root_pos, root_quat_wxyz
        d.qpos[self.qadr] = joint_pos
        d.ctrl[self.act] = joint_pos
        mujoco.mj_forward(self.model, d)

    def step(self, target):
        self.data.ctrl[self.act] = target
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)

    # ---- state used by observations (all in the canonical joint order)
    def joint_pos(self):
        return self.data.qpos[self.qadr]

    def joint_vel(self):
        return self.data.qvel[self.vadr]

    def root_quat(self):
        return self.data.qpos[3:7].copy()

    def root_ang_vel_b(self):
        return self.data.qvel[3:6].copy()  # free joint angular velocity is in the body frame

    def anchor_quat(self):
        return self.data.xquat[self.anchor].copy()

    def pelvis_height(self):
        return float(self.data.xpos[self.pelvis, 2])


# ----------------------------------------------------------------------------------------------- math
def quat_conj(q):
    return q * np.array([1, -1, -1, -1])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def quat_rotate_inv(q, v):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3).T @ v


def mat6(q):
    """First two columns of the rotation matrix, row-major flattened (Isaac matrix_from_quat(q)[..., :2])."""
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q / np.linalg.norm(q))
    return m.reshape(3, 3)[:, :2].reshape(6)


# ----------------------------------------------------------------------------------------------- motions
class Motion:
    def __init__(self, path):
        d = np.load(path)
        self.joint_pos = d["joint_pos"].astype(np.float32)
        self.joint_vel = d["joint_vel"].astype(np.float32)
        self.body_pos = d["body_pos_w"].astype(np.float64)
        self.body_quat = d["body_quat_w"].astype(np.float64)  # wxyz
        self.num_frames = len(self.joint_pos)


def load_motions():
    return {k: Motion(v) for k, v in MOTION_FILES.items()}


# ----------------------------------------------------------------------------------------------- policy
def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, n))
        self.register_buffer("std", torch.ones(1, n))

    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-2)


class TracePolicy(nn.Module):
    """Deterministic TRACE actor: (policy obs [N,508], gate obs [N,300]) -> (action [N,23], w_T [N,1]).

    gate obs = 4 frames x 75 (frame-major, oldest first): base_ang_vel3, joint_pos23, joint_vel23,
    last_action23, projected_gravity3.
    """

    def __init__(self, checkpoint=CHECKPOINT):
        super().__init__()
        self.obs_norm = Normalizer(508)
        self.tracking = mlp([508, 512, 256, 128, 23])
        self.recovery = mlp([508, 512, 256, 128, 23])
        self.gate = mlp([300, 512, 64, 2])
        self.gate_norm = Normalizer(75)
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
        self.obs_norm.mean.copy_(sd["obs_normalizer._mean"]); self.obs_norm.std.copy_(sd["obs_normalizer._std"])
        self.gate_norm.mean.copy_(sd["gate_normalizer._mean"]); self.gate_norm.std.copy_(sd["gate_normalizer._std"])
        for name, module in (("actor_experts.tracking", self.tracking), ("actor_experts.recovery", self.recovery),
                             ("gate", self.gate)):
            module.load_state_dict({k[len(name)+1:]: v for k, v in sd.items() if k.startswith(name + ".")})
        self.eval()

    def forward(self, policy_obs, gate_obs):
        x = self.obs_norm(policy_obs)
        mu_t, mu_r = self.tracking(x), self.recovery(x)
        frames = self.gate_norm(gate_obs.reshape(-1, 4, 75)).flatten(1)
        w = torch.softmax(self.gate(frames), -1)
        action = w[:, :1] * mu_t + w[:, 1:] * mu_r
        return action.clamp(-CLIP_ACTIONS, CLIP_ACTIONS), w[:, :1]


# ----------------------------------------------------------------------------------------------- observations
POLICY_TERMS = (("command", 46), ("anchor_ori", 6), ("ang_vel", 3), ("joint_pos", 23), ("joint_vel", 23),
                ("last_action", 23), ("gravity", 3))
GATE_TERMS = ("ang_vel", "joint_pos", "joint_vel", "last_action", "gravity")
HISTORY = 4


class Observer:
    """Term-wise H4 history (oldest first), reset by repeating the first sample, as Isaac Lab does."""

    def __init__(self):
        self.buffers = None

    def terms(self, robot: K1, dance: Motion, phase: int, last_action):
        ref_anchor = dance.body_quat[phase, ANCHOR_ID]
        relative = quat_mul(quat_conj(robot.anchor_quat()), ref_anchor)
        return {
            "command": np.concatenate([dance.joint_pos[phase], dance.joint_vel[phase]]),
            "anchor_ori": mat6(relative),
            "ang_vel": robot.root_ang_vel_b(),
            "joint_pos": robot.joint_pos() - DEFAULT_JOINT_POS,
            "joint_vel": robot.joint_vel(),
            "last_action": last_action,
            "gravity": quat_rotate_inv(robot.root_quat(), np.array([0., 0., -1.])),
        }

    def __call__(self, robot, dance, phase, last_action):
        t = self.terms(robot, dance, phase, last_action)
        if self.buffers is None:
            self.buffers = {k: [v.copy() for _ in range(HISTORY)] for k, v in t.items()}
        else:
            for k, v in t.items():
                self.buffers[k] = self.buffers[k][1:] + [v.copy()]
        policy = np.concatenate([np.concatenate(self.buffers[k]) for k, _ in POLICY_TERMS])
        gate = np.concatenate([np.concatenate([self.buffers[k][h] for k in GATE_TERMS]) for h in range(HISTORY)])
        return policy.astype(np.float32), gate.astype(np.float32)


# ----------------------------------------------------------------------------------------------- rollout
def start_state(motions, start):
    """Robot pose at frame 0 of the dance ("stand") or of a get-up reference ("supine" / "prone")."""
    ref = motions["tracking" if start == "stand" else start]
    return ref.body_pos[0, 0], ref.body_quat[0, 0], ref.joint_pos[0]


class Controller:
    def __init__(self, robot: K1, policy: TracePolicy, motions):
        self.robot, self.policy, self.motions = robot, policy, motions
        self.dance = motions["tracking"]

    def reset(self, start="stand"):
        self.robot.set_state(*start_state(self.motions, start))
        self.observer = Observer()
        self.phase = 0
        self.last_action = np.zeros(23, np.float32)

    @torch.no_grad()
    def step(self):
        policy_obs, gate_obs = self.observer(self.robot, self.dance, self.phase, self.last_action)
        action, w_t = self.policy(torch.from_numpy(policy_obs)[None], torch.from_numpy(gate_obs)[None])
        action = action[0].numpy()
        self.robot.step(DEFAULT_JOINT_POS + ACTION_SCALE * action)
        self.last_action = action
        self.phase = (self.phase + 1) % self.dance.num_frames
        return float(w_t[0, 0])
