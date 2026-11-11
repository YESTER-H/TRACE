"""Build the TRACE browser demo with mjswan (MuJoCo WASM + ONNX Runtime Web).

usage:
  .venv/bin/python build_web.py                    # build into web/dist and open a local preview
  MJSWAN_NO_LAUNCH=1 MJSWAN_BASE_PATH=/TRACE/demo/ .venv/bin/python build_web.py   # for GitHub Pages

Layout convention: everything mjswan sees (observations, actions, motion clip) is in the MuJoCo
model's own joint/body order. The exported ONNX policy converts to the canonical order the policy
was trained in, and builds the gate input from the policy observation, so it has one input.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import mujoco
import numpy as np
import onnx
import torch

import mjswan
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp import observations as obs_fns
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjswan.managers.event_manager import EventTermCfg  # mjswan's: adds `label` for manual buttons
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.mdp import observations as tracking_obs
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

import trace_mujoco as tm

HERE = Path(__file__).resolve().parent
GEN = HERE / "generated"
GEN.mkdir(exist_ok=True)
SCENE_NAME = "K1 TRACE"


# ----------------------------------------------------------------------------------------------- robot
def k1_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(tm.K1_MJCF))
    for actuator in tuple(spec.actuators):
        spec.delete(actuator)
    return spec


_REF = mujoco.MjSpec.from_file(str(tm.K1_MJCF)).compile()
ENTITY_JOINTS = [_REF.joint(i).name for i in range(_REF.njnt) if _REF.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
ENTITY_BODIES = [_REF.body(i).name for i in range(1, _REF.nbody)]
assert sorted(ENTITY_JOINTS) == sorted(tm.JOINT_NAMES), ENTITY_JOINTS
assert sorted(ENTITY_BODIES) == sorted(tm.LEGACY_BODY_NAMES), ENTITY_BODIES
# canonical index of each entity joint / legacy index of each entity body
J_E2C = [tm.JOINT_NAMES.index(n) for n in ENTITY_JOINTS]
B_E2L = [tm.LEGACY_BODY_NAMES.index(n) for n in ENTITY_BODIES]


def _group(name):
    return tm.SMALL if tm._is_small(name) else tm.BIG


def actuator_cfgs():
    out = []
    for name in ENTITY_JOINTS:
        g = _group(name)
        out.append(BuiltinPositionActuatorCfg(
            target_names_expr=(name,), stiffness=g["kp"], damping=g["kd"], effort_limit=g["effort"],
            armature=0.00564892 if g is tm.SMALL else 0.01936542))
    return tuple(out)


def robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.764),
            joint_pos={n: float(tm.DEFAULT_JOINT_POS[tm.JOINT_NAMES.index(n)]) for n in tm.JOINT_NAMES},
            joint_vel={".*": 0.0},
        ),
        spec_fn=k1_spec,
        articulation=EntityArticulationInfoCfg(actuators=actuator_cfgs(), soft_joint_pos_limit_factor=0.9),
    )


# ----------------------------------------------------------------------------------------------- motions
def write_entity_order_motion(src: Path, dst: Path):
    d = np.load(src)
    out = {"fps": np.array([50])}
    out["joint_pos"] = d["joint_pos"][:, J_E2C].astype(np.float32)
    out["joint_vel"] = d["joint_vel"][:, J_E2C].astype(np.float32)
    for k in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        out[k] = d[k][:, B_E2L].astype(np.float32)
    # Not np.savez: it streams entries (ZIP64 + data descriptor), leaving the sizes in the local
    # headers empty, and mjswan's browser npz reader takes the sizes from those headers.
    import io
    import zipfile
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:
        for k, v in out.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(v), allow_pickle=False)
            z.writestr(k + ".npy", buf.getvalue())
    return dst


DANCE_NPZ = write_entity_order_motion(tm.MOTION_FILES["tracking"], GEN / "dance1_k1_entity_order.npz")


# ----------------------------------------------------------------------------------------------- events
def lie_down(env, env_ids, root_pos_z: float, root_quat: tuple, joint_pos: tuple,
             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Put the robot on the floor in the first pose of a get-up reference, where it stands now."""
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    state = torch.zeros(n, 13, device=env.device)
    state[:, :2] = asset.data.root_link_pos_w[env_ids, :2]
    state[:, 2] = root_pos_z
    state[:, 3:7] = torch.tensor(root_quat, device=env.device)
    asset.write_root_state_to_sim(state, env_ids=env_ids)
    q = torch.tensor(joint_pos, device=env.device).repeat(n, 1)
    asset.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)


def lie_event(kind: str, label: str) -> EventTermCfg:
    """A control-panel button that lays the robot down in frame 0 of a get-up reference.

    The pose keeps its stored heading, which is the heading relation to the dance used in training.
    """
    m = np.load(tm.MOTION_FILES[kind])
    return EventTermCfg(
        func=lie_down, mode="manual", label=label,
        params={"root_pos_z": float(m["body_pos_w"][0, 0, 2]),
                "root_quat": tuple(float(x) for x in m["body_quat_w"][0, 0]),
                "joint_pos": tuple(float(x) for x in m["joint_pos"][0][J_E2C])},
    )


# ----------------------------------------------------------------------------------------------- env cfg
def env_cfg():
    cfg = make_tracking_env_cfg()
    cfg.scene.num_envs = 1
    cfg.scene.entities = {"robot": robot_cfg()}
    cfg.scene.sensors = ()
    cfg.decimation = tm.DECIMATION
    cfg.episode_length_s = 1e6
    cfg.sim.mujoco.timestep = tm.SIM_DT
    cfg.sim.mujoco.tolerance = 1e-6
    cfg.sim.mujoco.iterations = 100
    cfg.sim.mujoco.ls_iterations = 50
    cfg.sim.mujoco.ccd_iterations = 35
    hist = dict(history_length=4)
    cfg.observations = {"actor": ObservationGroupCfg(terms={
        "command": ObservationTermCfg(func=obs_fns.generated_commands, params={"command_name": "motion"}, **hist),
        "motion_anchor_ori_b": ObservationTermCfg(func=tracking_obs.motion_anchor_ori_b,
                                                  params={"command_name": "motion"}, **hist),
        "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel, **hist),
        "joint_pos": ObservationTermCfg(func=obs_fns.joint_pos_rel, **hist),
        "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel, **hist),
        "actions": ObservationTermCfg(func=obs_fns.last_action, **hist),
        "projected_gravity": ObservationTermCfg(func=obs_fns.projected_gravity, **hist),
    }, concatenate_terms=True, enable_corruption=False)}
    scale = {n: float(tm.ACTION_SCALE[tm.JOINT_NAMES.index(n)]) for n in ENTITY_JOINTS}
    cfg.actions = {"joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",), scale=scale,
                                                       use_default_offset=True)}
    motion = cfg.commands["motion"]
    motion.motion_file = str(DANCE_NPZ)
    motion.anchor_body_name = tm.ANCHOR_BODY
    motion.body_names = tuple(ENTITY_BODIES)
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.joint_position_range = (0.0, 0.0)
    motion.sampling_mode = "start"
    motion.debug_vis = True
    cfg.events = {
        "lie_face_up": lie_event("supine", "Lie down face up (Supine)"),
        "lie_face_down": lie_event("prone", "Lie down face down (Prone)"),
    }
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.curriculum = {}
    cfg.metrics = {}
    return cfg


# ----------------------------------------------------------------------------------------------- policy
class EntityOrderPolicy(torch.nn.Module):
    """mjswan actor obs [1,508] (term-major H4, entity joint order) -> action [1,23] (entity order)."""

    TERMS = (("command", 46), ("anchor", 6), ("ang_vel", 3), ("joint_pos", 23), ("joint_vel", 23),
             ("last_action", 23), ("gravity", 3))

    def __init__(self):
        super().__init__()
        self.policy = tm.TracePolicy()
        e2c = torch.tensor(J_E2C)
        c2e = torch.argsort(e2c)  # canonical position -> entity index
        self.register_buffer("c_from_e", c2e)    # canonical[i] = entity[c_from_e[i]]
        self.register_buffer("e_from_c", e2c)    # entity[j] = canonical[e_from_c[j]]

    def forward(self, obs):
        parts, i = {}, 0
        for name, d in self.TERMS:
            parts[name] = obs[:, i:i + 4*d].reshape(-1, 4, d)
            i += 4*d
        jp = lambda x: x[..., self.c_from_e]
        parts["command"] = torch.cat([jp(parts["command"][..., :23]), jp(parts["command"][..., 23:])], -1)
        for k in ("joint_pos", "joint_vel", "last_action"):
            parts[k] = jp(parts[k])
        policy_obs = torch.cat([parts[n].flatten(1) for n, _ in self.TERMS], -1)
        gate_obs = torch.cat([parts[k] for k in ("ang_vel", "joint_pos", "joint_vel", "last_action", "gravity")],
                             -1).flatten(1)
        action, _ = self.policy(policy_obs, gate_obs)
        return action[:, self.e_from_c]


def export_onnx(path: Path):
    model = EntityOrderPolicy().eval()
    torch.onnx.export(model, torch.zeros(1, 508), str(path), input_names=["actor"], output_names=["action"],
                      dynamic_axes={"actor": {0: "batch"}, "action": {0: "batch"}}, opset_version=17, dynamo=False)
    return onnx.load(str(path))


# ----------------------------------------------------------------------------------------------- build
def _patch_scene_gui_recorder():
    """mjswan 0.12.12 records mjlab's viewer GUI on a stub model without ``ncam``, which mjviser
    0.0.16 reads. Give the stub ``ncam = 0`` (no cameras), and skip the record if it still fails."""
    import types
    import mjswan.build.manifest as manifest
    import mjswan.mjlab.gui as gui

    class _Stub(types.SimpleNamespace):
        ncam = 0

    gui.SimpleNamespace = _Stub
    original = gui.record_scene_gui

    def record():
        try:
            return original()
        except AttributeError as e:
            print(f"[build_web] viewer GUI record skipped: {e}")
            return None

    manifest.record_scene_gui = record


def main():
    _patch_scene_gui_recorder()
    policy_onnx = export_onnx(GEN / "trace_run23.onnx")
    builder = mjswan.Builder(base_path=os.environ.get("MJSWAN_BASE_PATH", "/"))
    project = builder.add_project(name="TRACE")
    cfg = env_cfg()
    scene = project.add_scene_mjlab("Cyclo-TRACE-K1-Rev1", name=SCENE_NAME, env_cfg=cfg)
    policy = scene.add_policy(
        "TRACE run 23", policy_onnx, clip_actions=tm.CLIP_ACTIONS,
        policy_joint_names=["robot/" + n for n in ENTITY_JOINTS],  # names in the composed scene
        default_joint_pos=[float(tm.DEFAULT_JOINT_POS[tm.JOINT_NAMES.index(n)]) for n in ENTITY_JOINTS],
        default=True,
    )
    motion = policy.add_motion(name="dance1", source=str(DANCE_NPZ), fps=50.0, anchor_body_name=tm.ANCHOR_BODY,
                               body_names=tuple(ENTITY_BODIES), default=True, loop=True)
    # Reference frame = control step (sim time x 50 Hz), looping without a reset, as in training.
    motion._config.time_source = "sim"
    app = builder.build()
    if not os.environ.get("MJSWAN_NO_LAUNCH"):
        app.launch()


if __name__ == "__main__":
    main()
