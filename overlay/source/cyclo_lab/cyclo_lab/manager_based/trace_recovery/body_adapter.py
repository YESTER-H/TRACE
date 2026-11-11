# SPDX-License-Identifier: Apache-2.0
"""Read-only logical link poses / legacy source-COM velocities on the new stack."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import torch

LAYOUT = json.loads(Path(__file__).with_name("legacy_k1_layout.json").read_text())
BODY_NAMES = LAYOUT["body_names"]
JOINT_NAMES = LAYOUT["joint_names"]


def tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


def rotate(q, v):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    xyz, v = torch.broadcast_tensors(q[..., :3], v)
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + q[..., 3:] * t + torch.cross(xyz, t, dim=-1)


def refresh_reset_cache(robot):
    """Invalidate derived views without fake elapsed time, physics or RNG calls.

    Pinned Isaac Lab 6 cache adapter. Do not invalidate acceleration history.
    Root/joint state writers already invalidate their own authoritative/FK data.
    """
    data = robot.data
    if not hasattr(data, "_projected_gravity_b"):
        raise RuntimeError("Unsupported articulation cache API; gravity refresh cannot be guaranteed")
    for name in ("_projected_gravity_b", "_heading_w", "_root_link_vel_b",
                 "_root_com_vel_b", "_root_link_lin_vel_b", "_root_link_ang_vel_b",
                 "_root_com_lin_vel_b", "_root_com_ang_vel_b"):
        buffer = getattr(data, name, None)
        if buffer is not None:
            buffer.timestamp = -1.0


class LogicalBodyAdapter:
    """Canonical 25-body view. Native tensors stay 24 (or 25 on another backend).

    Pose is link-frame, velocity is the original reference URDF's COM-point
    velocity. This deliberately does not claim old/new inertial/dynamics parity.
    All internal new-stack quaternions are XYZW.
    """

    def __init__(self, robot, urdf_path):
        from isaaclab.utils.math import quat_from_euler_xyz
        self.robot = robot
        physical = list(robot.body_names)
        missing = set(BODY_NAMES) - set(physical)
        if missing not in (set(), {"head_link"}) or set(physical) - set(BODY_NAMES):
            raise ValueError(f"Unsupported K1 body layout: missing={missing}, physical={physical}")
        joint = ET.parse(urdf_path).getroot().find("./joint[@name='head_joint']")
        if (joint is None or joint.get("type") != "fixed" or
                joint.find("parent").get("link") != "torso_link" or
                joint.find("child").get("link") != "head_link"):
            raise ValueError("Expected explicit fixed torso->head joint")
        origin = joint.find("origin")
        xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
        rpy = torch.tensor([float(x) for x in origin.get("rpy", "0 0 0").split()], device=robot.device)
        self.fixed_xyz = torch.tensor(xyz, device=robot.device)
        self.fixed_quat = quat_from_euler_xyz(*rpy.unbind())
        self.physical_names = physical
        self.virtual_head = bool(missing)
        self.torso = physical.index("torso_link")
        self.head = BODY_NAMES.index("head_link")
        self.indices = torch.tensor([physical.index(n) if n in physical else self.torso for n in BODY_NAMES],
                                    dtype=torch.long, device=robot.device)
        self.source_com = torch.tensor([LAYOUT["source_com_by_name"][n] for n in BODY_NAMES],
                                       device=robot.device)

    def states(self):
        from isaaclab.utils.math import quat_mul
        data = self.robot.data
        pose = tensor(data.body_link_pose_w)
        velocity = tensor(data.body_com_vel_w)
        com = tensor(data.body_com_pos_b)
        p, q = pose[..., :3], pose[..., 3:]
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        v, w = velocity[..., :3], velocity[..., 3:]
        # Actual merged/native COM -> link origin, then -> legacy source COM.
        v_link = v - torch.cross(w, rotate(q, com), dim=-1)
        out_p = p[:, self.indices].clone()
        out_q = q[:, self.indices].clone()
        out_w = w[:, self.indices].clone()
        out_v = v_link[:, self.indices].clone()
        if self.virtual_head:
            offset = rotate(q[:, self.torso], self.fixed_xyz)
            out_p[:, self.head] = p[:, self.torso] + offset
            out_q[:, self.head] = quat_mul(q[:, self.torso], self.fixed_quat.expand(len(p), -1))
            out_v[:, self.head] = v_link[:, self.torso] + torch.cross(w[:, self.torso], offset, dim=-1)
        out_v += torch.cross(out_w, rotate(out_q, self.source_com), dim=-1)
        return out_p, out_q, out_v, out_w


def logical_contact_history(env, adapter):
    """Exact world normal-force history; never duplicate merged torso into head.

    Newton's registered shape sensor updates in the same physics pipeline and
    at the same rate/history as the native body sensor. Missing support raises.
    """
    if "contact_forces" not in env.scene.sensors:
        if adapter.virtual_head:
            raise RuntimeError("A virtual head requires exact Newton shape contact reporting")
        rows = []
        for i,name in enumerate(BODY_NAMES):
            sensor = env.scene[f"logical_contact_{i:02d}"]
            if sensor.num_sensors != 1 or sensor.body_names != [name]:
                raise RuntimeError(f"PhysX contact sensor must select only {name}")
            rows.append(tensor(sensor.data.net_forces_w_history)[:,:,0])
        return torch.stack(rows,dim=2)
    sensor = env.scene["contact_forces"]
    names = list(sensor.sensor_names if hasattr(sensor,"sensor_names") else sensor.body_names)
    if set(names) != set(adapter.physical_names):
        raise RuntimeError("Contact sensor body names must exactly cover the physical articulation")
    physical = tensor(sensor.data.net_forces_w_history)
    ids = [names.index(n) if n in names else names.index("torso_link") for n in BODY_NAMES]
    result = physical[:, :, ids].clone()
    if adapter.virtual_head:
        if "head_contact_forces" not in env.scene.sensors:
            raise RuntimeError("Merged head requires an exact shape contact sensor; no approximation is allowed")
        head = tensor(env.scene["head_contact_forces"].data.net_forces_w_history)
        if head.shape[:2] != physical.shape[:2] or head.shape[2] != 1:
            raise RuntimeError("Head/body contact histories are not synchronized")
        result[:, :, adapter.head] = head[:, :, 0]
        result[:, :, BODY_NAMES.index("torso_link")] -= head[:, :, 0]
    return result
