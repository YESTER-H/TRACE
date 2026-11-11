# SPDX-License-Identifier: Apache-2.0
"""Deployment-only heading alignment of the live dance reference after a fall (no retraining).

Recovery was trained lying in one heading relative to the dance reference (reset yaw noise +-.05 rad), and the
actor observes the dance anchor orientation relative to its own torso. A push that turns the robot before it lands
leaves it lying in another heading, which the Recovery expert has never seen (from frame 0: 90 deg off -> 0-6 %
success). Once the fallen robot lies still, this rotates the dance reference about world z so that the
robot-to-reference relation equals the trained one; the robot then gets up and dances on facing the new heading.

The yaw of a lying body is taken from its head direction (torso z axis on the floor plane), compared with the
same direction in frame 0 of the Supine or Prone reference (chosen by whether the chest faces up).
"""
import math
import torch
from isaaclab.utils.math import quat_apply, quat_mul, quat_from_euler_xyz, subtract_frame_transforms, matrix_from_quat

SETTLE_STEPS = 15         # lying still this long (0.3 s at 50 Hz) before the heading is read
RELOCK_SECONDS = 3.       # still lying this long after a lock (an attempt failed): read it again
LYING_PELVIS = .3
STILL_LINEAR, STILL_ANGULAR = .5, 1.5


def _head_angle(quat):
    head = quat_apply(quat, torch.tensor([0., 0., 1.], device=quat.device).expand(len(quat), 3))
    return torch.atan2(head[:, 1], head[:, 0])


def _chest_up(quat):
    return quat_apply(quat, torch.tensor([1., 0., 0.], device=quat.device).expand(len(quat), 3))[:, 2] > 0


class HeadingAlignment:
    def __init__(self, command, dt):
        self.c = command
        n, dev = command.num_envs, command.device
        self.offset = torch.zeros(n, device=dev)  # yaw (rad) applied to the dance reference
        self.still = torch.zeros(n, dtype=torch.long, device=dev)
        self.since_lock = torch.full((n,), 1e9, device=dev)
        self.locks = torch.zeros(n, dtype=torch.long, device=dev)
        self.dt = dt
        self.enabled = True
        anchor = command.logical_anchor_id
        # head angle of each recovery reference at frame 0 (index 0 Supine, 1 Prone)
        self.reference_head = torch.stack([_head_angle(ref.body_orientation_w[:1, anchor])[0]
                                           for ref in command.recovery_references])

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.offset[ids] = 0.
        self.still[ids] = 0
        self.since_lock[ids] = 1e9

    def update(self, fallen):
        """fallen: [N] bool, the robot is down (exhibition state). Returns envs that locked this step."""
        c = self.c
        pelvis = c.robot_body_pos_w[:, 0, 2]
        lin = c.robot_body_lin_vel_w[:, 0].norm(dim=-1)
        ang = c.robot_body_ang_vel_w[:, 0].norm(dim=-1)
        lying_still = fallen & (pelvis < LYING_PELVIS) & (lin < STILL_LINEAR) & (ang < STILL_ANGULAR)
        self.still = torch.where(lying_still, self.still+1, torch.zeros_like(self.still))
        self.since_lock += self.dt
        lock = (self.enabled & (self.still >= SETTLE_STEPS) & (self.since_lock >= RELOCK_SECONDS))
        if bool(lock.any()):
            q = c.robot_anchor_quat_w[lock]
            reference = torch.where(_chest_up(q), self.reference_head[0], self.reference_head[1])
            delta = _head_angle(q)-reference
            self.offset[lock] = torch.atan2(torch.sin(delta), torch.cos(delta))
            self.since_lock[lock] = 0.
            self.locks[lock] += 1
        return lock

    def rotated_reference_quat(self, quat):
        zero = torch.zeros_like(self.offset)
        return quat_mul(quat_from_euler_xyz(zero, zero, self.offset), quat)


def install(env_cfg, holder):
    """Route the policy's live anchor-orientation term through holder['alignment'] (set after env creation)."""
    def live_anchor_orientation_aligned(env, command_name):
        c = env.command_manager.get_term(command_name)
        q = c.reference.body_orientation_w[c.tracking_phase, c.logical_anchor_id]
        alignment = holder.get('alignment')
        if alignment is not None:
            q = alignment.rotated_reference_quat(q)
        _, relative = subtract_frame_transforms(c.robot_anchor_pos_w, c.robot_anchor_quat_w, c.robot_anchor_pos_w, q)
        return matrix_from_quat(relative)[..., :2].reshape(env.num_envs, 6)
    env_cfg.observations.policy.motion_anchor_ori_b.func = live_anchor_orientation_aligned
