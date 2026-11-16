# SPDX-License-Identifier: Apache-2.0
"""Checks for Integrated-Gated-ProneTrim-YawRand (run 23 + Recovery reset yaw uniform in [-pi, pi)). No training.

usage: /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/smoke_trace_gated_prone_trim_yawrand.py --headless
"""
import argparse
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import cyclo_lab  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_error_magnitude, yaw_quat, quat_mul, quat_inv, euler_xyz_from_quat
from cyclo_lab.manager_based.trace_recovery.integrated import RECOVERY, TRACKING
from cyclo_lab.manager_based.trace_recovery.body_adapter import tensor

N = 64


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("PASS", message, flush=True)


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


for task in ("Cyclo-TRACE-K1-Rev1-Integrated-Gated-ProneTrim-YawRand", "Cyclo-TRACE-K1-Rev1-Integrated-Gated-ProneTrim"):
    randomized = task.endswith("YawRand")
    env = gym.make(task, cfg=parse_env_cfg(task, device="cuda:0", num_envs=N))
    base = env.unwrapped
    c = base.command_manager.get_term("reference_trajectory")
    dev = base.device
    ids = torch.arange(N, device=dev)
    half = N//2
    c.forced_regimes[:half] = RECOVERY  # first half Recovery (Supine / Prone alternating), second half Tracking
    c.forced_reference_ids[:half] = ids[:half] % 2
    c.forced_frames[:half] = 0
    c.forced_regimes[half:] = TRACKING
    env.reset()
    rec = ids[:half]
    yaw = c.recovery_yaw[rec]
    if randomized:
        check(float(yaw.min()) < -2. and float(yaw.max()) > 2., f"{task}: Recovery reset yaws spread over [-pi, pi) "
              f"({math.degrees(float(yaw.min())):.0f} .. {math.degrees(float(yaw.max())):.0f} deg)")
    else:
        check(not c._yaw_random and bool((c.recovery_yaw == 0).all()), f"{task}: yaw randomization off (runs <= 23)")
    # The reset state matches the turned reference: anchor orientation and body positions. The targets are the hidden
    # successor frame k+1 and the reset carries noise, so both tasks get the same small error (compared in the log).
    anchor_error = quat_error_magnitude(c.anchor_quat_w[rec], c.robot_anchor_quat_w[rec])
    check(float(anchor_error.max()) < .35, f"{task}: robot anchor == turned Recovery reference (max {float(anchor_error.max()):.3f} rad)")
    body_error = (c.body_pos_w[rec]-c.robot_body_pos_w[rec]).norm(dim=-1).mean(-1)
    check(float(body_error.max()) < .15, f"{task}: robot bodies == turned Recovery reference (max {float(body_error.max()):.3f} m)")
    # The robot lies turned by recovery_yaw relative to the unturned reference, hence relative to the dance heading.
    unturned = torch.stack([c.recovery_references[i].body_orientation_w[0, c.logical_anchor_id] for i in (ids[:half] % 2).tolist()])
    relative = yaw_quat(quat_mul(c.robot_anchor_quat_w[rec], quat_inv(unturned)))
    relative_yaw = euler_xyz_from_quat(relative)[2]
    # Yaw of a lying body mixes in the roll/pitch/joint reset noise, so allow ~15 deg (both tasks are logged).
    diff = wrap(relative_yaw-yaw).abs()
    check(float(diff.max()) < .26, f"{task}: lying heading = reference heading + recovery_yaw "
          f"(|diff| mean {math.degrees(float(diff.mean())):.1f}, max {math.degrees(float(diff.max())):.1f} deg)")
    check(float(c.recovery_target_error()[rec].max()) < 1., f"{task}: Recovery target error at reset < 1 (persistent rule needs > 4)")
    # Tracking rows are never turned: their target is the live dance frame.
    live = c.live_body_quat_w[half:, c.reference_anchor_body_id]
    check(float(quat_error_magnitude(c.anchor_quat_w[half:], live).max()) < 1e-5, f"{task}: Tracking targets untouched")
    zero = torch.zeros((N, 23), device=dev)
    for _ in range(50):
        env.step(zero)
    check(bool(torch.isfinite(tensor(c.robot.data.joint_pos)).all()), f"{task}: 50 steps finite")
    env.close()
app.close()
