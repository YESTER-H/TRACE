# SPDX-License-Identifier: Apache-2.0
"""Checks for Integrated-Gated-ProneTrim (run 22 + Prone trim80).

usage: /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/smoke_trace_gated_prone_trim.py --headless
"""
import argparse
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import cyclo_lab  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from cyclo_lab.manager_based.trace_recovery.integrated import RECOVERY


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("PASS", message, flush=True)


source = np.load(os.path.join(os.path.dirname(cyclo_lab.__file__), "..", "data/motions/K1_rev1/trace_inputs/prone.npz"))
for task in ("Cyclo-TRACE-K1-Rev1-Integrated-Gated-ProneTrim", "Cyclo-TRACE-K1-Rev1-Integrated-Gated-ProneTrim-Play"):
    env = gym.make(task, cfg=parse_env_cfg(task, device="cuda:0", num_envs=16))
    c = env.unwrapped.command_manager.get_term("reference_trajectory")
    check([r.num_frames for r in c.recovery_references] == [336, 390], f"{task}: Supine 336 / Prone 390 frames")
    prone = c.recovery_references[1]
    check(np.allclose(np.sort(prone.joint_position[0].cpu().numpy()), np.sort(source["joint_pos"][80]), atol=1e-5)
          and np.allclose(prone.body_position_w[:, 0, 2].cpu().numpy(), source["body_pos_w"][80:, 0, 2], atol=1e-5),
          f"{task}: Prone frame k == original frame k+80 (joints, pelvis height)")
    if not task.endswith("Play"):
        check(c.adaptive_kernels[0] is not None and c.adaptive_kernels[1] is not None
              and c.cfg.adaptive_inner_uniform == 0., "look-ahead on for both motions, 50:50 mix")
    ids = torch.arange(16, device=env.unwrapped.device)
    c.forced_regimes[ids] = RECOVERY
    c.forced_reference_ids[ids] = 1
    c.forced_frames[ids] = 0
    env.reset()
    zero = torch.zeros((16, 23), device=env.unwrapped.device)
    for _ in range(20):
        env.step(zero)
    check(bool((c.regimes == RECOVERY).all()) and bool((c.recovery_reference_ids == 1).all()),
          f"{task}: frame-0 Prone reset stays in Recovery for 20 steps")
    env.close()
app.close()
