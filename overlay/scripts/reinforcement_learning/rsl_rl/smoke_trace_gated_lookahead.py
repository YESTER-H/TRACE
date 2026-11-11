# SPDX-License-Identifier: Apache-2.0
"""Checks for the BeyondMimic failure look-ahead in Integrated-Gated-Lookahead.

usage: /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/smoke_trace_gated_lookahead.py --headless
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import cyclo_lab  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("PASS", message, flush=True)


def command(task):
    env = gym.make(task, cfg=parse_env_cfg(task, device="cuda:0", num_envs=8))
    return env, env.unwrapped.command_manager.get_term("reference_trajectory")


env, c = command("Cyclo-TRACE-K1-Rev1-Integrated-Gated")
check(c.adaptive_kernels == [None, None] and c.cfg.adaptive_inner_uniform == .2, "run 21 task unchanged: look-ahead off, inner uniform .2")
env.close()

env, c = command("Cyclo-TRACE-K1-Rev1-Integrated-Gated-Lookahead")
K = c.adaptive_kernels[0]
bin_s = (c.recovery_references[0].num_frames-1)/20*env.unwrapped.step_dt
check(K is not None and K.shape == (20, 20), f"Supine kernel built (bin {bin_s:.3f} s)")
row = K[5]
check(torch.allclose(row[5:11], torch.tensor([.8**(u*bin_s) for u in range(6)], device=row.device))
      and not bool(row[:5].any()) and not bool(row[11:].any()), "bin 5 takes bins 5..10 (2 s ahead), weight .8**seconds")
check(torch.isclose(K[19, 19], K[5, 5:11].sum()), "last bin replicated past the end")
c.failure_score_by_reference_bin[0] = 0
c.failure_score_by_reference_bin[0, 9] = 1
p = c.adaptive_bin_probabilities(0)
check(c.cfg.adaptive_inner_uniform == 0., "adaptive branch has no extra uniform (50:50 overall)")
check(torch.isclose(p.sum(), torch.tensor(1., device=p.device)), "probabilities sum to 1")
check(bool((p[4:10] > p[0]).all()) and bool((p[4:9] < p[9]).all()) and torch.isclose(p[10], p[0]),
      "a failure in bin 9 raises bins 4-9 (decaying backwards), not bin 10 or later")
env.close()
app.close()
