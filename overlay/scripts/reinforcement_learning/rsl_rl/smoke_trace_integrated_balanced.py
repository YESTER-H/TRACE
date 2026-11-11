# SPDX-License-Identifier: Apache-2.0
"""Simulator checks for Integrated-U15-Balanced (12 s Tracking-origin time-out, 30/70 resets).

usage: /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/smoke_trace_integrated_balanced.py --headless
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
from cyclo_lab.manager_based.trace_recovery.integrated import TRACKING, RECOVERY

TASK = "Cyclo-TRACE-K1-Rev1-Integrated-U15-Balanced-Play"
N = 512


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("PASS", message, flush=True)


env = gym.make(TASK, cfg=parse_env_cfg(TASK, device="cuda:0", num_envs=N))
u = env.unwrapped
c = u.command_manager.get_term("reference_trajectory")
zero = torch.zeros((N, 23), device=u.device)
terms = u.termination_manager.active_terms
print("termination order:", terms, flush=True)
check(terms[-1] == "post_tracking_horizon_complete", "post-Tracking horizon is still evaluated last")
check("tracking_origin_horizon_complete" in terms and "time_out" in terms, "12 s Tracking-origin time-out added, 30 s global kept")
check(u.max_episode_length == 1500, "global episode limit stays 30 s")

env.reset()
tracking = float((c.episode_origins == TRACKING).float().mean())
identities = c.recovery_reference_ids[c.episode_origins == RECOVERY]
supine = float((identities == 0).float().mean())
print(f"reset split: Tracking {tracking:.3f}, Supine {(1-tracking)*supine:.3f}, Prone {(1-tracking)*(1-supine):.3f}", flush=True)
check(abs(tracking-.3) < .07 and abs(supine-.5) < .08, "resets are about Tracking 30 / Supine 35 / Prone 35")

ids = torch.arange(N, device=u.device)
c.forced_regimes[ids] = torch.where(ids < N//2, TRACKING, RECOVERY)
env.reset()
u.episode_length_buf[:] = 598
_, _, terminated, truncated, _ = env.step(zero)
term = u.termination_manager.get_term("tracking_origin_horizon_complete")
check(not bool(term.any()), "no Tracking-origin time-out before 600 steps")
u.episode_length_buf[:] = 599
before = c.episode_origins.clone()
_, _, terminated, truncated, _ = env.step(zero)
term = u.termination_manager.get_term("tracking_origin_horizon_complete")
check(torch.equal(term, before == TRACKING), "600th step times out exactly the Tracking-origin envs")
check(bool(truncated[before == TRACKING].all()), "it is reported as a time-out (value bootstrap)")
env.close()
app.close()
