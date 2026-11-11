# SPDX-License-Identifier: Apache-2.0
"""Simulator checks for Integrated-Gated (gate observations/labels, trimmed Supine, early-upright handoff).

usage: /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/smoke_trace_integrated_gated.py --headless
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
from cyclo_lab.manager_based.trace_recovery.integrated import TRACKING, RECOVERY, TRANSITION
from cyclo_lab.manager_based.trace_recovery.gated import gate_frames, GATE_HISTORY

TASK = "Cyclo-TRACE-K1-Rev1-Integrated-Gated-Play"
N = 64


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("PASS", message, flush=True)


env = gym.make(TASK, cfg=parse_env_cfg(TASK, device="cuda:0", num_envs=N))
u = env.unwrapped
c = u.command_manager.get_term("reference_trajectory")
zero = torch.zeros((N, 23), device=u.device)
check([r.num_frames for r in c.recovery_references] == [336, 470], "Supine reference trimmed 436 -> 336, Prone unchanged")
check(c.cfg.early_upright_steps == 25, "early-upright handoff after 25 upright steps")

ids = torch.arange(N, device=u.device)
c.forced_regimes[ids] = torch.where(ids < N//2, TRACKING, RECOVERY)
c.forced_reference_ids[ids] = 0
c.forced_frames[ids] = torch.where(ids < N//2, -1, 0)
obs, _ = env.reset()
check(set(obs) >= {"policy", "critic", "gate", "gate_target"} and "router" not in obs, "groups policy/critic/gate/gate_target, no oracle router")
check(obs["policy"].shape == (N, 508) and obs["gate"].shape == (N, 375) and obs["gate_target"].shape == (N, 3),
      "policy508 / gate 5x75 / gate_target3")
target = obs["gate_target"]
check(torch.equal(target[:, 0], (ids < N//2).float()), "gate target y_T: Tracking 1, frame-0 Supine 0")
check(not bool(target[:, 1].any()), "consistency mask is off on the reset observation")

for _ in range(3):
    obs, *_ = env.step(zero)
frames = gate_frames(obs["gate"])
policy_q = obs["policy"][:, 4*(46+6+3):4*(46+6+3+23)].reshape(N, 4, 23)
check(torch.allclose(frames[:, -4:, 3:26], policy_q), "gate joint-position frames equal the policy history, newest last")
check(bool(obs["gate_target"][:, 1].all()), "consistency mask on within an unchanged regime")
check(frames.shape[1] == GATE_HISTORY, "five gate frames stored (H4 at t and t-1)")

# A standing Supine frame: recovered-like label, then early-upright handoff, error rule exempt.
late = c.recovery_references[0].num_frames-40
c.forced_regimes[ids] = RECOVERY
c.forced_reference_ids[ids] = 0
c.forced_frames[ids] = late
env.reset()
check(bool(c.recovered_like.all()), f"Supine frame {late} reset is recovered-like (keeps the Tracking gate target)")
check(bool((c.gate_supervision()[:, 0] == 1).all()), "recovered-like Recovery samples carry y_T = 1")
seen_transition, step = torch.zeros(N, dtype=torch.bool, device=u.device), 0
while step < 40 and not bool(seen_transition.all()):
    env.step(zero)
    step += 1
    seen_transition |= c.regimes == TRANSITION
upright = c.upright_mask()
print(f"after {step} steps: Transition {int(seen_transition.sum())}/{N}, upright {int(upright.sum())}/{N}", flush=True)
check(bool(seen_transition.any()) and step >= 25, "upright Recovery hands off to Transition early (>= 25 steps), before the final frame")
check(int(c.events["supine_transition_from_early_upright"]) > 0, "handoff reason logged as early_upright")
t = c.gate_supervision()
mid = c.regimes == TRANSITION
check(bool((t[mid, 2] == 1).all()) and torch.allclose(t[mid, 0], c.transition_alpha[mid]), "Transition label = alpha, mask on")
env.close()
app.close()
