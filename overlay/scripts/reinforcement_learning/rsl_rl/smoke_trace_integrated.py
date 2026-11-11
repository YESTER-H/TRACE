# SPDX-License-Identifier: Apache-2.0
"""Simulator checks for the U1.5 integrated TRACE state machine (no training, no checkpoint).

usage: /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/smoke_trace_integrated.py --headless
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

TASK = "Cyclo-TRACE-K1-Rev1-Integrated-U15-Play"
N = 8


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("PASS", message, flush=True)


env = gym.make(TASK, cfg=parse_env_cfg(TASK, device="cuda:0", num_envs=N))
u = env.unwrapped
c = u.command_manager.get_term("reference_trajectory")
zero = torch.zeros((N, 23), device=u.device)


def step(k=1):
    for _ in range(k):
        obs, reward, terminated, truncated, _ = env.step(zero)
    return obs, reward, terminated, truncated


def force_reset(regimes, identities=None, frames=None):
    ids = torch.arange(N, device=u.device)
    c.forced_regimes[ids] = torch.as_tensor(regimes, device=u.device)
    if identities is not None:
        c.forced_reference_ids[ids] = torch.as_tensor(identities, device=u.device)
        c.forced_frames[ids] = torch.as_tensor(frames, device=u.device)
    return env.reset()[0]


obs = force_reset([TRACKING]*4+[RECOVERY]*4)
check(tuple(obs["policy"].shape) == (N, 508) and tuple(obs["critic"].shape) == (N, 256)
      and tuple(obs["router"].shape) == (N, 1), "observation groups policy508/critic256/router1")
check(torch.equal(c.regimes, c.episode_origins) and c.regimes.tolist() == [0]*4+[1]*4, "forced direct-reset regimes")
check(torch.equal(obs["router"].squeeze(-1), (c.regimes == TRACKING).float()), "router w_T is 1 Tracking / 0 Recovery")
check(bool((c.recovery_target_error()[:4] == 0).all()), "persistent target error is Recovery-only")
live = c.reference.joint_position[c.tracking_phase]
check(torch.allclose(c.joint_pos[:4], live[:4]), "Tracking target is the live Tracking frame")

# Recovery completion from the penultimate frame (standing) -> Transition, then Tracking.
last = [ref.num_frames-2 for ref in c.recovery_references]
force_reset([RECOVERY]*N, [0, 1]*4, [last[0], last[1]]*4)
before = {m: int(c.events[f"{m}_transition_from_terminal"]) for m in ("supine", "prone")}
step(3)
upright = c.regimes == TRANSITION
print("regimes after completion steps:", c.regimes.tolist(), flush=True)
check(bool(upright.any()), "upright final-frame Recovery enters Transition instead of terminating")
check(sum(int(c.events[f"{m}_transition_from_terminal"])-before[m] for m in before) == int(upright.sum()),
      "terminal transition events match")
bonus = u.reward_manager.active_terms.index("recovery_success")
step(1)
paid = u.reward_manager._step_reward[:, bonus]
check(torch.equal(paid > 0, upright) and bool(torch.allclose(paid[upright], paid.new_tensor(500.))),
      "success bonus is paid on the first Transition action")
step(1)
check(bool((u.reward_manager._step_reward[:, bonus] == 0).all()), "success bonus is paid once")
obs, *_ = step(8)
ids = torch.where(c.regimes == TRANSITION)[0]
expected = c.transition_steps[ids].float()/(c.transition_steps_required-1)
check(torch.allclose(obs["router"].squeeze(-1)[ids], expected), "router w_T equals transition alpha")
check(bool((c.transition_steps[ids] == 10).all()), "transition counter advances once per step")
c.transition_steps[ids] = c.transition_steps_required-2
step(1)
check(bool((c.regimes[ids] == TRANSITION).all()), "74 executed Transition actions do not finish")
step(1)
check(bool((c.regimes[ids] == TRACKING).all() and c.post_tracking_active[ids].all()),
      "75th Transition action hands off to bounded post-Tracking")
xy = (c.robot_anchor_pos_w[ids, :2]-c.live_anchor_pos_w[ids, :2]).norm(dim=-1)
print("live anchor xy offset after handoff [m]:", xy.tolist(), flush=True)

# Post-Tracking horizon: completion is a time-out, unless a failure already ended the episode.
c.post_tracking_steps[ids] = c.post_tracking_steps_required-1
done_before = sum(int(c.events[f"{m}_post_tracking_completions"]) for m in ("supine", "prone"))
stop_before = sum(int(c.events[f"{m}_post_tracking_interruptions"]) for m in ("supine", "prone"))
_, _, terminated, truncated = step(1)
done = sum(int(c.events[f"{m}_post_tracking_completions"]) for m in ("supine", "prone"))-done_before
stop = sum(int(c.events[f"{m}_post_tracking_interruptions"]) for m in ("supine", "prone"))-stop_before
print(f"post-Tracking horizon: completed {done}, interrupted {stop}", flush=True)
check(done+stop == len(ids), "every post-Tracking env completes or is interrupted exactly once")
check(bool(truncated[ids].sum() >= done), "horizon completion is reported as a time-out")

# Tracking failure: a Tracking env teleported to the supine floor pose fails next step.
force_reset([TRACKING]*N)
before = int(c.events["tracking_origin_failures"])
c.write_reference_state(torch.arange(N, device=u.device), c.recovery_references[0],
                        torch.zeros(N, dtype=torch.long, device=u.device))
_, _, terminated, _ = step(1)
check(bool(terminated.all()) and int(c.events["tracking_origin_failures"])-before == N,
      "fallen Tracking env terminates as a Tracking-origin failure")

# Reacquisition: a Recovery env placed on the live Tracking pose hands off after 5 steps.
force_reset([RECOVERY]*N, [0]*N, [0]*N)
c.write_reference_state(torch.arange(N, device=u.device), c.reference, c.tracking_phase.clone())
before = int(c.events["supine_transition_from_reacquisition"])
paid = torch.zeros(N, device=u.device)
for k in range(9):
    step(1)
    paid += u.reward_manager._step_reward[:, bonus]
    print(f"  step {k+1}: candidate {c.reacquisition_candidate.sum().item()} "
          f"persistence {c.reacquisition_steps.tolist()} regimes {c.regimes.tolist()}", flush=True)
gained = int(c.events["supine_transition_from_reacquisition"])-before
print("reacquisition transitions:", gained, flush=True)
check(gained > 0, "Recovery env matching the live Tracking pose reacquires")
handed = (c.regimes == TRANSITION)&(c.transition_steps >= 1)  # first Transition action already rewarded
check(torch.equal(paid > 0, handed) and bool(torch.allclose(paid[handed], paid.new_tensor(500.))),
      "reacquisition handoff earns the same one-time success bonus")
env.close()
app.close()
