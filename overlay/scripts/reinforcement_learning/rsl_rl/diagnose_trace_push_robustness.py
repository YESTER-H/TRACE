# SPDX-License-Identifier: Apache-2.0
"""Why does a policy fall less after a push?  Batch, no-reset, display-free; the balance phase only.

Same pushes as diagnose_trace_push_recovery.py (dance from frame 0, one pelvis velocity push per env: 8 directions x
--speeds x --repeats, spread over 6 s). Each env is followed for --after seconds after its push. With --gate:
  native     the learned gate picks the experts (deployment)
  tracking   gate forced to w_T = 1: the Tracking expert alone has to keep the robot up
  recovery   gate forced to w_T = 0 from the push on: the Recovery expert alone
Per env: fell (pelvis < .45 m), lowest pelvis, largest torso tilt, lowest gate w_T and when the gate first
handed over to Recovery (w_T < .5) relative to the push, feet-on-ground steps taken (touchdowns), and the
largest pelvis horizontal speed.

Written to envs.csv / summary.json in --output-dir (never overwritten).

usage (inside the container, /workspace/cyclo_lab):
  /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/diagnose_trace_push_robustness.py --headless \
    --gate native --output-dir logs/eval/diag_push_robustness_<time> --checkpoint <run>/model.pt
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

diag_parser = argparse.ArgumentParser(add_help=False)
diag_parser.add_argument('--output-dir', required=True)
diag_parser.add_argument('--eval-task', default='Cyclo-TRACE-K1-Rev1-Integrated-Gated-ProneTrim-Play')
diag_parser.add_argument('--gate', choices=('native', 'tracking', 'recovery'), default='native')
diag_parser.add_argument('--speeds', default='1.8,2.4,3.0')
diag_parser.add_argument('--directions', type=int, default=8)
diag_parser.add_argument('--repeats', type=int, default=4)
diag_parser.add_argument('--push-at', type=float, default=2.)
diag_parser.add_argument('--after', type=float, default=3., help='seconds followed after each push')
diag, remaining = diag_parser.parse_known_args()
output = Path(diag.output_dir).resolve()
output.mkdir(parents=True, exist_ok=False)
SPEEDS = [float(s) for s in diag.speeds.split(',')]
ENVS = [(d, s, r) for d in range(diag.directions) for s in SPEEDS for r in range(diag.repeats)]
sys.argv = [sys.argv[0], *remaining, '--task', diag.eval_task, '--num_envs', str(len(ENVS)),
            '--seed', '42', '--skip-export', '--play-log-dir', str(output)]

import play  # Launch Kit first, then import simulator modules.
import torch
from cyclo_lab.manager_based.trace_recovery.integrated import TraceIntegratedCommand, TRACKING
from cyclo_lab.manager_based.trace_recovery.body_adapter import tensor, refresh_reset_cache


class DanceStartCommand(TraceIntegratedCommand):
    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.forced_regimes[ids] = TRACKING
        super()._resample_command(ids)


original_run = play._run


def configured_run(env_cfg, agent_cfg):
    command = env_cfg.commands.reference_trajectory
    command.class_type = DanceStartCommand
    command.reset_perturbations = False
    command.tracking_start_from_zero = True
    env_cfg.events.joint_home_offset_noise = None
    env_cfg.events.push_robot = None
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.terminations.post_tracking_horizon_complete = None
    if getattr(env_cfg.terminations, 'tracking_origin_horizon_complete', None) is not None:
        env_cfg.terminations.tracking_origin_horizon_complete = None
    env_cfg.terminations.time_out = None
    env_cfg.episode_length_s = 1e6
    return original_run(env_cfg, agent_cfg)


def diagnose(env, policy, *_unused):
    base = env.unwrapped
    c = base.command_manager.get_term('reference_trajectory')
    robot = c.robot
    N, dt, dev = base.num_envs, base.step_dt, base.device
    manager = base.termination_manager
    original_compute = manager.compute

    def nan_only_compute():
        done = original_compute()
        keep = (manager.get_term('nan_detected').clone() if 'nan_detected' in manager.active_terms
                else torch.zeros_like(done))
        manager._terminated_buf &= keep
        manager._truncated_buf &= keep
        return done & keep
    manager.compute = nan_only_compute

    angle = torch.tensor([e[0]*2*math.pi/diag.directions for e in ENVS], device=dev)
    speed = torch.tensor([e[1] for e in ENVS], device=dev)
    push_step = torch.tensor([round((diag.push_at+6.*e[2]/diag.repeats)/dt) for e in ENVS], device=dev)
    push = torch.stack((torch.sin(angle), torch.cos(angle)), -1)*speed[:, None]
    state = {'step': 0}

    if diag.gate != 'native':
        if not hasattr(policy, 'gate_logits'):
            raise TypeError('expected the gated actor as the inference policy')
        native_logits = policy.gate_logits

        def forced_logits(frames):
            logits = native_logits(frames)
            w = torch.ones(N, device=dev) if diag.gate == 'tracking' else \
                torch.where(state['step'] >= push_step, 0., 1.).float()
            forced = torch.log(torch.stack((w, 1-w), -1).clamp_min(1e-6))  # softmax(log w) = w
            return forced
        policy.gate_logits = forced_logits

    feet = [i for i, n in enumerate(robot.body_names) if 'ankle' in n.lower() or 'foot' in n.lower()]
    feet = feet[-2:] if len(feet) >= 2 else feet
    fell = torch.zeros(N, dtype=torch.bool, device=dev)
    fall_step = torch.full((N,), -1, device=dev)
    handover_step = torch.full((N,), -1, device=dev)
    min_pelvis = torch.full((N,), 9., device=dev)
    max_tilt = torch.zeros(N, device=dev)
    min_gate = torch.ones(N, device=dev)
    max_speed = torch.zeros(N, device=dev)
    touchdowns = torch.zeros(N, device=dev)
    low = None
    obs = env.get_observations()
    last = int(push_step.max()) + round(diag.after/dt)
    with torch.inference_mode():
        for step in range(last+1):
            state['step'] = step
            action = policy(obs)
            gate_w_t = policy.last_gate['weights'][:, 0] if getattr(policy, 'last_gate', None) is not None \
                else torch.ones(N, device=dev)
            now = push_step == step
            if bool(now.any()):
                ids = torch.where(now)[0]
                velocity = tensor(robot.data.root_com_vel_w).clone()
                velocity[ids, :2] += push[ids]
                robot.write_root_com_velocity_to_sim_index(root_velocity=velocity[ids].contiguous(),
                                                           env_ids=ids.to(torch.int32))
                refresh_reset_cache(robot)
            obs, _, done, _ = env.step(action)
            policy.reset(done)
            window = (step >= push_step) & (step <= push_step + round(diag.after/dt))
            pelvis = c.robot_body_pos_w[:, 0, 2]
            tilt = torch.rad2deg(torch.acos((-tensor(robot.data.projected_gravity_b)[:, 2]).clamp(-1, 1)))
            vxy = tensor(robot.data.root_com_vel_w)[:, :2].norm(dim=-1)
            min_pelvis = torch.where(window, torch.minimum(min_pelvis, pelvis), min_pelvis)
            max_tilt = torch.where(window & ~fell, torch.maximum(max_tilt, tilt), max_tilt)
            min_gate = torch.where(window, torch.minimum(min_gate, gate_w_t), min_gate)
            max_speed = torch.where(window & (step > push_step), torch.maximum(max_speed, vxy), max_speed)
            new_hand = window & (handover_step < 0) & (gate_w_t < .5)
            handover_step[new_hand] = step
            new_fall = window & ~fell & (pelvis < .45)
            fall_step[new_fall] = step
            fell |= new_fall
            if feet:
                foot_z = tensor(robot.data.body_pos_w)[:, feet, 2]
                down = foot_z < (foot_z.min(dim=0).values.min() + .04)  # near the ground
                if low is not None:
                    touchdowns += torch.where(window & ~fell, (down & ~low).sum(-1).float(), 0.)
                low = down
    rows = []
    for i, (d, s, r) in enumerate(ENVS):
        sec = lambda k: round(float(k-push_step[i])*dt, 2) if k >= 0 else None
        rows.append({'env': i, 'direction_deg': d*360//diag.directions, 'speed': s, 'repeat': r,
                     'fell': bool(fell[i]), 'fall_after_push_s': sec(int(fall_step[i])),
                     'handover_after_push_s': sec(int(handover_step[i])), 'min_gate_w_t': round(float(min_gate[i]), 3),
                     'min_pelvis_m': round(float(min_pelvis[i]), 3), 'max_tilt_deg_before_fall': round(float(max_tilt[i]), 1),
                     'max_pelvis_speed': round(float(max_speed[i]), 2), 'touchdowns': int(touchdowns[i])})
    with (output/'envs.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by_speed = {}
    for s in SPEEDS:
        rr = [x for x in rows if x['speed'] == s]
        held = [x for x in rr if not x['fell']]
        by_speed[str(s)] = {'n': len(rr), 'fell': sum(x['fell'] for x in rr),
                            'handover': sum(x['handover_after_push_s'] is not None for x in rr),
                            'handover_and_held': sum(x['handover_after_push_s'] is not None for x in held),
                            'mean_touchdowns_held': round(sum(x['touchdowns'] for x in held)/max(len(held), 1), 2),
                            'mean_max_tilt_held': round(sum(x['max_tilt_deg_before_fall'] for x in held)/max(len(held), 1), 1)}
    summary = {'checkpoint': play.args_cli.checkpoint, 'gate': diag.gate, 'after_s': diag.after, 'envs': N,
               'feet_bodies': [robot.body_names[i] for i in feet], 'fell': int(fell.sum()), 'by_speed': by_speed}
    (output/'summary.json').write_text(json.dumps(summary, indent=1))
    print('[ROBUST-DIAG]', json.dumps(summary), flush=True)
    env.close()


play._run = configured_run
play._play = diagnose
if __name__ == '__main__':
    play.play_diagnostics.run(play.main)
