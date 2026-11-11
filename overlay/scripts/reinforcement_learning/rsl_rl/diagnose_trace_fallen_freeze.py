# SPDX-License-Identifier: Apache-2.0
"""Is a fallen robot that does not get up frozen, and why?  Copy of diagnose_trace_push_recovery.py plus, per env,
leg pose / motion / action / torque statistics while down (freeze_summary.json).

Which pushes can the exhibition robot get up from?  Batch, no-reset, display-free.

Every env dances from frame 0 (Tracking, no reset noise) and gets one pelvis velocity push of a given
direction (8 screen directions, 45 deg apart; 0 deg = away from the camera, +y) and speed, at a time spread over
one dance loop. Terminations are ignored (NaN guard only), as in exhibition_trace_push.py, and each env runs
--seconds. Per env: fell (pelvis < .45 m), how it lay at its lowest (supine / prone / side, from the pelvis forward
axis), stood up (pelvis >= .6 m, tilt <= 30 deg), back to the dance (upright with gate w_T >= .9 for 1 s) and when.

Written to envs.csv / cells.csv / summary.json in --output-dir (never overwritten).

usage (inside the container, /workspace/cyclo_lab):
  /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/diagnose_trace_push_recovery.py --headless \
    --output-dir logs/eval/diag_push_recovery_<time> --checkpoint <run>/model_40600.pt
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
diag_parser.add_argument('--speeds', default='1.2,1.8,2.4')
diag_parser.add_argument('--directions', type=int, default=8)
diag_parser.add_argument('--repeats', type=int, default=4)
diag_parser.add_argument('--push-at', type=float, default=2., help='first push time (s); repeats spread over 6 s')
diag_parser.add_argument('--seconds', type=float, default=25.)
diag_parser.add_argument('--align-heading', action='store_true',
                         help='rotate the dance reference to the lying heading once fallen (trace_heading_alignment)')
diag, remaining = diag_parser.parse_known_args()
output = Path(diag.output_dir).resolve()
output.mkdir(parents=True, exist_ok=False)
SPEEDS = [float(s) for s in diag.speeds.split(',')]
CELLS = [(d, s) for d in range(diag.directions) for s in SPEEDS]
ENVS = [(d, s, r) for d, s in CELLS for r in range(diag.repeats)]
sys.argv = [sys.argv[0], *remaining, '--task', diag.eval_task, '--num_envs', str(len(ENVS)),
            '--seed', '42', '--skip-export', '--play-log-dir', str(output)]

import play  # Launch Kit first, then import simulator modules.
import torch
import trace_heading_alignment
from cyclo_lab.manager_based.trace_recovery.integrated import TraceIntegratedCommand, TRACKING
from cyclo_lab.manager_based.trace_recovery.body_adapter import tensor, refresh_reset_cache
from isaaclab.utils.math import quat_apply


class DanceStartCommand(TraceIntegratedCommand):
    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.forced_regimes[ids] = TRACKING
        super()._resample_command(ids)


original_run = play._run
holder = {}


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
    trace_heading_alignment.install(env_cfg, holder)  # identity until an alignment is attached
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
    # 0 deg = away from the camera (+y), 90 deg = to the screen right (+x)
    push = torch.stack((torch.sin(angle), torch.cos(angle)), -1)*speed[:, None]
    fell = torch.zeros(N, dtype=torch.bool, device=dev)
    fall_step = torch.full((N,), -1, device=dev)
    stood_step = torch.full((N,), -1, device=dev)
    back_step = torch.full((N,), -1, device=dev)
    hold = torch.zeros(N, device=dev)
    min_pelvis = torch.full((N,), 9., device=dev)
    lying_fwd = torch.zeros((N, 3), device=dev)  # pelvis forward (x) axis in world at the lowest point
    nan_resets = torch.zeros(N, device=dev)
    down_gate_sum = torch.zeros(N, device=dev)
    down_steps = torch.zeros(N, device=dev)
    last_gate = torch.ones(N, device=dev)
    heading = None
    if diag.align_heading:
        heading = holder['alignment'] = trace_heading_alignment.HeadingAlignment(c, dt)
    obs = env.get_observations()
    steps = round(diag.seconds/dt)
    names = list(robot.joint_names)
    J = len(names)
    ref_names = [names[int(j)] for j in c.joint_ids]
    ref_cols = torch.tensor([ref_names.index(n) for n in names], device=dev)
    ref0 = torch.stack([r.joint_position[0][ref_cols] for r in c.recovery_references])  # [2, J] frame 0, robot order
    lim_lo = tensor(robot.data.joint_pos_limits)[:, :, 0]
    lim_hi = tensor(robot.data.joint_pos_limits)[:, :, 1]
    effort = tensor(robot.data.joint_effort_limits)
    t_down = torch.zeros(N, device=dev)
    snap = {k: torch.zeros((N, J), device=dev) for k in ('1s', '5s')}
    sum_speed = torch.zeros(N, device=dev); sum_dact = torch.zeros(N, device=dev)
    sum_err = torch.zeros((N, J), device=dev); sum_sat = torch.zeros((N, J), device=dev)
    sum_lim = torch.zeros((N, J), device=dev); sum_act = torch.zeros(N, device=dev)
    prev_action = None
    cos30 = math.cos(math.radians(30))
    with torch.inference_mode():
        for step in range(steps):
            action = policy(obs)
            gate_w_t = policy.last_gate['weights'][:, 0] if getattr(policy, 'last_gate', None) is not None else torch.ones(N, device=dev)
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
            nan_resets += done.float()
            pelvis = c.robot_body_pos_w[:, 0, 2]
            alignment = -tensor(robot.data.projected_gravity_b)[:, 2]
            pushed = step >= push_step
            lower = pushed & (pelvis < min_pelvis)
            min_pelvis = torch.where(lower, pelvis, min_pelvis)
            fwd = quat_apply(c.robot_body_quat_w[:, 0], torch.tensor([1., 0., 0.], device=dev).expand(N, 3))
            lying_fwd[lower] = fwd[lower]
            down = pushed & (pelvis < .3)
            down_gate_sum += torch.where(down, gate_w_t, 0.)
            down_steps += down.float()
            last_gate = gate_w_t.clone()
            new_fall = pushed & ~fell & (pelvis < .45)
            fall_step[new_fall] = step
            fell |= new_fall
            upright = (pelvis >= .6) & (alignment >= cos30)
            new_stood = fell & (stood_step < 0) & upright
            stood_step[new_stood] = step
            hold = torch.where(fell & (back_step < 0) & upright & (gate_w_t >= .9), hold+1, torch.zeros_like(hold))
            new_back = (back_step < 0) & (hold*dt >= 1.)
            back_step[new_back] = step
            if heading is not None:
                heading.update(fell & (back_step < 0))
            dm = fell & (back_step < 0) & (pelvis < .3) & (stood_step < 0)
            q = tensor(robot.data.joint_pos); qd = tensor(robot.data.joint_vel)
            tgt = tensor(robot.data.joint_pos_target); tau = tensor(robot.data.applied_torque)
            t_down += dm.float()
            for k, sec_ in (('1s', 1.), ('5s', 5.)):
                hit = dm & (t_down == round(sec_/dt))
                snap[k][hit] = q[hit]
            w = dm.float()
            sum_speed += w*qd.abs().mean(-1)
            if prev_action is not None:
                sum_dact += w*(action-prev_action).abs().mean(-1)
            sum_act += w*action.abs().mean(-1)
            prev_action = action.clone()
            sum_err += w[:, None]*(tgt-q).abs()
            sum_sat += w[:, None]*(tau.abs() >= .95*effort).float()
            sum_lim += w[:, None]*(((q-lim_lo) < .05) | ((lim_hi-q) < .05)).float()
    rows = []
    for i, (d, s, r) in enumerate(ENVS):
        f = lying_fwd[i]  # lying on the back: the chest (forward axis) faces up
        lying = ('' if not fell[i] else 'supine' if f[2] > .5 else 'prone' if f[2] < -.5 else 'side')
        sec = lambda k: round(float(k-push_step[i])*dt, 2) if k >= 0 else None
        rows.append({'env': i, 'direction_deg': d*360//diag.directions, 'speed': s, 'repeat': r,
                     'push_s': round(float(push_step[i])*dt, 2), 'fell': bool(fell[i]), 'lying': lying,
                     'min_pelvis_m': round(float(min_pelvis[i]), 3),
                     'stood_up_after_push_s': sec(int(stood_step[i])), 'back_to_dance_after_push_s': sec(int(back_step[i])),
                     'final_pelvis_m': round(float(c.robot_body_pos_w[i, 0, 2]), 3),
                     'gate_w_t_while_down': round(float(down_gate_sum[i]/down_steps[i].clamp_min(1)), 3) if down_steps[i] else None,
                     'final_gate_w_t': round(float(last_gate[i]), 3), 'nan_resets': int(nan_resets[i]),
                     'heading_offset_deg': round(math.degrees(float(heading.offset[i])), 1) if heading is not None else None,
                     'heading_locks': int(heading.locks[i]) if heading is not None else None})
    fz = []
    for i, r in enumerate(rows):
        n = float(t_down[i])
        if not r['fell'] or n < round(1./dt):
            continue
        ident = 0 if r['lying'] == 'supine' else 1
        d = {'env': i, 'back': r['back_to_dance_after_push_s'] is not None, 'lying': r['lying'], 'down_s': round(n*dt, 1),
             'mean_joint_speed': round(float(sum_speed[i])/n, 3), 'mean_action_change': round(float(sum_dact[i])/n, 4),
             'mean_abs_action': round(float(sum_act[i])/n, 3), 'gate_w_t_while_down': r['gate_w_t_while_down']}
        for k in ('1s', '5s'):
            if snap[k][i].abs().sum() > 0:
                d['rms_to_ref0_'+k] = round(float((snap[k][i]-ref0[ident]).square().mean().sqrt()), 3)
                d['pose_'+k] = {nm: round(float(snap[k][i, j]), 2) for j, nm in enumerate(names) if any(x in nm for x in ('hip', 'knee', 'ankle'))}
        d['top_target_error'] = {names[j]: round(float(sum_err[i, j])/n, 2) for j in sum_err[i].argsort(descending=True)[:4].tolist()}
        d['torque_sat_frac'] = {names[j]: round(float(sum_sat[i, j])/n, 2) for j in range(J) if sum_sat[i, j]/n > .2}
        d['at_limit_frac'] = {names[j]: round(float(sum_lim[i, j])/n, 2) for j in range(J) if sum_lim[i, j]/n > .2}
        fz.append(d)
    (output/'freeze_summary.json').write_text(json.dumps({'ref0': {m: {nm: round(float(ref0[k, j]), 2) for j, nm in enumerate(names)
        if any(x in nm for x in ('hip', 'knee', 'ankle'))} for k, m in enumerate(('supine', 'prone'))}, 'envs': fz}, indent=1))
    with (output/'envs.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cells = []
    for d, s in CELLS:
        rr = [x for x in rows if x['direction_deg'] == d*360//diag.directions and x['speed'] == s]
        fallen = [x for x in rr if x['fell']]
        back = [x for x in fallen if x['back_to_dance_after_push_s'] is not None]
        cells.append({'direction_deg': d*360//diag.directions, 'speed': s, 'n': len(rr), 'fell': len(fallen),
                      'back_to_dance': len(back),
                      'lying': {k: sum(x['lying'] == k for x in fallen) for k in ('supine', 'prone', 'side')},
                      'mean_back_s': round(sum(x['back_to_dance_after_push_s'] for x in back)/len(back), 2) if back else None})
    with (output/'cells.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0]))
        w.writeheader()
        w.writerows(cells)
    by_lying = {}
    for k in ('supine', 'prone', 'side'):
        f = [x for x in rows if x['lying'] == k]
        by_lying[k] = {'fell': len(f), 'back_to_dance': sum(x['back_to_dance_after_push_s'] is not None for x in f)}
    summary = {'checkpoint': play.args_cli.checkpoint, 'seconds': diag.seconds, 'envs': N, 'align_heading': diag.align_heading,
               'fell': int(fell.sum()), 'back_to_dance': sum(x['back_to_dance_after_push_s'] is not None for x in rows),
               'by_lying': by_lying, 'cells': cells}
    (output/'summary.json').write_text(json.dumps(summary, indent=1))
    print('[PUSH-DIAG]', json.dumps({k: v for k, v in summary.items() if k != 'cells'}), flush=True)
    for x in cells:
        print('[PUSH-CELL]', json.dumps(x), flush=True)
    env.close()


play._run = configured_run
play._play = diagnose
if __name__ == '__main__':
    play.play_diagnostics.run(play.main)
