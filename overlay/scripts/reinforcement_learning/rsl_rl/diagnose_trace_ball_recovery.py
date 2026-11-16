# SPDX-License-Identifier: Apache-2.0
"""Which thrown balls can the exhibition robot get up from?  Batch, no-reset, display-free.

The ball-mode counterpart of diagnose_trace_push_recovery.py, with the exhibition's balls (exhibition_trace_push.py):
every env dances from frame 0 (Tracking, no reset noise) and gets one ball thrown at its torso from 2.5 m on a
ballistic arc, from the side opposite to the knock direction (8 screen directions, 45 deg apart; 0 deg = away from the
camera, +y), at a time spread over one dance loop. The ball is removed --ball-seconds later. Terminations are ignored
(NaN guard only), no auto reset, heading alignment as in the exhibition unless --no-align-heading.

Per env: whether the ball touched the robot, fell (pelvis < .45 m), how it lay (supine / prone / side), stood up
(pelvis >= .6 m, tilt <= 30 deg), back to the dance (upright with gate w_T >= .9 for 1 s) and when.
Written to envs.csv / cells.csv / summary.json in --output-dir (never overwritten).

usage (inside the container, /workspace/cyclo_lab):
  /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/diagnose_trace_ball_recovery.py --headless \
    --output-dir logs/eval/diag_ball_recovery_<time> --checkpoint <run>/model_40600.pt
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
diag_parser.add_argument('--levels', default='1,2,3', help='ball levels: 1 soccer ball, 2 3 kg ball, 3 10 kg ball')
diag_parser.add_argument('--ball-speeds', type=float, nargs=3, default=(10., 10., 10.))
diag_parser.add_argument('--ball-seconds', type=float, default=3.)
diag_parser.add_argument('--directions', type=int, default=8)
diag_parser.add_argument('--repeats', type=int, default=4)
diag_parser.add_argument('--push-at', type=float, default=2., help='first throw time (s); repeats spread over 6 s')
diag_parser.add_argument('--seconds', type=float, default=25.)
diag_parser.add_argument('--spacing', type=float, default=8., help='env spacing (m), keeps balls in their own env')
diag_parser.add_argument('--no-align-heading', action='store_true')
diag, remaining = diag_parser.parse_known_args()
output = Path(diag.output_dir).resolve()
output.mkdir(parents=True, exist_ok=False)
LEVELS = [int(s)-1 for s in diag.levels.split(',')]
CELLS = [(d, lv) for d in range(diag.directions) for lv in LEVELS]
ENVS = [(d, lv, r) for d, lv in CELLS for r in range(diag.repeats)]
sys.argv = [sys.argv[0], *remaining, '--task', diag.eval_task, '--num_envs', str(len(ENVS)),
            '--seed', '42', '--skip-export', '--play-log-dir', str(output)]

import play  # Launch Kit first, then import simulator modules.
import torch
import trace_heading_alignment
from cyclo_lab.manager_based.trace_recovery.integrated import TraceIntegratedCommand, TRACKING
from cyclo_lab.manager_based.trace_recovery.body_adapter import tensor
from isaaclab.utils.math import quat_apply

# Same balls as exhibition_trace_push.py: (mass kg, radius m, colour)
BALL_TYPES = ((.45, .11, (.95, .95, .95)), (3., .13, (.95, .6, .1)), (10., .18, (.8, .15, .15)))
LEVEL_NAMES = ('light', 'medium', 'strong')
BALL_DISTANCE = 2.5


def park_offset(level):  # resting place of an idle ball, in its env's corner
    return (diag.spacing*.42, diag.spacing*.42-.5*level, BALL_TYPES[level][1])


class DanceStartCommand(TraceIntegratedCommand):
    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.forced_regimes[ids] = TRACKING
        super()._resample_command(ids)


original_run = play._run
holder = {}


def configured_run(env_cfg, agent_cfg):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
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
    env_cfg.scene.env_spacing = diag.spacing
    trace_heading_alignment.install(env_cfg, holder)
    for level, (mass, radius, color) in enumerate(BALL_TYPES):
        setattr(env_cfg.scene, f'ball_{level}', RigidObjectCfg(
            prim_path=f'{{ENV_REGEX_NS}}/Ball_{level}',
            spawn=sim_utils.SphereCfg(radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                                      rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                      mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                                      collision_props=sim_utils.CollisionPropertiesCfg()),
            init_state=RigidObjectCfg.InitialStateCfg(pos=park_offset(level))))
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

    balls = [base.scene[f'ball_{level}'] for level in range(3)]
    origins = tensor(base.scene.env_origins)
    level_of = torch.tensor([e[1] for e in ENVS], device=dev)
    angle = torch.tensor([e[0]*2*math.pi/diag.directions for e in ENVS], device=dev)
    push_step = torch.tensor([round((diag.push_at+6.*e[2]/diag.repeats)/dt) for e in ENVS], device=dev)
    park_step = push_step+round(diag.ball_seconds/dt)
    # 0 deg = away from the camera (+y), 90 deg = to the screen right (+x)
    knock = torch.stack((torch.sin(angle), torch.cos(angle)), -1)
    fell = torch.zeros(N, dtype=torch.bool, device=dev)
    fall_step = torch.full((N,), -1, device=dev)
    stood_step = torch.full((N,), -1, device=dev)
    back_step = torch.full((N,), -1, device=dev)
    hold = torch.zeros(N, device=dev)
    min_pelvis = torch.full((N,), 9., device=dev)
    lying_fwd = torch.zeros((N, 3), device=dev)
    nan_resets = torch.zeros(N, device=dev)
    min_gap = torch.full((N,), 9., device=dev)
    peak_pelvis_speed = torch.zeros(N, device=dev)
    heading = None
    if not diag.no_align_heading:
        heading = holder['alignment'] = trace_heading_alignment.HeadingAlignment(c, dt)
    obs = env.get_observations()
    steps = round(diag.seconds/dt)
    cos30 = math.cos(math.radians(30))
    g = 9.81
    with torch.inference_mode():
        for step in range(steps):
            action = policy(obs)
            gate_w_t = policy.last_gate['weights'][:, 0] if getattr(policy, 'last_gate', None) is not None else torch.ones(N, device=dev)
            for level in range(3):
                mine = level_of == level
                now = torch.where(mine & (push_step == step))[0]
                if len(now):  # throw from behind the knock direction, arc ending at the torso centre
                    speed = diag.ball_speeds[level]
                    target = c.robot_anchor_pos_w[now]
                    start = target.clone()
                    start[:, :2] -= knock[now]*BALL_DISTANCE
                    pose = torch.cat((start, torch.tensor([0., 0., 0., 1.], device=dev).expand(len(now), 4)), -1)
                    vel = torch.zeros((len(now), 6), device=dev)
                    vel[:, :2] = knock[now]*speed
                    vel[:, 2] = .5*g*BALL_DISTANCE/speed
                    ids = now.to(torch.int32)
                    balls[level].write_root_pose_to_sim_index(root_pose=pose.contiguous(), env_ids=ids)
                    balls[level].write_root_com_velocity_to_sim_index(root_velocity=vel, env_ids=ids)
                done_ids = torch.where(mine & (park_step == step))[0]
                if len(done_ids):
                    rest = origins[done_ids]+torch.tensor(park_offset(level), device=dev)
                    pose = torch.cat((rest, torch.tensor([0., 0., 0., 1.], device=dev).expand(len(done_ids), 4)), -1)
                    ids = done_ids.to(torch.int32)
                    balls[level].write_root_pose_to_sim_index(root_pose=pose.contiguous(), env_ids=ids)
                    balls[level].write_root_com_velocity_to_sim_index(root_velocity=torch.zeros((len(done_ids), 6), device=dev),
                                                                      env_ids=ids)
            obs, _, done, _ = env.step(action)
            policy.reset(done)
            nan_resets += done.float()
            in_flight = (step >= push_step) & (step < park_step)
            bodies = c.robot_body_pos_w
            for level in range(3):
                mine = in_flight & (level_of == level)
                if bool(mine.any()):
                    centre = tensor(balls[level].data.root_pos_w)
                    gap = (bodies-centre[:, None]).norm(dim=-1).min(-1).values-BALL_TYPES[level][1]
                    min_gap = torch.where(mine, torch.minimum(min_gap, gap), min_gap)
            speed_now = tensor(robot.data.root_com_vel_w)[:, :2].norm(dim=-1)
            peak_pelvis_speed = torch.where(in_flight, torch.maximum(peak_pelvis_speed, speed_now), peak_pelvis_speed)
            pelvis = c.robot_body_pos_w[:, 0, 2]
            alignment = -tensor(robot.data.projected_gravity_b)[:, 2]
            pushed = step >= push_step
            lower = pushed & (pelvis < min_pelvis)
            min_pelvis = torch.where(lower, pelvis, min_pelvis)
            fwd = quat_apply(c.robot_body_quat_w[:, 0], torch.tensor([1., 0., 0.], device=dev).expand(N, 3))
            lying_fwd[lower] = fwd[lower]
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
    rows = []
    for i, (d, lv, r) in enumerate(ENVS):
        f = lying_fwd[i]
        lying = ('' if not fell[i] else 'supine' if f[2] > .5 else 'prone' if f[2] < -.5 else 'side')
        sec = lambda k: round(float(k-push_step[i])*dt, 2) if k >= 0 else None
        rows.append({'env': i, 'direction_deg': d*360//diag.directions, 'level': LEVEL_NAMES[lv],
                     'ball_kg': BALL_TYPES[lv][0], 'ball_speed': diag.ball_speeds[lv], 'repeat': r,
                     'throw_s': round(float(push_step[i])*dt, 2), 'hit': bool(min_gap[i] <= .02),
                     'min_gap_m': round(float(min_gap[i]), 3), 'peak_pelvis_speed': round(float(peak_pelvis_speed[i]), 2),
                     'fell': bool(fell[i]), 'lying': lying, 'min_pelvis_m': round(float(min_pelvis[i]), 3),
                     'stood_up_after_throw_s': sec(int(stood_step[i])), 'back_to_dance_after_throw_s': sec(int(back_step[i])),
                     'final_pelvis_m': round(float(c.robot_body_pos_w[i, 0, 2]), 3), 'nan_resets': int(nan_resets[i]),
                     'heading_offset_deg': round(math.degrees(float(heading.offset[i])), 1) if heading is not None else None})
    with (output/'envs.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cells = []
    for d, lv in CELLS:
        rr = [x for x in rows if x['direction_deg'] == d*360//diag.directions and x['level'] == LEVEL_NAMES[lv]]
        fallen = [x for x in rr if x['fell']]
        back = [x for x in fallen if x['back_to_dance_after_throw_s'] is not None]
        cells.append({'direction_deg': d*360//diag.directions, 'level': LEVEL_NAMES[lv], 'n': len(rr),
                      'hit': sum(x['hit'] for x in rr), 'fell': len(fallen), 'back_to_dance': len(back),
                      'mean_back_s': round(sum(x['back_to_dance_after_throw_s'] for x in back)/len(back), 2) if back else None})
    with (output/'cells.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0]))
        w.writeheader()
        w.writerows(cells)
    summary = {'checkpoint': play.args_cli.checkpoint, 'seconds': diag.seconds, 'envs': N,
               'align_heading': not diag.no_align_heading, 'ball_speeds': list(diag.ball_speeds),
               'hit': sum(x['hit'] for x in rows), 'fell': int(fell.sum()),
               'back_to_dance': sum(x['back_to_dance_after_throw_s'] is not None for x in rows), 'cells': cells}
    (output/'summary.json').write_text(json.dumps(summary, indent=1))
    print('[BALL-DIAG]', json.dumps({k: v for k, v in summary.items() if k != 'cells'}), flush=True)
    env.close()


play._run = configured_run
play._play = diagnose
if __name__ == '__main__':
    play.play_diagnostics.run(play.main)
