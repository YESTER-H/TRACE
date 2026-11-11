# SPDX-License-Identifier: Apache-2.0
"""Deterministic frame-zero side-by-side view of the integrated U1.5 TRACE policy.

Three envs in one window, on one straight line (left to right): env 0 Tracking from
Hiphop frame 0, env 1 Supine and env 2 Prone from reference frame 0 (live Tracking
phase also 0). By default the
U1.5 2 s post-Tracking horizon is removed so each recovery continues through
Transition into Tracking until a failure or the 30 s time-out; --bounded-horizon
keeps it. An env that terminates is not reset: its terminal pose is held (rewritten
each step with zero velocity) while the others keep running. Display/evaluation only.

--loop: no time limit. An env that fails restarts from its own frame 0 at once, and
the window runs until it is closed (or --loop-seconds). Every attempt is written to
attempts.csv / summary.json.

--loop --no-resets: deployment-style run. Every termination (recovery failure, tracking
failure) is still computed (its error bookkeeping feeds the env) but ignored, so each env starts once from its own frame
0 and the one policy runs on without any reset; only the command's own Recovery ->
Transition -> Tracking handoff still happens. A NaN state still resets, as a guard.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

evaluation_parser = argparse.ArgumentParser(add_help=False)
evaluation_parser.add_argument('--output-dir', required=True)
evaluation_parser.add_argument('--hold-result-seconds', type=float, default=30.)
evaluation_parser.add_argument('--record-eval-video', action='store_true')
evaluation_parser.add_argument('--bounded-horizon', action='store_true')
evaluation_parser.add_argument('--loop', action='store_true')
evaluation_parser.add_argument('--loop-seconds', type=float, default=None)
evaluation_parser.add_argument('--spacing', type=float, default=4.)
evaluation_parser.add_argument('--no-resets', action='store_true')
evaluation_parser.add_argument('--eval-task', default='Cyclo-TRACE-K1-Rev1-Integrated-U15-Play',
                               help='Play task (e.g. Cyclo-TRACE-K1-Rev1-Integrated-Gated-Play)')
evaluation, remaining = evaluation_parser.parse_known_args()
if evaluation.no_resets and not evaluation.loop:
    evaluation_parser.error('--no-resets needs --loop')
if evaluation.loop_seconds is not None and not evaluation.loop:
    evaluation_parser.error('--loop-seconds needs --loop')
if evaluation.loop and evaluation.loop_seconds is None and '--headless' in remaining:
    evaluation_parser.error('headless --loop needs --loop-seconds')
if evaluation.hold_result_seconds < 0:
    evaluation_parser.error('hold-result-seconds must be nonnegative')
output = Path(evaluation.output_dir).resolve()
output.mkdir(parents=True, exist_ok=False)  # Never overwrite another evaluation.
sys.argv = [sys.argv[0], *remaining, '--task', evaluation.eval_task,
            '--num_envs', '3', '--seed', '42', '--skip-export', '--play-log-dir', str(output)]

import play  # Launch Kit first, then import simulator modules.
import torch
from cyclo_lab.manager_based.trace_recovery.integrated import TraceIntegratedCommand, TRACKING, RECOVERY, TRANSITION
from cyclo_lab.manager_based.trace_recovery.body_adapter import tensor, refresh_reset_cache

if play.args_cli.video or play.args_cli.mouse_drag or play.args_cli.max_steps:
    raise ValueError('Use --record-eval-video; mouse drag and arbitrary step cutoffs are not allowed')
ROLES = ('tracking', 'supine', 'prone')
REGIME_NAMES = {TRACKING: 'tracking', RECOVERY: 'recovery', TRANSITION: 'transition'}


class ThreeViewCommand(TraceIntegratedCommand):
    """env 0 Tracking, env 1 Supine, env 2 Prone; every one starts at frame 0."""
    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.forced_regimes[ids] = torch.where(ids == 0, TRACKING, RECOVERY)
        self.forced_reference_ids[ids] = (ids-1).clamp_min(0)
        self.forced_frames[ids] = 0
        super()._resample_command(ids)
        self.tracking_phase[ids] = 0  # Recovery resets otherwise draw a random live phase.
        self._update_alignment_cache()


original_run = play._run


def line_transforms(N, spacing=1., up_axis='z', device='cpu'):
    """Display-only replacement of the near-square env grid: envs on the x axis, env 0 at -x.

    Every reference stands facing yaw -90 deg (-y), so a row along x puts the robots side
    by side, all facing the camera placed on the -y side (as in the PhysX Eval-Play view).
    """
    assert up_axis.lower() == 'z'
    x = (torch.arange(N, device=device, dtype=torch.float32)-(N-1)/2)*spacing
    pos = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], 1)
    ori = torch.zeros((N, 4), device=device)
    ori[:, 3] = 1.
    return pos, ori


def configured_run(env_cfg, agent_cfg):
    command = env_cfg.commands.reference_trajectory
    command.class_type = ThreeViewCommand
    command.reset_perturbations = False
    command.tracking_start_from_zero = True
    env_cfg.events.joint_home_offset_noise = None
    env_cfg.events.push_robot = None
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.scene.env_spacing = evaluation.spacing  # Wider than training's 2.5 m: no on-screen overlap.
    # Scene cloning and terrain origins both read grid_transforms at call time.
    import isaaclab.cloner, isaaclab.cloner.cloner_utils
    isaaclab.cloner.grid_transforms = isaaclab.cloner.cloner_utils.grid_transforms = line_transforms
    if not evaluation.bounded_horizon:
        env_cfg.terminations.post_tracking_horizon_complete = None
    if getattr(env_cfg.terminations, 'tracking_origin_horizon_complete', None) is not None:
        env_cfg.terminations.tracking_origin_horizon_complete = None  # training-only 12 s budget
    if evaluation.loop:
        env_cfg.terminations.time_out = None
        env_cfg.episode_length_s = 1e6
    return original_run(env_cfg, agent_cfg)


def place_camera(base, viewer):
    """Camera on the -y side looking +y: the robots face it, envs read left to right."""
    origins = tensor(base.scene.env_origins)[:, :2].cpu()
    center = origins.mean(0)
    forward = torch.tensor([0., 1.])  # opposite of the standing heading (yaw -90 deg)
    right = torch.tensor([forward[1], -forward[0]])  # forward x z-up
    x = (origins-center) @ right
    gap = float((x[1:]-x[:-1]).abs().min())
    distance = max(4.5, float((origins-center).norm(dim=-1).max())*1.5)  # row fills ~2/3 of the width
    eye = (float(center[0]-forward[0]*distance), float(center[1]-forward[1]*distance), 2.0)
    viewer._apply_camera_pose((eye, (float(center[0]), float(center[1]), .5)))
    order = sorted(range(3), key=lambda i: float(x[i]))
    return {'eye': eye, 'left_to_right': [ROLES[i] for i in order], 'screen_gap_m': gap,
            'env_origins_xy': origins.tolist()}


def attach_viewer(base, stop_when):
    """Native Newton viewer, line camera and optional video; stop_when() pauses recording."""
    if not base.sim.visualizers:
        return None, None, None
    from isaaclab_visualizers.newton.newton_visualizer import NewtonViewerGL
    visualizer = next((v for v in base.sim.visualizers if isinstance(getattr(v, '_viewer', None), NewtonViewerGL)), None)
    if visualizer is None:
        raise RuntimeError('GUI evaluation requires native Newton viewer')
    camera = place_camera(base, visualizer)
    print('[THREE-VIEW] screen left to right:', ' | '.join(camera['left_to_right']), flush=True)
    video_writer = None
    if evaluation.record_eval_video:
        import imageio.v2 as imageio
        video_writer = imageio.get_writer(str(output/'rollout.mp4'), fps=round(1/base.step_dt))
        original_end_frame = NewtonViewerGL.end_frame

        def observed_end_frame(this, *args, **kwargs):
            result = original_end_frame(this, *args, **kwargs)
            if not stop_when():
                video_writer.append_data(this.get_frame().numpy())
            return result
        NewtonViewerGL.end_frame = observed_end_frame
    return visualizer._viewer, camera, video_writer


def evaluate_loop(env, policy, base, c):
    """Endless display: native resets restart a failed env from its own frame 0."""
    obs = env.get_observations()
    step, attempts, current, step_rows = 0, [], [None]*3, []

    def begin(i):
        current[i] = {'env': i, 'role': ROLES[i], 'attempt': sum(a['env'] == i for a in attempts)+1,
                      'start_step': step, 'handoff_step': None, 'handoff_recovery_frame': None,
                      'tracking_start_step': 0 if i == 0 else None, 'end_step': None, 'end_reason': None}
    for i in range(3):
        begin(i)
    original_compute = base.termination_manager.compute

    def observed_compute():
        done = original_compute()
        manager = base.termination_manager
        if evaluation.no_resets:  # ignore every termination except a NaN state
            keep = (manager.get_term('nan_detected').clone() if 'nan_detected' in manager.active_terms
                    else torch.zeros_like(done))
            manager._terminated_buf &= keep
            manager._truncated_buf &= keep
            done = done & keep
        for i in range(3):
            if done[i]:
                a = current[i]
                a['end_step'] = step
                a['end_reason'] = '|'.join(n for n in manager.active_terms if bool(manager.get_term(n)[i]))
                attempts.append(a)
                print('[THREE-VIEW-ATTEMPT]', json.dumps(a), flush=True)
        return done
    base.termination_manager.compute = observed_compute
    viewer, camera, video_writer = attach_viewer(base, lambda: False)
    limit = None if evaluation.loop_seconds is None else round(evaluation.loop_seconds/base.step_dt)
    previous = c.regimes.clone()
    with torch.inference_mode():
        while play.simulation_app.is_running() and all(v.is_running() for v in base.sim.visualizers):
            if limit is not None and step >= limit:
                break
            start = time.monotonic()
            action = policy(obs)
            step += 1
            recovery_frames = c.recovery_phase.clone()
            gate = getattr(policy, 'last_gate', None)
            for i in range(3):  # regime / oracle weight / gate weight that produced this action
                step_rows.append({'step': step, 'env': i, 'role': ROLES[i], 'regime': REGIME_NAMES[int(c.regimes[i])],
                                  'oracle_w_t': round(float(c.tracking_weight[i]), 4),
                                  'gate_w_t': '' if gate is None else round(float(gate['weights'][i, 0]), 4),
                                  'pelvis_m': round(float(c.robot_body_pos_w[i, 0, 2]), 4)})
            obs, _, done, _ = env.step(action)
            policy.reset(done)
            for i in range(3):
                if done[i]:
                    begin(i)
                    continue
                regime = int(c.regimes[i])
                if regime == TRANSITION and int(previous[i]) == RECOVERY:
                    current[i]['handoff_step'] = step
                    current[i]['handoff_recovery_frame'] = int(recovery_frames[i])
                if regime == TRACKING and current[i]['tracking_start_step'] is None:
                    current[i]['tracking_start_step'] = step
            previous = c.regimes.clone()
            if step % 250 == 0:
                print('[THREE-VIEW-STEP]', step, ' '.join(
                    f"{ROLES[i]}#{current[i]['attempt']}={REGIME_NAMES[int(c.regimes[i])]}"
                    f"(pelvis {float(c.robot_body_pos_w[i, 0, 2]):.2f} m"
                    + (f", gate w_T {float(policy.last_gate['weights'][i, 0]):.2f})"
                       if getattr(policy, 'last_gate', None) is not None else ")") for i in range(3)), flush=True)
            if play.args_cli.real_time:
                time.sleep(max(0., base.step_dt-(time.monotonic()-start)))
    if video_writer is not None:
        video_writer.close()
    for i in range(3):  # still running when the window closed
        attempts.append({**current[i], 'end_step': step, 'end_reason': 'running_at_exit'})
    fields = list(attempts[0])
    with (output/'attempts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(attempts)
    with (output/'steps.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(step_rows[0]))
        writer.writeheader()
        writer.writerows(step_rows)
    summary = {'mode': 'loop_no_resets' if evaluation.no_resets else 'loop', 'control_steps': step, 'elapsed_simulation_s': step*base.step_dt,
               'checkpoint': play.args_cli.checkpoint, 'camera': camera,
               'env_origins_xy': tensor(base.scene.env_origins)[:, :2].tolist(),
               'envs': {role: {'attempts': sum(a['env'] == i for a in attempts),
                               'failures': sum(a['env'] == i and a['end_reason'] != 'running_at_exit' for a in attempts),
                               'handoffs': sum(a['env'] == i and a['handoff_step'] is not None for a in attempts),
                               'final_regime': REGIME_NAMES[int(c.regimes[i])],
                               'final_pelvis_m': float(c.robot_body_pos_w[i, 0, 2])}
                        for i, role in enumerate(ROLES)}}
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('[THREE-VIEW-RESULT]', json.dumps(summary), flush=True)
    env.close()


def evaluate(env, policy, *_unused):
    base = env.unwrapped
    c = base.command_manager.get_term('reference_trajectory')
    robot = c.robot
    obs = env.get_observations()
    assert c.regimes.tolist() == [TRACKING, RECOVERY, RECOVERY]
    assert c.recovery_reference_ids[1:].tolist() == [0, 1] and c.recovery_phase.tolist() == [0, 0, 0]
    if evaluation.loop:
        return evaluate_loop(env, policy, base, c)
    rows = []
    state = {'step': 0, 'done': torch.zeros(3, dtype=torch.bool, device=base.device), 'ended': [None]*3,
             'held': None}
    original_compute = base.termination_manager.compute
    original_reset = base._reset_idx

    def observed_compute():
        done = original_compute()  # All native predicates and counters untouched.
        manager = base.termination_manager
        for i in range(3):
            if state['done'][i]:
                continue
            reasons = [name for name in manager.active_terms if bool(manager.get_term(name)[i])]
            regime = int(c.regimes[i])
            rows.append({
                'env': i, 'role': ROLES[i], 'control_step': state['step'], 'time_s': state['step']*base.step_dt,
                'regime': REGIME_NAMES[regime], 'tracking_weight': float(c.tracking_weight[i]),
                'recovery_frame': int(c.recovery_phase[i]) if regime == RECOVERY else -1,
                'live_tracking_frame': int(c.tracking_phase[i]),
                'root_height': float(c.logical_body_pos_w[i, 0, 2]),
                'up_alignment': float(-tensor(robot.data.projected_gravity_b)[i, 2]),
                'persistent_error': float(c.recovery_target_error()[i]),
                'transition_steps': int(c.transition_steps[i]),
                'termination_reason': '|'.join(reasons), 'terminated': bool(done[i]),
            })
            if done[i]:
                state['ended'][i] = {'step': state['step'], 'reason': '|'.join(reasons), 'regime': REGIME_NAMES[regime]}
        state['done'] |= done
        return done

    def partial_reset_guard(ids):
        ids = torch.as_tensor(ids, device=base.device, dtype=torch.long)
        keep = ids[~state['done'][ids]]
        if len(keep):  # Never expected: only terminated envs request resets here.
            return original_reset(keep)

    def hold_terminal_poses():
        ids = torch.where(state['done'])[0]
        if not len(ids):
            return
        if state['held'] is None or len(state['held'][0]) != len(ids):
            state['held'] = (ids, tensor(robot.data.root_link_pose_w)[ids].clone(),
                             tensor(robot.data.joint_pos)[ids].clone())
        ids, pose, joints = state['held']
        write = ids.to(torch.int32)
        robot.write_root_link_pose_to_sim_index(root_pose=pose.contiguous(), env_ids=write)
        robot.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros((len(ids), 6), device=base.device),
                                                   env_ids=write)
        robot.write_joint_state_to_sim_index(position=joints.contiguous(), velocity=torch.zeros_like(joints),
                                             env_ids=write, joint_ids=all_joints)
        refresh_reset_cache(robot)

    all_joints = torch.arange(len(robot.joint_names), dtype=torch.int32, device=base.device)
    base.termination_manager.compute = observed_compute
    base._reset_idx = partial_reset_guard
    viewer, camera, video_writer = attach_viewer(
        base, lambda: state['step'] < 1 or bool(state['done'].all()))

    with torch.inference_mode():
        while state['step'] < base.max_episode_length and not bool(state['done'].all()):
            if not play.simulation_app.is_running() or any(not v.is_running() for v in base.sim.visualizers):
                raise RuntimeError('Evaluation interrupted before every episode ended')
            start = time.monotonic()
            action = policy(obs)  # Native stochastic_output=False deterministic mean.
            state['step'] += 1
            obs, _, done, _ = env.step(action)
            hold_terminal_poses()
            if state['step'] % 50 == 0:
                print('[THREE-VIEW-STEP]', state['step'], ' '.join(
                    f"{ROLES[i]}={'ENDED' if state['done'][i] else REGIME_NAMES[int(c.regimes[i])]}" for i in range(3)), flush=True)
            if play.args_cli.real_time:
                time.sleep(max(0., base.step_dt-(time.monotonic()-start)))
    if video_writer is not None:
        video_writer.close()
    with (output/'rollout.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {'bounded_horizon': evaluation.bounded_horizon, 'control_steps': state['step'],
               'checkpoint': play.args_cli.checkpoint, 'camera': camera,
               'env_origins_xy': tensor(base.scene.env_origins)[:, :2].tolist(), 'envs': {}}
    for i, role in enumerate(ROLES):
        mine = [r for r in rows if r['env'] == i]
        handoff = next((k for k in range(1, len(mine)) if mine[k-1]['regime'] == 'recovery'
                        and mine[k]['regime'] == 'transition'), None)
        tracking_start = next((r['control_step'] for r in mine if r['regime'] == 'tracking'), None)
        ended = state['ended'][i]
        reason = ended['reason'] if ended else 'time_out_not_reached'
        if role == 'tracking':
            outcome = 'TRACKING_SURVIVED' if reason in ('time_out', 'time_out_not_reached') else 'TRACKING_FAILURE'
        elif tracking_start is not None and reason in ('time_out', 'post_tracking_horizon_complete', 'time_out_not_reached'):
            outcome = 'RECOVERED_AND_TRACKING'
        elif handoff is not None:
            outcome = 'RECOVERED_THEN_FAILED'
        else:
            outcome = 'RECOVERY_FAILURE'
        last_frame = c.recovery_references[i-1].num_frames-1 if i else None
        summary['envs'][role] = {
            'outcome': outcome, 'end': ended,
            'handoff_step': mine[handoff]['control_step'] if handoff is not None else None,
            'handoff_reason': None if handoff is None else (
                'terminal' if mine[handoff-1]['recovery_frame'] == last_frame else 'reacquisition'),
            'handoff_recovery_frame': mine[handoff-1]['recovery_frame'] if handoff is not None else None,
            'tracking_start_step': tracking_start,
            'tracking_steps': sum(r['regime'] == 'tracking' for r in mine),
            'max_persistent_error': max(r['persistent_error'] for r in mine),
        }
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('[THREE-VIEW-RESULT]', json.dumps(summary), flush=True)
    if viewer is not None:
        end = time.monotonic()+evaluation.hold_result_seconds
        # Render-only pause. No physics, resets or policy calls after the last episode ended.
        while time.monotonic() < end and play.simulation_app.is_running():
            if any(not v.is_running() for v in base.sim.visualizers):
                break
            base.sim.render()
            time.sleep(.02)
    env.close()


play._run = configured_run
play._play = evaluate
if __name__ == '__main__':
    play.play_diagnostics.run(play.main)
