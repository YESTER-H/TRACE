# SPDX-License-Identifier: Apache-2.0
"""One-episode deterministic frame-zero evaluation of the integrated U1.5 TRACE policy.

Supine/Prone start a Recovery-origin episode at reference frame 0 and run through
Recovery -> Transition -> bounded post-Tracking; --motion tracking starts a
Tracking-origin episode at Hiphop frame 0. Reuses play.py's launcher, strict
checkpoint loading and deterministic mean actions; only reset noise/events differ.
--continue-tracking removes the U1.5 2 s horizon (display only) so Tracking runs
until a failure or the 30 s time-out. The terminal autoreset is suppressed.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

evaluation_parser = argparse.ArgumentParser(add_help=False)
evaluation_parser.add_argument('--motion', required=True, choices=('supine', 'prone', 'tracking'))
evaluation_parser.add_argument('--output-dir', required=True)
evaluation_parser.add_argument('--hold-result-seconds', type=float, default=30.)
evaluation_parser.add_argument('--record-eval-video', action='store_true')
evaluation_parser.add_argument('--continue-tracking', action='store_true')
evaluation, remaining = evaluation_parser.parse_known_args()
if evaluation.hold_result_seconds < 0:
    evaluation_parser.error('hold-result-seconds must be nonnegative')
output = Path(evaluation.output_dir).resolve()
output.mkdir(parents=True, exist_ok=False)  # Never overwrite another evaluation.
sys.argv = [sys.argv[0], *remaining, '--task', 'Cyclo-TRACE-K1-Rev1-Integrated-U15-Play',
            '--num_envs', '1', '--seed', '42', '--skip-export', '--play-log-dir', str(output)]

import play  # Launch Kit first, then import simulator modules.
import torch
from cyclo_lab.manager_based.trace_recovery.integrated import TraceIntegratedCommand, TRACKING, RECOVERY, TRANSITION
from cyclo_lab.manager_based.trace_recovery.body_adapter import tensor

if play.args_cli.video or play.args_cli.mouse_drag or play.args_cli.max_steps:
    raise ValueError('Use --record-eval-video; mouse drag and arbitrary step cutoffs are not allowed')
REGIME_NAMES = {TRACKING: 'tracking', RECOVERY: 'recovery', TRANSITION: 'transition'}


class FrameZeroIntegratedCommand(TraceIntegratedCommand):
    """Forced regime/identity/frame 0 bypass every random reset branch."""
    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if evaluation.motion == 'tracking':
            self.forced_regimes[ids] = TRACKING
        else:
            self.forced_regimes[ids] = RECOVERY
            self.forced_reference_ids[ids] = ('supine', 'prone').index(evaluation.motion)
            self.forced_frames[ids] = 0
        super()._resample_command(ids)


original_run = play._run


def configured_run(env_cfg, agent_cfg):
    command = env_cfg.commands.reference_trajectory
    command.class_type = FrameZeroIntegratedCommand
    command.reset_perturbations = False
    command.tracking_start_from_zero = True
    env_cfg.events.joint_home_offset_noise = None
    env_cfg.events.push_robot = None
    env_cfg.observations.policy.enable_corruption = False
    if evaluation.continue_tracking:
        env_cfg.terminations.post_tracking_horizon_complete = None
    return original_run(env_cfg, agent_cfg)


def evaluate(env, policy, *_unused):
    base = env.unwrapped
    c = base.command_manager.get_term('reference_trajectory')
    obs = env.get_observations()
    assert obs['policy'].shape == (1, 508) and obs['router'].shape == (1, 1)
    rows, state = [], {'step': 0, 'terminal': False, 'autoresets_suppressed': 0}
    original_compute = base.termination_manager.compute
    original_reset = base._reset_idx

    def observed_compute():
        done = original_compute()  # All native predicates and counters untouched.
        reasons = [name for name in base.termination_manager.active_terms
                   if bool(base.termination_manager.get_term(name)[0])]
        regime = int(c.regimes[0])
        rows.append({
            'control_step': state['step'], 'time_s': state['step']*base.step_dt,
            'regime': REGIME_NAMES[regime], 'tracking_weight': float(c.tracking_weight[0]),
            'recovery_frame': int(c.recovery_phase[0]) if regime == RECOVERY else -1,
            'live_tracking_frame': int(c.tracking_phase[0]),
            'root_height': float(c.logical_body_pos_w[0, 0, 2]),
            'up_alignment': float(-tensor(c.robot.data.projected_gravity_b)[0, 2]),
            'upright': bool(c.upright_mask()[0]),
            'persistent_error': float(c.recovery_target_error()[0]),
            'persistent_counter': int(c.persistent_steps[0]),
            'reacquisition_steps': int(c.reacquisition_steps[0]),
            'transition_steps': int(c.transition_steps[0]),
            'post_tracking_steps': int(c.post_tracking_steps[0]),
            'live_anchor_height_error': float((c.live_anchor_pos_w[0, 2]-c.robot_anchor_pos_w[0, 2]).abs()),
            'termination_reason': '|'.join(reasons), 'terminated': bool(done[0]),
        })
        state['terminal'] = bool(done[0])
        return done

    def terminal_reset_guard(ids):
        if state['terminal']:
            state['autoresets_suppressed'] += 1
            return  # No second episode, terminal physical pose remains visible.
        return original_reset(ids)

    base.termination_manager.compute = observed_compute
    base._reset_idx = terminal_reset_guard
    video_writer = viewer = None
    if base.sim.visualizers:
        from isaaclab_visualizers.newton.newton_visualizer import NewtonViewerGL
        viewer = next((v for v in (getattr(v, '_viewer', None) for v in base.sim.visualizers)
                       if isinstance(v, NewtonViewerGL)), None)
        if viewer is None:
            raise RuntimeError('GUI evaluation requires native Newton viewer')
        if evaluation.record_eval_video:
            import imageio.v2 as imageio
            video_writer = imageio.get_writer(str(output/'rollout.mp4'), fps=round(1/base.step_dt))
            original_end_frame = NewtonViewerGL.end_frame

            def observed_end_frame(this, *args, **kwargs):
                result = original_end_frame(this, *args, **kwargs)
                if 1 <= state['step'] and not state['terminal']:
                    video_writer.append_data(this.get_frame().numpy())
                return result
            NewtonViewerGL.end_frame = observed_end_frame

    with torch.inference_mode():
        while state['step'] < base.max_episode_length and not state['terminal']:
            if not play.simulation_app.is_running() or any(not v.is_running() for v in base.sim.visualizers):
                raise RuntimeError('Evaluation interrupted before native episode termination')
            start = time.monotonic()
            action = policy(obs)  # Native stochastic_output=False deterministic mean.
            state['step'] += 1
            obs, _, done, _ = env.step(action)
            policy.reset(done)
            if state['step'] % 50 == 0 or state['terminal']:
                print('[INTEGRATED-STEP]', json.dumps(rows[-1]), flush=True)
            if play.args_cli.real_time:
                time.sleep(max(0., base.step_dt-(time.monotonic()-start)))
    if video_writer is not None:
        video_writer.close()
    with (output/'rollout.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    last = rows[-1]
    reasons = last['termination_reason'].split('|') if last['termination_reason'] else []
    handoff = next((i for i in range(1, len(rows)) if rows[i-1]['regime'] == 'recovery'
                    and rows[i]['regime'] == 'transition'), None)
    tracking_start = next((r['control_step'] for r in rows if r['regime'] == 'tracking'), None)
    if evaluation.motion == 'tracking':
        outcome = 'TRACKING_TIMEOUT' if reasons == ['time_out'] or not state['terminal'] else 'TRACKING_FAILURE'
    elif reasons == ['post_tracking_horizon_complete']:
        outcome = 'U15_COMPLETE'
    elif evaluation.continue_tracking and tracking_start is not None and reasons == ['time_out']:
        outcome = 'RECOVERED_TRACKING_TIMEOUT'
    else:
        outcome = 'FAILURE'
    summary = {
        'motion': evaluation.motion, 'outcome': outcome, 'termination_reason': last['termination_reason'],
        'continue_tracking': evaluation.continue_tracking, 'control_steps': state['step'],
        'elapsed_simulation_s': state['step']*base.step_dt,
        'handoff_step': rows[handoff]['control_step'] if handoff is not None else None,
        # Completion hands off at the final frame; anything earlier was a reacquisition.
        'handoff_recovery_frame': rows[handoff-1]['recovery_frame'] if handoff is not None else None,
        'handoff_reason': None if handoff is None else (
            'terminal' if rows[handoff-1]['recovery_frame'] == c.recovery_references[
                ('supine', 'prone').index(evaluation.motion)].num_frames-1 else 'reacquisition'),
        'tracking_start_step': tracking_start,
        'tracking_steps': sum(r['regime'] == 'tracking' for r in rows),
        'max_persistent_error': max(r['persistent_error'] for r in rows),
        'final_root_height': last['root_height'], 'final_up_alignment': last['up_alignment'],
        'checkpoint': play.args_cli.checkpoint, 'execution_checks': state, 'last_step': last,
    }
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('[INTEGRATED-RESULT]', json.dumps(summary), flush=True)
    if viewer is not None:
        end = time.monotonic()+evaluation.hold_result_seconds
        # Render-only pause. No physics, resets or policy calls after termination.
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
