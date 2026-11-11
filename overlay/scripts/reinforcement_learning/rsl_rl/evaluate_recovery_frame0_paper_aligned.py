# SPDX-License-Identifier: Apache-2.0
"""One-episode frame-zero evaluation of the paper-aligned unified Recovery expert.

Copy of evaluate_recovery_frame0.py; only the task and checkpoint record differ,
so outputs stay directly comparable.

Reuses play.py's launcher, strict checkpoint loading and deterministic PyTorch
inference. Only evaluation resets/noise/events differ. All native termination
predicates run unchanged; the terminal autoreset is suppressed to retain the
first episode's final state. Metrics are sampled after physics/termination and
BEFORE command advancement, preserving training's k -> k+1 successor target.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

evaluation_parser = argparse.ArgumentParser(add_help=False)
evaluation_parser.add_argument('--motion', required=True, choices=('supine', 'prone'))
evaluation_parser.add_argument('--output-dir', required=True)
evaluation_parser.add_argument('--hold-result-seconds', type=float, default=30.)
evaluation_parser.add_argument('--record-eval-video', action='store_true')
evaluation, remaining = evaluation_parser.parse_known_args()
if evaluation.hold_result_seconds < 0:
    evaluation_parser.error('hold-result-seconds must be nonnegative')
output = Path(evaluation.output_dir).resolve()
output.mkdir(parents=True, exist_ok=False)  # Never overwrite another evaluation.
sys.argv = [sys.argv[0], *remaining, '--task', 'Cyclo-TRACE-K1-Rev1-Recovery-Paper-Aligned-Play',
            '--num_envs', '1', '--seed', '42', '--skip-export', '--play-log-dir', str(output)]

import play  # Launch Kit first, then import simulator modules.
import numpy as np
import torch
from isaaclab.utils.math import quat_error_magnitude
from cyclo_lab.manager_based.trace_recovery.commands import RecoveryOnlyCommand
from cyclo_lab.manager_based.trace_recovery.body_adapter import BODY_NAMES, JOINT_NAMES, tensor, rotate

if play.args_cli.video or play.args_cli.mouse_drag or play.args_cli.max_steps:
    raise ValueError('Use --record-eval-video; mouse drag and arbitrary step cutoffs are not allowed')
IDENTITY = ('supine', 'prone').index(evaluation.motion)


def scalar(value):
    return value.detach().cpu().item() if isinstance(value, torch.Tensor) else value


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def write_json(name, value):
    (output/name).write_text(json.dumps(value, indent=2, allow_nan=False))


class FrameZeroRecoveryCommand(RecoveryOnlyCommand):
    """Forced inputs bypass both random frame branches; native state write/H4 reset."""
    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.forced_reference_ids[ids] = IDENTITY
        self.forced_frames[ids] = 0
        super()._resample_command(ids)


original_run = play._run


def configured_run(env_cfg, agent_cfg):
    # These are fresh Eval-Play config objects, not the registered task defaults.
    env_cfg.commands.reference_trajectory.class_type = FrameZeroRecoveryCommand
    env_cfg.commands.reference_trajectory.reset_perturbations = False
    env_cfg.events.joint_home_offset_noise = None
    env_cfg.events.push_robot = None
    env_cfg.observations.policy.enable_corruption = False
    return original_run(env_cfg, agent_cfg)


def evaluate(env, policy, *_unused):
    base = env.unwrapped
    c = base.command_manager.get_term('reference_trajectory')
    ref = c.recovery_references[IDENTITY]
    obs = env.get_observations()
    native_initial = c.logical_states()
    origin = tensor(base.scene.env_origins)[0]
    initial_error = {
        'root_position_max_abs_m': scalar((native_initial[0][0, 0]-origin-ref.body_position_w[0, 0]).abs().max()),
        'root_orientation_rad': scalar(quat_error_magnitude(native_initial[1][0, 0], ref.body_orientation_w[0, 0])),
        'root_source_com_linear_velocity_max_abs_m_s': scalar((native_initial[2][0, 0]-ref.body_linear_velocity_w[0, 0]).abs().max()),
        'root_angular_velocity_max_abs_rad_s': scalar((native_initial[3][0, 0]-ref.body_angular_velocity_w[0, 0]).abs().max()),
        'joint_position_max_abs_rad': scalar((tensor(c.robot.data.joint_pos)[0, c.joint_ids]-ref.joint_position[0]).abs().max()),
        'joint_velocity_max_abs_rad_s': scalar((tensor(c.robot.data.joint_vel)[0, c.joint_ids]-ref.joint_velocity[0]).abs().max()),
    }
    histories = {}
    for name, buffer in base.observation_manager._group_obs_term_history_buffer['policy'].items():
        history = buffer.buffer
        histories[name] = {'shape': list(history.shape), 'length': buffer.max_length,
                           'reset_history_max_difference': scalar((history-history[:, -1:]).abs().max())}
    from isaaclab_newton.physics import NewtonManager
    backend = {'manager': type(NewtonManager.get_manager()).__name__} if hasattr(NewtonManager, 'get_manager') else {
        'manager': str(getattr(NewtonManager, '_manager', 'NewtonManager'))}
    backend['physics_config'] = str(type(base.cfg.sim.physics))
    backend['solver_type'] = str(base.cfg.sim.physics.solver_cfg.solver_type)
    initial = {'motion': evaluation.motion, 'reference_frame': int(c.recovery_phase[0]),
        'reference_id': int(c.recovery_reference_ids[0]), 'live_tracking_phase': int(c.tracking_phase[0]),
        'backend': backend, 'physical_bodies': len(c.robot.body_names), 'logical_bodies': len(BODY_NAMES),
        'policy_shape': list(obs['policy'].shape), 'critic_shape': list(obs['critic'].shape),
        'router': array(obs['router']).tolist(), 'router_semantics': 'scalar w_T=0, w_R=1',
        'action_dim': env.num_actions, 'shared_log_std_shape': list(policy.log_std.shape),
        'dt': base.step_dt, 'physics_dt': base.physics_dt, 'grace_steps': c.grace_steps,
        'persistent_required_steps': c.persistent_error_steps_required,
        'max_episode_steps': base.max_episode_length, 'initial_state_errors': initial_error,
        'history': histories, 'reset_branch': int(c.last_direct_reset_branch[0]),
        'reset_events': {k: int(v) for k, v in c.events.items()},
        'initial_root_pose_xyzw': array(tensor(c.robot.data.root_link_pose_w)[0]).tolist(),
        'initial_root_com_velocity': array(tensor(c.robot.data.root_com_vel_w)[0]).tolist(),
        'joint_home_offset_noise': False, 'push_robot': False, 'reset_perturbations': False,
        'random_recovery_selection_or_phase': False, 'deterministic_mean_action': True,
        'checkpoint': play.args_cli.checkpoint,
        'metric_timing': 'post-physics/pre-command-advance; reward target=min(action_frame+1,last)',
        'tracking_live_phase': 'native seeded reset/progression retained; Recovery reference hidden from actor'}
    write_json('initial_state.json', initial)
    assert initial['reference_id'] == IDENTITY and initial['reference_frame'] == 0
    assert initial['policy_shape'] == [1, 508] and initial['critic_shape'] == [1, 256]
    assert initial['router'] == [[0.0]] and initial['action_dim'] == 23
    assert initial['shared_log_std_shape'] == [23]
    assert initial['physical_bodies'] == 24 and initial['logical_bodies'] == 25
    assert 'Newton' in initial['backend']['physics_config'] and initial['backend']['solver_type'] == 'mujoco_warp'
    assert all(v < 1.e-4 for v in initial_error.values()), initial_error
    assert all(h['length'] == 4 and h['reset_history_max_difference'] == 0 for h in histories.values())
    assert initial['reset_branch'] == -1
    assert initial['reset_events']['uniform_episode_resets'] == 0
    assert initial['reset_events']['adaptive_episode_resets'] == 0

    changes = {'joint_reference_l2_rad': float((ref.joint_position[20]-ref.joint_position[0]).norm()),
        'joint_reference_path_length_rad': float((ref.joint_position[1:21]-ref.joint_position[:20]).norm(dim=-1).sum()),
        'root_height_delta_m': float(ref.body_position_w[20, 0, 2]-ref.body_position_w[0, 0, 2]),
        'body_orientation_mean_change_rad': float(quat_error_magnitude(ref.body_orientation_w[20], ref.body_orientation_w[0]).mean())}
    write_json('reference_initial_0_20.json', changes)
    np.savez_compressed(output/'reference.npz', joint_position=array(ref.joint_position),
        joint_velocity=array(ref.joint_velocity), body_position=array(ref.body_position_w),
        body_orientation=array(ref.body_orientation_w), joint_names=JOINT_NAMES, body_names=BODY_NAMES)

    rows, payloads = [], []
    state = {'step': 0, 'terminal': False, 'autoresets_suppressed': 0,
             'tracking_expert_calls': 0, 'recovery_expert_calls': 0, 'action_mean_difference': 0.}
    policy.actor_experts['tracking'].register_forward_hook(
        lambda *_args: state.__setitem__('tracking_expert_calls', state['tracking_expert_calls']+1))
    policy.actor_experts['recovery'].register_forward_hook(
        lambda *_args: state.__setitem__('recovery_expert_calls', state['recovery_expert_calls']+1))
    previous_q = tensor(c.robot.data.joint_pos)[0, c.joint_ids].clone()
    current_action = None
    original_compute = base.termination_manager.compute
    original_reset = base._reset_idx

    def observed_compute():
        nonlocal previous_q
        done = original_compute()  # All native predicates and counters untouched.
        reasons = [name for name in base.termination_manager.active_terms
                   if bool(base.termination_manager.get_term(name)[0])]
        p, q, v, w = c.logical_states()
        jq = tensor(c.robot.data.joint_pos)[0, c.joint_ids]
        jv = tensor(c.robot.data.joint_vel)[0, c.joint_ids]
        components = c.target_error_components()
        row = {'control_step': state['step'], 'time_s': state['step']*base.step_dt,
            'motion': evaluation.motion, 'reference_frame': int(c.recovery_phase[0]),
            'reward_target_frame': int(c.recovery_reward_phase[0]),
            'live_tracking_frame': int(c.tracking_phase[0]),
            'root_height': float(p[0, 0, 2]), 'reference_root_height': float(c.body_pos_w[0, 0, 2]),
            'root_orientation_error': float(quat_error_magnitude(q[0, 0], c._raw('body_orientation_w')[0, 0])),
            'anchor_orientation_error': float(components['anchor_orientation'][0]),
            'body_position_error': float(components['body_position'][0]),
            'body_orientation_error': float(components['body_orientation'][0]),
            'root_height_error': float(components['root_height'][0]),
            'joint_position_l2_error': float((jq-c.joint_pos[0]).norm()),
            'joint_velocity_l2_error': float((jv-c.joint_vel[0]).norm()),
            'actual_joint_step_l2': float((jq-previous_q).norm()),
            'persistent_error': float(c.recovery_target_error()[0]),
            'persistent_counter': int(c.persistent_steps[0]),
            'violation_onset_frame': int(c.failure_onset_phase[0]),
            'persistent_detection_frame': int(c.failure_detection_phase[0]),
            'upright': bool(c.upright_mask()[0]), 'final_frame': bool(c.final_mask()[0]),
            'hold_steps_pre_advance': int(c.hold_steps[0]),
            'up_alignment': float(-tensor(c.robot.data.projected_gravity_b)[0, 2]),
            'action_norm': float(current_action.norm()),
            'termination_reason': '|'.join(reasons), 'terminated': bool(done[0]),
        }
        for axis, val in zip(('x', 'y', 'z', 'w'), q[0, 0]):
            row['root_quat_'+axis] = float(val)
        for axis, val in zip(('x', 'y', 'z'), tensor(c.robot.data.projected_gravity_b)[0]):
            row['projected_gravity_'+axis] = float(val)
        target = env.unwrapped.action_manager.get_term('joint_pos').processed_actions[0]
        for i, name in enumerate(JOINT_NAMES):
            for key, values in (('reference_q', c.joint_pos[0]), ('actual_q', jq),
                                ('reference_qd', c.joint_vel[0]), ('actual_qd', jv),
                                ('target_q', target), ('action', current_action[0])):
                row[name+'__'+key] = float(values[i])
        bp = (c.aligned_body_position_w-c.robot_body_pos_w).norm(dim=-1)[0]
        bq = quat_error_magnitude(c.aligned_body_orientation_w, c.robot_body_quat_w)[0]
        for i, name in enumerate(c.cfg.body_names):
            row[name+'__position_error'] = float(bp[i])
            row[name+'__orientation_error'] = float(bq[i])
        rows.append(row)
        payloads.append({'q': array(jq).copy(), 'qd': array(jv).copy(),
            'body_pos': array(p[0]).copy(), 'body_quat': array(q[0]).copy(),
            'action': array(current_action[0]).copy(), 'target': array(target).copy()})
        previous_q = jq.clone()
        state['terminal'] = bool(done[0])
        return done

    def terminal_reset_guard(ids):
        if state['terminal']:
            state['autoresets_suppressed'] += 1
            return  # No second episode, terminal physical pose remains visible.
        return original_reset(ids)

    base.termination_manager.compute = observed_compute
    base._reset_idx = terminal_reset_guard
    video_writer = None
    viewer = None
    capture_state = {'frames': 0, 'screenshots': [], 'renderer': None}
    if base.sim.visualizers:
        from PIL import Image
        from pyglet import gl
        from isaaclab_visualizers.newton.newton_visualizer import NewtonViewerGL
        viewers = [getattr(v, '_viewer', None) for v in base.sim.visualizers]
        viewer = next((v for v in viewers if isinstance(v, NewtonViewerGL)), None)
        if viewer is None:
            raise RuntimeError('GUI evaluation requires native Newton viewer')
        if evaluation.record_eval_video:
            import imageio.v2 as imageio
            video_writer = imageio.get_writer(str(output/'rollout.mp4'), fps=round(1/base.step_dt))
        original_end_frame = NewtonViewerGL.end_frame
        def observed_end_frame(this, *args, **kwargs):
            result = original_end_frame(this, *args, **kwargs)
            if 1 <= state['step'] <= base.max_episode_length and not state['terminal']:
                pixels = this.get_frame().numpy()
                capture_state['frames'] += 1
                capture_state['renderer'] = gl.gl_info.get_renderer()
                if video_writer is not None:
                    video_writer.append_data(pixels)
                phase = int(c.recovery_phase[0])
                keyframes = (0, 20, 87, 109, 130, 174, 195, 258, 281, 329, 340, 351, 423, 445,
                             ref.num_frames-1)
                if phase in keyframes and phase not in capture_state['screenshots']:
                    Image.fromarray(pixels).save(output/f'frame_{phase:03d}.png')
                    capture_state['screenshots'].append(phase)
            return result
        NewtonViewerGL.end_frame = observed_end_frame

    print('[FRAME0-INITIAL]', json.dumps(initial), flush=True)
    # Includes native 30s timeout; no arbitrary smoke cutoff.
    with torch.inference_mode():
        while state['step'] < base.max_episode_length and not state['terminal']:
            if not play.simulation_app.is_running() or any(not v.is_running() for v in base.sim.visualizers):
                raise RuntimeError('Evaluation interrupted before native episode termination')
            start = time.monotonic()
            assert torch.equal(obs['router'], torch.zeros_like(obs['router']))
            current_action = policy(obs)  # Native stochastic_output=False deterministic mean.
            state['action_mean_difference'] = max(state['action_mean_difference'],
                float((current_action-policy.distribution.mean).abs().max()))
            state['step'] += 1
            obs, reward, done, extras = env.step(current_action)
            policy.reset(done)
            if state['step'] % 50 == 0 or state['terminal']:
                print('[FRAME0-STEP]', json.dumps(rows[-1]), flush=True)
            if play.args_cli.real_time:
                time.sleep(max(0., base.step_dt-(time.monotonic()-start)))
    if video_writer is not None:
        video_writer.close()
    with (output/'rollout.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output/'states.npz', **{key: np.stack([p[key] for p in payloads]) for key in payloads[0]})
    last = rows[-1]
    reasons = last['termination_reason'].split('|')
    outcome = 'SUCCESS' if reasons == ['recovery_success'] else 'TIMEOUT' if reasons == ['time_out'] else 'FAILURE'
    if not state['terminal']:
        outcome = 'TIMEOUT'
    first_violation = next((r['reference_frame'] for r in rows if r['persistent_error'] > 4), None)
    summary = {'motion': evaluation.motion, 'outcome': outcome, 'termination_reason': last['termination_reason'],
        'start_frame': 0, 'maximum_action_frame': max(r['reference_frame'] for r in rows),
        'maximum_successor_frame': max(r['reward_target_frame'] for r in rows),
        'termination_action_frame': last['reference_frame'], 'termination_target_frame': last['reward_target_frame'],
        'control_steps': state['step'], 'elapsed_simulation_s': state['step']*base.step_dt,
        'first_target_error_above_4_frame': first_violation,
        'confirmed_persistent_onset_frame': last['violation_onset_frame'] if 'recovery_persistent_failure' in reasons else None,
        'persistent_detection_frame': last['persistent_detection_frame'],
        'final_frame_reached': any(r['final_frame'] for r in rows),
        'final_frame_exposure_steps': sum(r['final_frame'] for r in rows),
        'initial_0_20_reference': changes, 'last_step': last,
        'execution_checks': state, 'gui': capture_state,
        'initial_state_errors': initial_error, 'episode_count': 1}
    write_json('summary.json', summary)
    print('[FRAME0-RESULT]', json.dumps(summary), flush=True)
    if viewer is not None:
        from PIL import Image
        Image.fromarray(viewer.get_frame().numpy()).save(output/'terminal.png')
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
