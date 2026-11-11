# SPDX-License-Identifier: Apache-2.0
"""Which expert can finish a recovery from where?  Batch, no-reset diagnostic for the gated TRACE policy.

Every env starts once in Recovery at a fixed (motion, reference frame) and runs for --seconds with
every termination ignored (deployment style, NaN guard only). Training reset perturbations stay on,
so the --repeats envs of one cell differ. Each cell is run under three routings of the one checkpoint:

  gate       the learned gate (deployment)
  recovery   oracle w_T: Recovery expert while in Recovery, alpha in Transition, Tracking after
  tracking   Tracking expert while in Recovery (w_T = 1), oracle afterwards

Per env: reached Tracking, final / max pelvis height, stuck half-up (final pelvis 0.3-0.6 m), gate
w_T statistics while in Recovery, and the gate training label (recovered_like) of that start.
--trace adds trace.npz (gate-mode Supine envs: recovery frame, robot/reference joint positions and
body positions, every step). Written to cells.csv / envs.csv / summary.json in --output-dir (never overwritten). Display-free.

usage (inside the container, /workspace/cyclo_lab):
  /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/diagnose_trace_gated_routing.py --headless \
    --output-dir logs/eval/diag_gated_routing_<time> --checkpoint <run>/model_49999.pt
"""
import argparse
import csv
import json
import sys
from pathlib import Path

diag_parser = argparse.ArgumentParser(add_help=False)
diag_parser.add_argument('--output-dir', required=True)
diag_parser.add_argument('--seconds', type=float, default=15.)
diag_parser.add_argument('--repeats', type=int, default=16)
diag_parser.add_argument('--supine-frames', default='0,60,100,130,150,170,200,260')
diag_parser.add_argument('--prone-frames', default='0,250,330,400')
diag_parser.add_argument('--trace', action='store_true',
                         help='also save trace.npz: per-step joint/body states vs reference, gate-mode Supine envs')
diag_parser.add_argument('--eval-task', default='Cyclo-TRACE-K1-Rev1-Integrated-Gated-Play')
diag, remaining = diag_parser.parse_known_args()
output = Path(diag.output_dir).resolve()
output.mkdir(parents=True, exist_ok=False)
MODES = ('gate', 'recovery', 'tracking')
CELLS = [(0, int(f)) for f in diag.supine_frames.split(',')]+[(1, int(f)) for f in diag.prone_frames.split(',')]
ENVS = [(m, ident, frame) for m in MODES for ident, frame in CELLS for _ in range(diag.repeats)]
sys.argv = [sys.argv[0], *remaining, '--task', diag.eval_task, '--num_envs', str(len(ENVS)),
            '--seed', '42', '--skip-export', '--play-log-dir', str(output)]

import play  # Launch Kit first, then import simulator modules.
import torch
from cyclo_lab.manager_based.trace_recovery.integrated import TraceIntegratedCommand, TRACKING, RECOVERY
from cyclo_lab.manager_based.trace_recovery.body_adapter import tensor
from cyclo_lab.manager_based.trace_recovery.commands import JOINT_NAMES, BODY_NAMES

MOTIONS = ('supine', 'prone')


class FixedStartCommand(TraceIntegratedCommand):
    """Every env starts in Recovery at its own fixed cell; later resets (NaN guard only) repeat it."""
    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.forced_regimes[ids] = RECOVERY
        self.forced_reference_ids[ids] = self._cell_ids[ids]
        self.forced_frames[ids] = self._cell_frames[ids]
        super()._resample_command(ids)


original_run = play._run


def configured_run(env_cfg, agent_cfg):
    command = env_cfg.commands.reference_trajectory
    command.class_type = FixedStartCommand
    command.reset_perturbations = True  # training-like spread between repeats
    env_cfg.events.push_robot = None
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.terminations.post_tracking_horizon_complete = None
    if getattr(env_cfg.terminations, 'tracking_origin_horizon_complete', None) is not None:
        env_cfg.terminations.tracking_origin_horizon_complete = None
    env_cfg.terminations.time_out = None
    env_cfg.episode_length_s = 1e6
    # The command is built inside gym.make; hand it the cells through the class.
    original_init = FixedStartCommand.__init__

    def init(self, cfg, env):
        self._cell_ids = torch.tensor([e[1] for e in ENVS], device=env.device)
        self._cell_frames = torch.tensor([e[2] for e in ENVS], device=env.device)
        original_init(self, cfg, env)
    FixedStartCommand.__init__ = init
    return original_run(env_cfg, agent_cfg)


def diagnose(env, policy, *_unused):
    base = env.unwrapped
    c = base.command_manager.get_term('reference_trajectory')
    if not hasattr(policy, 'gate_logits'):
        raise TypeError('expected the gated actor as the inference policy')
    N = base.num_envs
    mode = torch.tensor([MODES.index(e[0]) for e in ENVS], device=base.device)
    for i, ref in enumerate(c.recovery_references):
        z = ref.body_position_w[:, 0, 2]
        print(f'[DIAG] {MOTIONS[i]}: {ref.num_frames} frames; reference pelvis >= .64 m from frame '
              f'{int(torch.nonzero(z >= .64)[0])}', flush=True)
    native_logits = policy.gate_logits

    def routed_logits(frames):
        logits = native_logits(frames)
        w = c.tracking_weight.clone()
        w[c.regimes == RECOVERY] = torch.where(mode == 2, 1., 0.)[c.regimes == RECOVERY]
        forced = torch.log(torch.stack((w, 1-w), -1).clamp_min(1e-6))  # softmax(log w) = w
        return torch.where((mode > 0)[:, None], forced, logits)
    policy.gate_logits = routed_logits

    manager = base.termination_manager
    original_compute = manager.compute

    def observed_compute():
        done = original_compute()
        keep = (manager.get_term('nan_detected').clone() if 'nan_detected' in manager.active_terms
                else torch.zeros_like(done))
        manager._terminated_buf &= keep
        manager._truncated_buf &= keep
        return done & keep
    manager.compute = observed_compute

    env.reset()
    obs = env.get_observations()
    assert bool((c.regimes == RECOVERY).all())
    recovered_like = c.recovered_like.clone()
    steps = round(diag.seconds/base.step_dt)
    zeros = torch.zeros(N, device=base.device)
    reached = torch.zeros(N, dtype=torch.bool, device=base.device)
    reach_step = torch.full((N,), -1, device=base.device)
    max_pelvis, nan_resets = zeros.clone(), zeros.clone()
    gate_sum, gate_n, gate_max = zeros.clone(), zeros.clone(), zeros.clone()
    traced = torch.tensor([i for i, e in enumerate(ENVS) if e[0] == 'gate' and e[1] == 0], device=base.device)
    trace = {k: [] for k in ('frame', 'regime', 'robot_joint_pos', 'ref_joint_pos', 'robot_body_pos', 'ref_body_pos')}
    with torch.inference_mode():
        for step in range(steps):
            action = policy(obs)
            native = torch.softmax(native_logits(policy.last_gate['frames'][:, -4:]), -1)[:, 0]
            rec = c.regimes == RECOVERY
            gate_sum += torch.where(rec, native, 0.)
            gate_n += rec.float()
            gate_max = torch.where(rec, torch.maximum(gate_max, native), gate_max)
            obs, _, done, _ = env.step(action)
            policy.reset(done)
            nan_resets += done.float()
            pelvis = c.robot_body_pos_w[:, 0, 2]
            if diag.trace:
                trace['frame'].append(c.recovery_phase[traced].cpu())
                trace['regime'].append(c.regimes[traced].cpu())
                trace['robot_joint_pos'].append(tensor(c.robot.data.joint_pos)[traced][:, c.joint_ids].cpu())
                trace['ref_joint_pos'].append(c.joint_pos[traced].cpu())
                trace['robot_body_pos'].append(c.robot_body_pos_w[traced].cpu())
                trace['ref_body_pos'].append(c.body_pos_w[traced].cpu())
            max_pelvis = torch.maximum(max_pelvis, pelvis)
            new = (c.regimes == TRACKING) & ~reached
            reach_step[new] = step+1
            reached |= c.regimes == TRACKING
    final = c.robot_body_pos_w[:, 0, 2]
    if diag.trace:
        import numpy as np
        np.savez_compressed(output/'trace.npz', env=traced.cpu().numpy(),
                            start_frame=np.array([ENVS[i][2] for i in traced.tolist()]),
                            joint_names=np.array(JOINT_NAMES), body_names=np.array(BODY_NAMES),
                            **{k: torch.stack(v, 1).numpy() for k, v in trace.items()})
    rows = []
    for i, (m, ident, frame) in enumerate(ENVS):
        rows.append({'env': i, 'mode': m, 'motion': MOTIONS[ident], 'start_frame': frame,
                     'recovered_like': bool(recovered_like[i]), 'reached_tracking': bool(reached[i]),
                     'tracking_step': int(reach_step[i]), 'final_pelvis_m': round(float(final[i]), 3),
                     'max_pelvis_m': round(float(max_pelvis[i]), 3),
                     'stuck_half_up': bool(.3 <= final[i] < .6), 'nan_resets': int(nan_resets[i]),
                     'gate_w_t_mean_in_recovery': round(float(gate_sum[i]/gate_n[i].clamp_min(1)), 4),
                     'gate_w_t_max_in_recovery': round(float(gate_max[i]), 4)})
    with open(output/'envs.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cells = []
    for m in MODES:
        for ident, frame in CELLS:
            r = [x for x in rows if x['mode'] == m and x['motion'] == MOTIONS[ident] and x['start_frame'] == frame]
            ok = [x['reached_tracking'] and x['final_pelvis_m'] >= .6 for x in r]
            cells.append({'mode': m, 'motion': MOTIONS[ident], 'start_frame': frame,
                          'recovered_like': sum(x['recovered_like'] for x in r)/len(r),
                          'success': sum(ok)/len(r), 'stuck_half_up': sum(x['stuck_half_up'] for x in r)/len(r),
                          'fallen': sum(x['final_pelvis_m'] < .3 for x in r)/len(r),
                          'gate_w_t_mean_in_recovery': round(sum(x['gate_w_t_mean_in_recovery'] for x in r)/len(r), 4),
                          'gate_w_t_max_in_recovery': max(x['gate_w_t_max_in_recovery'] for x in r)})
    with open(output/'cells.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    json.dump({'seconds': diag.seconds, 'repeats': diag.repeats, 'checkpoint': play.args_cli.checkpoint,
               'cells': cells}, open(output/'summary.json', 'w'), indent=1)
    for x in cells:
        print('[DIAG-CELL]', json.dumps(x), flush=True)
    env.close()


play._run = configured_run
play._play = diagnose
if __name__ == '__main__':
    play.play_diagnostics.run(play.main)
