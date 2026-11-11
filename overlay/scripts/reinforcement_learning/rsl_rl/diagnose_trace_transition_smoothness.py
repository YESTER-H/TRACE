# SPDX-License-Identifier: Apache-2.0
"""How smooth is the Recovery -> Tracking handoff of a gated TRACE policy?  Fixed frame-0 starts, gate only.

Per step it records the deterministic action, the gate w_T, the oracle alpha and both expert means, then
reports per regime the action-rate P95 (|a_t - a_{t-1}|/dt, max over joints, the paper's target-rate
statistic on the raw action), and in Transition the gap between the gate blend and the alpha blend
(|(w_T - alpha)(mu_T - mu_R)|, max over joints) and |mu_T - mu_R|. Earlier description, kept for the shared parts:
Can a checkpoint finish a recovery from a fixed reference frame?

Every env starts once in Recovery at a fixed (motion, reference frame) with training reset perturbations
on, and the policy runs for --seconds with its own routing (gate or oracle router). Terminations are only
observed, not applied (NaN guard only), so each env reports both:

  training_success   reached Tracking before any training termination fired (what the training metric counts)
  stood_success      reached Tracking at some point and ended with pelvis >= .6 m (deployment view)

plus the first termination term that fired and when. Written to cells.csv / envs.csv / summary.json in
--output-dir (never overwritten). Display-free.

usage (inside the container, /workspace/cyclo_lab):
  /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/diagnose_trace_fixed_starts.py --headless \
    --eval-task Cyclo-TRACE-K1-Rev1-Integrated-U15-Balanced-Play --supine-frames 0,100,140 --prone-frames '' \
    --output-dir logs/eval/diag_fixed_starts_<time> --checkpoint <run>/model.pt
"""
import argparse
import csv
import json
import sys
from pathlib import Path

diag_parser = argparse.ArgumentParser(add_help=False)
diag_parser.add_argument('--output-dir', required=True)
diag_parser.add_argument('--eval-task', required=True)
diag_parser.add_argument('--seconds', type=float, default=15.)
diag_parser.add_argument('--repeats', type=int, default=32)
diag_parser.add_argument('--settle-seconds', type=float, default=2., help='Tracking time after Transition counted as "early tracking"')
diag_parser.add_argument('--supine-frames', default='0')
diag_parser.add_argument('--prone-frames', default='')
diag, remaining = diag_parser.parse_known_args()
output = Path(diag.output_dir).resolve()
output.mkdir(parents=True, exist_ok=False)
frames_of = lambda s: [int(f) for f in s.split(',') if f.strip()]
CELLS = [(0, f) for f in frames_of(diag.supine_frames)]+[(1, f) for f in frames_of(diag.prone_frames)]
ENVS = [(ident, frame) for ident, frame in CELLS for _ in range(diag.repeats)]
sys.argv = [sys.argv[0], *remaining, '--task', diag.eval_task, '--num_envs', str(len(ENVS)),
            '--seed', '42', '--skip-export', '--play-log-dir', str(output)]

import play  # Launch Kit first, then import simulator modules.
import torch
from cyclo_lab.manager_based.trace_recovery.integrated import TraceIntegratedCommand, TRACKING, RECOVERY

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
    env_cfg.episode_length_s = 1e6
    original_init = FixedStartCommand.__init__

    def init(self, cfg, env):
        self._cell_ids = torch.tensor([e[0] for e in ENVS], device=env.device)
        self._cell_frames = torch.tensor([e[1] for e in ENVS], device=env.device)
        original_init(self, cfg, env)
    FixedStartCommand.__init__ = init
    return original_run(env_cfg, agent_cfg)


def diagnose(env, policy, *_unused):
    base = env.unwrapped
    c = base.command_manager.get_term('reference_trajectory')
    N = base.num_envs
    for i, ref in enumerate(c.recovery_references):
        z = ref.body_position_w[:, 0, 2]
        print(f'[DIAG] {MOTIONS[i]}: {ref.num_frames} frames; reference pelvis >= .64 m from frame '
              f'{int(torch.nonzero(z >= .64)[0])}', flush=True)
    manager = base.termination_manager
    names = list(manager.active_terms)
    # Only failure terms count against training_success (success / horizon / time-out terms do not).
    failure_names = [n for n in names if 'success' not in n and 'horizon' not in n and n not in ('time_out', 'nan_detected')]
    print(f'[DIAG] termination terms {names}; counted as failure {failure_names}', flush=True)
    original_compute = manager.compute
    first_term = torch.full((N,), -1, device=base.device, dtype=torch.long)
    first_term_step = torch.full((N,), -1, device=base.device, dtype=torch.long)
    step_now = [0]

    def observed_compute():
        done = original_compute()
        for k, name in enumerate(names):
            if name not in failure_names:
                continue
            hit = manager.get_term(name).clone() & (first_term < 0)
            first_term[hit] = k
            first_term_step[hit] = step_now[0]
        keep = manager.get_term('nan_detected').clone() if 'nan_detected' in names else torch.zeros_like(done)
        manager._terminated_buf &= keep
        manager._truncated_buf &= keep
        return done & keep
    manager.compute = observed_compute

    env.reset()
    obs = env.get_observations()
    assert bool((c.regimes == RECOVERY).all())
    first_term[:] = -1
    first_term_step[:] = -1
    steps = round(diag.seconds/base.step_dt)
    reached = torch.zeros(N, dtype=torch.bool, device=base.device)
    reach_step = torch.full((N,), -1, device=base.device, dtype=torch.long)
    max_pelvis = torch.zeros(N, device=base.device)
    rec = {k: [] for k in ('action', 'w_t', 'alpha', 'gap', 'expert_diff', 'regime')}
    with torch.inference_mode():
        for step in range(steps):
            step_now[0] = step+1
            regime_before = c.regimes.clone()
            alpha = torch.where(regime_before == RECOVERY, 0., c.transition_alpha)
            alpha = torch.where(regime_before == TRACKING, 1., alpha)
            action = policy(obs)
            g = policy.last_gate
            w_t = g['weights'][:, 0]
            diff = g['mu_t']-g['mu_r']
            rec['action'].append(action.clone())
            rec['w_t'].append(w_t.clone())
            rec['alpha'].append(alpha)
            rec['gap'].append(((w_t-alpha)[:, None]*diff).abs().max(-1).values)
            rec['expert_diff'].append(diff.abs().max(-1).values)
            rec['regime'].append(regime_before)
            obs, _, done, _ = env.step(action)
            policy.reset(done)
            max_pelvis = torch.maximum(max_pelvis, c.robot_body_pos_w[:, 0, 2])
            new = (c.regimes == TRACKING) & ~reached
            reach_step[new] = step+1
            reached |= c.regimes == TRACKING
    R = {k: torch.stack(v, 1) for k, v in rec.items()}  # [N, T, ...]
    rate = torch.cat((torch.zeros_like(R['action'][:, :1]), R['action'].diff(dim=1)), 1).abs().max(-1).values/base.step_dt
    rate[:, 0] = float('nan')
    smooth = {}
    from cyclo_lab.manager_based.trace_recovery.integrated import TRANSITION
    settle = round(diag.settle_seconds/base.step_dt)
    for ident, motion in enumerate(MOTIONS):
        m = torch.tensor([e[0] == ident for e in ENVS], device=base.device)
        if not m.any():
            continue
        reg = R['regime'][m]
        first_tracking = torch.where((reg == TRACKING).any(1), (reg == TRACKING).float().argmax(1), reg.shape[1])
        t_idx = torch.arange(reg.shape[1], device=base.device)[None]
        windows = {'recovery': reg == RECOVERY, 'transition': reg == TRANSITION,
                   'early_tracking': (reg == TRACKING) & (t_idx < first_tracking[:, None]+settle),
                   'late_tracking': (reg == TRACKING) & (t_idx >= first_tracking[:, None]+settle)}
        out = {}
        for name, w in windows.items():
            r = rate[m][w]
            r = r[torch.isfinite(r)]
            out[f'{name}_action_rate_p95'] = round(float(r.quantile(.95)), 2) if len(r) else None
            out[f'{name}_action_rate_p50'] = round(float(r.quantile(.5)), 2) if len(r) else None
        tw = windows['transition']
        out['transition_gate_minus_alpha_mean'] = round(float((R['w_t'][m]-R['alpha'][m])[tw].mean()), 3)
        out['transition_blend_gap_p95'] = round(float(R['gap'][m][tw].quantile(.95)), 3)
        out['transition_expert_diff_p95'] = round(float(R['expert_diff'][m][tw].quantile(.95)), 3)
        out['recovery_expert_diff_p95'] = round(float(R['expert_diff'][m][windows['recovery']].quantile(.95)), 3)
        smooth[motion] = out
        print('[DIAG-SMOOTH]', motion, json.dumps(out), flush=True)
    json.dump(smooth, open(output/'smoothness.json', 'w'), indent=1)
    final = c.robot_body_pos_w[:, 0, 2]
    rows = []
    for i, (ident, frame) in enumerate(ENVS):
        term = names[int(first_term[i])] if first_term[i] >= 0 else ''
        term_step = int(first_term_step[i])
        train_ok = bool(reached[i]) and (term_step < 0 or term_step >= int(reach_step[i]))
        rows.append({'env': i, 'motion': MOTIONS[ident], 'start_frame': frame,
                     'training_success': train_ok, 'stood_success': bool(reached[i]) and float(final[i]) >= .6,
                     'tracking_step': int(reach_step[i]), 'first_termination': term,
                     'first_termination_step': term_step, 'final_pelvis_m': round(float(final[i]), 3),
                     'max_pelvis_m': round(float(max_pelvis[i]), 3)})
    with open(output/'envs.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cells = []
    for ident, frame in CELLS:
        r = [x for x in rows if x['motion'] == MOTIONS[ident] and x['start_frame'] == frame]
        before = [x['first_termination'] for x in r if not x['training_success'] and x['first_termination']]
        cells.append({'motion': MOTIONS[ident], 'start_frame': frame,
                      'training_success': sum(x['training_success'] for x in r)/len(r),
                      'stood_success': sum(x['stood_success'] for x in r)/len(r),
                      'stuck_half_up': sum(.3 <= x['final_pelvis_m'] < .6 for x in r)/len(r),
                      'fallen': sum(x['final_pelvis_m'] < .3 for x in r)/len(r),
                      'failure_terms': {t: before.count(t) for t in sorted(set(before))}})
    with open(output/'cells.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    json.dump({'eval_task': diag.eval_task, 'seconds': diag.seconds, 'repeats': diag.repeats,
               'checkpoint': play.args_cli.checkpoint, 'termination_terms': names, 'cells': cells},
              open(output/'summary.json', 'w'), indent=1)
    for x in cells:
        print('[DIAG-CELL]', json.dumps(x), flush=True)
    env.close()


play._run = configured_run
play._play = diagnose
if __name__ == '__main__':
    play.play_diagnostics.run(play.main)
