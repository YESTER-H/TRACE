# TRACE Recovery-only semantic port (Newton / RSL-RL5)

This is an additive diagnostic, not the integrated TRACE method. No original
PhysX worktree, upstream mimic/locomotion source, URDF, NPZ, Newton importer,
Isaac Lab source, or RSL-RL source is overwritten.

## Registered tasks

- Cyclo-TRACE-K1-Rev1-Recovery-Only (Newton)
- Cyclo-TRACE-K1-Rev1-Recovery-Only-Play (no reset perturbation/actor corruption)
- Cyclo-TRACE-K1-Rev1-Recovery-Only-PhysX (same-stack comparison)

Runner default is deliberately **one iteration**, 64 environments, seed 42,
scratch, no resume. A later training run requires an explicit iteration override.
No integrated Tracking/Transition/post-Tracking training is registered here.

## Immutable reference inputs

TRACE_MOTION_INPUT_DIR defaults to /diag_inputs in the existing Newton container.
It must contain tracking.npz, supine.npz, prone.npz, currently read-only bind mounts
of the original PhysX reference files. The trajectories have 300/436/470 frames
at 50 Hz. Neither source files nor motion data are copied/edited.

The source layout is legacy_k1_layout.json: 23 canonical joints, 25 logical bodies,
WXYZ source quaternions and original source-COM offsets. ReferenceTrajectory is
reused with explicit names; the upstream 24-name fallback is never used.

## Compatibility and action contract

Native Newton stays 23 joints / 24 bodies. Logical poses are link-frame poses;
logical linear velocities are velocities at the legacy source-URDF COM point.
All internal new-stack quaternions are XYZW. For every body:

1. native actual-COM velocity -> native link-origin velocity;
2. reconstruct head link pose/velocity through the fixed torso-head transform;
3. link-origin velocity -> original source-COM velocity;
4. reorder by canonical body names.

Reset uses native root-link pose, actual root-COM velocity and named joint-state
writers, with int32 write indices shared by Newton and PhysX. Derived gravity and
body-frame velocity caches are invalidated without changing elapsed time, adding
a physics step, clearing acceleration history, or drawing random numbers.

The Newton body-force and shape-head sensor have synchronized history length 3.
Logical head force is the real head collision-shape force; logical torso force is
merged torso force minus head force. Missing/incorrect sensor layouts raise.

Actions use canonical joint ordering, default-offset absolute position targets,
the unchanged upstream named action scales and native RSL-RL clipping at +/-10.
The upstream startup home-offset event assumes action order == physical order.
That assumption is false on Newton. canonical_home_joint_offset_noise reuses its
exact noise draw/state update and only synchronizes action offsets by joint names.
The physical q_default, scales, gains, and random-number consumption are preserved.

## Observations and policy

U1.5 history was **term-wise H4**, not a global [4,127] flattening. Its real order
is preserved: live motion command46, live anchor orientation6, angular velocity3,
canonical q-relative23, canonical dq-relative23, last action23, gravity3 (last).
H4 is 508; per-term native reset buffers repeat the initial sample four times.
Recovery identity/frame/target never enter this actor group.

The critic is 256: hidden successor command46, target-anchor relative position3,
orientation6, measured 14-body positions42/orientations84, source-COM root linear
velocity3 in root frame, angular velocity3, q-relative23, dq-relative23, action23.
Router is separate [N,1] w_T and is stored in native TensorDict rollout storage.

U14ProposalActor ports two 508->512->256->128->23 ELU actors. The native separate
critic is 256->512->256->128->1. One native state-independent Gaussian has a shared
log_std[23] around w_T*mu_T + (1-w_T)*mu_R. Native RSL-RL5 owns sampling,
log-probability, entropy, KL, storage, GAE, PPO, clipping, optimizer and checkpoint
serialization. Tracking is retained in the module/optimizer but w_T=0 throughout
this diagnostic, so it receives no updates. Both references use the same actor.

Parameter counts: Tracking427799, Recovery427799, critic295937, log_std23,
total1151558. Native two-input JIT/ONNX export is not implemented; unsupported
single-input export raises explicitly instead of dropping the oracle router.

## MDP semantics

Only Recovery actions/rewards are active. Live Tracking phase remains independent
and advances each control step except the existing just-reset suppression.
Recovery target is k+1 (clamped at the last frame). The original six imitation
kernels/weights/stds and 2.5 Recovery scale are preserved; the positional root
kernel is root height in Recovery. Original regularizers are retained, including
allowing recovery knee contacts. There is no success bonus or Tracking reward.

Persistent E = mean of four normalized squared errors: root height/.3,
anchor orientation/.4, mean body-position norm/.3, mean body-angle/.4.
E>4 for 100 consecutive 50-Hz control steps is a failure; non-violation clears
the counter. Confirmed failure records onset and detection separately.
Supine/Prone onset histograms and EMA scores are independent.

Direct resets choose Supine/Prone 50/50, then choose uniform/adaptive frame
sampling 50/50. There are 20 normalized bins, EMA alpha=.1, score floor=.001,
temperature1, priority/uniform mixture .8/.2. Only frames with a valid successor
are sampled (terminal frame excluded), matching the original sampler formula.

Final frame + upright (pelvis height>=.65, gravity alignment>=cos30deg) is true
completion/terminated, never failure or time_out/bootstrap. Completion has no
failure histogram/EMA update and immediately resets to the normal distribution.
At final frame while non-upright, grace expires after exactly 50 control steps
(1 s). Persistent/grace/safety failures are true terminations. Episode time_out is
still truncated/bootstrap. There is no Transition, learned gate, re-entry or
post-Tracking in this Recovery-only diagnostic.

The native termination/reward managers run before command progression/autoreset
observations. Exposure labels use the policy state that generated the action.
Error snapshots are post-physics, before autoreset, against that action's k+1
target. Final completion checks the current final-frame state in this pipeline.

## Exact logging

Trace/{supine,prone}_mode_step_fraction and phase_bin_00..19_step_fraction are
pre-action occupancy divided by num_envs; Tracking occupancy is zero.
Final-frame hold/not-upright and Supine120..180 exposure use the same masks.
Direct-reset, success, persistent-failure and grace-failure events are separate.
No runtime Tracking->Recovery event exists in a Recovery-only task.

Every event scalar is step count / num_envs. Native RSL-RL averages each scalar
over a rollout. Recover iteration count = mean scalar * num_envs * rollout steps.
Across iterations sum these counts. Direct-reset/failure histograms use deltas,
not cumulative counters or reset-subset command-metric averages.

Error/reward contribution = sum(value * reference-mask) / num_envs.
Period conditioned mean = sum(iteration contribution * rollout_steps) /
sum(iteration mode fraction * rollout_steps). Do not average conditional means.
Joint diagnostic errors are joint-vector L2 norms, body diagnostics mean norms/
geodesic angles, root height absolute error; total reward is per-control-step
reward after manager weights and dt. Logging does not enter observations/rewards.

## Same-stack PhysX representation

PhysX keeps fixed head unmerged (25 physical bodies). Its link hierarchy is
nested under Robot/Geometry/pelvis, and the pinned multi-body sensor assumes
siblings. physx_compat uses the unchanged native URDF spawner, enables contact
reporting on each rigid link, and uses one native single-link sensor per named
body. These exact native forces form the same logical25 contact tensor.
No head-force duplication, collision/model editing or importer-source patch
is used. Newton configuration stays untouched. Fixed-body representation and
backend-specific contact-query cost differ; identical dynamics/performance are
not claimed. New upstream K1 inertials also differ from old PhysX source inertials;
the source-COM adapter preserves feature/reward meanings, not old dynamics.

## Validation artifacts and limits

Host artifacts: /tmp/trace_newton_runtime/trace_port_20261005
Container artifacts: /diag/trace_port_20261005
smoke.py: 64-env correctness + scratch one iteration.
smoke.py --verify-only: forced success/grace/contact, Gaussian/ratio/clipping,
RNG parity and strict checkpoint verification; no additional PPO update.
backend_check.py: 2-env PhysX five steps only.
preserve.py: before/after protected worktree, index, logs, submodule and file hashes.

Early offset-bug checkpoints are INVALID. All smoke checkpoints, even final valid
ones, must never be warm-start/resume sources for subsequent research training.
Long training was not started. 4096-env memory/throughput remains unverified.
A short 4096-env capacity smoke is needed before authorizing a long scratch run.

The stock upstream Mimic default source-body metadata has 24 names while its
existing Dance1 NPZ has25 bodies. Its default loader is blocked independently
of this additive port. Do not label that default task PASS; no upstream fix was
made in this task. Upstream K1 locomotion Newton smoke passes.

## Per-motion Recovery expert split (diagnostic)

- Cyclo-TRACE-K1-Rev1-Recovery-Motion-Experts (+ -Play)

Same env, observations, critic, rewards, sampler, terminations and PPO as
Recovery-Only. Only the actor differs: RecoveryMotionExpertActor holds one
508->512->256->128->23 ELU expert per reference (supine, prone; 427799 each,
same as the U14 Recovery expert) with the shared log_std[23]. The training-only
router group is a [Supine, Prone] one-hot of the hidden identity and is never
concatenated to the network input; each sample's gradient reaches only its
expert. This tests whether sharing one Recovery expert causes the Prone
329-351 plateau. Deployment would require a learned proprioceptive gate;
no JIT/ONNX export. Evaluate with evaluate_recovery_frame0_motion_experts.py.
Logs: logs/rsl_rl/trace_newton_recovery_motion_experts. Runner default stays
one iteration; long training requires an explicit override.

## Paper-aligned unified Recovery (StableMimic Table V settings)

- Cyclo-TRACE-K1-Rev1-Recovery-Paper-Aligned (+ -Play)

Unified Recovery-only MDP and U14 actor (w_T=0) unchanged except:
actor/critic observation normalization (policy group only, router excluded),
RSL-RL "scalar" std, entropy 0.005 (Table V 0.05 diverged to a NaN reward
by iteration 297 in run 2026-10-27_14-44-20), Table V reset ranges (pose xy +/-0.05, z +/-0.01,
roll/pitch +/-0.1, yaw +/-0.2; velocity xy +/-0.5, z +/-0.2, roll/pitch +/-0.52,
yaw +/-0.78; joints +/-0.1, joint velocity +/-0.1 kept), and a terminal success
bonus (weight 500 x dt 0.02 = return 10 on final frame + upright). The 2 s
persistent-error termination is unchanged. Logs:
logs/rsl_rl/trace_newton_recovery_paper_aligned. Evaluate with
evaluate_recovery_frame0_paper_aligned.py.

## Integrated U1.5 TRACE (Tracking + unified Recovery)

- Cyclo-TRACE-K1-Rev1-Integrated-U15 (+ -Play)

Port of the PhysX U1.4/U1.5 state machine (stable_mimic proposal_transition_commands
and bounded_post_tracking_commands) onto the paper-aligned Recovery-only MDP, in
integrated.py. Direct resets pick Tracking/Recovery 50/50. Tracking resets use the
U1.4 sampler (uniform/adaptive 50/50 over 20 Tracking bins, failure EMA .1, floor
1e-3, mix .2) and the original Mimic reset ranges with joint noise +/-0.1, no joint
velocity noise. Recovery resets, sampler, persistent error, grace and reset ranges
are the Recovery-only ones, now active only in the Recovery regime.

Handoff: final frame + upright (the pre-progression test Recovery-only terminated
on), or reacquisition (all eight U1.4 live-Tracking error limits for 5 steps),
enters Transition. 75 Transition actions blend w_T = 0, 1/74, ..., 1 and reward
the live Tracking kernels x alpha. Both handoffs align the live Tracking xy to
the robot anchor. After Transition, Recovery-origin episodes run Tracking for
100 more actions and end as a time-out (U1.5 bounded horizon); any earlier
failure or time-out wins. Tracking fails on the U1 fall detector (pelvis <= .514
and up-alignment <= .2) or the U1.4 deviation limits (anchor height .25, anchor
gravity .8, end-body height .25 against the live reference).

Rewards: Tracking kernels x1 (full anchor xyz), Recovery x2.5 (root height),
Transition x alpha. The paper-aligned +10 success bonus is paid on the first
Transition action for completion and reacquisition alike. Actor/critic/PPO are
TraceRecoveryPaperAlignedRunnerCfg (U14 two-expert actor, obs normalization,
scalar std, entropy .005); w_T is the oracle tracking weight. Logs:
logs/rsl_rl/trace_newton_integrated_u15. Simulator checks:
scripts/reinforcement_learning/rsl_rl/smoke_trace_integrated.py (19 checks).


## Integrated gated (final structure)

Cyclo-TRACE-K1-Rev1-Integrated-Gated(-Play), env_cfg.TraceIntegratedGatedEnvCfg,
runner_cfg.TraceIntegratedGatedRunnerCfg, gated.py. Balanced resets/horizon as run 19, plus:
- Proprioceptive gate (StableMimic / project proposal): softmax(f_G(H4(o_prop))/tau) blends
  the two expert means; o_prop = angular velocity, joint position/velocity, previous
  action, projected gravity (75/frame, same terms and noise as the policy group). The
  oracle router group is removed; the regime only labels the gate ("gate_target":
  y_T, same-regime mask, Transition mask). TracePPO adds CE .1 (x4 on Transition),
  consistency .01 (gate on the t-1 window of the 5-frame "gate" group), alignment .01.
  A Recovery reset whose reference pelvis is >= .8 x the live command pelvis keeps y_T=1.
- Supine reference: supine_getup_trim100.npz (first 100 still frames removed, 336 frames).
- Early-upright handoff: 25 consecutive upright Recovery steps enter Transition before the
  final frame (reason early_upright); upright Recovery states are exempt from the 2 s rule.
Simulator checks: scripts/reinforcement_learning/rsl_rl/smoke_trace_integrated_gated.py (15).

## Integrated gated + failure look-ahead (BeyondMimic)

`Cyclo-TRACE-K1-Rev1-Integrated-Gated-Lookahead` = run 21 (`-Gated`) with `adaptive_kernel_seconds=2`,
`adaptive_kernel_decay=.8`, `adaptive_inner_uniform=0` (overall uniform:adaptive 50:50 as in
StableMimic; runs 14-21 had an extra 20% uniform inside the adaptive branch, 60:40). A failure is still recorded at its onset phase, but the adaptive
sampler also raises the bins up to 2 s before it (weight .8**seconds ahead), as BeyondMimic's
non-causal kernel does (1 s bins, u in {0,1,2}, rho .8). Default 0 keeps runs 14-21 unchanged.
Evaluate with `-Gated-Play` (the sampler does not act there). Smoke: `smoke_trace_gated_lookahead.py`.

## Integrated gated + look-ahead + Prone trim80

- Cyclo-TRACE-K1-Rev1-Integrated-Gated-ProneTrim (evaluate with -Gated-ProneTrim-Play)

Run 22 (gate, Supine trim100, early-upright handoff, 2 s failure look-ahead, 50:50 reset mix) with the Prone
reference trimmed by 80 frames (data/motions/K1_rev1/get_up/prone/prone_getup_trim80.npz, written by
scripts/tools/motion/trim_reference_motion.py). Both get-ups now start moving at ~0.56 s (frame ~28).
