# Experiment history

There were 23 training runs from 2026-08-10 to 2026-11-06. Unless noted, numbers are
**recovery success averaged over the last 2,500 iterations**, and every run used seed 42 once.

Each family of runs defines "recovery success" a little differently. Compare curve shapes, and compare
values within a family; don't compare absolute values across families.

Notation: **Supine** is the face-up get-up and **Prone** is the face-down get-up.
**Frame-0 eval** means one deterministic episode that starts at reference frame 0.

## Phase 1: single skills (PhysX)

| # | Run | Change | Result |
|---|---|---|---|
| 01 | Hiphop tracking | Mimic baseline, the actor sees the reference | Completion 0.99 |
| 03 | Supine only | Same Mimic structure | Completion 0.71; failures near frame 243 |
| 04 | Prone only | Same | Completion 0.99 (jump at ~11k) |
| 05 | Supine only + focus | 70 % of resets in the failing frames 120–180 | Completion 0.98 |

All three skills can be learned alone.

## Phase 2: one policy for all three skills (PhysX)

| # | Run | Change | Result |
|---|---|---|---|
| 02 | SM3 shared actor | One actor; resets 50/25/25; no early termination | Recovery 0 / 0 |
| 06 | SM3 + focus | Focus curriculum and exposure logging | Recovery 0 / 0 |
| 07 | SM3 + final-frame grace | Terminate if not upright 1 s after the final frame | Recovery 0 / 0 |
| 08 | SM4, 3 experts | Separate MLP per skill, oracle router | Supine 0, Prone 0.66 (mostly fall-started) |

Recovery reaches the last frame but never stands up.

## Phase 3: Tracking + unified Recovery expert (PhysX, U1–U1.5)

| # | Run | Change | Result |
|---|---|---|---|
| 09 | U1 | Two experts; **recovery reference hidden from the actor** | 0.39 / 0.33 |
| 10 | U1.2 paper core | H4 history, adaptive sampling, 2 s error termination, Recovery ×2.5 | 0.72 / 0.89, but an artifact of fall → recover → fall loops |
| 11 | U1.3 cohorts | Fixed 45/45/10 training cohorts | 0.97 / **0.50 plateau** |
| 12 | U1.4 soft transition | 1.5 s Transition blending in the Tracking reward | 0.69 / 0.48 |
| 13 | U1.5 bounded post-tracking | Episode completes after 2 s of Tracking | 0.73 / 0.48 |

Prone stays at about 0.5 whatever the structure.

## Phase 4: isolate Recovery (Newton)

| # | Run | Change | Result |
|---|---|---|---|
| 14 | Recovery only, unified | Move to Newton/MJWarp, w_T = 0 | 0.99 / 0.66; **99.6 % of Prone failures start at frames 329–351** (torso lift) |
| 15 | Per-motion experts | Supine and Prone experts split | 0.99 / 0.65, same failure frames, so expert interference is ruled out |
| 16 | + StableMimic Table V | Obs normalization, scalar std, entropy 0.05, reset perturbation, +10 success bonus | Diverged (NaN) at iteration 297; entropy too large for this reward scale |
| 17 | Same, entropy 0.005 | | 0.964 / **0.999**. Prone jumps at ~13k. Frame-0 Prone succeeds. Frame-0 Supine stands about 90 frames early and is flagged by the 2 s error rule. |

The StableMimic setting bundle breaks the Prone local optimum. In it, the robot stalls folded at the hips.

## Phase 5: integrated TRACE (Newton)

| # | Run | Change | Result |
|---|---|---|---|
| 18 | Integrated U1.5 | Run 17's Recovery settings + Tracking expert + Transition | 0.77 / 0.47. Recovery gets only 10 % of steps. |
| 19 | + exposure balance | Resets 30/35/35; Tracking-origin episodes 12 s | 0.96 / 0.64 |
| 20 | 19 resumed to 50k | | Prone jumps 0.65 → 0.99 at ~35.8k. Frame-0 Prone reaches the dance. |
| 21 | **Learned gate** | Proprioceptive gate replaces the oracle; Supine trim100; early-upright handoff | 0.83 / 0.65, Prone jumps only at ~49.2k. Frame-0 Supine is stuck half-up. |
| 22 | + failure look-ahead | BeyondMimic 2 s kernel (decay 0.8), uniform:adaptive 50:50 | 0.997 / 0.998. **Frame-0 both reach the dance; 192/192 perturbed starts.** |
| 23 | **+ Prone trim80 (final)** | Prone lead-in trimmed so both get-ups start moving at ~0.56 s | 0.997 / 0.999, **no plateau** (0.95 at ~2.1k / ~4.0k). Frame-0 Tracking starts at step 261 / 338 (run 22: 257 / 507). 192/192. |

### Notes on the final runs

- **Plateaus look random.** Each motion stays on a plateau and then jumps to ~1.0 within about 1k iterations.
  With nearly the same setup, Prone jumped at ~36k (run 20), ~49.4k (run 21) and ~8.3k (run 22).
- **Run 21's Supine plateau was a local optimum, not missing settle time.** In fixed-start tests, run 20
  recovers 100 % from original frames 100–200. These are the states that trimmed frames 0–100 start from.
  Run 21 recovers 0 % from those same states. Run 22's look-ahead sampler fixes it.
- **Transition smoothness (run 22).** Action-rate P95 is 38.6 in Transition and 36.7 in late Tracking (Supine).
  For Prone it is 40.3 and 36.8. The gate moves faster than the α ramp, but the blended action stays smooth.
  The expert-alignment loss keeps the two expert means close during Transition: P95 of |μ_T − μ_R| is 0.74–0.90.

## Deployment: heading alignment and pushes

- Recovery was trained with reset yaw noise of ±0.05 rad only. Frame-0 starts rotated by 90°, 180° or 270°
  succeed 0–6 % of the time.
- `trace_heading_alignment.py` waits until the robot lies still for 0.3 s. It then rotates the dance reference
  about z by (robot head angle − reference frame-0 head angle).
- Push test with 96 environments, 8 directions × 3 speeds: falls that return to the dance went from **15/44 to 45/45**.
- Ball mode in the exhibition: a 3 kg ball at 10 m/s is absorbed. A 10 kg ball knocks the robot down
  5 times out of 5, and all 5 times it is back to the dance within 5.7–10.7 s.
- Robots that fail to get up are not at joint limits or torque limits. The policy output almost freezes
  (action change 0.04 vs. 0.18).
