# Sim-to-sim in plain MuJoCo

Both tools reuse [`web/trace_mujoco.py`](../web/trace_mujoco.py): the K1 MJCF with the training PD gains, the
Newton MuJoCo-Warp solver settings, the policy in plain PyTorch and observations identical to training.
Choose the policy with `TRACE_POLICY=run24` (default, final) or `TRACE_POLICY=run23`. Use the `web/.venv`
environment (see [`web/README.md`](../web/README.md)).

## Native viewer

```bash
web/.venv/bin/python sim2sim/play_mujoco.py [--start stand|supine|prone] [--condition delay_40ms] [--no-align]
```

| Key | Action |
|---|---|
| Home / End / Insert | reset: Stand / Supine / Prone (frame 0 of the dance or get-up reference) |
| Arrow keys / Delete | push the pelvis in a screen direction / a random direction |
| Page Up / Page Down | push strength up / down (0.5–4.0 m/s) |
| Ctrl + right drag | MuJoCo's own perturbation (spring force on a body) |

Heading alignment is on by default. Run 23 needs it; run 24 learned to get up from any heading (`--no-align`).
`--condition` takes any condition name from `strict_sweep.py`. Terminal commands also work: `stand`,
`supine`, `prone`, `cond <name>`, `list`, `push <deg>`, `random`, `quit`.

## Strict sweep

```bash
systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 \
    web/.venv/bin/python sim2sim/strict_sweep.py --seeds 8 --workers 6
```

Each condition changes one thing away from training: solver, contact, friction, mass, PD gains, armature,
joint friction, action delay or sensor noise. There are also randomized combinations and limit probes.
Each condition runs four scenarios (stand, Supine, Prone, push) × `--seeds` trials with training-level
observation noise. Keep the worker count low on a laptop: MuJoCo models are not always freed between trials,
so memory grows; the `systemd-run` cap kills only the sweep if that happens.

`results/strict_20261111_2325/` holds the **run 23** sweep (1184 trials, with heading alignment in the push
scenario): [`summary.md`](results/strict_20261111_2325/summary.md), `summary.json` and `trials.json`.

- **Passes:** solver changes, friction 0.4–1.5, mass ±15–20 %, kp/kd ±20 %, effort ×0.7, delay ≤ 20 ms,
  random combinations.
- **Weak:** 40 ms delay and zero armature.
- **Breaks:** delay ≥ 60 ms and kp/kd ×0.5.
