# Sim-to-sim in plain MuJoCo

Runs the policy in CPU MuJoCo with the training PD gains and solver settings. Uses the `web/.venv`
environment (see [`web/README.md`](../web/README.md)). Choose the policy with `TRACE_POLICY=run24`
(default) or `TRACE_POLICY=run23`.

## Viewer

```bash
web/.venv/bin/python sim2sim/play_mujoco.py [--start stand|supine|prone] [--condition delay_40ms] [--no-align]
```

| Key | Action |
| --- | --- |
| Home / End / Insert | reset: Stand / Supine / Prone |
| Arrow keys / Delete | push in a screen direction / a random direction |
| Page Up / Page Down | push strength up / down |
| Ctrl + right drag | drag a body |

Use `--no-align` with run 24 (it does not need heading alignment). `--condition` takes a condition name from
`strict_sweep.py`.

## Strict sweep

```bash
systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 \
    web/.venv/bin/python sim2sim/strict_sweep.py --seeds 8 --workers 6
```

The sweep changes one physics, actuator or sensing setting at a time and runs four scenarios: stand, Supine,
Prone and push. Keep the worker count low, because memory grows over trials. A saved run 23 sweep is in
[`results/strict_20261111_2325/summary.md`](results/strict_20261111_2325/summary.md).
