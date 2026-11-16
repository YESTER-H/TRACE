# Browser demo

Live: **https://yester-h.github.io/TRACE/**

The policy runs in the browser with [mjswan](https://github.com/ttktjmt/mjswan) (MuJoCo 3.11 WebAssembly +
ONNX Runtime Web). No server is needed.

## Setup

```bash
cd web
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "mjswan[mjlab]==0.12.12" onnxruntime opencv-python-headless
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
```

`K1_MJCF` points to the K1 MJCF from cyclo_lab's `ai_sapiens` submodule. Default:
`~/humanoid/cyclo_lab_newton/third_party/ai_sapiens/ai_sapiens_description/mujoco/k1/k1.xml`.

## Build and run

```bash
.venv/bin/python build_web.py     # build web/dist and open a local preview
./build_pages.sh                  # GitHub Pages site (landing page + demo/) -> web/pages_out
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `TRACE_POLICY` | `run24` | `run24` (final) or `run23` |
| `TRACE_ALIGN_HEADING` | `0` for run 24, `1` for run 23 | turn the dance reference to a fallen robot's heading |

Demo controls: *Lie down face up / face down*, *Shove (random push)*, *Interact → Push / Pull*, *Reset*.

## Checks

```bash
.venv/bin/python sim2sim.py       # Stand / Supine / Prone starts in CPU MuJoCo
.venv/bin/python check_dist.py    # the built bundle (scene + ONNX) in Python
.venv/bin/python push_test.py     # pushes, heading alignment off vs on
MUJOCO_GL=egl .venv/bin/python make_teaser.py   # README teaser GIF
```
