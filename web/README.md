# Web demo (browser, MuJoCo WebAssembly)

Live: **https://yester-h.github.io/TRACE/**

The run-23 policy runs in the browser with [mjswan](https://github.com/ttktjmt/mjswan)
(MuJoCo 3.11 WebAssembly + ONNX Runtime Web). Nothing runs on a server.

| File | Purpose |
|---|---|
| `trace_mujoco.py` | The policy in plain PyTorch, the K1 MuJoCo model with the training PD gains and solver settings, and observations identical to training |
| `sim2sim.py` | Sim-to-sim check in CPU MuJoCo from the Stand / Supine / Prone frame-0 starts |
| `build_web.py` | Builds the mjswan app: mjlab env config, ONNX export, motion clip, lie-down buttons |
| `check_dist.py` | Runs the built bundle (`dist/` scene + ONNX via onnxruntime) in Python |
| `build_pages.sh` | Builds the GitHub Pages site (landing page + `demo/`) into `pages_out/` |
| `site/index.html` | Landing page |
| `tests/*.cjs` | Headless-Chromium smoke tests (Playwright) |

```bash
cd web
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "mjswan[mjlab]==0.12.12" onnxruntime opencv-python-headless
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu

.venv/bin/python sim2sim.py              # 1. does the policy work in plain MuJoCo?
.venv/bin/python build_web.py            # 2. build web/dist and open a local preview
.venv/bin/python check_dist.py           # 3. run the built bundle in Python
./build_pages.sh                         # 4. GitHub Pages site -> web/pages_out
```

`K1_MJCF` points to the K1 MJCF (default:
`~/humanoid/cyclo_lab_newton/third_party/ai_sapiens/ai_sapiens_description/mujoco/k1/k1.xml`, from the
cyclo_lab `ai_sapiens` submodule). Its collision geometry matches the URDF used in training.

## How the policy is adapted

- **Physics.** Training ran Newton with the MuJoCo-Warp solver. The settings are matched: dt 0.005,
  decimation 4, Newton solver, implicitfast, pyramidal cone, 100/50 iterations, implicit PD
  (kp/kd/armature/effort per joint group).
- **One ONNX input.** The gate's input (4 frames of angular velocity, joint position and velocity,
  last action, gravity) is a subset of the policy observation. So the exported graph rebuilds it
  internally and takes only mjswan's `actor` observation.
- **Joint order.** mjswan works in the MuJoCo model's joint and body order. The exported graph
  permutes to the training (canonical) order and back. The motion clip is rewritten in model
  order.
- **Reference clock.** The dance frame follows simulation time (`time_source="sim"`), one frame
  per control step. It loops without a reset, as in training.
- **Lie-down buttons** write frame 0 of the Supine / Prone get-up reference. The heading relative
  to the dance is the trained one.

## Known limitations

- The deployment-time heading alignment from the exhibition script is not ported. A robot pushed
  into another lying heading may not get up. Use *Reset*.
- `build_web.py` patches one mjswan 0.12.12 / mjviser 0.0.16 incompatibility (the viewer-GUI
  record) and writes the motion `.npz` with sizes in the ZIP local headers. mjswan's browser
  reader needs this, and `np.savez` does not write them.
