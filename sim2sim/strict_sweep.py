"""Strict sim-to-sim sweep of a TRACE policy (TRACE_POLICY=run24 default, run23) in CPU MuJoCo under physics / actuator / sensing changes.

The plain check (web/sim2sim.py) copies the Newton MuJoCo-Warp training settings exactly. Here each
condition changes one thing away from training (one-factor-at-a-time), plus randomized combinations.

What training randomized (so anything else is out of distribution):
  obs noise (anchor ori +-0.05, ang vel +-0.2, joint pos +-0.01, joint vel +-0.5, gravity +-0.05),
  home joint offset +-0.01 rad, torso COM offset (+-2.5/5/5 cm), pelvis velocity pushes <= 0.5 m/s.
  NOT randomized: friction (fixed 1.0), masses, PD gains, armature, joint friction, latency, solver.

Scenarios per condition (all with training-level observation noise, different seed per trial):
  stand   dance from frame 0 for --seconds; success = never falls (pelvis < 0.45 m)
  supine  frame 0 of the Supine get-up ref; success = upright (pelvis >= 0.6, tilt <= 30 deg) for the last 2 s
  prone   same from the Prone ref
  push    dance, one 2.4 m/s pelvis push at 3 s in 8 directions, heading alignment on;
          reports falls and how many of those got back up

usage (TRACE repo root): web/.venv/bin/python sim2sim/strict_sweep.py [--seeds 8] [--workers 14] [--only a,b]
"""
import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# name -> dict of modifications (see Sim.apply). "group" is only for the report.
CONDITIONS = {
    "nominal":            dict(group="baseline"),
    "nominal_no_noise":   dict(group="baseline", obs_noise=0.0),
    # --- solver / integration (training: Newton solver, implicitfast, pyramidal, 100/50 it, dt 5 ms)
    "euler":              dict(group="solver", integrator="euler"),
    "elliptic_impratio10": dict(group="solver", cone="elliptic", impratio=10.0),
    "cg_10iter":          dict(group="solver", solver="cg", iterations=10, ls_iterations=10),
    "dt_2ms":             dict(group="solver", sim_dt=0.002),
    "dt_1ms":             dict(group="solver", sim_dt=0.001),
    "soft_contact":       dict(group="solver", solref_timeconst=0.04),
    # --- contact / body
    "friction_0.4":       dict(group="dynamics", friction=0.4),
    "friction_0.7":       dict(group="dynamics", friction=0.7),
    "friction_1.5":       dict(group="dynamics", friction=1.5),
    "mass_x0.85":         dict(group="dynamics", mass_scale=0.85),
    "mass_x1.15":         dict(group="dynamics", mass_scale=1.15),
    "mass_rand_20pct":    dict(group="dynamics", mass_rand=0.20),
    "torso_com_x+5cm":    dict(group="dynamics", torso_com=(0.05, 0, 0)),
    # --- actuators (training: implicit PD, fixed kp/kd/effort/armature, no joint friction)
    "kp_kd_x0.8":         dict(group="actuator", gain_scale=0.8),
    "kp_kd_x1.2":         dict(group="actuator", gain_scale=1.2),
    "effort_x0.7":        dict(group="actuator", effort_scale=0.7),
    "armature_x0":        dict(group="actuator", armature_scale=0.0),
    "armature_x2":        dict(group="actuator", armature_scale=2.0),
    "joint_friction_0.3": dict(group="actuator", frictionloss=0.3),
    "joint_damping_0.5":  dict(group="actuator", damping=0.5),
    "explicit_pd":        dict(group="actuator", explicit_pd=True),
    # --- sensing / timing
    "delay_5ms":          dict(group="timing", delay_s=0.005),
    "delay_10ms":         dict(group="timing", delay_s=0.010),
    "delay_20ms":         dict(group="timing", delay_s=0.020),
    "delay_40ms":         dict(group="timing", delay_s=0.040),
    "obs_noise_x2":       dict(group="timing", obs_noise=2.0),
    "gyro_bias_0.1":      dict(group="timing", gyro_bias=0.1),
    # --- limit probes: deliberately extreme, to find where it breaks (and to prove the knobs act)
    "limit_delay_60ms":   dict(group="limit", delay_s=0.060),
    "limit_delay_100ms":  dict(group="limit", delay_s=0.100),
    "limit_friction_0.2": dict(group="limit", friction=0.2),
    "limit_kp_kd_x0.5":   dict(group="limit", gain_scale=0.5),
    "limit_effort_x0.4":  dict(group="limit", effort_scale=0.4),
    "limit_mass_x1.4":    dict(group="limit", mass_scale=1.4),
    # --- everything at once, re-sampled per trial (realistic hardware-like spread)
    "random_mild":        dict(group="combined", random="mild"),
    "random_hard":        dict(group="combined", random="hard"),
}

RANDOM_RANGES = {
    #           friction     mass_rand gain   effort  armature delay_s     frictionloss  com(x)
    "mild": dict(friction=(0.7, 1.2), mass_rand=0.10, gain=(0.9, 1.1), effort=(0.9, 1.0), arm=(0.8, 1.2),
                 delay=(0.0, 0.010), floss=(0.0, 0.1), com=0.02),
    "hard": dict(friction=(0.4, 1.5), mass_rand=0.20, gain=(0.8, 1.2), effort=(0.75, 1.0), arm=(0.5, 1.5),
                 delay=(0.0, 0.020), floss=(0.0, 0.3), com=0.04),
}
TRAIN_NOISE = dict(anchor_ori=0.05, ang_vel=0.2, joint_pos=0.01, joint_vel=0.5, gravity=0.05)


def sample_random(kind, rng):
    r = RANDOM_RANGES[kind]
    return dict(friction=rng.uniform(*r["friction"]), mass_rand=r["mass_rand"], gain_scale=rng.uniform(*r["gain"]),
                effort_scale=rng.uniform(*r["effort"]), armature_scale=rng.uniform(*r["arm"]),
                delay_s=rng.uniform(*r["delay"]), frictionloss=rng.uniform(*r["floss"]),
                torso_com=(rng.uniform(-r["com"], r["com"]), rng.uniform(-r["com"], r["com"]), 0.0))


_WORKER = None
MODEL_FIELDS = ("geom_solref", "geom_friction", "body_mass", "body_inertia", "body_ipos", "actuator_gainprm",
                "actuator_biasprm", "actuator_forcerange", "actuator_ctrllimited", "dof_armature",
                "dof_frictionloss", "dof_damping")
MODEL_OPTS = ("integrator", "cone", "impratio", "solver", "iterations", "ls_iterations", "timestep")


def snapshot(robot):
    """Training-matched model values, so a condition can be applied (and undone) on the same model."""
    m = robot.model
    return dict(fields={f: getattr(m, f).copy() for f in MODEL_FIELDS}, opts={o: getattr(m.opt, o) for o in MODEL_OPTS})


def configure(robot, base, cond_name, rng):
    """Apply a condition to `robot` (restoring `base` first).

    Returns dict(cond, physics, make_observer, after_reset):
      robot.step = physics; after every Controller.reset(): ctrl.observer = make_observer(ctrl.aligner); after_reset().
    """
    import mujoco
    import trace_mujoco as tm

    cond = dict(CONDITIONS[cond_name])
    if "random" in cond:
        cond.update(sample_random(cond["random"], rng))
    m = robot.model
    for f, v in base["fields"].items():
        getattr(m, f)[:] = v
    for o, v in base["opts"].items():
        setattr(m.opt, o, v)
    # ---- solver options
    if cond.get("integrator") == "euler":
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    if cond.get("cone") == "elliptic":
        m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    if "impratio" in cond:
        m.opt.impratio = cond["impratio"]
    if cond.get("solver") == "cg":
        m.opt.solver = mujoco.mjtSolver.mjSOL_CG
    if "iterations" in cond:
        m.opt.iterations, m.opt.ls_iterations = cond["iterations"], cond["ls_iterations"]
    sim_dt = cond.get("sim_dt", tm.SIM_DT)
    m.opt.timestep = sim_dt
    substeps = int(round(tm.CONTROL_DT / sim_dt))
    if "solref_timeconst" in cond:
        m.geom_solref[:, 0] = cond["solref_timeconst"]
    # ---- body / contact
    if "friction" in cond:
        m.geom_friction[:, 0] = cond["friction"]       # MuJoCo takes the max of a pair: set every geom
    if "mass_scale" in cond:
        m.body_mass[1:] *= cond["mass_scale"]; m.body_inertia[1:] *= cond["mass_scale"]
    if "mass_rand" in cond:
        s = rng.uniform(1 - cond["mass_rand"], 1 + cond["mass_rand"], m.nbody - 1)
        m.body_mass[1:] *= s; m.body_inertia[1:] *= s[:, None]
    if "torso_com" in cond:
        m.body_ipos[robot.anchor] += np.asarray(cond["torso_com"])
    # ---- actuators
    kp, kd, effort = tm.KP.copy(), tm.KD.copy(), tm.EFFORT.copy()
    kp *= cond.get("gain_scale", 1.0); kd *= cond.get("gain_scale", 1.0); effort *= cond.get("effort_scale", 1.0)
    m.actuator_gainprm[robot.act, 0] = kp
    m.actuator_biasprm[robot.act, 1] = -kp
    m.actuator_biasprm[robot.act, 2] = -kd
    m.actuator_forcerange[robot.act] = np.stack([-effort, effort], 1)
    m.dof_armature[robot.vadr] *= cond.get("armature_scale", 1.0)
    m.dof_frictionloss[robot.vadr] = cond.get("frictionloss", 0.0)
    m.dof_damping[robot.vadr] = cond.get("damping", 0.0)
    explicit = cond.get("explicit_pd", False)
    if explicit:   # torque actuators; PD evaluated by us once per physics step (no implicit damping)
        m.actuator_gainprm[robot.act, 0] = 1.0
        m.actuator_biasprm[robot.act, :3] = 0.0
        m.actuator_ctrllimited[robot.act] = 0
    delay_steps = int(round(cond.get("delay_s", 0.0) / sim_dt))

    # ---- sensing: training-level noise (scaled) + optional gyro bias, applied per term before the history
    noise_scale = cond.get("obs_noise", 1.0)
    gyro_bias = rng.normal(0, cond.get("gyro_bias", 0.0), 3) if cond.get("gyro_bias") else np.zeros(3)

    class NoisyObserver(tm.Observer):
        def terms(self, *a, **k):
            t = super().terms(*a, **k)
            for key, amp in TRAIN_NOISE.items():
                if noise_scale > 0:
                    t[key] = t[key] + rng.uniform(-amp, amp, t[key].shape) * noise_scale
            t["ang_vel"] = t["ang_vel"] + gyro_bias
            return t

    pending = []   # targets waiting in the delay line, one entry per physics step

    def physics(target):
        for _ in range(substeps):
            pending.append(target)
            applied = pending.pop(0) if len(pending) > delay_steps else pending[0]
            if explicit:
                tau = kp * (applied - robot.joint_pos()) - kd * robot.joint_vel()
                robot.data.ctrl[robot.act] = np.clip(tau, -effort, effort)
            else:
                robot.data.ctrl[robot.act] = applied
            mujoco.mj_step(m, robot.data)

    def after_reset():
        pending[:] = [robot.joint_pos().copy() for _ in range(delay_steps)]
        if explicit:
            robot.data.ctrl[robot.act] = 0.0

    return dict(cond=cond, physics=physics, make_observer=NoisyObserver, after_reset=after_reset)


def run_trial(job):
    import torch
    torch.set_num_threads(1)
    import sys
    sys.path.insert(0, str(HERE.parent / "web"))
    import trace_mujoco as tm

    cond_name, scenario, seed, seconds = job
    rng = np.random.default_rng(seed)
    # One model / policy / motion set per worker process; each trial restores the model fields it may change.
    # (Building a new K1 per trial leaked ~330 MB per trial through reference cycles -> OOM with many workers.)
    global _WORKER
    if _WORKER is None:
        robot = tm.K1()
        _WORKER = dict(robot=robot, policy=tm.TracePolicy(), motions=tm.load_motions(), base=snapshot(robot))
    robot = _WORKER["robot"]
    setup = configure(robot, _WORKER["base"], cond_name, rng)
    robot.step = setup["physics"]
    align = scenario == "push"
    ctrl = tm.Controller(robot, _WORKER["policy"], _WORKER["motions"], align_heading=align)
    start = scenario if scenario in ("supine", "prone") else "stand"
    ctrl.reset(start)
    ctrl.observer = setup["make_observer"](ctrl.aligner)
    setup["after_reset"]()

    push_angle = 2 * math.pi * (seed % 8) / 8
    steps, push_step = int(seconds / tm.CONTROL_DT), int(3.0 / tm.CONTROL_DT)
    h, up, wt, jerr = [], [], [], []
    nan = False
    for k in range(steps):
        if scenario == "push" and k == push_step:
            robot.data.qvel[0] += 2.4 * math.cos(push_angle)
            robot.data.qvel[1] += 2.4 * math.sin(push_angle)
        w = ctrl.step()
        if not np.isfinite(robot.data.qpos).all():
            nan = True
            break
        h.append(robot.pelvis_height())
        up.append(robot.data.xmat[robot.pelvis].reshape(3, 3)[2, 2])
        wt.append(w)
        jerr.append(float(np.abs(robot.joint_pos() - ctrl.dance.joint_pos[ctrl.phase]).mean()))
    del robot.step   # drop the per-trial closure (restores K1.step)
    h, up, wt, jerr = map(np.array, (h, up, wt, jerr))
    last = int(2 / tm.CONTROL_DT)
    upright_end = (not nan) and len(h) == steps and bool((h[-last:] >= .6).all() and
                                                         (up[-last:] >= math.cos(math.radians(30))).all())
    fell = nan or bool((h < .45).any())
    if scenario == "stand":
        fell_at = None if not fell or nan else round(float(np.argmax(h < .45) * tm.CONTROL_DT), 2)
        success = not fell
    elif scenario == "push":
        fell_at = None
        success = (not fell) or upright_end
    else:
        fell_at = None
        success = upright_end
    # get-up time: first moment the pelvis is >= 0.6 m and stays there for 1 s
    t_up = None
    if scenario != "stand" and len(h):
        ok = h >= .6
        win = int(1 / tm.CONTROL_DT)
        for k in range(len(ok) - win):
            if ok[k:k + win].all():
                t_up = round(k * tm.CONTROL_DT, 2)
                break
    tail = slice(-last, None)
    return dict(condition=cond_name, scenario=scenario, seed=seed, success=bool(success), fell=fell, nan=nan,
                upright_end=upright_end, fell_at=fell_at, t_up=t_up,
                joint_err_end=round(float(jerr[tail].mean()), 4) if len(jerr) else None,
                wT_end=round(float(wt[tail].mean()), 3) if len(wt) else None)


def summarize(rows):
    out = {}
    for name in dict.fromkeys(r["condition"] for r in rows):
        rc = [r for r in rows if r["condition"] == name]
        s = dict(group=CONDITIONS[name]["group"])
        for sc in ("stand", "supine", "prone", "push"):
            r = [x for x in rc if x["scenario"] == sc]
            if not r:
                continue
            d = dict(n=len(r), success=sum(x["success"] for x in r))
            if sc in ("supine", "prone"):
                t = [x["t_up"] for x in r if x["success"] and x["t_up"] is not None]
                d["t_up_median"] = round(float(np.median(t)), 2) if t else None
            if sc == "push":
                falls = [x for x in r if x["fell"]]
                d["falls"], d["recovered"] = len(falls), sum(x["upright_end"] for x in falls)
            ok = [x["joint_err_end"] for x in r if x["success"] and x["joint_err_end"] is not None]
            d["joint_err_end"] = round(float(np.mean(ok)), 3) if ok else None
            d["nan"] = sum(x["nan"] for x in r)
            s[sc] = d
        out[name] = s
    return out


def markdown(summary, seeds, seconds):
    lines = [f"Strict sim2sim sweep, {seeds} trials per scenario, {seconds:.0f} s each. "
             "stand = no fall; supine/prone = upright at the end (t = median get-up time); "
             "push = 2.4 m/s in 8 directions (falls -> recovered). err = mean |joint - dance ref| (rad) "
             "over the last 2 s of successful trials.", "",
             "| group | condition | stand | supine (t) | prone (t) | push: falls -> recovered | err |",
             "|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        def gu(sc):
            d = s.get(sc)
            if not d:
                return "-"
            t = f" ({d['t_up_median']:.1f}s)" if d.get("t_up_median") is not None else ""
            return f"{d['success']}/{d['n']}{t}"
        p = s.get("push")
        push = f"{p['falls']} -> {p['recovered']}" if p else "-"
        err = s.get("stand", {}).get("joint_err_end")
        lines.append(f"| {s['group']} | {name} | {gu('stand')} | {gu('supine')} | {gu('prone')} | {push} | "
                     f"{'-' if err is None else f'{err:.3f}'} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--only", default=None, help="comma-separated condition names")
    parser.add_argument("--out", default=str(HERE / "results" / time.strftime("strict_%Y%m%d_%H%M")))
    args = parser.parse_args()
    names = args.only.split(",") if args.only else list(CONDITIONS)
    jobs = [(n, sc, seed, args.seconds) for n in names for sc in ("stand", "supine", "prone", "push")
            for seed in range(args.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_trial, jobs, chunksize=4))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    (out / "trials.json").write_text(json.dumps(rows, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    md = markdown(summary, args.seeds, args.seconds)
    (out / "summary.md").write_text(md + "\n")
    print(md)
    print(f"\n{len(jobs)} trials in {time.time() - t0:.0f} s -> {out}")
