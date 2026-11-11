# SPDX-License-Identifier: Apache-2.0
"""Write a small, comparable TensorBoard view of TRACE Recovery runs.

Original run logs are only read. For each LABEL=RUN_DIR the curated scalars go to
OUT/LABEL (regenerated on every call, so it can be rerun while a run is training).
Event counts are reconstructed exactly as count = scalar * num_envs * steps_per_env
and failure/success rates use a rolling window of iterations (sum of counts / sum
of counts), never an average of per-iteration ratios.

usage (container): /isaac-sim/python.sh curate_trace_tensorboard.py \
    --out logs/tensorboard_curated --window 100 baseline=logs/rsl_rl/... paper=logs/rsl_rl/...
"""
import argparse
import glob
import json
import shutil
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

MOTIONS = ("supine", "prone")
PRONE_FAILURE_BIN = 14  # Prone frames 329-351 (bin = frame*20//469).
ERRORS = ("root_height", "anchor_orientation", "body_position", "joint_position")
PASSTHROUGH = {
    "Train/mean_reward": "2_reward/mean_episode_return",
    "Train/mean_episode_length": "2_reward/mean_episode_length",
    "Episode_Reward/recovery_success": "2_reward/success_bonus_per_episode_second",
    "Policy/mean_std": "5_ppo/action_std",
    "Loss/value": "5_ppo/value_loss",
    "Loss/surrogate": "5_ppo/surrogate_loss",
    "Loss/entropy": "5_ppo/entropy",
    "Loss/learning_rate": "5_ppo/learning_rate",
}


def varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1; r |= (c & 0x7f) << s; s += 7
        if c < 0x80:
            return r, i


def fields(b):
    i = 0
    while i < len(b):
        k, i = varint(b, i); f, t = k >> 3, k & 7
        if t == 0: v, i = varint(b, i)
        elif t == 1: v = b[i:i+8]; i += 8
        elif t == 5: v = b[i:i+4]; i += 4
        elif t == 2: n, i = varint(b, i); v = b[i:i+n]; i += n
        else: raise ValueError(t)
        yield f, t, v


def read_scalars(path, wanted):
    """Simple-value scalars by tag; a truncated last record (live run) is ignored."""
    out = defaultdict(dict)
    with open(path, "rb") as fh:
        while True:
            header = fh.read(12)
            if len(header) < 12:
                return out
            n = struct.unpack("<Q", header[:8])[0]
            data = fh.read(n)
            if len(fh.read(4)) < 4 or len(data) < n:
                return out
            step, summary = 0, None
            for f, _, v in fields(data):
                if f == 2: step = v
                elif f == 5: summary = v
            for f, _, v in fields(summary or b""):
                if f != 1:
                    continue
                tag = value = None
                for g, _, w in fields(v):
                    if g == 1: tag = w.decode()
                    elif g == 2: value = struct.unpack("<f", w)[0]
                if tag is not None and value is not None and wanted(tag):
                    out[tag][step] = value


def wanted(tag):
    return tag in PASSTHROUGH or (tag.startswith("Trace/") and (
        "_bins_event_fraction" in tag or tag.endswith(("_episode_resets_event_fraction", "_mode_step_fraction",
                                                         "_reward_contribution", "_error_contribution"))
        or tag in ("Trace/success_completions_event_fraction", "Trace/grace_failures_event_fraction",
                   f"Trace/prone_phase_bin_{PRONE_FAILURE_BIN:02d}_step_fraction",
                   f"Trace/prone_phase_bin_{PRONE_FAILURE_BIN:02d}_adaptive_probability")))


def run_metadata(run):
    for line in open(Path(run) / "termination_diagnostic.jsonl"):
        event = json.loads(line)
        if event["event"] == "training_start":
            return event["num_envs"], event["num_steps_per_env"]
    raise ValueError(f"{run}: no training_start event")


def curate(label, run, out, window):
    events = sorted(glob.glob(str(Path(run) / "events.out.tfevents.*")))
    if not events:
        raise SystemExit(f"{run}: no tfevents file")
    num_envs, steps = run_metadata(run)
    raw = read_scalars(events[-1], wanted)
    last = max(max(v) for v in raw.values())
    iters = np.arange(last + 1)

    def series(tag):
        values = raw.get(tag, {})
        return np.array([values.get(i, np.nan) for i in iters])

    def counts(tag):
        return np.nan_to_num(series(tag)) * num_envs * steps

    def rolling(x):
        c = np.cumsum(x)
        return c - np.concatenate((np.zeros(window), c[:-window])) if len(c) > window else c

    target = Path(out) / label
    if target.exists():
        shutil.rmtree(target)  # Only this script's derived output lives here.
    writer = SummaryWriter(str(target))
    resets = {m: rolling(counts(f"Trace/{m}_episode_resets_event_fraction")) for m in MOTIONS}
    for m in MOTIONS:
        failures = rolling(sum(counts(f"Trace/{m}_phase_bin_{b:02d}_failure_detection_bins_event_fraction")
                               for b in range(20)))
        rate = failures / np.maximum(resets[m], 1)
        writer_rows = [(f"1_outcome/{m}_persistent_failure_rate", rate)]
        if m == "prone":
            onset = rolling(counts(f"Trace/prone_phase_bin_{PRONE_FAILURE_BIN:02d}_failure_onset_bins_event_fraction"))
            writer_rows.append(("1_outcome/prone_failure_onset_share_frames_329_351", onset / np.maximum(failures, 1)))
            mode = series("Trace/prone_mode_step_fraction")
            writer_rows.append(("4_exposure/prone_step_share_frames_329_351",
                                series(f"Trace/prone_phase_bin_{PRONE_FAILURE_BIN:02d}_step_fraction") / mode))
            writer_rows.append(("4_exposure/prone_reset_probability_frames_329_351",
                                series(f"Trace/prone_phase_bin_{PRONE_FAILURE_BIN:02d}_adaptive_probability")))
        mode = series(f"Trace/{m}_mode_step_fraction")
        writer_rows.append((f"2_reward/{m}_reward_per_step", series(f"Trace/{m}_reward_contribution") / mode))
        for e in ERRORS:
            writer_rows.append((f"3_error/{m}_{e}", series(f"Trace/{m}_{e}_error_contribution") / mode))
        for tag, values in writer_rows:
            for i, v in zip(iters, values):
                if np.isfinite(v):
                    writer.add_scalar(tag, float(v), int(i))
    total = np.maximum(resets["supine"] + resets["prone"], 1)
    for name, tag in (("success_rate_all", "Trace/success_completions_event_fraction"),
                      ("final_frame_grace_failure_rate_all", "Trace/grace_failures_event_fraction")):
        for i, v in zip(iters, rolling(counts(tag)) / total):
            writer.add_scalar(f"1_outcome/{name}", float(v), int(i))
    for tag, name in PASSTHROUGH.items():
        for i, v in raw.get(tag, {}).items():
            writer.add_scalar(name, float(v), int(i))
    writer.close()
    print(f"{label}: iterations 0..{last}, num_envs {num_envs} -> {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=int, default=100, help="Rolling window in iterations for rates.")
    parser.add_argument("runs", nargs="+", help="LABEL=RUN_DIR")
    args = parser.parse_args()
    for spec in args.runs:
        label, run = spec.split("=", 1)
        curate(label, run, args.out, args.window)
