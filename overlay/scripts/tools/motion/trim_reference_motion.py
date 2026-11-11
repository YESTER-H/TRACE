# SPDX-License-Identifier: Apache-2.0
"""Write a copy of a reference .npz without its first N frames (source file is never modified).

usage: python trim_reference_motion.py <source.npz> <output.npz> --drop 100
Every array whose first axis is the frame axis is sliced; scalars (fps) are kept.
"""
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("output")
parser.add_argument("--drop", type=int, required=True)
args = parser.parse_args()
if os.path.exists(args.output):
    raise FileExistsError(f"{args.output} exists; never overwrite a reference")
data = np.load(args.source)
frames = data["joint_pos"].shape[0]
if not 0 < args.drop < frames - 1:
    raise ValueError(f"drop must be in (0, {frames - 1})")
out = {k: (v[args.drop:] if v.ndim and v.shape[0] == frames else v) for k, v in data.items()}
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
np.savez(args.output, **out)
print(f"{args.source}: {frames} frames -> {args.output}: {frames - args.drop} frames",
      {k: v.shape for k, v in out.items()})
