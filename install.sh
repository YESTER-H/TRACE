#!/usr/bin/env bash
# Install TRACE into a cyclo_lab checkout (Newton branch).
# usage: ./install.sh /path/to/cyclo_lab
set -euo pipefail

TARGET=${1:?usage: ./install.sh /path/to/cyclo_lab}
HERE=$(cd "$(dirname "$0")" && pwd)

[ -d "$TARGET/source/cyclo_lab/cyclo_lab/manager_based" ] || {
    echo "error: $TARGET does not look like a cyclo_lab checkout" >&2; exit 1; }

# 1. Small changes to the upstream train/play scripts (diagnostics hooks, --skip-export, --max-steps).
if git -C "$TARGET" apply --check "$HERE/patches/rsl_rl_play_train.patch" 2>/dev/null; then
    git -C "$TARGET" apply "$HERE/patches/rsl_rl_play_train.patch"
    echo "patched scripts/reinforcement_learning/rsl_rl/{train,play}.py"
elif git -C "$TARGET" apply --check -R "$HERE/patches/rsl_rl_play_train.patch" 2>/dev/null; then
    echo "train/play patch already applied"
else
    echo "error: patches/rsl_rl_play_train.patch does not apply; see README (tested base commit)" >&2; exit 1
fi

# 2. New files only: the task package, scripts, tests and reference motions.
#    Existing files are never overwritten.
cd "$HERE/overlay"
find . -type f | while read -r f; do
    if [ -e "$TARGET/$f" ]; then
        cmp -s "$f" "$TARGET/$f" || { echo "skip (exists, differs): $f" >&2; }
    else
        mkdir -p "$TARGET/$(dirname "$f")"
        cp -p "$f" "$TARGET/$f"
    fi
done
echo "TRACE installed into $TARGET"
