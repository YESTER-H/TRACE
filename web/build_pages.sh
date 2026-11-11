#!/usr/bin/env bash
# Build the GitHub Pages site: landing page at /, mjswan demo at /demo/.
# usage: web/build_pages.sh [base]   (base defaults to /TRACE/, the project-page path)
# Output: web/pages_out/ (publish it as the root of the gh-pages branch).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${1:-/TRACE/}
NODE_BIN="$HERE/.venv/lib/python3.12/site-packages/mjswan/template/.nodeenv/bin"
cd "$HERE"
PATH="$NODE_BIN:$PATH" MJSWAN_NO_LAUNCH=1 MJSWAN_BASE_PATH="${BASE}demo/" .venv/bin/python build_web.py
OUT="$HERE/pages_out"
mkdir -p "$OUT/static"
cp site/index.html "$OUT/index.html"
cp ../assets/teaser_run23.gif "$OUT/static/"
rsync -a --delete dist/ "$OUT/demo/"
touch "$OUT/.nojekyll"
echo "pages site: $OUT"
