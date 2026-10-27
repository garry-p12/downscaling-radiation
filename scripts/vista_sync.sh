#!/usr/bin/env bash
# Push code to / pull results from TACC Vista (same pattern as the
# precipitation repo's vista_sync.sh: globs, never a file list, then verify).
#
#   scripts/vista_sync.sh push      # code -> $WORK/downscaling-radiation, verified
#   scripts/vista_sync.sh statics   # data_store/statics.npz -> $SCRATCH
#   scripts/vista_sync.sh pull      # runs/ (models, eval tables, figures) -> here
set -euo pipefail
HOST=${VISTA_HOST:-vista2}
REMOTE=${VISTA_DIR:-/work/11755/gurup12/vista/downscaling-radiation}
SCR=${VISTA_SCRATCH:-/scratch/11755/gurup12/downscaling-radiation}
cd "$(dirname "$0")/.."

code_files() {
  ls common/*.py data/*.py features/*.py models/*.py training/*.py evaluation/*.py \
     scripts/*.py scripts/*.sh slurm/*.sh configs/*.yaml tests/*.py pyproject.toml 2>/dev/null || true
}

case "${1:-}" in
  push)
    code_files | tar --no-xattrs -czf - -T - \
      | ssh "$HOST" "mkdir -p '$REMOTE' && cd '$REMOTE' && tar xzf -"
    files=$(code_files | sort | tr '\n' ' ')
    # shellcheck disable=SC2086
    shasum $files | sort > /tmp/_rad_local
    ssh "$HOST" "cd '$REMOTE' && shasum $files" | sort > /tmp/_rad_remote
    if diff -q /tmp/_rad_local /tmp/_rad_remote >/dev/null; then
      echo "pushed + verified $(wc -l < /tmp/_rad_local | tr -d ' ') files"
    else
      echo "verify: MISMATCH" >&2; diff /tmp/_rad_local /tmp/_rad_remote >&2; exit 1
    fi
    ;;
  statics)
    ssh "$HOST" "mkdir -p '$SCR'"
    scp data_store/statics.npz "$HOST:$SCR/statics.npz"
    ;;
  pull)
    ssh "$HOST" "cd '$REMOTE' && tar czf - runs logs 2>/dev/null || true" | tar xzf -
    echo "pulled runs/ and logs/"
    ;;
  *) echo "usage: $0 {push|statics|pull}" >&2; exit 1 ;;
esac
