#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <source.mpc> [log-path]" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MP_SPDZ_ROOT="${MP_SPDZ_ROOT:-}"
MP_SPDZ_PYTHON="${MP_SPDZ_PYTHON:-python3}"
SOURCE="$REPO_ROOT/mpc/programs/bp/$1"
PROGRAM="$(basename "$1" .mpc)"
LOG="${2:-$REPO_ROOT/results/mpc/bp/${PROGRAM}.log}"

if [[ -z "$MP_SPDZ_ROOT" || ! -d "$MP_SPDZ_ROOT" ]]; then
    echo "MP_SPDZ_ROOT must point to an existing MP-SPDZ installation" >&2
    exit 1
fi
if [[ ! -x "$MP_SPDZ_ROOT/Scripts/compile-run.py" ]]; then
    echo "Missing executable MP-SPDZ runner: $MP_SPDZ_ROOT/Scripts/compile-run.py" >&2
    exit 1
fi
if ! command -v "$MP_SPDZ_PYTHON" >/dev/null 2>&1; then
    echo "MP_SPDZ_PYTHON is not executable: $MP_SPDZ_PYTHON" >&2
    exit 1
fi
if [[ "$1" != *.mpc || "$PROGRAM" == "$1" || "$PROGRAM" == */* || ! -f "$SOURCE" ]]; then
    echo "Source must be a basename of an authoritative file under mpc/programs/bp/: $1" >&2
    exit 1
fi

mkdir -p "$(dirname "$LOG")"
TARGET="$MP_SPDZ_ROOT/Programs/Source/$PROGRAM.mpc"
TMP="$TARGET.$$.tmp"
cleanup() { rm -f "$TMP"; }
trap cleanup EXIT

cp -- "$SOURCE" "$TMP"
cmp -- "$SOURCE" "$TMP"
mv -f -- "$TMP" "$TARGET"

echo "Running authoritative source: $SOURCE"
echo "MP-SPDZ root: $MP_SPDZ_ROOT"
echo "Log: $LOG"
(
    cd "$MP_SPDZ_ROOT"
    env -u DEBUG "$MP_SPDZ_PYTHON" Scripts/compile-run.py -M -E ring "$PROGRAM"
) 2>&1 | tee "$LOG"
