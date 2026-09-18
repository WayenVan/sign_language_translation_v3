#!/usr/bin/env bash

# Run the Qwen3-14B + C-RADIOv4-H layers sequentially in this terminal.
# Usage: bash h_14b_outputlayer_ablation/sweep.sh [positive_epochs] [debug] [share] [--dry-run]
# Default: -10 through -15 at 80 epochs each. A missing scorer is fitted
# before its layer trains. This script does not use a job scheduler.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SINGLE_SCRIPT="$SCRIPT_DIR/run_outputlayer.sh"
SCORER_SCRIPT="$SCRIPT_DIR/fit_scorer.sh"
LAYERS=(-10 -11 -12 -13 -14 -15)

EPOCHS_SEEN=false
DEBUG_SEEN=false
SHARE_SEEN=false
DRY_RUN=false
SINGLE_ARGS=()
for arg in "$@"; do
  case "$arg" in
    debug)
      if [[ "$DEBUG_SEEN" == true ]]; then
        echo "duplicate argument: debug" >&2
        exit 2
      fi
      DEBUG_SEEN=true
      SINGLE_ARGS+=(debug)
      ;;
    share)
      if [[ "$SHARE_SEEN" == true ]]; then
        echo "duplicate argument: share" >&2
        exit 2
      fi
      SHARE_SEEN=true
      SINGLE_ARGS+=(share)
      ;;
    --dry-run) DRY_RUN=true ;;
    *)
      if [[ "$arg" =~ ^[1-9][0-9]*$ && "$EPOCHS_SEEN" == false ]]; then
        EPOCHS_SEEN=true
        SINGLE_ARGS+=("$arg")
      else
        echo "unknown or repeated argument: $arg" >&2
        exit 2
      fi
      ;;
  esac
done

# Validate scripts before starting the sweep.
if [[ ! -f "$SINGLE_SCRIPT" || ! -f "$SCORER_SCRIPT" ]]; then
  echo "single-layer or scorer-fitting script is missing in $SCRIPT_DIR" >&2
  exit 3
fi

cd "$PROJECT_DIR"
for layer in "${LAYERS[@]}"; do
  scorer_dir="outputs/hand_patch_scorer_cradio-h_L${layer#-}"
  if [[ ! -f "$scorer_dir/config.json" || ! -f "$scorer_dir/fit_report.json" || ! -f "$scorer_dir/model.safetensors" ]]; then
    if [[ "$DRY_RUN" == true ]]; then
      printf 'bash %q %q\n' "$SCORER_SCRIPT" "$layer"
    else
      echo "=== fitting scorer for output layer $layer ==="
      bash "$SCORER_SCRIPT" "$layer"
    fi
  fi
  if [[ "$DRY_RUN" == true ]]; then
    printf 'bash %q %q' "$SINGLE_SCRIPT" "$layer"
    if ((${#SINGLE_ARGS[@]} > 0)); then
      printf ' %q' "${SINGLE_ARGS[@]}"
    fi
    printf '\n'
  else
    echo "=== training output layer $layer ==="
    bash "$SINGLE_SCRIPT" "$layer" "${SINGLE_ARGS[@]}"
  fi
done
