#!/usr/bin/env bash

# Fit the C-RADIOv4-H hand-patch scorer for one output layer in this terminal.
# sweep.sh calls this only when that layer has no complete scorer.

set -euo pipefail

LAYER="${1:-}"
if [[ ! "$LAYER" =~ ^-[1-9][0-9]*$ || $# -ne 1 ]]; then
  echo "usage: $0 <negative_output_layer>" >&2
  exit 2
fi

if [[ "$(hostname -f)" == "tubbs.eng.gla.ac.uk" ]]; then
  PROJECT_DIR=/home/2533494W/project/sign_language_translation_v3
else
  PROJECT_DIR=/users/2533494w/projects/sign_language_translation_v3
fi
cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE=1

LAYER_ABS="${LAYER#-}"
FEATURES_DIR="dataset/ph14_scorer_features_cradio-h_L${LAYER_ABS}"
SCORER_DIR="outputs/hand_patch_scorer_cradio-h_L${LAYER_ABS}"
BACKBONE_CONFIG="{\"id\": \"nvidia/C-RADIOv4-H\", \"output_layer\": ${LAYER}}"

# A partial cache may belong to another still-running extraction. Never
# overwrite it automatically. A complete cache must match this exact layer.
CACHE_STATE=$(python - "$FEATURES_DIR" "$BACKBONE_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
expected = json.loads(sys.argv[2])
metadata = directory / "meta.json"
if not metadata.exists():
    print("partial" if directory.exists() and any(directory.iterdir()) else "missing")
else:
    meta = json.loads(metadata.read_text())
    if meta["backbone"]["config"] != expected:
        print("mismatch")
    elif meta.get("truncated", False) or not {"train", "test"} <= set(meta["splits"]):
        print("partial")
    elif not all((directory / split["file"]).exists() for split in meta["splits"].values()):
        print("partial")
    else:
        print("complete")
PY
)

case "$CACHE_STATE" in
  complete) echo "reusing complete feature cache: $FEATURES_DIR" ;;
  missing)
    python preprocess/ph14t/extract_scorer_features.py \
      --backbone c_radio_v4 \
      --backbone-config "$BACKBONE_CONFIG" \
      --out "$FEATURES_DIR" \
      --num-workers 8
    ;;
  *)
    echo "feature cache $FEATURES_DIR is $CACHE_STATE; refusing to overwrite it" >&2
    exit 3
    ;;
esac

if [[ -f "$SCORER_DIR/config.json" && -f "$SCORER_DIR/fit_report.json" && -f "$SCORER_DIR/model.safetensors" ]]; then
  echo "scorer already fitted: $SCORER_DIR"
else
  python preprocess/ph14t/train_scorer.py \
    --features "$FEATURES_DIR" \
    --out "$SCORER_DIR" \
    --epochs 40
fi

if [[ ! -f "$SCORER_DIR/config.json" || ! -f "$SCORER_DIR/fit_report.json" || ! -f "$SCORER_DIR/model.safetensors" ]]; then
  echo "scorer fit did not produce complete outputs in $SCORER_DIR" >&2
  exit 3
fi
