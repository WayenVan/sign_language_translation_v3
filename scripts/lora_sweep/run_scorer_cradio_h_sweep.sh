#! /bin/bash
#
# Usage:
#   sbatch scripts/swaps/run_scorer_cradio_h_sweep.sh              # -9 -8 -10 -7 -11
#   sbatch scripts/swaps/run_scorer_cradio_h_sweep.sh -9           # one layer
#   sbatch scripts/swaps/run_scorer_cradio_h_sweep.sh -9 force     # refit, reuse the feature cache
#
# C-RADIOv4-H hand-patch scorer sweep; see below.

#SBATCH --job-name=scorer_cradio_h
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --time=02:00:00

set -euo pipefail

if [[ "$(hostname -f)" == "tubbs.eng.gla.ac.uk" ]]; then
  SCRIPT_DIR=/home/2533494W/project/sign_language_translation_v3
else
  SCRIPT_DIR=/users/2533494w/projects/sign_language_translation_v3
fi

cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"

export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# --------------------------------------------------------------------------- #
# Hand-patch scorer sweep for C-RADIOv4-H (vit_huge_patch16, 32 blocks, 1280-d).
#
# A scorer is one affine map over a single backbone's features at a single
# layer, so none of the SO400M scorers (outputs/hand_patch_scorer_L*, 1152-d)
# carry over. This fits one per candidate H layer:
#
#   -9   block 23/32 (71.9%) -- same relative depth as SO400M's -8 (19/27,
#        70.4%); what configs/model/qwen3-8b-cradio-h-*-28m.yaml points at
#   -8   block 24/32 (75.0%)
#   -10  block 22/32 (68.8%)
#   -7   block 25/32 (78.1%)
#   -11  block 21/32 (65.6%)
#
# The SO400M sweep was flat across -7..-10; these five span that plateau with
# a layer of margin on each side.
#
# Per layer, two steps, each skipped when its output is already complete:
#   1. preprocess/extract_scorer_features.py -- the only GPU step: one pass of
#      the backbone over dataset/ph14_train_scorer_dataset, every patch cached
#      as float16 (~6.5 GB per layer at 1280-d).
#   2. preprocess/train_scorer.py -- the linear fit, with the SO400M sweep's
#      settings (--epochs 40, everything else at its default) so the fit
#      reports compare across backbones.
#
# Outputs, named so they cannot collide with the SO400M L* directories:
#   dataset/ph14_scorer_features_cradio-h_L<N>
#   outputs/hand_patch_scorer_cradio-h_L<N>
#
# `force` refits scorers that already exist; the feature cache is still reused.
# To re-extract, delete the feature directory. A feature directory whose
# meta.json records a different backbone config is refused rather than
# overwritten.
#
# Time limit: the SO400M per-layer sweep on 2026-09-07 took ~12 min per layer
# end to end (~2 min extraction, ~10 min fit). H is ~1.5x the backbone compute
# and 1.11x the feature width, so ~14 min per layer, ~70 min for five, plus a
# first-use download of the 2.5 GB checkpoint (HF_HOME is node-local /tmp).
# 02:00:00 is ~1.6x that. It is kept tight on purpose: the partition default
# is 3 days, which backfills badly, and a job that does run out loses nothing
# -- every finished layer is skipped on resubmission.
# --------------------------------------------------------------------------- #

BACKBONE_ID="nvidia/C-RADIOv4-H"
# Settings of the SO400M per-layer fits (outputs/hand_patch_scorer_L*/fit_report.json).
SCORER_EPOCHS=40

LAYERS=()
FORCE=false
for arg in "$@"; do
  case "$arg" in
  force) FORCE=true ;;
  *)
    if [[ "$arg" =~ ^-[0-9]+$ ]]; then
      LAYERS+=("$arg")
    else
      echo "Unknown argument: $arg (supported: negative layer ints, force)" >&2
      exit 2
    fi
    ;;
  esac
done
if ((${#LAYERS[@]} == 0)); then
  LAYERS=(-9 -8 -10 -7 -11)
fi

# 设置 TQDM_DISABLE：非交互终端下关掉进度条，避免刷爆日志。
if [[ -t 2 ]]; then
  unset TQDM_DISABLE
else
  export TQDM_DISABLE=1
fi

# Exit 0: complete cache for exactly this backbone config. Exit 1: no finished
# cache (meta.json is written last, so a partial run has none). Exit 3: a
# finished cache for a different backbone config.
check_feature_cache() {
  python - "$1" "$2" <<'EOF'
import json, sys
from pathlib import Path

directory, expected = Path(sys.argv[1]), json.loads(sys.argv[2])
meta_path = directory / "meta.json"
if not meta_path.exists():
    sys.exit(1)
meta = json.loads(meta_path.read_text())
if meta["backbone"]["config"] != expected:
    print(f"{meta_path} records {meta['backbone']['config']}, expected {expected}",
          file=sys.stderr)
    sys.exit(3)
complete = (
    not meta.get("truncated", False)
    and {"train", "test"} <= set(meta["splits"])
    and all((directory / s["file"]).exists() for s in meta["splits"].values())
)
sys.exit(0 if complete else 1)
EOF
}

echo "BACKBONE_ID=$BACKBONE_ID  LAYERS=${LAYERS[*]}  FORCE=$FORCE"

for LAYER in "${LAYERS[@]}"; do
  LAYER_ABS="${LAYER#-}"
  FEATURES_DIR="dataset/ph14_scorer_features_cradio-h_L${LAYER_ABS}"
  SCORER_DIR="outputs/hand_patch_scorer_cradio-h_L${LAYER_ABS}"
  BACKBONE_CONFIG="{\"id\": \"${BACKBONE_ID}\", \"output_layer\": ${LAYER}}"

  echo "=== layer ${LAYER}: ${FEATURES_DIR} -> ${SCORER_DIR}"

  cache_status=0
  check_feature_cache "$FEATURES_DIR" "$BACKBONE_CONFIG" || cache_status=$?
  case "$cache_status" in
  0) echo "features: complete cache found, skipping extraction" ;;
  1)
    python preprocess/extract_scorer_features.py \
      --backbone c_radio_v4 \
      --backbone-config "$BACKBONE_CONFIG" \
      --out "$FEATURES_DIR" \
      --num-workers "${SLURM_CPUS_PER_TASK:-8}"
    ;;
  *)
    echo "refusing to overwrite $FEATURES_DIR; move it aside first." >&2
    exit 3
    ;;
  esac

  if [[ "$FORCE" != true && -f "$SCORER_DIR/config.json" && -f "$SCORER_DIR/fit_report.json" ]]; then
    echo "scorer: already fitted, skipping (pass 'force' to refit)"
  else
    python preprocess/train_scorer.py \
      --features "$FEATURES_DIR" \
      --out "$SCORER_DIR" \
      --epochs "$SCORER_EPOCHS"
  fi
done

# Held-out summary next to the SO400M reference the current recipe uses (-8),
# at k=24, the adapter's top_k.
python - "${LAYERS[@]}" <<'EOF'
import json, sys
from pathlib import Path

rows = [("SO400M -8 (ref)", Path("outputs/hand_patch_scorer_L8"))]
rows += [(f"H {layer}", Path(f"outputs/hand_patch_scorer_cradio-h_L{layer.lstrip('-')}"))
         for layer in sys.argv[1:]]
print(f"\n{'scorer':>16}{'AUC':>8}{'recall@24':>11}{'frames<0.5':>12}{'hand%':>8}")
for name, directory in rows:
    report_path = directory / "fit_report.json"
    if not report_path.exists():
        print(f"{name:>16}  (no fit_report.json)")
        continue
    report = json.loads(report_path.read_text())["eval"]
    k24 = report["k=24"]
    print(f"{name:>16}{report['auc']:>8.4f}{k24['recall_macro']:>11.4f}"
          f"{k24['frames_below_half']:>12.2%}{k24['hand_share']:>8.1%}")
EOF
