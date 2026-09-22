#!/usr/bin/env bash
#
# CSL-Daily adapter pretraining with the architecture of the best PH14T 14B
# checkpoint, with relaxed pseudo-gloss CTC supervision.
#
# Usage:
#   sbatch scripts/csl/run_cognition_14b.sh
#   sbatch scripts/csl/run_cognition_14b.sh debug
#   sbatch scripts/csl/run_cognition_14b.sh share
#
# `debug` disables WandB and writes below outputs/csl_ckpt/debug.
# `share` reads the already-extracted shared dataset instead of staging its tar
# to local scratch.

#SBATCH --job-name=slt_csl_qwen3_14b
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=21
#SBATCH --mem=256g
#SBATCH --time=2-12:00:00

set -euo pipefail

HOST_FQDN="$(hostname -f)"
if [[ "$HOST_FQDN" == "tubbs.eng.gla.ac.uk" ]]; then
  SCRIPT_DIR=/home/2533494W/project/sign_language_translation_v3
  IS_TUBBS=true
elif [[ -d /users/2533494w/projects/sign_language_translation_v3 ]]; then
  SCRIPT_DIR=/users/2533494w/projects/sign_language_translation_v3
  IS_TUBBS=false
else
  echo "unrecognized host '$HOST_FQDN': no known sign_language_translation_v3 checkout here; refusing to run." >&2
  exit 4
fi

if [[ "$IS_TUBBS" == true ]]; then
  unset NCCL_P2P_DISABLE
else
  export NCCL_P2P_DISABLE=1
fi

cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/.venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

DEBUG=false
SHARED_DATASET=false
for arg in "$@"; do
  case "$arg" in
  debug) DEBUG=true ;;
  share) SHARED_DATASET=true ;;
  *)
    echo "Unknown argument: $arg (supported: debug, share)" >&2
    exit 2
    ;;
  esac
done

RUN_NAME=qwen3-14b-cradio-l-nextframe-handroi-cls-31m-posenc-learned-ctc-relaxed-diverse-zh
if [[ "$DEBUG" == true ]]; then
  REPORT_TO=none
  OUTPUT_DIR="outputs/csl_ckpt/debug-$RUN_NAME"
else
  export WANDB_PROJECT=sign_language_translation_v5.0-dev
  export WANDB_TAGS="csl-daily,zh,qwen3-14b,cradio-l,next-frame,hand-roi,cls,31m,proj-dropout,posenc-learned,output-layer--8,ctc-relaxed,diverse-train,fixed-eval,no-warmup,ep40"
  REPORT_TO=wandb
  OUTPUT_DIR="outputs/csl_ckpt/$RUN_NAME"
fi

if [[ -t 2 ]]; then
  unset TQDM_DISABLE
  HG_TQDM_DISABLE=False
else
  export TQDM_DISABLE=1
  HG_TQDM_DISABLE=True
fi

# prepare_dataset extracts the frame archive directly into the returned dataroot.
if [[ "$IS_TUBBS" == true || "$SHARED_DATASET" == true ]]; then
  DATA_ROOT="$SCRIPT_DIR/dataset/CSL-Daily-HG/preprocessed/full-trim-256x256px"
else
  source "$SCRIPT_DIR/scripts/csl/prepare_dataset.sh"
  DATA_ROOT=$(prepare_dataset \
    /mnt/scratch/users/2533494w/dataset/full-trim-256x256px.tar \
    "$HOME/localscratch/full-trim-256x256px")
fi

if [[ ! -f "$DATA_ROOT/.complete" ]]; then
  echo "CSL-Daily frame dataset is incomplete or missing: $DATA_ROOT" >&2
  exit 3
fi
python - "$DATA_ROOT" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

data_root = Path(sys.argv[1])
required_column = "pseudo_gloss_relaxed"
for filename in ("train.parquet", "dev.parquet", "test.parquet"):
    path = data_root / filename
    if not path.is_file():
        raise SystemExit(f"missing CSL-Daily split index: {path}")
    if required_column not in pq.read_schema(path).names:
        raise SystemExit(
            f"{path} has no {required_column!r} column; refresh the staged "
            "dataset/archive after running build_pseudo_gloss.py"
        )
PY

CTC_TOKENIZER_DIR=outputs/ctc_tokenizer_csl_daily_relaxed
CTC_VOCAB_SIZE=7353
CTC_BLANK_ID=2
if [[ ! -f "$SCRIPT_DIR/$CTC_TOKENIZER_DIR/tokenizer.json" ]]; then
  echo "CSL-Daily CTC tokenizer is missing: $SCRIPT_DIR/$CTC_TOKENIZER_DIR" >&2
  exit 3
fi
read -r ACTUAL_CTC_VOCAB_SIZE ACTUAL_CTC_BLANK_ID < <(
  python - "$SCRIPT_DIR/$CTC_TOKENIZER_DIR" <<'PY'
import sys

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
print(len(tokenizer), tokenizer.convert_tokens_to_ids("<blank>"))
PY
)
if [[ "$ACTUAL_CTC_VOCAB_SIZE" != "$CTC_VOCAB_SIZE" || \
      "$ACTUAL_CTC_BLANK_ID" != "$CTC_BLANK_ID" ]]; then
  echo "CTC tokenizer/config mismatch: expected vocab=$CTC_VOCAB_SIZE blank=$CTC_BLANK_ID, got vocab=$ACTUAL_CTC_VOCAB_SIZE blank=$ACTUAL_CTC_BLANK_ID" >&2
  exit 3
fi

# The scorer was fitted on CSL-Daily C-RADIOv4-SO400M output-layer -8 features.
OUTPUT_LAYER=-8
SCORER_PATH=outputs/hand_patch_scorer_csl_L8
if [[ ! -f "$SCRIPT_DIR/$SCORER_PATH/config.json" ]]; then
  echo "no fitted CSL scorer at $SCRIPT_DIR/$SCORER_PATH" >&2
  exit 3
fi

echo "LANGUAGE=zh  PROMPT=diverse-train/fixed-eval  CTC=relaxed  CTC_VOCAB_SIZE=$CTC_VOCAB_SIZE  CTC_BLANK_ID=$CTC_BLANK_ID  EPOCHS=40  WARMUP=0"
echo "DATA_ROOT=$DATA_ROOT  OUTPUT_DIR=$OUTPUT_DIR  SCORER_PATH=$SCORER_PATH"

CMD_ARGS=(
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  --config-name=train/pretrain_adapter/baseline_ablation

  # CSL-Daily is intrinsically Chinese-only; the dataset emits lang=zh.
  data='csl_*x224x224_qwen_single_language'
  # Random canonical/diverse prompts for training; fixed canonical prompt for eval.
  prompt=diverse_train
  data.data_root="$DATA_ROOT"
  # This adapter emits one token per two input frames.
  data.processor.video_token_scale=0.5

  # Match the best 14B checkpoint's adapter architecture and regularization.
  model=qwen3-14b-cradio-l-spatiotemporal-next-frame-handroi-cls-31m
  model.config.visual_backbone_config.output_layer="$OUTPUT_LAYER"
  model.config.visual_adapter_kwargs.scorer_path="$SCORER_PATH"
  model.config.visual_adapter_kwargs.projection_dropout=0.5
  model.config.visual_adapter_kwargs.roi_projection_dropout=0.5
  model.config.visual_adapter_kwargs.cls_projection_dropout=0.5
  model.config.visual_position_embedding_type=learned
  engine.trainability.visual_position_embedding.parameter_mode=full
  +engine.optimization.visual_adapter.parameter_groups.gates.learning_rate=1e-3

  # Auxiliary CTC supervision from CSL-Daily's relaxed pseudo glosses.
  model.config.ctc_enabled=true
  model.config.ctc_loss_weight=1.0
  model.config.ctc_vocab_size="$CTC_VOCAB_SIZE"
  model.config.ctc_blank_id="$CTC_BLANK_ID"
  engine.trainability.ctc_head.parameter_mode=full

  # Train for 40 epochs without learning-rate warmup.
  engine.training_args.num_train_epochs=40
  engine.training_args.warmup_steps=0
  engine.training_args.dataloader_num_workers=8
  engine.training_args.eval_steps=6000
  engine.training_args.logging_steps=15
  engine.training_args.ddp_find_unused_parameters=false
  engine.training_args.output_dir="$OUTPUT_DIR"
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
)

accelerate launch "${CMD_ARGS[@]}"
