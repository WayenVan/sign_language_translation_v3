#! /bin/bash

#SBATCH --job-name=slt_label_smoothing_de_ablation
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=21
#SBATCH --mem=256g

set -euo pipefail

# Host handling, mirrored from scripts/run_visual_lora_ablation.sh: force
# NCCL_P2P_DISABLE on for every host (the cluster fabric misbehaves without it),
# then pick the checkout path by hostname, defaulting to the Glasgow HPC
# cluster. No allowlist / hard stop; IS_TUBBS is retained below so tubbs can
# use the already extracted shared dataset instead of staging it to scratch.
export NCCL_P2P_DISABLE=1 # NOTE: 测试的时候集群通信容易出问题 集群出现了问题

if [[ "$(hostname -f)" == "tubbs.eng.gla.ac.uk" ]]; then
  SCRIPT_DIR=/home/2533494W/project/sign_language_translation_v3
  IS_TUBBS=true
else
  SCRIPT_DIR=/users/2533494w/projects/sign_language_translation_v3
  IS_TUBBS=false
fi

cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"

export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# --------------------------------------------------------------------------- #
# Label-smoothing ablation, German only.
#
# The model structure, regularization and optimizer are byte-identical to
#
#   scripts/run_cognition_target_language.sh de
#
# which itself reproduces the architecture stored in
#
#   outputs/v5.0-qwen3-4b-cradio-l-nextframe-handroi-cls-20m-gate1-hardmatch-
#   wr3-projdrop0.5-posenc-learned-ol-8-ep80-0907.224x224
#
# Two deliberate departures from that script:
#   1. num_train_epochs is NOT overridden -- it stays at the base-config default
#      (configs/train/pretrain_adapter/base.yaml -> 50), instead of the 80 the
#      reference script hard-codes.
#   2. The sole experimental variable is language-model-side label smoothing,
#      set through model.config.label_smoothing. It must NOT go through
#      TrainingArguments.label_smoothing_factor -- SltTrainer raises if that is
#      non-zero, because Trainer would then pop `labels` and silently drop the
#      CTC term and every logging scalar. The CTC loss is never smoothed.
#
# Default epsilon when smoothing is enabled without a value: 0.1, the canonical
# label-smoothing weight for MT / seq2seq (Vaswani et al., 2017).
#
# Usage:
#   sbatch scripts/run_label_smoothing_de.sh                 # off (eps = 0.0)
#   sbatch scripts/run_label_smoothing_de.sh smooth          # on,  eps = 0.1
#   sbatch scripts/run_label_smoothing_de.sh smooth=0.05     # on,  eps = 0.05
#   sbatch scripts/run_label_smoothing_de.sh smooth debug
#   sbatch scripts/run_label_smoothing_de.sh smooth share
# --------------------------------------------------------------------------- #

LABEL_SMOOTHING=0.0
DEBUG=false
SHARED_DATASET=false
for arg in "$@"; do
  case "$arg" in
  smooth | labelsmooth | ls) LABEL_SMOOTHING=0.1 ;;
  smooth=* | labelsmooth=* | ls=*) LABEL_SMOOTHING="${arg#*=}" ;;
  debug) DEBUG=true ;;
  share) SHARED_DATASET=true ;;
  *)
    echo "Unknown argument: $arg (supported: smooth[=EPS], debug, share)" >&2
    exit 2
    ;;
  esac
done

# SltConfig enforces label_smoothing in [0.0, 1.0); reject bad values here so a
# typo fails at submit time rather than after the job is scheduled.
if ! awk -v e="$LABEL_SMOOTHING" \
  'BEGIN { exit !(e ~ /^[0-9]*\.?[0-9]+$/ && e + 0 >= 0 && e + 0 < 1) }'; then
  echo "label smoothing epsilon must be a number in [0.0, 1.0), got: $LABEL_SMOOTHING" >&2
  exit 2
fi

if awk -v e="$LABEL_SMOOTHING" 'BEGIN { exit !(e + 0 > 0) }'; then
  LS_ON=true
  RUN_TAG="ls${LABEL_SMOOTHING}"
else
  LS_ON=false
  RUN_TAG="ls-off"
fi

if [[ "$DEBUG" == true ]]; then
  echo "Debug mode: Disabling reporting to WandB, outputs go to outputs/debug-labelsmooth-de-${RUN_TAG}."
  REPORT_TO=none
  OUTPUT_DIR="outputs/debug-labelsmooth-de-${RUN_TAG}"
else
  export WANDB_PROJECT=sign_language_translation_v5.0-dev
  export WANDB_TAGS="next-frame,hand-roi,cls,20m,fixed-prompt,proj-dropout,posenc-learned,output-layer--8,de,label-smoothing-ablation,${RUN_TAG}"
  REPORT_TO=wandb
  OUTPUT_DIR="outputs/v5.0-qwen3-4b-cradio-l-nextframe-handroi-cls-20m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-ol-8-ep50-de-labelsmooth-${RUN_TAG}-0910.224x224"
fi

if [[ -t 2 ]]; then
  unset TQDM_DISABLE
  HG_TQDM_DISABLE=False
else
  export TQDM_DISABLE=1
  HG_TQDM_DISABLE=True
fi

# tubbs 上直接使用共享数据集；集群上只有非 share 模式才复制到本地 scratch。
if [[ "$IS_TUBBS" == true || "$SHARED_DATASET" == true ]]; then
  DATASET_PATH="$SCRIPT_DIR/dataset/PHOENIX-2014-T-release-v3"
else
  source "$SCRIPT_DIR/scripts/prepare_dataset.sh"
  DATASET_PATH=$(prepare_dataset \
    "$SCRIPT_DIR/dataset/phoenix-2014-T.v3.tar.gz" \
    "$HOME/localscratch/ph14t")
fi
echo "LABEL_SMOOTHING=$LABEL_SMOOTHING  DATASET_PATH=$DATASET_PATH  OUTPUT_DIR=$OUTPUT_DIR"

SCORER_PATH=outputs/hand_patch_scorer_L8
if [[ ! -f "$SCRIPT_DIR/$SCORER_PATH/config.json" ]]; then
  echo "no fitted scorer at $SCORER_PATH" >&2
  exit 3
fi

CMD_ARGS=(
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  --config-name=train/pretrain_adapter/baseline_ablation
  # Fixed for every split; its resolver maps en/de/zh to their canonical IDs.
  prompt=fixed_prompt
  # German only; this ablation does not vary the target language.
  data.language=de
  # Best-checkpoint architecture and regularization settings -- identical to
  # scripts/run_cognition_target_language.sh.
  model=qwen3-4b-cradio-l-spatiotemporal-next-frame-handroi-cls-20m
  model.config.visual_backbone_config.output_layer=-8
  model.config.visual_adapter_kwargs.scorer_path="$SCORER_PATH"
  model.config.visual_adapter_kwargs.projection_dropout=0.5
  model.config.visual_adapter_kwargs.roi_projection_dropout=0.5
  model.config.visual_adapter_kwargs.cls_projection_dropout=0.5
  model.config.visual_position_embedding_type=learned
  engine.trainability.visual_position_embedding.parameter_mode=full
  +engine.optimization.visual_adapter.parameter_groups.gates.learning_rate=1e-3
  # Logging cadence: same as the reference. num_train_epochs is intentionally
  # NOT set here, so it stays at the base-config default (50).
  engine.training_args.dataloader_num_workers=8
  engine.training_args.eval_steps=6000
  engine.training_args.logging_steps=15
  engine.training_args.ddp_find_unused_parameters=false
  engine.training_args.output_dir="$OUTPUT_DIR"
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  data.data_root="$DATASET_PATH"
)

# The sole experimental variable. Left unset (config default 0.0) when off, so
# an "off" run is bit-for-bit the reference architecture.
if [[ "$LS_ON" == true ]]; then
  CMD_ARGS+=(+model.config.label_smoothing="$LABEL_SMOOTHING")
fi

accelerate launch "${CMD_ARGS[@]}"
