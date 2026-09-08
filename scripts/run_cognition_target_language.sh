#! /bin/bash

#SBATCH --job-name=slt_target_language_ablation
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=21
#SBATCH --mem=256g

set -euo pipefail

export NCCL_P2P_DISABLE=1

if [[ "$(hostname -f)" == "tubbs.eng.gla.ac.uk" ]]; then
  SCRIPT_DIR=/home/2533494W/project/sign_language_translation_v3
else
  SCRIPT_DIR=/users/2533494w/projects/sign_language_translation_v3
fi

cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"

export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# --------------------------------------------------------------------------- #
# Target-language ablation using the exact settings of the best checkpoint:
#
#   outputs/v5.0-qwen3-4b-cradio-l-nextframe-handroi-cls-20m-gate1-hardmatch-
#   wr3-projdrop0.5-posenc-learned-ol-8-ep80-0907.224x224/checkpoint-96000
#
# The only experimental variable is data.language. All splits use
# prompt=fixed_prompt, which selects the canonical prompt for that language.
# The default target language is English:
#
#   sbatch scripts/run_cognition_target_language.sh
#   sbatch scripts/run_cognition_target_language.sh zh
#   sbatch scripts/run_cognition_target_language.sh de share
# --------------------------------------------------------------------------- #

TARGET_LANGUAGE=en
DEBUG=false
SHARED_DATASET=false
for arg in "$@"; do
  case "$arg" in
  de | en | zh) TARGET_LANGUAGE="$arg" ;;
  debug) DEBUG=true ;;
  share) SHARED_DATASET=true ;;
  *)
    echo "Unknown argument: $arg (supported: de, en, zh, debug, share)" >&2
    exit 2
    ;;
  esac
done

RUN_TAG="targetlang-${TARGET_LANGUAGE}"
if [[ "$DEBUG" == true ]]; then
  echo "Debug mode: Disabling reporting to WandB, outputs go to outputs/debug-${RUN_TAG}."
  REPORT_TO=none
  OUTPUT_DIR="outputs/debug-${RUN_TAG}"
else
  export WANDB_PROJECT=sign_language_translation_v5.0-dev
  export WANDB_TAGS="next-frame,hand-roi,cls,20m,fixed-prompt,proj-dropout,posenc-learned,output-layer--8,ep80,target-language-ablation,${RUN_TAG}"
  REPORT_TO=wandb
  OUTPUT_DIR="outputs/v5.0-qwen3-4b-cradio-l-nextframe-handroi-cls-20m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-ol-8-ep80-${RUN_TAG}-0908.224x224"
fi

if [[ -t 2 ]]; then
  unset TQDM_DISABLE
  HG_TQDM_DISABLE=False
else
  export TQDM_DISABLE=1
  HG_TQDM_DISABLE=True
fi

if [[ "$SHARED_DATASET" == true ]]; then
  DATASET_PATH="$SCRIPT_DIR/dataset/PHOENIX-2014-T-release-v3"
else
  source "$SCRIPT_DIR/scripts/prepare_dataset.sh"
  DATASET_PATH=$(prepare_dataset \
    "$SCRIPT_DIR/dataset/phoenix-2014-T.v3.tar.gz" \
    "$HOME/localscratch/ph14t")
fi
echo "TARGET_LANGUAGE=$TARGET_LANGUAGE  DATASET_PATH=$DATASET_PATH  OUTPUT_DIR=$OUTPUT_DIR"

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
  # The sole ablation variable (default: en).
  data.language="$TARGET_LANGUAGE"
  # Best-checkpoint architecture and regularization settings.
  model=qwen3-4b-cradio-l-spatiotemporal-next-frame-handroi-cls-20m
  model.config.visual_backbone_config.output_layer=-8
  model.config.visual_adapter_kwargs.scorer_path="$SCORER_PATH"
  model.config.visual_adapter_kwargs.projection_dropout=0.5
  model.config.visual_adapter_kwargs.roi_projection_dropout=0.5
  model.config.visual_adapter_kwargs.cls_projection_dropout=0.5
  model.config.visual_position_embedding_type=learned
  engine.trainability.visual_position_embedding.parameter_mode=full
  +engine.optimization.visual_adapter.parameter_groups.gates.learning_rate=1e-3
  # Best-checkpoint training schedule and logging cadence.
  engine.training_args.num_train_epochs=80
  engine.training_args.dataloader_num_workers=8
  engine.training_args.eval_steps=6000
  engine.training_args.logging_steps=15
  engine.training_args.ddp_find_unused_parameters=false
  engine.training_args.output_dir="$OUTPUT_DIR"
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  data.data_root="$DATASET_PATH"
)

accelerate launch "${CMD_ARGS[@]}"
