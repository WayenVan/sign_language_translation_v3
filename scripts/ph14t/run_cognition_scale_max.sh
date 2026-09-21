#! /bin/bash
#
# Usage:
#   bash scripts/ph14t/run_cognition_scale_max.sh              # tubbs: use the extracted dataset directly
#   sbatch scripts/ph14t/run_cognition_scale_max.sh            # de+en+zh, diverse train prompts (default)
#   sbatch scripts/ph14t/run_cognition_scale_max.sh multi 60   # override epochs
#   sbatch scripts/ph14t/run_cognition_scale_max.sh zh fixed   # single language: 80 epochs
#   sbatch scripts/ph14t/run_cognition_scale_max.sh multi fixed share
#   sbatch scripts/ph14t/run_cognition_scale_max.sh debug      # full training, no WandB
#
# Qwen3-32B dense + C-RADIO-L scale-max launcher. See below for the fixed
# recipe (two H100s, FSDP2 sharding).

#SBATCH --job-name=slt_scale_max_qwen3_32b
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=21
#SBATCH --mem=256g
#SBATCH --time=2-12:00:00

set -euo pipefail

# Host allowlist. An unrecognized host is a hard stop so a stray launch
# (laptop, wrong login node) fails immediately instead of cd-ing into a path
# that does not exist and half-running.
#
# Per-host, IS_TUBBS also decides two things below:
#   * dataset: tubbs has it extracted in-repo, so skip the stage-to-scratch step;
#   * NCCL P2P: only the cluster fabric needs NCCL_P2P_DISABLE=1; tubbs's local
#     GPUs keep P2P enabled.
HOST_FQDN="$(hostname -f 2>/dev/null || hostname)"
if [[ "$HOST_FQDN" == "tubbs.eng.gla.ac.uk" || "$HOST_FQDN" == "tubbs" ]]; then
  SCRIPT_DIR=/home/2533494W/project/sign_language_translation_v3
  IS_TUBBS=true
elif [[ -d /users/2533494w/projects/sign_language_translation_v3 ]]; then
  # Glasgow HPC cluster.
  SCRIPT_DIR=/users/2533494w/projects/sign_language_translation_v3
  IS_TUBBS=false
else
  echo "unrecognized host '$HOST_FQDN': no known sign_language_translation_v3 checkout here; refusing to run." >&2
  exit 4
fi

if [[ "$IS_TUBBS" == true ]]; then
  export NCCL_P2P_DISABLE=0
else
  export NCCL_P2P_DISABLE=1
fi

cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"

export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# --------------------------------------------------------------------------- #
# Qwen3-32B dense + C-RADIO-L, with the same 30,847,501-parameter adapter
# as the 14B model (projection ranks 2349 / 1265 / 620).
# Default to multilingual de+en+zh with diverse train prompts, 25 epochs; frozen LLM and
# backbone, trainable adapter/CTC/visual embeddings, no LoRA.
#
# Two H100s with FSDP2 parameter sharding; activation checkpointing disabled.
# CPU RAM covers both ranks loading full weights before sharding. GPU peak
# memory and throughput still require measurement on the training cluster.
# Measure full forward/backward peak memory with recomputation disabled.
# --------------------------------------------------------------------------- #

TARGET_LANGUAGE=multi
PROMPT_CONFIG=diverse_train
EPOCHS_OVERRIDE=
DEBUG=false
SHARED_DATASET=false
for arg in "$@"; do
  case "$arg" in
  de | en | zh) TARGET_LANGUAGE="$arg" ;;
  multi | multilang) TARGET_LANGUAGE=multi ;;
  fixed | fixed_prompt) PROMPT_CONFIG=fixed_prompt ;;
  diverse | diverse_train) PROMPT_CONFIG=diverse_train ;;
  debug) DEBUG=true ;;
  share) SHARED_DATASET=true ;;
  [0-9]*)
    if [[ ! "$arg" =~ ^[0-9]+$ ]] || ((10#$arg < 1)); then
      echo "epochs must be a positive integer, got: $arg" >&2
      exit 2
    fi
    EPOCHS_OVERRIDE="$arg"
    ;;
  *)
    echo "Unknown argument: $arg (supported: de, en, zh, multi, fixed, diverse, debug, share, <epochs>)" >&2
    exit 2
    ;;
  esac
done

if [[ "$TARGET_LANGUAGE" == multi ]]; then
  # Shorter 32B multilingual run; evaluate about every 2.25 epochs.
  NUM_TRAIN_EPOCHS=25
  EVAL_STEPS=12000
  TRAIN_CONFIG=train/pretrain_adapter/best_adapter_multilang
  LANG_TAG=multilang
else
  NUM_TRAIN_EPOCHS=80
  EVAL_STEPS=6000
  TRAIN_CONFIG=train/pretrain_adapter/baseline_ablation
  LANG_TAG="$TARGET_LANGUAGE"
fi
NUM_TRAIN_EPOCHS="${EPOCHS_OVERRIDE:-$NUM_TRAIN_EPOCHS}"
RUN_TAG="scale-max-${LANG_TAG}-${PROMPT_CONFIG}"
if [[ "$DEBUG" == true ]]; then
  echo "Debug mode: Disabling reporting to WandB, outputs go to outputs/debug-${RUN_TAG}-ep${NUM_TRAIN_EPOCHS}."
  REPORT_TO=none
  OUTPUT_DIR="outputs/debug-${RUN_TAG}-ep${NUM_TRAIN_EPOCHS}"
else
  export WANDB_PROJECT=sign_language_translation_v5.0-dev
  export WANDB_TAGS="next-frame,hand-roi,cls,31m,qwen3-32b,fsdp2,${PROMPT_CONFIG},proj-dropout,posenc-learned,output-layer--8,ep${NUM_TRAIN_EPOCHS},scale-max,${RUN_TAG}"
  REPORT_TO=wandb
  OUTPUT_DIR="outputs/v5.0-qwen3-32b-cradio-l-nextframe-handroi-cls-31m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-ol-8-ep${NUM_TRAIN_EPOCHS}-${RUN_TAG}-0914.224x224"
fi

if [[ -t 2 ]]; then
  unset TQDM_DISABLE
  HG_TQDM_DISABLE=False
else
  export TQDM_DISABLE=1
  HG_TQDM_DISABLE=True
fi

# tubbs 上数据集已在仓库内解压好，直接用；share 模式同理走共享路径。
# 只有集群需要把数据集迁移(stage)到本地 scratch。
if [[ "$IS_TUBBS" == true || "$SHARED_DATASET" == true ]]; then
  DATASET_PATH="$SCRIPT_DIR/dataset/PHOENIX-2014-T-release-v3"
else
  source "$SCRIPT_DIR/scripts/ph14t/prepare_dataset.sh"
  DATASET_PATH=$(prepare_dataset \
    "$SCRIPT_DIR/dataset/phoenix-2014-T.v3.tar.gz" \
    "$HOME/localscratch/ph14t")
fi
echo "TARGET_LANGUAGE=$TARGET_LANGUAGE  PROMPT=$PROMPT_CONFIG  EPOCHS=$NUM_TRAIN_EPOCHS  EVAL_STEPS=$EVAL_STEPS  DATASET_PATH=$DATASET_PATH  OUTPUT_DIR=$OUTPUT_DIR"

SCORER_PATH=outputs/hand_patch_scorer_L8
if [[ ! -f "$SCRIPT_DIR/$SCORER_PATH/config.json" ]]; then
  echo "no fitted scorer at $SCORER_PATH" >&2
  exit 3
fi

CMD_ARGS=(
  --config_file=configs/accelerate/fsdp2.yaml
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  --config-name="$TRAIN_CONFIG"
  # Diverse training prompts by default; val/test keep canonical prompts.
  prompt="$PROMPT_CONFIG"
  # Best-checkpoint architecture and regularization settings.
  model=qwen3-32b-cradio-l-spatiotemporal-next-frame-handroi-cls-31m
  model.config.visual_backbone_config.output_layer=-8
  model.config.visual_adapter_kwargs.scorer_path="$SCORER_PATH"
  model.config.visual_adapter_kwargs.projection_dropout=0.5
  model.config.visual_adapter_kwargs.roi_projection_dropout=0.5
  model.config.visual_adapter_kwargs.cls_projection_dropout=0.5
  model.config.visual_position_embedding_type=learned
  engine.trainability.visual_position_embedding.parameter_mode=full
  +engine.optimization.visual_adapter.parameter_groups.gates.learning_rate=1e-3
  # Best-checkpoint training schedule and logging cadence.
  engine.training_args.num_train_epochs="$NUM_TRAIN_EPOCHS"
  # Maximum on-disk weight shard size; independent of FSDP GPU sharding.
  +engine.training_args.checkpoint_max_shard_size=8GB
  engine.training_args.dataloader_num_workers=8
  engine.training_args.eval_steps="$EVAL_STEPS"
  engine.training_args.logging_steps=15
  engine.training_args.ddp_find_unused_parameters=false
  engine.training_args.output_dir="$OUTPUT_DIR"
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  data.data_root="$DATASET_PATH"
)

# The multilingual dataset yields de/en/zh together and has no language key.
if [[ "$TARGET_LANGUAGE" != multi ]]; then
  CMD_ARGS+=(data.language="$TARGET_LANGUAGE")
fi

accelerate launch "${CMD_ARGS[@]}"
