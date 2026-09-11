#! /bin/bash
#
# Usage:
#   sbatch scripts/run_cognition_scaleup_8b_h.sh              # de (default)
#   sbatch scripts/run_cognition_scaleup_8b_h.sh en           # target language: de | en | zh
#   sbatch scripts/run_cognition_scaleup_8b_h.sh de share     # shared dataset path, no scratch staging
#   sbatch scripts/run_cognition_scaleup_8b_h.sh debug        # no WandB; outputs/debug-scaleup-<lang>
#
# Qwen3-8B + C-RADIOv4-H scale-up of the best Qwen3-4B checkpoint; see below.

#SBATCH --job-name=slt_scaleup_qwen3_8b_cradio_h
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
HOST_FQDN="$(hostname -f)"
if [[ "$HOST_FQDN" == "tubbs.eng.gla.ac.uk" ]]; then
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
  unset NCCL_P2P_DISABLE
else
  export NCCL_P2P_DISABLE=1
fi

cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"

export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# --------------------------------------------------------------------------- #
# Scale-up run: Qwen3-8B + C-RADIOv4-H (output layer -9), with the adapter and
# every training setting of the best Qwen3-4B checkpoint:
#
#   outputs/v5.0-qwen3-4b-cradio-l-nextframe-handroi-cls-20m-gate1-hardmatch-
#   wr3-projdrop0.5-posenc-learned-ol-8-ep80-0907.224x224/checkpoint-96000
#   (de, dev BLEU-4 0.1715; its en run reached 0.1843)
#
# The overrides below are scripts/run_cognition_target_language.sh's, changed
# only where the scale-up requires it: model, output layer, and scorer.
#
# Partition: H100, not L40S. The 4B reference ran on H100 NVL (node12) and
# peaked at 39.0 GiB on one rank; Qwen3-8B adds ~7.8 GiB of frozen bf16 weights
# on top, past the ~45 GiB an L40S exposes.
#
# Time limit: the 4B reference took 16.5 h (59,309 s) for 141,920 steps, about
# 12 h training and 4.5 h evaluation / train-probe generation. Doubling both
# for 8B + H gives ~31 h; 2-12:00:00 is ~1.9x that. save_strategy=best keeps no
# resumable last checkpoint, so the margin errs on the generous side.
# --------------------------------------------------------------------------- #

# de by default: the language of the 4B checkpoint this run is compared against.
TARGET_LANGUAGE=de
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

RUN_TAG="${TARGET_LANGUAGE}"
if [[ "$DEBUG" == true ]]; then
  echo "Debug mode: Disabling reporting to WandB, outputs go to outputs/debug-scaleup-${RUN_TAG}."
  REPORT_TO=none
  OUTPUT_DIR="outputs/debug-scaleup-${RUN_TAG}"
else
  export WANDB_PROJECT=sign_language_translation_v5.0-dev
  export WANDB_TAGS="next-frame,hand-roi,cls,28m,qwen3-8b,cradio-h,fixed-prompt,proj-dropout,posenc-learned,output-layer--9,ep80,scale-up,${RUN_TAG}"
  REPORT_TO=wandb
  OUTPUT_DIR="outputs/v5.0-qwen3-8b-cradio-h-nextframe-handroi-cls-28m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-ol-9-ep80-${RUN_TAG}-0911.224x224"
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
  source "$SCRIPT_DIR/scripts/prepare_dataset.sh"
  DATASET_PATH=$(prepare_dataset \
    "$SCRIPT_DIR/dataset/phoenix-2014-T.v3.tar.gz" \
    "$HOME/localscratch/ph14t")
fi
echo "TARGET_LANGUAGE=$TARGET_LANGUAGE  DATASET_PATH=$DATASET_PATH  OUTPUT_DIR=$OUTPUT_DIR"

# Fitted by scripts/swaps/run_scorer_cradio_h_sweep.sh. Must match OUTPUT_LAYER:
# the adapter raises if the scorer's recorded layer disagrees with the backbone.
OUTPUT_LAYER=-9
SCORER_PATH=outputs/hand_patch_scorer_cradio-h_L9
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
  data.language="$TARGET_LANGUAGE"
  # Scale-up architecture: Qwen3-8B + C-RADIOv4-H, same adapter ranks (27.5M).
  model=qwen3-8b-cradio-h-spatiotemporal-next-frame-handroi-cls-28m
  model.config.visual_backbone_config.output_layer="$OUTPUT_LAYER"
  model.config.visual_adapter_kwargs.scorer_path="$SCORER_PATH"
  # Best-checkpoint regularization. The model file already defaults to these;
  # restated so the run's recipe reads off this command in full.
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
