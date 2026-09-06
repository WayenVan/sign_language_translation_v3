#! /bin/bash

#SBATCH --job-name=slt_qwen3_4b_nextframe_handroi_cls_20m_projdrop05_posenc_learned_outputlayer
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=21
#SBATCH --mem=256g

set -euo pipefail

export NCCL_P2P_DISABLE=1 # NOTE: 测试的时候集群通信容易出问题 集群出现了问题

SCRIPT_DIR=/users/2533494w/projects/sign_language_translation_v3

cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"

export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# --------------------------------------------------------------------------- #
# C-RADIO output-layer ablation, on top of the best adapter + learned posenc.
#
# Reference (differs from this by exactly one variable, the output layer):
#   scripts/run_cognition_posenc.sh with visual_position_embedding_type=learned
#   = best adapter (next-frame fusion + gated hand-ROI residual + gated CLS
#     residual + 0.5 projection dropout on every branch) + learned visual
#     positional table.
#
# The ablation variable is model.config.visual_backbone_config.output_layer:
# which C-RADIOv4-SO400M ViT block feeds the adapter. It is passed as the first
# positional argument (-1 | -4 | -8 | -12); -1 is the reference layer.
#
# The frozen hand-patch scorer is fitted per layer -- its coefficients only
# match the feature distribution of the block they were fitted on -- so the
# scorer_path is swapped in lockstep with the layer. SltModel hands the live
# backbone to the adapter at load time and the adapter now raises if the
# scorer's recorded output_layer disagrees with the backbone's, so a
# forgotten swap fails loudly instead of silently degrading the ranking.
#
# Fit the per-layer scorers first (once):
#   for L in 4 8 12; do
#     python preprocess/extract_scorer_features.py --backbone c_radio_v4 \
#       --backbone-config "{\"id\": \"nvidia/C-RADIOv4-SO400M\", \"output_layer\": -${L}}" \
#       --out dataset/ph14_scorer_features_L${L}
#     python preprocess/train_scorer.py --features dataset/ph14_scorer_features_L${L} \
#       --out outputs/hand_patch_scorer_L${L} --epochs 40
#   done
# (-1 is outputs/hand_patch_scorer_L1, already fitted.)
#
# Sweep all four:
#   for L in -1 -4 -8 -12; do
#     sbatch -J slt_ol${L} scripts/run_cognition_outputlayer.sh ${L}
#   done
# --------------------------------------------------------------------------- #

OUTPUT_LAYER="${1:-}"
if [[ -z "$OUTPUT_LAYER" ]]; then
  echo "usage: $0 <output_layer> [debug] [share]   (output_layer: -1 | -4 | -8 | -12)" >&2
  exit 2
fi
if [[ ! "$OUTPUT_LAYER" =~ ^-[0-9]+$ ]]; then
  echo "output_layer must be a negative integer like -8, got: $OUTPUT_LAYER" >&2
  exit 2
fi
shift

# -8 -> L8. The per-layer scorer directory produced by preprocess/train_scorer.py.
LAYER_ABS="${OUTPUT_LAYER#-}"
SCORER_PATH="outputs/hand_patch_scorer_L${LAYER_ABS}"
if [[ ! -f "$SCRIPT_DIR/$SCORER_PATH/config.json" ]]; then
  echo "no fitted scorer at $SCORER_PATH; fit it before launching (see header)." >&2
  exit 3
fi

# 可选参数：debug 关闭 WandB，share 直接使用共享数据集。
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

RUN_TAG="ol${OUTPUT_LAYER}"

if [[ "$DEBUG" == true ]]; then
  echo "Debug mode: Disabling reporting to WandB, outputs go to outputs/debug."
  REPORT_TO=none
  OUTPUT_DIR="outputs/debug"
else
  export WANDB_PROJECT=sign_language_translation_v5.0-dev
  export WANDB_TAGS="next-frame,hand-roi,cls,20m,fixed-prompt,proj-dropout,posenc-learned,output-layer-ablation,${RUN_TAG}"
  REPORT_TO=wandb
  OUTPUT_DIR="outputs/v5.0-qwen3-4b-cradio-l-nextframe-handroi-cls-20m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-${RUN_TAG}-0907.224x224"
fi

# 设置 TQDM_DISABLE 和 HG_TQDM_DISABLE 用于 accelerate launch
if [[ -t 2 ]]; then
  unset TQDM_DISABLE
  HG_TQDM_DISABLE=False
else
  export TQDM_DISABLE=1
  HG_TQDM_DISABLE=True
fi

# share 模式下直接使用共享路径，否则准备数据集到本地 scratch。
if [[ "$SHARED_DATASET" == true ]]; then
  DATASET_PATH="$SCRIPT_DIR/dataset/PHOENIX-2014-T-release-v3"
else
  source "$SCRIPT_DIR/scripts/prepare_dataset.sh"
  DATASET_PATH=$(prepare_dataset \
    "$SCRIPT_DIR/dataset/phoenix-2014-T.v3.tar.gz" \
    "$HOME/localscratch/ph14t")
fi
echo "DATASET_PATH=$DATASET_PATH"
echo "OUTPUT_LAYER=$OUTPUT_LAYER  SCORER_PATH=$SCORER_PATH  OUTPUT_DIR=$OUTPUT_DIR"

# Array form instead of backslash-continued lines: inside (...) each element
# can live on its own line and be commented out individually with a leading
# "#" without breaking the rest of the command.
CMD_ARGS=(
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  --config-name=train/pretrain_adapter/baseline_ablation
  model=qwen3-4b-cradio-l-spatiotemporal-next-frame-handroi-cls-20m
  # --- the ablation variable: which C-RADIO ViT block feeds the adapter ---
  model.config.visual_backbone_config.output_layer="$OUTPUT_LAYER"
  # Swapped in lockstep: the frozen scorer is fitted per layer. The adapter
  # verifies the scorer's recorded output_layer against the backbone at load.
  model.config.visual_adapter_kwargs.scorer_path="$SCORER_PATH"
  # --- everything below is copied verbatim from run_cognition_posenc.sh's
  #     learned variant: best adapter + learned visual positional table ---
  model.config.visual_adapter_kwargs.projection_dropout=0.5
  model.config.visual_adapter_kwargs.roi_projection_dropout=0.5
  model.config.visual_adapter_kwargs.cls_projection_dropout=0.5
  model.config.visual_position_embedding_type=learned
  # base.yaml freezes visual_position_embedding; the learned table only exists
  # and only trains when this is full.
  engine.trainability.visual_position_embedding.parameter_mode=full
  # Run the three scalar gates (patch-fusion / roi / cls) 10x above the
  # adapter's own rate so they reach equilibrium before the projections lock in.
  +engine.optimization.visual_adapter.parameter_groups.gates.learning_rate=1e-3
  engine.training_args.output_dir="$OUTPUT_DIR"
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  data.data_root="$DATASET_PATH"
)

accelerate launch "${CMD_ARGS[@]}"
