#!/usr/bin/env bash

# Qwen3-14B + C-RADIOv4-H output-layer ablation.
# Run exactly one layer in the current terminal. Use sweep.sh for -10..-15.
# Example: bash h_14b_outputlayer_ablation/run_outputlayer.sh -10
# Optional arguments after the layer: a positive epoch count, debug, share.

set -euo pipefail

OUTPUT_LAYER="${1:-}"
if [[ ! "$OUTPUT_LAYER" =~ ^-[1-9][0-9]*$ ]]; then
  echo "usage: $0 <negative_output_layer> [positive_epochs] [debug] [share]" >&2
  exit 2
fi
shift

NUM_TRAIN_EPOCHS=80
DEBUG=false
SHARED_DATASET=false
EPOCHS_SEEN=false
for arg in "$@"; do
  case "$arg" in
    debug) DEBUG=true ;;
    share) SHARED_DATASET=true ;;
    *)
      if [[ "$arg" =~ ^[1-9][0-9]*$ && "$EPOCHS_SEEN" == false ]]; then
        NUM_TRAIN_EPOCHS="$arg"
        EPOCHS_SEEN=true
      else
        echo "unknown or repeated argument: $arg" >&2
        exit 2
      fi
      ;;
  esac
done

HOST_FQDN="$(hostname -f)"
if [[ "$HOST_FQDN" == "tubbs.eng.gla.ac.uk" ]]; then
  PROJECT_DIR=/home/2533494W/project/sign_language_translation_v3
  IS_TUBBS=true
elif [[ -d /users/2533494w/projects/sign_language_translation_v3 ]]; then
  PROJECT_DIR=/users/2533494w/projects/sign_language_translation_v3
  IS_TUBBS=false
else
  echo "unrecognized host '$HOST_FQDN': repository path is unknown" >&2
  exit 4
fi

if [[ "$IS_TUBBS" == true ]]; then
  unset NCCL_P2P_DISABLE
else
  export NCCL_P2P_DISABLE=1
fi

cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# A scorer must be fitted on precisely the same backbone and output layer.
LAYER_ABS="${OUTPUT_LAYER#-}"
SCORER_PATH="outputs/hand_patch_scorer_cradio-h_L${LAYER_ABS}"
if [[ ! -f "$SCORER_PATH/config.json" || ! -f "$SCORER_PATH/model.safetensors" ]]; then
  echo "no fitted C-RADIOv4-H scorer at $SCORER_PATH" >&2
  exit 3
fi

RUN_TAG="ol${OUTPUT_LAYER}-ep${NUM_TRAIN_EPOCHS}"
OUTPUT_ROOT="outputs/h_14b_outputlayer_ablation"
if [[ "$DEBUG" == true ]]; then
  REPORT_TO=none
  OUTPUT_DIR="$OUTPUT_ROOT/debug/$RUN_TAG"
else
  export WANDB_PROJECT=sign_language_translation_v5.0-dev
  export WANDB_TAGS="next-frame,hand-roi,cls,32m,qwen3-14b,cradio-h,fixed-prompt,proj-dropout,posenc-learned,output-layer-ablation,${RUN_TAG}"
  REPORT_TO=wandb
  OUTPUT_DIR="$OUTPUT_ROOT/$RUN_TAG"
fi

if [[ -t 2 ]]; then
  unset TQDM_DISABLE
  HG_TQDM_DISABLE=False
else
  export TQDM_DISABLE=1
  HG_TQDM_DISABLE=True
fi

if [[ "$IS_TUBBS" == true || "$SHARED_DATASET" == true ]]; then
  DATASET_PATH="$PROJECT_DIR/dataset/PHOENIX-2014-T-release-v3"
else
  source "$PROJECT_DIR/scripts/ph14t/prepare_dataset.sh"
  DATASET_PATH=$(prepare_dataset \
    "$PROJECT_DIR/dataset/phoenix-2014-T.v3.tar.gz" \
    "$HOME/localscratch/ph14t")
fi

echo "OUTPUT_LAYER=$OUTPUT_LAYER  SCORER_PATH=$SCORER_PATH"
echo "DATASET_PATH=$DATASET_PATH  OUTPUT_DIR=$OUTPUT_DIR  EPOCHS=$NUM_TRAIN_EPOCHS"

# Match the single-language 14B best-adapter scale-up recipe. Only the
# backbone output layer and its inseparable, layer-fitted scorer vary.
CMD_ARGS=(
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  --config-name=train/pretrain_adapter/baseline_ablation
  model=qwen3-14b-cradio-h-spatiotemporal-next-frame-handroi-cls-32m
  model.config.visual_backbone_config.output_layer="$OUTPUT_LAYER"
  model.config.visual_adapter_kwargs.scorer_path="$SCORER_PATH"
  model.config.visual_adapter_kwargs.projection_dropout=0.5
  model.config.visual_adapter_kwargs.roi_projection_dropout=0.5
  model.config.visual_adapter_kwargs.cls_projection_dropout=0.5
  model.config.visual_position_embedding_type=learned
  engine.trainability.visual_position_embedding.parameter_mode=full
  +engine.optimization.visual_adapter.parameter_groups.gates.learning_rate=1e-3
  engine.training_args.num_train_epochs="$NUM_TRAIN_EPOCHS"
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
