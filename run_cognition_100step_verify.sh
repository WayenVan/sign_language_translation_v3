#! /bin/bash

#SBATCH --job-name=slt_qwen_ckpt_verify
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=21
#SBATCH --mem=256g

set -euo pipefail

export NCCL_P2P_DISABLE=1

SCRIPT_DIR=/users/2533494w/projects/sign_language_translation_v3
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/.venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# Usage: sbatch run_cognition_100step_verify.sh [debug] [share]
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

if [[ "$DEBUG" == true ]]; then
  REPORT_TO=none
else
  export WANDB_PROJECT=sign_language_translation_v4.0-dev
  export WANDB_TAGS="checkpoint-roundtrip-verification"
  REPORT_TO=wandb
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
  source "$SCRIPT_DIR/prepare_dataset.sh"
  DATASET_PATH=$(prepare_dataset \
    "$SCRIPT_DIR/dataset/phoenix-2014-T.v3.tar.gz" \
    "$HOME/localscratch/ph14t")
fi
echo "DATASET_PATH=$DATASET_PATH"

# Override this when running several checks without deleting old results.
OUTPUT_DIR=${SLT_VERIFY_OUTPUT_DIR:-outputs/cognition-100step-checkpoint-verify}
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output already exists: $OUTPUT_DIR" >&2
  echo "Set SLT_VERIFY_OUTPUT_DIR to a new path to avoid mixing runs." >&2
  exit 2
fi

CMD_ARGS=(
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  model=qwen3-1.7b-cradio-l-dinoframecrossv28shuffle-micro
  peft=none
  engine.trainability.llm.parameter_mode=frozen
  engine.trainability.visual_backbone.parameter_mode=frozen
  engine.trainability.visual_adapter.parameter_mode=full
  engine.trainability.ctc_head.parameter_mode=full
  engine.trainability.visual_position_embedding.parameter_mode=full
  engine.trainability.visual_boundary_embeddings.parameter_mode=full
  engine.training_args.learning_rate=1e-4
  +engine.optimization.visual_adapter.learning_rate=1e-4
  prompt=fixed_prompt
  engine.training_args.output_dir="$OUTPUT_DIR"
  +engine.training_args.max_steps=100
  engine.training_args.save_strategy=steps
  +engine.training_args.save_steps=100
  engine.training_args.save_total_limit=1
  engine.training_args.eval_strategy=no
  engine.training_args.load_best_model_at_end=False
  engine.training_args.do_predict=False
  +engine.verify_checkpoint_roundtrip=True
  engine.training_args.per_device_train_batch_size=2
  engine.training_args.per_device_eval_batch_size=1
  engine.training_args.dataloader_num_workers=6
  engine.training_args.logging_steps=15
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  engine.training_args.ddp_find_unused_parameters=False
  data=ph14t_*x224x224_qwen_single_language
  data.language=de
  data.processor.video_token_scale=1.0
  data.data_root="$DATASET_PATH"
  data.processor.num_extra_video_tokens=2
  data.processor.video_processor.padding_to_multiple_of=4
  data.processor.video_processor.do_resize=False
  data.processor.video_processor.do_normalize=False
)

accelerate launch "${CMD_ARGS[@]}"
