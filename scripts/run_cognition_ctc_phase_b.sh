#!/bin/bash

#SBATCH --job-name=slt_ctc_phase_b
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=21
#SBATCH --mem=256g

set -euo pipefail

# The cluster's peer-to-peer path has been unreliable in earlier runs.
export NCCL_P2P_DISABLE=1

# This script lives in scripts/; resolve the project root one level up so it
# works whether submitted from the root or from scripts/.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/.venv/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# Which Phase-B config to run. Edit this line to switch variants:
#   train/ctc/phase_b_codebook  -- codebook only, everything else frozen
#   train/ctc/phase_b_joint     -- also trains the visual adapter + CTC head
CONFIG_NAME=train/ctc/phase_b_joint

# Optional arguments:
#   debug  disable WandB reporting
#   share  read the shared dataset instead of preparing local scratch
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
  echo "Debug mode: disabling WandB reporting."
  REPORT_TO=none
else
  export WANDB_PROJECT=sign_language_translation_ctc
  export WANDB_TAGS="phase-b,codebook,next-frame-handroi,conv,20m"
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
  DATASET_PATH="$PROJECT_DIR/dataset/PHOENIX-2014-T-release-v3"
else
  source "$PROJECT_DIR/scripts/prepare_dataset.sh"
  DATASET_PATH=$(prepare_dataset \
    "$PROJECT_DIR/dataset/phoenix-2014-T.v3.tar.gz" \
    "$HOME/localscratch/ph14t")
fi
echo "DATASET_PATH=$DATASET_PATH"

# Phase-A CTC checkpoint this continues: best eval_ctc_wer = 0.363 (dev) at
# checkpoint-29548, train-probe WER 0.008. Phase B restores the whole model from
# here. Override on the CLI to resume from a different one. The output_dir is
# set by the chosen config (phase_b_codebook / phase_b_joint) so the two
# variants never collide.
PHASE_A_CKPT=/mnt/scratch/users/2533494w/slt_outputs/ctc-phase-a-qwen3-4b-cradio-l-nextframe-handroi-conv-20m-gloss/checkpoint-29548

CMD_ARGS=(
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  --config-name="$CONFIG_NAME"
  model.checkpoint_dir="$PHASE_A_CKPT"
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  data.data_root="$DATASET_PATH"
)

accelerate launch "${CMD_ARGS[@]}"

# --- Follow-ups, one variable each, after this run's step-0 (before_train)
#     evaluation is recorded.
#
# Sharper tail if mean_top1_prob_nonblank stays low and dev trails the probe:
#   engine.ctc_codebook_temperature_schedule.end=0.15
#
# Temperature-matched eval instead of forced argmax (needs the bridge change
# discussed -- thread an eval temperature through the prediction step):
#   keeps soft mixtures at eval, removes the train/eval prefix gap entirely.
#
# Stage 3: unfreeze the LLM through LoRA on top of the best Phase-B codebook,
#   cf. configs/train/cognition/stage2_llm_lora.yaml (adapt to forward_mode=joint
#   and ctc_codebook already trained).
