#! /bin/bash
#
# Usage:
#   sbatch scripts/csl/run_llm_lora_inject.sh                      # default ckpt, zh, diverse, r768, ep12
#   sbatch scripts/csl/run_llm_lora_inject.sh 512                  # rank override
#   sbatch scripts/csl/run_llm_lora_inject.sh 768 18               # epoch-count override
#   sbatch scripts/csl/run_llm_lora_inject.sh <CKPT_DIR>           # another stage-1 checkpoint
#   bash   scripts/csl/run_llm_lora_inject.sh share debug          # local smoke test
#
# <CKPT_DIR> is a CSL-Daily stage-1 checkpoint from scripts/csl/run_cognition_14b.sh;
# it defaults to the relaxed-CTC diverse-prompt run's checkpoint-84000 and, when
# given, always comes first. After that, arguments are order-free: debug, share,
# plus one bare positive integer for the LoRA rank and a second one for the
# epoch count.
# Help: scripts/csl/run_llm_lora_inject.sh --help
#
# CSL-Daily counterpart of scripts/ph14t/run_llm_lora_inject.sh: q/k/v/o LoRA
# over all LLM layers (alpha = 2 * rank, lr 1e-4), with the adapter, CTC head
# and visual embeddings trainable at 1e-5, as in train/lora_llm/base.

#SBATCH --job-name=slt_csl_llm_lora_inject
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=21
#SBATCH --mem=256g
#SBATCH --time=2-12:00:00

set -euo pipefail

usage() {
  sed -n '/^# Usage:$/,/^# Help:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

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

# Default stage-1 checkpoint: the relaxed-CTC diverse-prompt 14B run at epoch
# 18.3. Its eval BLEU4 (0.2228) matches the later checkpoint-126000 (0.2242,
# within run-to-run noise) while its train probe is far lower (0.461 vs 0.638),
# so it is the same held-out quality with much less memorization to carry into
# stage 2. A path argument overrides it; a path is anything containing a slash
# or naming an existing directory, so bare keywords and ranks still parse.
CHECKPOINT_DIR=outputs/csl_ckpt/qwen3-14b-cradio-l-nextframe-handroi-cls-31m-posenc-learned-ctc-relaxed-diverse-zh/checkpoint-84000
if [[ $# -gt 0 && ( "$1" == */* || -d "$1" ) ]]; then
  CHECKPOINT_DIR="$1"
  shift
fi
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "checkpoint directory not found: $CHECKPOINT_DIR" >&2
  exit 3
fi

RANK=
EPOCHS=
DEBUG=false
SHARED_DATASET=false
for arg in "$@"; do
  case "$arg" in
  debug) DEBUG=true ;;
  share) SHARED_DATASET=true ;;
  [0-9]*)
    if [[ ! "$arg" =~ ^[0-9]+$ ]] || ((10#$arg < 1)); then
      echo "numeric arguments must be positive integers, got: $arg" >&2
      exit 2
    fi
    if [[ -z "$RANK" ]]; then
      RANK="$arg"
    elif [[ -z "$EPOCHS" ]]; then
      EPOCHS="$arg"
    else
      echo "too many numeric arguments (rank and epochs already set), got: $arg" >&2
      exit 2
    fi
    ;;
  *)
    echo "Unknown argument: $arg (supported: debug, share, <rank>, <epochs>)" >&2
    exit 2
    ;;
  esac
done
RANK="${RANK:-768}"
ALPHA=$(( 2 * RANK ))
EPOCHS="${EPOCHS:-12}"
# The processor is rebuilt from the data group (default 1.0), not loaded from
# the checkpoint; the stage-1 adapter emits one token per two frames.
VIDEO_TOKEN_SCALE=0.5

# The checkpoint must come from the CSL diverse-prompt relaxed-CTC stage-1 run.
# data.language is not checked: CSL configs inherit an unused `de` there.
python3 - "$CHECKPOINT_DIR/hydra_config.yaml" "$VIDEO_TOKEN_SCALE" <<'PYEOF'
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
if not path.is_file():
    sys.exit(f"no {path}; cannot verify the stage-1 checkpoint")
cfg = yaml.safe_load(path.read_text())
data = cfg["data"]
checks = {
    "dataset": (data["train"]["dataset"]["_target_"].endswith("CSLDailyDataset"),
                data["train"]["dataset"]["_target_"]),
    "prompt": (cfg["prompt"]["train"]["_target_"].endswith("RandomPromptResolver"),
               cfg["prompt"]["train"]["_target_"]),
    "video_token_scale": (float(data["processor"]["video_token_scale"]) == float(sys.argv[2]),
                          data["processor"]["video_token_scale"]),
    "ctc_enabled": (cfg["model"]["config"].get("ctc_enabled") is True,
                    cfg["model"]["config"].get("ctc_enabled")),
}
bad = [f"{k}={v}" for k, (ok, v) in checks.items() if not ok]
if bad:
    sys.exit(f"checkpoint mismatch in {path}: " + ", ".join(bad))
print(f"Checkpoint verification OK ({path})")
PYEOF

CKPT_STEP="$(basename "$CHECKPOINT_DIR")"
CKPT_RUN_NAME="$(basename "$(dirname "$CHECKPOINT_DIR")")"
# WandB rejects tags over 64 characters; the full run name lives in OUTPUT_DIR.
CKPT_HASH="$(echo -n "$CKPT_RUN_NAME" | md5sum | cut -c1-8)"
RUN_TAG="llmlora-${CKPT_RUN_NAME}-${CKPT_STEP}-zh-qkvo-r${RANK}a${ALPHA}-ep${EPOCHS}-diverse"

if [[ "$DEBUG" == true ]]; then
  REPORT_TO=none
  OUTPUT_DIR="outputs/csl_ckpt/debug-$RUN_TAG"
else
  export WANDB_PROJECT=sign_language_translation_v5.0-dev
  export WANDB_TAGS="csl-daily,zh,llm-lora,targets-qkvo,diverse-prompt,rank${RANK},ep${EPOCHS},ckpt-${CKPT_HASH}-${CKPT_STEP#checkpoint-}"
  REPORT_TO=wandb
  OUTPUT_DIR="outputs/csl_ckpt/$RUN_TAG"
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

echo "LANGUAGE=zh  PROMPT=diverse-train/fixed-eval  RANK=$RANK  ALPHA=$ALPHA  EPOCHS=$EPOCHS"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "DATA_ROOT=$DATA_ROOT  OUTPUT_DIR=$OUTPUT_DIR"

CMD_ARGS=(
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  --config-name=train/lora_llm/base
  model.checkpoint_dir="$CHECKPOINT_DIR"

  peft.llm_lora_config.target_modules='[q_proj,k_proj,v_proj,o_proj]'
  peft.llm_lora_config.r="$RANK"
  peft.llm_lora_config.lora_alpha="$ALPHA"

  # Same data and prompts as the stage-1 run.
  data='csl_*x224x224_qwen_single_language'
  data.data_root="$DATA_ROOT"
  data.processor.video_token_scale="$VIDEO_TOKEN_SCALE"
  prompt=diverse_train

  engine.training_args.num_train_epochs="$EPOCHS"
  engine.training_args.dataloader_num_workers=8
  engine.training_args.ddp_find_unused_parameters=false
  engine.training_args.output_dir="$OUTPUT_DIR"
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
)

accelerate launch "${CMD_ARGS[@]}"
