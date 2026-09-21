#! /bin/bash
#
# Usage:
#   sbatch scripts/ph14t/run_llm_lora_inject.sh <CKPT_DIR>                       # de, fixed prompt, r512, ep12
#   sbatch scripts/ph14t/run_llm_lora_inject.sh <CKPT_DIR> 256                   # rank override
#   sbatch scripts/ph14t/run_llm_lora_inject.sh <CKPT_DIR> 256 en                # single target language: de | en | zh
#   sbatch scripts/ph14t/run_llm_lora_inject.sh <CKPT_DIR> multi                 # multilingual joint training (de+en+zh)
#   sbatch scripts/ph14t/run_llm_lora_inject.sh <CKPT_DIR> multi diverse         # joint training with diverse train prompts
#   sbatch scripts/ph14t/run_llm_lora_inject.sh <CKPT_DIR> 256 multi 20          # epoch-count override
#   bash   scripts/ph14t/run_llm_lora_inject.sh <CKPT_DIR> de share debug        # local smoke test
#   sbatch scripts/ph14t/run_llm_lora_inject.sh <CKPT_DIR> 768 en stable         # StableAdamW optimizer (torch-optimi)
#   sbatch scripts/ph14t/run_llm_lora_inject.sh <CKPT_DIR> 768 en spikeskip      # skip steps with grad norm > 20x median
#
# <CKPT_DIR> is required (a stage-1 checkpoint directory) and always comes
# first. After that, arguments are order-free keywords: de|en|zh|multi,
# fixed|diverse, debug, share, stable, spikeskip, plus one bare positive integer for the LoRA
# rank and a second one for the epoch count -- the first bare int seen is the
# rank, the second is the epoch override. `diverse` is accepted only together
# with `multi` -- see the prompt note below.
# Environment: LLM_LORA_OUTPUT_ROOT overrides the parent output directory.
#   LLM_LORA_SPIKE_SKIP_FACTOR (default 20) sets the `spikeskip` threshold.
# Help: scripts/ph14t/run_llm_lora_inject.sh --help
#
# Formal LLM-LoRA injection launcher: continues LoRA training from any
# stage-1 (frozen-LLM) adapter checkpoint. See below for the fixed recipe.

#SBATCH --job-name=slt_llm_lora_inject
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=21
#SBATCH --mem=256g

set -euo pipefail

usage() {
  sed -n '/^# Usage:$/,/^# Help:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

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
# Generalizes scripts/ph14t/run_llm_lora_ablation.sh (fixed ckpt96k checkpoint,
# single language de, qv/qkvo target ablation) into a formal launcher for
# injecting LoRA into any stage-1 (frozen-LLM) checkpoint:
#
#   Variable:
#     $1 checkpoint dir -- required, no default
#     language          -- de | en | zh (single), or multi (de+en+zh joint)
#     prompt            -- fixed (default) or diverse; diverse is multi-only,
#                           for the same reason run_cognition_scaleup_14b_l.sh
#                           refuses it for single-language runs: a
#                           single-language diverse resolver would sample
#                           paraphrases of one and the same target
#                           instruction, confounding the prompt with nothing.
#     epochs            -- default 12 (single) / 10 (multi)
#     rank              -- default 512; lora_alpha is always 2 * rank
#   Fixed:
#     target_modules    -- q_proj,k_proj,v_proj,o_proj (qkvo), all 36 layers
#     learning rate      -- llm LoRA 1e-4 (LLM_LORA_LR env override), visual
#                           adapter/CTC head/positions/boundary/scale 1e-5,
#                           exactly as in train/lora_llm/base(.yaml)
#     trainable modules  -- adapter, CTC head, learned positions, boundary
#                           embeddings, visual_scale stay trainable; the
#                           visual backbone stays frozen -- unchanged from the
#                           ablation recipe.
#
# Config: train/lora_llm/base (single language) or train/lora_llm/base_multilang
# (multi), which is base's exact recipe continued from
# train/pretrain_adapter/best_adapter_multilang's data/prompt group instead.
# --------------------------------------------------------------------------- #

CHECKPOINT_DIR="${1:-}"
if [[ -z "$CHECKPOINT_DIR" ]]; then
  echo "a checkpoint directory is required as the first argument." >&2
  usage
  exit 2
fi
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "checkpoint directory not found: $CHECKPOINT_DIR" >&2
  exit 3
fi
shift

LEARNING_RATE="${LLM_LORA_LR:-1e-4}"
TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj]'

TARGET_LANGUAGE=de
MULTILANG=false
LANG_ARG_SEEN=false
PROMPT_CONFIG=fixed_prompt
RANK=
EPOCHS_OVERRIDE=
DEBUG=false
SHARED_DATASET=false
STABLE_ADAMW=false
SPIKE_SKIP=false
for arg in "$@"; do
  case "$arg" in
  de | en | zh)
    TARGET_LANGUAGE="$arg"
    LANG_ARG_SEEN=true
    ;;
  multi | multilang) MULTILANG=true ;;
  fixed | fixed_prompt) PROMPT_CONFIG=fixed_prompt ;;
  diverse | diverse_train) PROMPT_CONFIG=diverse_train ;;
  debug) DEBUG=true ;;
  share) SHARED_DATASET=true ;;
  stable | stable_adamw) STABLE_ADAMW=true ;;
  spikeskip | spike_skip) SPIKE_SKIP=true ;;
  [0-9]*)
    if [[ ! "$arg" =~ ^[0-9]+$ ]] || ((10#$arg < 1)); then
      echo "numeric arguments must be positive integers, got: $arg" >&2
      exit 2
    fi
    # First bare int seen is the LoRA rank, second is the epoch override.
    if [[ -z "$RANK" ]]; then
      RANK="$arg"
    elif [[ -z "$EPOCHS_OVERRIDE" ]]; then
      EPOCHS_OVERRIDE="$arg"
    else
      echo "too many numeric arguments (rank and epochs already set), got: $arg" >&2
      exit 2
    fi
    ;;
  *)
    echo "Unknown argument: $arg (supported: de, en, zh, multi, fixed, diverse, debug, share, stable, spikeskip, <rank>, <epochs>)" >&2
    exit 2
    ;;
  esac
done
RANK="${RANK:-512}"
ALPHA=$(( 2 * RANK ))

if [[ "$MULTILANG" == true && "$LANG_ARG_SEEN" == true ]]; then
  echo "'multi' trains de+en+zh jointly; do not also pass a single language ($TARGET_LANGUAGE)." >&2
  exit 2
fi

if [[ "$PROMPT_CONFIG" == diverse_train && "$MULTILANG" != true ]]; then
  echo "diverse prompts are only allowed for multilingual joint training; pass 'multi diverse'." >&2
  exit 2
fi

if [[ "$MULTILANG" == true ]]; then
  CONFIG_NAME=train/lora_llm/base_multilang
  LANG_TAG=multilang
  DEFAULT_EPOCHS=10
else
  CONFIG_NAME=train/lora_llm/base
  LANG_TAG="$TARGET_LANGUAGE"
  DEFAULT_EPOCHS=12
fi
NUM_TRAIN_EPOCHS="${EPOCHS_OVERRIDE:-}"

if [[ "$PROMPT_CONFIG" == diverse_train ]]; then
  PROMPT_TAG=diverse
else
  PROMPT_TAG=fixed
fi

# --------------------------------------------------------------------------- #
# Sanity check: the requested language/prompt must match what this checkpoint
# was actually pretrained with. The adapter and CTC head being continued here
# were fit against one specific language/prompt distribution; training them
# against a different one is not a supported use of this launcher.
#
# hydra_config.yaml is the fully-resolved stage-1 config Hydra writes next to
# every checkpoint (data.language is present only for single-language runs;
# multilingual runs use Ph14TMultiLinglDataset and omit it; prompt.train's
# _target_ is FixedPromptResolver or RandomPromptResolver). A checkpoint
# without that file cannot be verified, so the check is skipped with a
# warning instead of blocking it outright.
# --------------------------------------------------------------------------- #
HYDRA_CONFIG_PATH="$CHECKPOINT_DIR/hydra_config.yaml"
if [[ -f "$HYDRA_CONFIG_PATH" ]]; then
  CKPT_CHECK=$(python3 - "$HYDRA_CONFIG_PATH" <<'PYEOF'
import sys
import yaml

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

language = (cfg.get("data") or {}).get("language")
prompt_target = (((cfg.get("prompt") or {}).get("train")) or {}).get("_target_", "")
prompt_mode = "diverse" if prompt_target.endswith("RandomPromptResolver") else "fixed"
print(language if language else "multilang")
print(prompt_mode)
PYEOF
  )
  CKPT_LANGUAGE="$(sed -n '1p' <<< "$CKPT_CHECK")"
  CKPT_PROMPT="$(sed -n '2p' <<< "$CKPT_CHECK")"

  if [[ "$CKPT_LANGUAGE" != "$LANG_TAG" ]]; then
    echo "checkpoint/language mismatch: $HYDRA_CONFIG_PATH was pretrained with language=$CKPT_LANGUAGE, but this run requested $LANG_TAG." >&2
    exit 5
  fi
  if [[ "$CKPT_PROMPT" != "$PROMPT_TAG" ]]; then
    echo "checkpoint/prompt mismatch: $HYDRA_CONFIG_PATH was pretrained with prompt=$CKPT_PROMPT, but this run requested $PROMPT_TAG." >&2
    exit 5
  fi
  echo "Checkpoint verification OK: pretrained language=$CKPT_LANGUAGE prompt=$CKPT_PROMPT (from $HYDRA_CONFIG_PATH)"
else
  echo "WARNING: no hydra_config.yaml at $CHECKPOINT_DIR; cannot verify its pretraining language/prompt match this run's ($LANG_TAG / $PROMPT_TAG)." >&2
fi

# Fold a non-default epoch count into the run tag so a longer run never
# overwrites the default-length one's output dir.
EP_SUFFIX=""
if [[ -n "$NUM_TRAIN_EPOCHS" ]]; then
  EP_SUFFIX="-ep${NUM_TRAIN_EPOCHS}"
fi
# Single-language fixed-prompt runs keep the historical tag (de / en / zh);
# only multilingual runs carry the prompt tag, since single is always fixed.
PROMPT_SUFFIX=""
if [[ "$MULTILANG" == true ]]; then
  PROMPT_SUFFIX="-${PROMPT_TAG}"
fi

# Derive a checkpoint tag from the path so runs from different stage-1
# checkpoints never collide in outputs/ or WandB.
CKPT_STEP="$(basename "$CHECKPOINT_DIR")"
CKPT_RUN_NAME="$(basename "$(dirname "$CHECKPOINT_DIR")")"
CKPT_TAG="${CKPT_RUN_NAME}-${CKPT_STEP}"
# Short, collision-resistant stand-in for CKPT_RUN_NAME: WandB rejects any
# single tag over 64 characters, and CKPT_TAG (the full stage-1 run directory
# name) routinely blows past that. The full name still lives in OUTPUT_DIR.
CKPT_HASH="$(echo -n "$CKPT_RUN_NAME" | md5sum | cut -c1-8)"
# checkpoint-54000 -> 54000; the wandb tag's own "ckpt-" prefix already says
# "checkpoint", no need to say it twice.
CKPT_STEP_NUM="${CKPT_STEP#checkpoint-}"

# StableAdamW changes the optimizer, so it gets its own output dir / wandb tag
# instead of overwriting the AdamW run with the same checkpoint and rank.
OPTIM_SUFFIX=""
if [[ "$STABLE_ADAMW" == true ]]; then
  OPTIM_SUFFIX="-stableadamw"
fi
# Spike skipping changes which updates are applied, so it also gets its own dir;
# the factor is part of the tag so different thresholds never share one.
SPIKE_SKIP_FACTOR="${LLM_LORA_SPIKE_SKIP_FACTOR:-20}"
SPIKE_SUFFIX=""
if [[ "$SPIKE_SKIP" == true ]]; then
  if [[ ! "$SPIKE_SKIP_FACTOR" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "LLM_LORA_SPIKE_SKIP_FACTOR must be a positive number, got: $SPIKE_SKIP_FACTOR" >&2
    exit 2
  fi
  SPIKE_SUFFIX="-spikeskip${SPIKE_SKIP_FACTOR}"
fi

RUN_TAG="llmlora-${CKPT_TAG}-${LANG_TAG}-qkvo-r${RANK}a${ALPHA}${EP_SUFFIX}${PROMPT_SUFFIX}${OPTIM_SUFFIX}${SPIKE_SUFFIX}"

if [[ "$DEBUG" == true ]]; then
  echo "Debug mode: Disabling reporting to WandB, outputs go to outputs/debug."
  REPORT_TO=none
  OUTPUT_DIR="outputs/debug"
else
  export WANDB_PROJECT=sign_language_translation_v5.0-dev
  # Every tag here must stay under WandB's 64-character-per-tag limit, so this
  # carries short pieces only; RUN_TAG/OUTPUT_DIR (unbounded) is the full
  # record and shows up in the run's config instead.
  export WANDB_TAGS="llm-lora,targets-qkvo,language-${LANG_TAG},${PROMPT_TAG}-prompt,rank${RANK},lr${LEARNING_RATE},ckpt-${CKPT_HASH}-${CKPT_STEP_NUM}${OPTIM_SUFFIX:+,optim-stable-adamw}${SPIKE_SUFFIX:+,spike-skip${SPIKE_SKIP_FACTOR}}"
  REPORT_TO=wandb
  OUTPUT_ROOT="${LLM_LORA_OUTPUT_ROOT:-outputs}"
  OUTPUT_DIR="${OUTPUT_ROOT%/}/${RUN_TAG}"
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

if [[ "$MULTILANG" == true ]]; then
  echo "LANGUAGES=de+en+zh (joint)  PROMPT=$PROMPT_CONFIG  EPOCHS=${NUM_TRAIN_EPOCHS:-config-default-$DEFAULT_EPOCHS}"
else
  echo "TARGET_LANGUAGE=$TARGET_LANGUAGE  PROMPT=$PROMPT_CONFIG  EPOCHS=${NUM_TRAIN_EPOCHS:-config-default-$DEFAULT_EPOCHS}"
fi
echo "TARGETS=qkvo  LAYERS=all  RANK=$RANK  ALPHA=$ALPHA  LEARNING_RATE=$LEARNING_RATE"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "DATASET_PATH=$DATASET_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"

CMD_ARGS=(
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  --config-name="$CONFIG_NAME"
  model.checkpoint_dir="$CHECKPOINT_DIR"
  # --- fixed LoRA injection: qkvo, all layers ---
  peft.llm_lora_config.target_modules="$TARGET_MODULES"
  peft.llm_lora_config.r="$RANK"
  peft.llm_lora_config.lora_alpha="$ALPHA"
  engine.optimization.llm.learning_rate="$LEARNING_RATE"
  # -----------------------------------------------
  # fixed_prompt resolves en/de/zh to their canonical IDs in every split;
  # diverse_train (multilingual only) randomizes the training prompt and keeps
  # the canonical one for val/test.
  prompt="$PROMPT_CONFIG"
  engine.training_args.output_dir="$OUTPUT_DIR"
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  data.data_root="$DATASET_PATH"
)

# The multilingual data group has no `language` key -- the dataset yields all
# three targets -- so this override belongs to the single-language recipe only.
if [[ "$MULTILANG" != true ]]; then
  CMD_ARGS+=(data.language="$TARGET_LANGUAGE")
fi

if [[ "$STABLE_ADAMW" == true ]]; then
  # Per-tensor update clipping (Wortsman et al. 2023); `++` because base.yaml has no optim key.
  CMD_ARGS+=("++engine.training_args.optim=stable_adamw")
  echo "OPTIMIZER = stable_adamw"
fi

if [[ "$SPIKE_SKIP" == true ]]; then
  # Skip the optimizer step when the pre-clip grad norm exceeds this multiple of
  # the recent median (csi_slt.engine.sft.spike_guard); `++` because base.yaml
  # has no key.
  CMD_ARGS+=("++engine.training_args.spike_skip_factor=$SPIKE_SKIP_FACTOR")
  echo "SPIKE_SKIP_FACTOR = $SPIKE_SKIP_FACTOR"
fi

# Optional longer/shorter run: override only when an epoch count was passed,
# otherwise the config keeps its per-mode default (12 single / 10 multi).
if [[ -n "$NUM_TRAIN_EPOCHS" ]]; then
  CMD_ARGS+=(engine.training_args.num_train_epochs="$NUM_TRAIN_EPOCHS")
  echo "NUM_TRAIN_EPOCHS override = $NUM_TRAIN_EPOCHS"
fi

accelerate launch "${CMD_ARGS[@]}"
