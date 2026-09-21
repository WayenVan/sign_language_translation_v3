#! /bin/bash
#
# Usage:
#   bash scripts/ph14t/eval/run_eval.sh <CKPT_DIR> multi                 # de+en+zh, canonical prompts
#   bash scripts/ph14t/eval/run_eval.sh <CKPT_DIR> de                    # one target language
#   bash scripts/ph14t/eval/run_eval.sh <CKPT_DIR> multi share           # in-repo dataset, no staging
#   bash scripts/ph14t/eval/run_eval.sh <CKPT_DIR> multi dry-run         # print the command only
#
# <CKPT_DIR> is required and always comes first. After it, arguments are
# order-free keywords: de|en|zh|multi (required), share, dry-run.
#
# Environment:
#   EVAL_OUTPUT_DIR     where predictions land (default: see OUTPUT_DIR below)
#   EVAL_PROMPT_BANK    prompt bank JSONL (default: configs/prompt/eval_fixed.yaml's)
#   EVAL_PROMPT_IDS     "de=<id>,en=<id>,zh=<id>"; default: the canonical prompts
#   EVAL_NUM_PROCESSES  GPUs for this run (default 2)
#   EVAL_NUM_WORKERS    dataloader workers (default 6)
#   EVAL_BATCH_SIZE     per-device eval batch size (default 1)
#   EVAL_MAX_NEW_TOKENS generation cap (default: eval/base's 128)
#   EVAL_MODEL_DTYPE    覆盖 engine.model_dtype（默认 checkpoint：按 ckpt 存储的 dtype 载入）
#   EVAL_DATAMODULE     datamodule group (default standard: the official test split)
#   EVAL_EXTRA_ARGS     extra Hydra overrides, whitespace separated
#   EVAL_INHERIT_PROCESSOR=0  do not replay the checkpoint's processor settings
#   FORCE=1             re-run even if predictions_metrics.json already exists
#   FORCE_LANG=1        evaluate a language the checkpoint was not trained on
#
# Help: scripts/ph14t/eval/run_eval.sh --help
#
# One checkpoint evaluation = one prompt condition = one output directory.
# This is the parameterized worker behind scripts/ph14t/eval/run_eval_*_prompt.sh;
# those pass EVAL_PROMPT_BANK / EVAL_PROMPT_IDS / EVAL_OUTPUT_DIR per prompt
# variant. Evaluation runs locally (no Slurm anywhere in scripts/ph14t/eval).

set -euo pipefail

usage() {
  sed -n '/^# Usage:$/,/^# Help:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Host allowlist, same contract as scripts/ph14t/run_llm_lora_inject.sh: an
# unrecognized host is a hard stop instead of a half-run in a path that does
# not exist. IS_TUBBS also decides dataset staging and NCCL P2P.
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

TARGET_LANGUAGE=
MULTILANG=false
SHARED_DATASET=false
DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
  de | en | zh) TARGET_LANGUAGE="$arg" ;;
  multi | multilang) MULTILANG=true ;;
  share) SHARED_DATASET=true ;;
  dry-run | --dry-run) DRY_RUN=true ;;
  *)
    echo "Unknown argument: $arg (supported: de, en, zh, multi, share, dry-run)" >&2
    exit 2
    ;;
  esac
done

if [[ "$MULTILANG" == true && -n "$TARGET_LANGUAGE" ]]; then
  echo "'multi' evaluates de+en+zh jointly; do not also pass a single language ($TARGET_LANGUAGE)." >&2
  exit 2
fi
if [[ "$MULTILANG" != true && -z "$TARGET_LANGUAGE" ]]; then
  echo "pass the evaluation languages: de, en, zh, or multi." >&2
  exit 2
fi

if [[ "$MULTILANG" == true ]]; then
  LANG_TAG=multi
  DATA_GROUP='ph14t_*x224x224_qwen_multiling'
else
  LANG_TAG="$TARGET_LANGUAGE"
  DATA_GROUP='ph14t_*x224x224_qwen_single_language'
fi

# --------------------------------------------------------------------------- #
# Replay the checkpoint's own preprocessing.
#
# Evaluation must feed the model exactly what training did: the older
# scripts/ph14t/run_cognition_eval.sh carries a hand-copied
# `do_normalize=False # NOTE: 很重要！！！`, and copying that wrong costs BLEU
# silently. hydra_config.yaml is the fully-resolved stage-1/stage-2 config
# Hydra writes next to every checkpoint, so every `data.processor.*` leaf is
# read back from it and re-applied with `++` (the eval data group may not
# declare a given key at all, e.g. Gemma's video_start_token).
#
# The data *group* still follows this run's request, not the checkpoint: a
# multilingual checkpoint may legitimately be evaluated on one language, which
# is the single-language view of the same data. Only the processor is inherited.
# --------------------------------------------------------------------------- #
CKPT_LANGUAGE=unknown
PROCESSOR_OVERRIDES=()
HYDRA_CONFIG_PATH="$CHECKPOINT_DIR/hydra_config.yaml"
if [[ ! -f "$HYDRA_CONFIG_PATH" ]]; then
  echo "WARNING: no hydra_config.yaml at $CHECKPOINT_DIR; cannot verify the checkpoint's training language or replay its processor settings." >&2
elif [[ "${EVAL_INHERIT_PROCESSOR:-1}" != "1" ]]; then
  echo "EVAL_INHERIT_PROCESSOR=0: using $DATA_GROUP's own processor defaults."
else
  CKPT_INFO="$(python3 - "$HYDRA_CONFIG_PATH" <<'PYEOF'
import re
import sys

import yaml

_BARE = re.compile(r"^[A-Za-z0-9_./*+-]+$")


def render(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(render(item) for item in value) + "]"
    if isinstance(value, str):
        if "'" in value:
            raise ValueError(f"cannot pass {value!r} as a Hydra override")
        return value if _BARE.match(value) else f"'{value}'"
    raise TypeError(f"unsupported config value: {value!r}")


def emit(prefix, node):
    for key, value in node.items():
        path = f"{prefix}.{key}"
        if isinstance(value, dict):
            emit(path, value)
        else:
            print(f"override=++{path}={render(value)}")


with open(sys.argv[1]) as handle:
    cfg = yaml.safe_load(handle) or {}

data = cfg.get("data") or {}
language = data.get("language")
print(f"ckpt_language={language if language else 'multilang'}")

processor = data.get("processor")
if isinstance(processor, dict):
    emit("data.processor", processor)
else:
    # Pre-v4 configs kept a processor per split; nothing to inherit here.
    print("inherit=unavailable")

ctc_tokenizer_dir = data.get("ctc_tokenizer_dir")
if ctc_tokenizer_dir:
    print(f"override=++data.ctc_tokenizer_dir={render(ctc_tokenizer_dir)}")
PYEOF
  )"
  while IFS= read -r line; do
    case "$line" in
    ckpt_language=*) CKPT_LANGUAGE="${line#ckpt_language=}" ;;
    override=*) PROCESSOR_OVERRIDES+=("${line#override=}") ;;
    inherit=unavailable)
      echo "WARNING: $HYDRA_CONFIG_PATH has no data.processor block; using $DATA_GROUP's own processor defaults." >&2
      ;;
    esac
  done <<<"$CKPT_INFO"
fi

# A single-language checkpoint only ever saw its own target language; scoring it
# on another one measures nothing. A multilingual checkpoint may be evaluated
# on all three or on any single one.
if [[ "$CKPT_LANGUAGE" != unknown && "$CKPT_LANGUAGE" != multilang ]]; then
  if [[ "$LANG_TAG" != "$CKPT_LANGUAGE" ]]; then
    if [[ "${FORCE_LANG:-0}" == "1" ]]; then
      echo "WARNING: FORCE_LANG=1; evaluating a $CKPT_LANGUAGE-only checkpoint on $LANG_TAG." >&2
    else
      echo "checkpoint/language mismatch: $HYDRA_CONFIG_PATH was trained with language=$CKPT_LANGUAGE, but this run requested $LANG_TAG. Set FORCE_LANG=1 to override." >&2
      exit 5
    fi
  fi
fi

CKPT_STEP="$(basename "$CHECKPOINT_DIR")"
CKPT_RUN_NAME="$(basename "$(dirname "$CHECKPOINT_DIR")")"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${EVAL_OUTPUT_ROOT:-outputs/eval}/${CKPT_RUN_NAME}/${CKPT_STEP}/manual/${LANG_TAG}}"

# The suite launchers loop over prompt variants and re-invoke this script, so a
# finished variant is skipped rather than recomputed.
if [[ -f "$OUTPUT_DIR/predictions_metrics.json" && "${FORCE:-0}" != "1" ]]; then
  echo "SKIP: $OUTPUT_DIR already holds predictions_metrics.json (FORCE=1 to re-run)."
  exit 0
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
# dry-run 只打印命令，不值得为此复制一遍数据集，因此直接用共享路径。
if [[ "$IS_TUBBS" == true || "$SHARED_DATASET" == true || "$DRY_RUN" == true ]]; then
  DATASET_PATH="$SCRIPT_DIR/dataset/PHOENIX-2014-T-release-v3"
else
  source "$SCRIPT_DIR/scripts/ph14t/prepare_dataset.sh"
  DATASET_PATH=$(prepare_dataset \
    "$SCRIPT_DIR/dataset/phoenix-2014-T.v3.tar.gz" \
    "$HOME/localscratch/ph14t")
fi

read -r -a HYDRA_EXTRA_ARGS <<<"${EVAL_EXTRA_ARGS:-}"

CMD_ARGS=(
  "--num_processes=${EVAL_NUM_PROCESSES:-2}"
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.evaluate
  --config-name eval/base
  model.checkpoint_dir="$CHECKPOINT_DIR"
  prompt=eval_fixed
  "datamodule=${EVAL_DATAMODULE:-standard}"
  data="$DATA_GROUP"
  data.data_root="$DATASET_PATH"
  engine.training_args.output_dir="$OUTPUT_DIR"
  "engine.training_args.per_device_eval_batch_size=${EVAL_BATCH_SIZE:-1}"
  "engine.training_args.dataloader_num_workers=${EVAL_NUM_WORKERS:-6}"
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to=none
)

if [[ "$MULTILANG" != true ]]; then
  CMD_ARGS+=(data.language="$TARGET_LANGUAGE")
fi

if ((${#PROCESSOR_OVERRIDES[@]} > 0)); then
  CMD_ARGS+=("${PROCESSOR_OVERRIDES[@]}")
fi

if [[ -n "${EVAL_PROMPT_BANK:-}" ]]; then
  CMD_ARGS+=(prompt.test.sampler.prompt_paths="$EVAL_PROMPT_BANK")
fi

# "de=<id>,en=<id>,zh=<id>" -> one override per language. Omitting it keeps
# configs/prompt/eval_fixed.yaml's canonical IDs.
if [[ -n "${EVAL_PROMPT_IDS:-}" ]]; then
  IFS=',' read -r -a PROMPT_ID_PAIRS <<<"$EVAL_PROMPT_IDS"
  for pair in "${PROMPT_ID_PAIRS[@]}"; do
    pair="${pair// /}"
    [[ -z "$pair" ]] && continue
    if [[ ! "$pair" =~ ^(de|en|zh)=[A-Za-z0-9_]+$ ]]; then
      echo "EVAL_PROMPT_IDS entries must look like de=canonical_en_de_001, got: $pair" >&2
      exit 2
    fi
    CMD_ARGS+=("prompt.test.prompt_ids.${pair%%=*}=${pair#*=}")
  done
fi

if [[ -n "${EVAL_MAX_NEW_TOKENS:-}" ]]; then
  CMD_ARGS+=("engine.generation_config.max_new_tokens=$EVAL_MAX_NEW_TOKENS")
fi

if [[ -n "${EVAL_MODEL_DTYPE:-}" ]]; then
  CMD_ARGS+=("engine.model_dtype=$EVAL_MODEL_DTYPE")
fi

if ((${#HYDRA_EXTRA_ARGS[@]} > 0)); then
  CMD_ARGS+=("${HYDRA_EXTRA_ARGS[@]}")
fi

echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "CKPT_TRAINING_LANGUAGE=$CKPT_LANGUAGE  EVAL_LANGUAGES=$LANG_TAG  DATA_GROUP=$DATA_GROUP"
echo "PROMPT_BANK=${EVAL_PROMPT_BANK:-configs/prompt/eval_fixed.yaml default}"
echo "PROMPT_IDS=${EVAL_PROMPT_IDS:-canonical (config default)}"
echo "DATASET_PATH=$DATASET_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "INHERITED_PROCESSOR_OVERRIDES=${#PROCESSOR_OVERRIDES[@]}"
echo "MODEL_DTYPE=${EVAL_MODEL_DTYPE:-checkpoint (eval/base default)}"

if [[ "$DRY_RUN" == true ]]; then
  printf 'DRY RUN: accelerate launch '
  printf '%q ' "${CMD_ARGS[@]}"
  printf '\n'
  exit 0
fi

accelerate launch "${CMD_ARGS[@]}"
