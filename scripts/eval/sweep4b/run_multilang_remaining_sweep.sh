#! /bin/bash
#
# Usage:
#   bash scripts/eval/sweep4b/run_multilang_remaining_sweep.sh            # 已跑完 diverse 的每个 ckpt
#   bash scripts/eval/sweep4b/run_multilang_remaining_sweep.sh share      # 用仓库内数据集
#   bash scripts/eval/sweep4b/run_multilang_remaining_sweep.sh dry-run    # 只打印命令
#
# Environment:
#   SUITES="unseen wrong_task"       只跑其中几个 suite（默认 unseen wrong_task unrelated）
#   CHECKPOINT_STEPS="37254"         只跑这些 step
#   SWEEP_OUTPUT_ROOT                默认 outputs/eval/4b-multilang-diverse-eval
#   FORCE / EVAL_NUM_PROCESSES / EVAL_LANGUAGE_DISTRIBUTION 等透传给 suite 与 scripts/eval/run_eval.sh
#
# Help: scripts/eval/sweep4b/run_multilang_remaining_sweep.sh --help
#
# run_multilang_diverse_sweep.sh 的续篇：对它已经跑完 diverse suite 的每个 checkpoint
# （SWEEP_OUTPUT_ROOT 下有 <run>/<checkpoint-step>/diverse/summary.json 的），补跑剩下的
# 三个 suite：
#
#   unseen      heldout_001..008（8 个变体）
#   wrong_task  wrong_task_001（1 个，diagnostic）
#   unrelated   unrelated_001（1 个，diagnostic）
#
# checkpoint 列表直接取 diverse sweep 的产出，不再重新挑"最好"的，保证五个 suite 评的是
# 同一批 checkpoint。结果与 diverse 放在一起：
#
#   outputs/eval/4b-multilang-diverse-eval/<run>/<checkpoint-step>/{unseen,wrong_task,unrelated}/
#
# 单个 suite 失败不会中断整个 sweep；已经有 predictions_metrics.json 的变体默认跳过
# （FORCE=1 重跑），中途断了直接重跑这个脚本即可续上。细节见 .ai/eval.md。

set -euo pipefail

usage() {
  sed -n '/^# Usage:$/,/^# Help:/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PROJECT_DIR=/users/2533494w/projects/sign_language_translation_v3
if [[ ! -d "$PROJECT_DIR" ]]; then
  PROJECT_DIR=/home/2533494W/project/sign_language_translation_v3
fi
cd "$PROJECT_DIR"

CKPT_ROOT="$PROJECT_DIR/outputs"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-outputs/eval/4b-multilang-diverse-eval}"
SUITES=(${SUITES:-unseen wrong_task unrelated})

# suite 名 -> 入口脚本 / 变体数（只用于打印总量）
declare -A SUITE_SCRIPTS=(
  [unseen]=run_eval_unseen_prompt.sh
  [wrong_task]=run_eval_wrong_task_prompt.sh
  [unrelated]=run_eval_unrelated_prompt.sh
)
declare -A SUITE_VARIANTS=([unseen]=8 [wrong_task]=1 [unrelated]=1)

total_variants=0
for suite in "${SUITES[@]}"; do
  script="${SUITE_SCRIPTS[$suite]:-}"
  if [[ -z "$script" ]]; then
    echo "Unknown suite in SUITES: $suite (supported: unseen, wrong_task, unrelated)" >&2
    exit 2
  fi
  if [[ ! -f "$PROJECT_DIR/scripts/eval/$script" ]]; then
    echo "suite launcher not found: $PROJECT_DIR/scripts/eval/$script" >&2
    exit 3
  fi
  total_variants=$((total_variants + SUITE_VARIANTS[$suite]))
done

SUITE_ARGS=()
for arg in "$@"; do
  case "$arg" in
  share | dry-run | --dry-run) SUITE_ARGS+=("$arg") ;;
  *)
    echo "Unknown argument: $arg (supported: share, dry-run)" >&2
    exit 2
    ;;
  esac
done

if [[ "$SWEEP_OUTPUT_ROOT" == /* ]]; then
  sweep_root_abs="$SWEEP_OUTPUT_ROOT"
else
  sweep_root_abs="$PROJECT_DIR/$SWEEP_OUTPUT_ROOT"
fi

# checkpoint 列表 = diverse sweep 已经完成（有 summary.json）的那些。
CHECKPOINTS=()
while IFS= read -r summary; do
  step_dir="$(dirname "$(dirname "$summary")")"
  step="$(basename "$step_dir")"
  run="$(basename "$(dirname "$step_dir")")"
  if [[ -n "${CHECKPOINT_STEPS:-}" && " $CHECKPOINT_STEPS " != *" ${step#checkpoint-} "* ]]; then
    continue
  fi
  checkpoint_dir="$CKPT_ROOT/$run/$step"
  if [[ ! -d "$checkpoint_dir" ]]; then
    echo "WARNING: diverse results exist for $run/$step but the checkpoint is gone: $checkpoint_dir" >&2
    continue
  fi
  CHECKPOINTS+=("$checkpoint_dir")
done < <(find "$sweep_root_abs" -mindepth 4 -maxdepth 4 -path '*/diverse/summary.json' 2>/dev/null | sort)

if ((${#CHECKPOINTS[@]} == 0)); then
  echo "nothing to evaluate: no <run>/<checkpoint>/diverse/summary.json under $sweep_root_abs" \
    "${CHECKPOINT_STEPS:+matching CHECKPOINT_STEPS='$CHECKPOINT_STEPS'}" >&2
  echo "run scripts/eval/sweep4b/run_multilang_diverse_sweep.sh first." >&2
  exit 2
fi

echo "Sweep: 4B multilingual, remaining suites: ${SUITES[*]}"
echo "Checkpoints: ${#CHECKPOINTS[@]} -> $((${#SUITES[@]} * ${#CHECKPOINTS[@]})) suite runs, $((total_variants * ${#CHECKPOINTS[@]})) evaluations"
echo "Output root: $SWEEP_OUTPUT_ROOT"
echo "Suite arguments: ${SUITE_ARGS[*]:-none}"
for checkpoint_dir in "${CHECKPOINTS[@]}"; do
  echo "  $(basename "$(dirname "$checkpoint_dir")")/$(basename "$checkpoint_dir")"
done
echo

FAILED=()
for checkpoint_dir in "${CHECKPOINTS[@]}"; do
  ckpt_label="$(basename "$(dirname "$checkpoint_dir")")/$(basename "$checkpoint_dir")"
  for suite in "${SUITES[@]}"; do
    echo "=========================================================================="
    echo "$ckpt_label  [$suite]"
    echo "=========================================================================="
    status=0
    EVAL_OUTPUT_ROOT="$SWEEP_OUTPUT_ROOT" \
      bash "$PROJECT_DIR/scripts/eval/${SUITE_SCRIPTS[$suite]}" "$checkpoint_dir" \
      "${SUITE_ARGS[@]+"${SUITE_ARGS[@]}"}" || status=$?
    if ((status != 0)); then
      FAILED+=("$ckpt_label [$suite]")
      echo "FAILED (exit $status); continuing with the remaining suites." >&2
    fi
    echo
  done
done

echo "Summaries:"
for checkpoint_dir in "${CHECKPOINTS[@]}"; do
  for suite in "${SUITES[@]}"; do
    echo "  $SWEEP_OUTPUT_ROOT/$(basename "$(dirname "$checkpoint_dir")")/$(basename "$checkpoint_dir")/$suite/summary.md"
  done
done

if ((${#FAILED[@]} > 0)); then
  echo
  echo "${#FAILED[@]} of $((${#SUITES[@]} * ${#CHECKPOINTS[@]})) suite runs failed:" >&2
  printf '  %s\n' "${FAILED[@]}" >&2
  exit 1
fi
