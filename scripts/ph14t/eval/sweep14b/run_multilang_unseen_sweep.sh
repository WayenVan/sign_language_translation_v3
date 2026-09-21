#! /bin/bash
#
# Usage:
#   bash scripts/ph14t/eval/sweep14b/run_multilang_unseen_sweep.sh            # 已跑完 diverse 的每个 ckpt
#   bash scripts/ph14t/eval/sweep14b/run_multilang_unseen_sweep.sh share      # 用仓库内数据集
#   bash scripts/ph14t/eval/sweep14b/run_multilang_unseen_sweep.sh dry-run    # 只打印命令
#
# Environment:
#   CHECKPOINT_STEPS="38064"         只跑这些 step
#   SWEEP_OUTPUT_ROOT                默认 outputs/eval/14b-multilang-diverse-eval
#   FORCE=1                          重跑已经有 predictions_metrics.json 的变体
#   EVAL_NUM_PROCESSES / EVAL_LANGUAGE_DISTRIBUTION 等透传给 suite 与 scripts/ph14t/eval/run_eval.sh
#
# Help: scripts/ph14t/eval/sweep14b/run_multilang_unseen_sweep.sh --help
#
# run_multilang_remaining_sweep.sh 里 unseen 那一个 suite 单拿出来重跑用：
# prompts/generic/heldout.jsonl 从 4 个模板扩到 8 个（4 类句式 x 2）之后，
# wrong_task / unrelated 的结果没有变化，没必要跟着一起重算。
#
#   unseen      heldout_001..008（8 个变体，每个变体三语同跑）
#
# checkpoint 列表跟其它 sweep 一样取 diverse sweep 的产出（<run>/<step>/diverse/
# summary.json），保证所有 suite 评的是同一批 checkpoint。结果照旧写回：
#
#   outputs/eval/14b-multilang-diverse-eval/<run>/<checkpoint-step>/unseen/
#
# 注意：已经有 predictions_metrics.json 的变体默认跳过，所以在旧 prompt bank 下跑出来的
# heldout_001..004 会被当成现成结果复用——换了 bank 之后要么先把旧的 unseen/ 目录删掉，
# 要么用 FORCE=1 全部重算。单个 checkpoint 失败不会中断整个 sweep，中途断了重跑即可续上。
# 细节见 .ai/eval.md。

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

CKPT_ROOT="$PROJECT_DIR/outputs/v5.0-14b-final-ckpts"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-outputs/eval/14b-multilang-diverse-eval}"

# 这个脚本就是为了只跑 unseen 而存在的，所以不开放 SUITES 覆盖；
# 要连 wrong_task / unrelated 一起跑就用 run_multilang_remaining_sweep.sh。
SUITE=unseen
SUITE_SCRIPT=run_eval_unseen_prompt.sh
VARIANTS_PER_SUITE=8

if [[ ! -f "$PROJECT_DIR/scripts/ph14t/eval/$SUITE_SCRIPT" ]]; then
  echo "suite launcher not found: $PROJECT_DIR/scripts/ph14t/eval/$SUITE_SCRIPT" >&2
  exit 3
fi

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
  echo "run scripts/ph14t/eval/sweep14b/run_multilang_diverse_sweep.sh first." >&2
  exit 2
fi

echo "Sweep: 14B multilingual, held-out instruction suite ($SUITE)"
echo "Checkpoints: ${#CHECKPOINTS[@]} -> ${#CHECKPOINTS[@]} suite runs, $((VARIANTS_PER_SUITE * ${#CHECKPOINTS[@]})) evaluations"
echo "Prompt bank: prompts/generic/heldout.jsonl (heldout_001..$(printf '%03d' "$VARIANTS_PER_SUITE"))"
echo "Output root: $SWEEP_OUTPUT_ROOT"
echo "Suite arguments: ${SUITE_ARGS[*]:-none}${FORCE:+   FORCE=$FORCE}"
for checkpoint_dir in "${CHECKPOINTS[@]}"; do
  echo "  $(basename "$(dirname "$checkpoint_dir")")/$(basename "$checkpoint_dir")"
done
echo

FAILED=()
for checkpoint_dir in "${CHECKPOINTS[@]}"; do
  ckpt_label="$(basename "$(dirname "$checkpoint_dir")")/$(basename "$checkpoint_dir")"
  echo "=========================================================================="
  echo "$ckpt_label  [$SUITE]"
  echo "=========================================================================="
  status=0
  EVAL_OUTPUT_ROOT="$SWEEP_OUTPUT_ROOT" \
    bash "$PROJECT_DIR/scripts/ph14t/eval/$SUITE_SCRIPT" "$checkpoint_dir" \
    "${SUITE_ARGS[@]+"${SUITE_ARGS[@]}"}" || status=$?
  if ((status != 0)); then
    FAILED+=("$ckpt_label [$SUITE]")
    echo "FAILED (exit $status); continuing with the remaining checkpoints." >&2
  fi
  echo
done

echo "Summaries:"
for checkpoint_dir in "${CHECKPOINTS[@]}"; do
  echo "  $SWEEP_OUTPUT_ROOT/$(basename "$(dirname "$checkpoint_dir")")/$(basename "$checkpoint_dir")/$SUITE/summary.md"
done

if ((${#FAILED[@]} > 0)); then
  echo
  echo "${#FAILED[@]} of ${#CHECKPOINTS[@]} suite runs failed:" >&2
  printf '  %s\n' "${FAILED[@]}" >&2
  exit 1
fi
