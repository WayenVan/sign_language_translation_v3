#! /bin/bash
#
# Usage:
#   bash scripts/eval/sweep4b/run_multilang_counterfactual_sweep.sh            # 已跑完 diverse 的每个 ckpt
#   bash scripts/eval/sweep4b/run_multilang_counterfactual_sweep.sh share      # 用仓库内数据集
#   bash scripts/eval/sweep4b/run_multilang_counterfactual_sweep.sh dry-run    # 只打印命令
#
# Environment:
#   CHECKPOINT_STEPS="37254"         只跑这些 step
#   SWEEP_OUTPUT_ROOT                默认 outputs/eval/4b-multilang-diverse-eval
#   SUMMARY_SAMPLES=3                每个条件在报告里引用几条失败输出
#   FORCE_LANGID=1                   重算缓存的输出语言判定
#   FORCE / EVAL_NUM_PROCESSES / EVAL_LANGUAGE_DISTRIBUTION 等透传给 suite 与 scripts/eval/run_eval.sh
#
# Help: scripts/eval/sweep4b/run_multilang_counterfactual_sweep.sh --help
#
# 反事实指令切换实验（.ai/experiment_plan.md）：对 diverse sweep 已经跑完的每个
# checkpoint 补跑两个反事实 suite，再把它们汇总成一份报告。
#
#   cf_first  cf_first_001..002（2 个变体）  Translate the signing into {目标}, not {干扰项}.
#   cf_last   cf_last_001..002 （2 个变体）  Do not translate the signing into {干扰项}; use {目标}.
#
# 两个 suite 合起来是 2 模板 x 3 语言对 x 2 方向 = 12 个条件 / 视频，每个变体一次
# 完整 predict（三种语言同跑），所以每个 checkpoint 4 次 predict、7704 次生成。
#
# checkpoint 列表直接取 diverse sweep 的产出（<run>/<step>/diverse/summary.json），
# 不再重新挑"最好"的，保证跟其它 suite 评的是同一批 checkpoint；主表的 canonical
# 那一列也直接读 diverse/canonical_001 已经算出来的 lacc，不重跑。
#
#   outputs/eval/4b-multilang-diverse-eval/<run>/<checkpoint-step>/
#       cf_first/ cf_last/                      两个 suite 各自的 summary.{json,md}
#       counterfactual_summary.{json,md}        LAcc / BSA / 混淆矩阵 / 主表那一行
#
# 单个 suite 失败不会中断整个 sweep，但该 checkpoint 的汇总会被跳过——BSA 必须建立在
# 完整的 12 个条件上，缺一个方向就没有意义。已经有 predictions_metrics.json 的变体默认
# 跳过（FORCE=1 重跑），中途断了直接重跑这个脚本即可续上。细节见 .ai/eval.md。

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

# 两个 suite 都是汇总的必需输入，所以这里不开放 SUITES 覆盖。
SUITES=(cf_first cf_last)
VARIANTS_PER_SUITE=2

for suite in "${SUITES[@]}"; do
  if [[ ! -f "$PROJECT_DIR/scripts/eval/run_eval_${suite}_prompt.sh" ]]; then
    echo "suite launcher not found: $PROJECT_DIR/scripts/eval/run_eval_${suite}_prompt.sh" >&2
    exit 3
  fi
done

SUITE_ARGS=()
DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
  share) SUITE_ARGS+=("$arg") ;;
  dry-run | --dry-run)
    SUITE_ARGS+=("$arg")
    DRY_RUN=true
    ;;
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

total_variants=$((${#SUITES[@]} * VARIANTS_PER_SUITE))
echo "Sweep: 4B multilingual, counterfactual instruction switching (${SUITES[*]})"
echo "Checkpoints: ${#CHECKPOINTS[@]} -> $((${#SUITES[@]} * ${#CHECKPOINTS[@]})) suite runs, $((total_variants * ${#CHECKPOINTS[@]})) evaluations"
echo "Conditions: 2 templates x 3 language pairs x 2 directions = 12 per video"
echo "Output root: $SWEEP_OUTPUT_ROOT"
echo "Suite arguments: ${SUITE_ARGS[*]:-none}"
for checkpoint_dir in "${CHECKPOINTS[@]}"; do
  echo "  $(basename "$(dirname "$checkpoint_dir")")/$(basename "$checkpoint_dir")"
done
echo

SUMMARIZE_ARGS=(--samples "${SUMMARY_SAMPLES:-3}")
[[ "${FORCE_LANGID:-0}" == "1" ]] && SUMMARIZE_ARGS+=(--force-langid)

FAILED=()
SUMMARIZED=()
for checkpoint_dir in "${CHECKPOINTS[@]}"; do
  run="$(basename "$(dirname "$checkpoint_dir")")"
  step="$(basename "$checkpoint_dir")"
  ckpt_label="$run/$step"
  suite_failed=0
  for suite in "${SUITES[@]}"; do
    echo "=========================================================================="
    echo "$ckpt_label  [$suite]"
    echo "=========================================================================="
    status=0
    EVAL_OUTPUT_ROOT="$SWEEP_OUTPUT_ROOT" \
      bash "$PROJECT_DIR/scripts/eval/run_eval_${suite}_prompt.sh" "$checkpoint_dir" \
      "${SUITE_ARGS[@]+"${SUITE_ARGS[@]}"}" || status=$?
    if ((status != 0)); then
      FAILED+=("$ckpt_label [$suite]")
      suite_failed=$((suite_failed + 1))
      echo "FAILED (exit $status); continuing with the remaining suites." >&2
    fi
    echo
  done

  if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run: would summarize $sweep_root_abs/$ckpt_label"
    printf '  python -m csi_slt.commands.summarize_counterfactual %q' "$SWEEP_OUTPUT_ROOT/$ckpt_label"
    printf ' %q' "${SUMMARIZE_ARGS[@]}"
    printf '\n\n'
    continue
  fi

  # BSA 必须建立在完整的 12 个条件上；缺一个方向的汇总不是"少一点数据"，而是错的。
  if ((suite_failed > 0)); then
    echo "SKIP summary for $ckpt_label: $suite_failed suite(s) failed." >&2
    echo
    continue
  fi

  echo "--- summarizing $ckpt_label"
  status=0
  PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PROJECT_DIR/.venv/bin/python" -m csi_slt.commands.summarize_counterfactual \
    "$SWEEP_OUTPUT_ROOT/$ckpt_label" "${SUMMARIZE_ARGS[@]}" || status=$?
  if ((status != 0)); then
    FAILED+=("$ckpt_label [summary]")
    echo "FAILED (exit $status)" >&2
  else
    SUMMARIZED+=("$ckpt_label")
  fi
  echo
done

if ((${#SUMMARIZED[@]} > 0)); then
  echo "Summaries:"
  for ckpt_label in "${SUMMARIZED[@]}"; do
    echo "  $SWEEP_OUTPUT_ROOT/$ckpt_label/counterfactual_summary.md"
  done
fi

if ((${#FAILED[@]} > 0)); then
  echo
  echo "${#FAILED[@]} step(s) failed:" >&2
  printf '  %s\n' "${FAILED[@]}" >&2
  exit 1
fi
