#! /bin/bash
#
# Usage:
#   bash scripts/eval/sweep14b/run_multilang_diverse_sweep.sh            # 每个 run 里最好的 ckpt
#   bash scripts/eval/sweep14b/run_multilang_diverse_sweep.sh share      # 用仓库内数据集
#   bash scripts/eval/sweep14b/run_multilang_diverse_sweep.sh dry-run    # 只打印命令
#
# Environment:
#   CHECKPOINT_STEPS="38064 49776"   手动指定 step，跳过自动选择
#   ALL_CHECKPOINTS=1                每个 run 目录下的全部 checkpoint 都跑
#   SWEEP_OUTPUT_ROOT                默认 outputs/eval/14b-multilang-diverse-eval
#   PROMPT_VARIANTS / FORCE / EVAL_NUM_PROCESSES 等透传给 scripts/eval/run_eval.sh
#
# Help: scripts/eval/sweep14b/run_multilang_diverse_sweep.sh --help
#
# 对两个 14B 多语言 stage-2 LoRA run（fixed-prompt 训练的和 diverse-prompt 训练的）
# 各取其中最好的一个 checkpoint，跑一遍 diverse prompt suite：
# canonical_001 + diverse_001..007，共 8 个 prompt 变体，三种语言在同一变体内一起评。
#
# "最好" = 验证集 eval_overall_macro_bleu4 最高的 checkpoint。分数从 trainer_state.json
# 的 log_history 读（只有最新那个 checkpoint 带完整历史），并且只在磁盘上仍存在的
# checkpoint 里挑——save_total_limit=2，每个 run 只剩两个。注意训练时
# metric_for_best_model 用的是 eval_overall_weighted_bleu4 而不是 macro；两者在这里恰好
# 一致（三种语言样本数都是 642），不一致时脚本会告警并仍按 macro 选。
#
#   outputs/eval/14b-multilang-diverse-eval/<run>/<checkpoint-step>/diverse/
#       <变体>/predictions.jsonl …   summary.json / summary.md
#
# 单个 checkpoint 失败不会中断整个 sweep；已经有 predictions_metrics.json 的变体默认跳过
# （FORCE=1 重跑），所以中途断了直接重跑这个脚本即可续上。细节见 .ai/eval.md。

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

SUITE_SCRIPT="$PROJECT_DIR/scripts/eval/run_eval_diverse_prompt.sh"
if [[ ! -f "$SUITE_SCRIPT" ]]; then
  echo "diverse suite launcher not found: $SUITE_SCRIPT" >&2
  exit 3
fi

CKPT_ROOT="$PROJECT_DIR/outputs/v5.0-14b-final-ckpts"
RUN_DIRS=(
  "$CKPT_ROOT/llmlora-v5.0-qwen3-14b-cradio-l-nextframe-handroi-cls-31m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-ol-8-ep40-multilang-fixed-0911.224x224-checkpoint-126000-multilang-qkvo-r768a1536-ep11-fixed"
  "$CKPT_ROOT/llmlora-v5.0-qwen3-14b-cradio-l-nextframe-handroi-cls-31m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-ol-8-ep40-multilang-diverse-0911.224x224-checkpoint-180000-multilang-qkvo-r768a1536-ep11-diverse"
)

SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-outputs/eval/14b-multilang-diverse-eval}"

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

# 选 checkpoint：默认每个 run 取 eval_overall_macro_bleu4 最高的那个；
# CHECKPOINT_STEPS 手动指定；ALL_CHECKPOINTS=1 全跑。
CHECKPOINTS=()
for run_dir in "${RUN_DIRS[@]}"; do
  if [[ ! -d "$run_dir" ]]; then
    echo "run directory not found: $run_dir" >&2
    exit 3
  fi

  if [[ -n "${CHECKPOINT_STEPS:-}" || "${ALL_CHECKPOINTS:-0}" == "1" ]]; then
    found=0
    while IFS= read -r checkpoint_dir; do
      step="$(basename "$checkpoint_dir")"
      step="${step#checkpoint-}"
      if [[ -n "${CHECKPOINT_STEPS:-}" && " $CHECKPOINT_STEPS " != *" $step "* ]]; then
        continue
      fi
      CHECKPOINTS+=("$checkpoint_dir")
      found=$((found + 1))
    done < <(find "$run_dir" -maxdepth 1 -type d -name 'checkpoint-*' -print |
      sort -t- -k2 -n)
    if ((found == 0)); then
      echo "WARNING: no checkpoint selected under $run_dir" >&2
    fi
    continue
  fi

  selected="$("$PROJECT_DIR/.venv/bin/python" - "$run_dir" <<'PYEOF'
import json
import sys
from pathlib import Path

METRIC = "eval_overall_macro_bleu4"


def step_of(path: Path) -> int:
    return int(path.name.split("-")[1])


run_dir = Path(sys.argv[1])
checkpoints = sorted(
    (path for path in run_dir.glob("checkpoint-*") if path.is_dir()), key=step_of
)
if not checkpoints:
    sys.exit(f"no checkpoint-* directory under {run_dir}")

# Only the newest checkpoint carries the full log_history.
state = json.loads((checkpoints[-1] / "trainer_state.json").read_text())
scores = {
    entry["step"]: entry[METRIC]
    for entry in state.get("log_history", [])
    if METRIC in entry
}

scored = [
    (scores[step_of(path)], step_of(path), path)
    for path in checkpoints
    if step_of(path) in scores
]
if not scored:
    sys.exit(
        f"{run_dir}: no {METRIC} recorded for any surviving checkpoint "
        f"({', '.join(path.name for path in checkpoints)})"
    )

# Ties go to the later step.
score, step, path = max(scored, key=lambda item: (item[0], item[1]))

tracked = state.get("best_model_checkpoint") or ""
if tracked and Path(tracked).name != path.name:
    # Training selected on eval_overall_weighted_bleu4; this script selects on
    # macro, so the two can disagree when the language groups differ in size.
    print(
        f"WARNING: {run_dir.name}: {METRIC} picks {path.name}, but training "
        f"tracked {Path(tracked).name} (best_metric={state.get('best_metric')!r}); "
        f"going with {METRIC}.",
        file=sys.stderr,
    )

candidates = " ".join(f"{item[1]}:{item[0]:.6f}" for item in sorted(scored, key=lambda i: i[1]))
print(f"{path}\t{score:.6f}\t{candidates}")
PYEOF
  )"
  checkpoint_dir="${selected%%$'\t'*}"
  rest="${selected#*$'\t'}"
  echo "$(basename "$run_dir")"
  echo "  best by eval_overall_macro_bleu4: $(basename "$checkpoint_dir") (${rest%%$'\t'*})"
  echo "  candidates (step:macro_bleu4): ${rest#*$'\t'}"
  CHECKPOINTS+=("$checkpoint_dir")
done
echo

if ((${#CHECKPOINTS[@]} == 0)); then
  echo "nothing to evaluate: no checkpoint matched CHECKPOINT_STEPS='${CHECKPOINT_STEPS:-}'" >&2
  exit 2
fi

echo "Sweep: 14B multilingual, diverse prompt suite (8 variants each)"
echo "Checkpoints: ${#CHECKPOINTS[@]} -> ${#CHECKPOINTS[@]} suite runs, $((8 * ${#CHECKPOINTS[@]})) evaluations"
echo "Output root: $SWEEP_OUTPUT_ROOT"
echo "Suite arguments: ${SUITE_ARGS[*]:-none}"
for checkpoint_dir in "${CHECKPOINTS[@]}"; do
  echo "  $(basename "$(dirname "$checkpoint_dir")")/$(basename "$checkpoint_dir")"
done
echo

FAILED=()
for checkpoint_dir in "${CHECKPOINTS[@]}"; do
  echo "=========================================================================="
  echo "$(basename "$(dirname "$checkpoint_dir")")/$(basename "$checkpoint_dir")"
  echo "=========================================================================="
  status=0
  EVAL_OUTPUT_ROOT="$SWEEP_OUTPUT_ROOT" \
    bash "$SUITE_SCRIPT" "$checkpoint_dir" "${SUITE_ARGS[@]+"${SUITE_ARGS[@]}"}" || status=$?
  if ((status != 0)); then
    FAILED+=("$(basename "$(dirname "$checkpoint_dir")")/$(basename "$checkpoint_dir")")
    echo "FAILED (exit $status); continuing with the remaining checkpoints." >&2
  fi
  echo
done

echo "Summaries:"
for checkpoint_dir in "${CHECKPOINTS[@]}"; do
  echo "  $SWEEP_OUTPUT_ROOT/$(basename "$(dirname "$checkpoint_dir")")/$(basename "$checkpoint_dir")/diverse/summary.md"
done

if ((${#FAILED[@]} > 0)); then
  echo
  echo "${#FAILED[@]} of ${#CHECKPOINTS[@]} checkpoints failed:" >&2
  printf '  %s\n' "${FAILED[@]}" >&2
  exit 1
fi
