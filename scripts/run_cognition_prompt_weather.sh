#! /bin/bash

#SBATCH --job-name=slt_qwen3_4b_nextframe_handroi_cls_20m_projdrop05_prompt_weather
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=21
#SBATCH --mem=256g

set -euo pipefail

export NCCL_P2P_DISABLE=1 # NOTE: 测试的时候集群通信容易出问题 集群出现了问题

SCRIPT_DIR=/users/2533494w/projects/sign_language_translation_v3

cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"

export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# 可选参数：debug 关闭 WandB，share 直接使用共享数据集。
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
  echo "Debug mode: Disabling reporting to WandB, outputs go to outputs/debug."
  REPORT_TO=none
  OUTPUT_DIR="outputs/debug"
else
  export WANDB_PROJECT=sign_language_translation_v5.0-dev
  export WANDB_TAGS="next-frame,hand-roi,cls,20m,fixed-prompt,proj-dropout,prompt-weather"
  REPORT_TO=wandb
  OUTPUT_DIR="outputs/v5.0-qwen3-4b-cradio-l-nextframe-handroi-cls-20m-gate1-hardmatch-wr3-projdrop0.5-prompt-weather-0906.224x224"
fi

# 设置 TQDM_DISABLE 和 HG_TQDM_DISABLE 用于 accelerate launch
if [[ -t 2 ]]; then
  unset TQDM_DISABLE
  HG_TQDM_DISABLE=False
else
  export TQDM_DISABLE=1
  HG_TQDM_DISABLE=True
fi

# share 模式下直接使用共享路径，否则准备数据集到本地 scratch。
if [[ "$SHARED_DATASET" == true ]]; then
  DATASET_PATH="$SCRIPT_DIR/dataset/PHOENIX-2014-T-release-v3"
else
  source "$SCRIPT_DIR/scripts/prepare_dataset.sh"
  DATASET_PATH=$(prepare_dataset \
    "$SCRIPT_DIR/dataset/phoenix-2014-T.v3.tar.gz" \
    "$HOME/localscratch/ph14t")
fi
echo "DATASET_PATH=$DATASET_PATH"

# Array form instead of backslash-continued lines: inside (...) each element
# can live on its own line and be commented out individually with a leading
# "#" without breaking the rest of the command (a "#" on a "\"-continued
# line eats that line's trailing backslash too and splits the command).
CMD_ARGS=(
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  # Weather-domain prompt ablation on top of the best adapter.
  #
  # Reference run (the one this branches from, currently training):
  #   outputs/v5.0-qwen3-4b-cradio-l-nextframe-handroi-cls-20m-gate1-hardmatch-wr3-projdrop0.5-0905
  #   = 表 B 行 3 + 行 10 + 行 2 + 行 9 all combined
  #     (next-frame fusion + gated hand-ROI residual + gated CLS residual +
  #      0.5 projection dropout on every branch), from run_cognition_new.sh.
  #
  # The ONLY change here is prompt=fixed_prompt_weather. baseline_ablation
  # defaults to prompt=fixed_prompt (the canonical, domain-agnostic template
  # from prompts/generic/train.jsonl); this swaps in the identically-id'd bank
  # prompts/weather/train.jsonl, whose templates name the PHOENIX-2014-T
  # weather-forecast domain ("signed weather forecast" instead of "the
  # signing"). Same canonical id at train / val / test, so it is a clean A/B
  # against the reference: does telling the frozen LLM the domain up front buy
  # anything.
  #
  # Every other flag is copied verbatim from run_cognition_new.sh.
  #
  # Adapter dropout needs engine.trainability.visual_adapter.runtime_mode =
  # train, which baseline_ablation inherits from base.yaml, or the three
  # projection dropouts silently stay off in eval.
  --config-name=train/pretrain_adapter/baseline_ablation
  model=qwen3-4b-cradio-l-spatiotemporal-next-frame-handroi-cls-20m
  model.config.visual_adapter_kwargs.projection_dropout=0.5
  model.config.visual_adapter_kwargs.roi_projection_dropout=0.5
  model.config.visual_adapter_kwargs.cls_projection_dropout=0.5
  # The ablation variable.
  prompt=fixed_prompt_weather
  # Temporary: run the three scalar gates (patch-fusion / roi / cls, exposed by
  # the adapter's optimization_parameter_groups() as the "gates" group) 10x
  # above the adapter's own rate so they can move to their equilibrium before
  # the projections lock in. "+" because engine.optimization is an empty {} in
  # baseline_ablation and struct mode rejects a plain deep-key add.
  +engine.optimization.visual_adapter.parameter_groups.gates.learning_rate=1e-3
  engine.training_args.output_dir="$OUTPUT_DIR"
  engine.training_args.disable_tqdm="$HG_TQDM_DISABLE"
  engine.training_args.report_to="$REPORT_TO"
  data.data_root="$DATASET_PATH"
)

accelerate launch "${CMD_ARGS[@]}"
