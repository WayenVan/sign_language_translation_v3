#! /bin/bash

#SBATCH --job-name=slt_qwen3_4b_nextframe_handroi_cls_20m_projdrop05_beam4_ls
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
  export WANDB_TAGS="next-frame,hand-roi,cls,20m,fixed-prompt,proj-dropout,beam4,label-smoothing"
  REPORT_TO=wandb
  OUTPUT_DIR="outputs/v5.0-qwen3-4b-cradio-l-nextframe-handroi-cls-20m-gate1-hardmatch-wr3-projdrop0.5-beam4-ls0.1-0906.224x224"
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
  # Beam search + label smoothing on top of the best adapter (one ablation
  # group: the two are complementary -- smoothing flattens the next-token
  # distribution, which is where a beam gets most of its BLEU gain over greedy).
  #
  # Reference run (the one this branches from, currently training):
  #   outputs/v5.0-qwen3-4b-cradio-l-nextframe-handroi-cls-20m-gate1-hardmatch-wr3-projdrop0.5-0905
  #   = 表 B 行 3 + 行 10 + 行 2 + 行 9 all combined, from run_cognition_new.sh.
  #
  # Two changes vs that reference:
  #
  #   1. generation config: greedy -> num_beams=4. Does NOT touch the training
  #      objective (teacher-forced CE + CTC either way). It changes every
  #      generate() call: the periodic val / train-probe evals, the
  #      best-checkpoint selection (metric_for_best_model =
  #      eval_overall_weighted_bleu4), and the final test decode.
  #      num_beams=4 with do_sample=false (base.yaml's value, unchanged) is
  #      plain deterministic beam search -- beams are not samples.
  #      early_stopping=true stops a beam once it emits EOS; length_penalty=1.0
  #      is neutral (raise toward 1.1-1.2 if outputs come out short and
  #      BLEU-4's brevity penalty bites, lower toward 0.8-0.9 if they run long).
  #
  #   2. model.config.label_smoothing 0.0 -> 0.1. Applied inside SltModel to
  #      the LM cross-entropy only (F.cross_entropy label_smoothing arg); the
  #      CTC term is never smoothed. This DOES change the training objective,
  #      so the optimization trajectory differs from the reference. Raises the
  #      reported ce_loss by a constant -- loss curves are not comparable to
  #      the reference, BLEU / gap still are. NOTE: this is a two-variable run
  #      by design; if it wins, split it (beam-only vs ls-only) to attribute.
  #
  # Cheaper alternative for the beam half on an existing checkpoint:
  #   python -m csi_slt.commands.evaluate <ckpt> +engine.generation_config.num_beams=4
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
  # Ablation variable 1: beam search. "+" because these keys are commented out
  # in base.yaml, so they are added, not overridden.
  +engine.generation_config.num_beams=4
  +engine.generation_config.early_stopping=true
  +engine.generation_config.length_penalty=1.0
  # Ablation variable 2: label smoothing on the LM cross-entropy (0.0 -> 0.1).
  # "+" because label_smoothing is an SltConfig.__init__ kwarg (default 0.0)
  # that no configs/model/*.yaml lists under `config:`, so it must be appended,
  # not overridden. It still reaches SltConfig via the model.config mapping.
  +model.config.label_smoothing=0.1
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
