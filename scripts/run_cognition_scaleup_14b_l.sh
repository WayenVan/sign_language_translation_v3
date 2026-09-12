#! /bin/bash
#
# Usage:
#   sbatch scripts/run_cognition_scaleup_14b_l.sh                # de, fixed prompt (default)
#   sbatch scripts/run_cognition_scaleup_14b_l.sh en             # single target language: de | en | zh
#   sbatch scripts/run_cognition_scaleup_14b_l.sh multi          # multilingual joint training (de+en+zh)
#   sbatch scripts/run_cognition_scaleup_14b_l.sh multi diverse  # joint training with diverse train prompts
#   sbatch scripts/run_cognition_scaleup_14b_l.sh multi 60       # override the epoch count
#   sbatch scripts/run_cognition_scaleup_14b_l.sh de share       # shared dataset path, no scratch staging
#   sbatch scripts/run_cognition_scaleup_14b_l.sh debug          # no WandB; outputs/debug-scaleup-14b-l-<tag>
#
# Arguments are order-free keywords: de|en|zh|multi, fixed|diverse, debug, share,
# plus a bare positive integer for the epoch count (default 80 single / 40 multi).
# `diverse` is accepted only together with `multi` -- see the prompt note below.
#
# Qwen3-14B + C-RADIOv4-SO400M scale-up of the best Qwen3-4B checkpoint; see below.

#SBATCH --job-name=slt_scaleup_qwen3_14b_cradio_l
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=21
#SBATCH --mem=256g
#SBATCH --time=2-12:00:00

set -euo pipefail

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
# Scale-up run: Qwen3-14B + C-RADIOv4-SO400M (output layer -8), with the adapter and
# every training setting of the best Qwen3-4B checkpoint:
#
#   outputs/v5.0-qwen3-4b-cradio-l-nextframe-handroi-cls-20m-gate1-hardmatch-
#   wr3-projdrop0.5-posenc-learned-ol-8-ep80-0907.224x224/checkpoint-96000
#   (de, dev BLEU-4 0.1715; its en run reached 0.1843)
#
# The overrides below are scripts/run_cognition_target_language.sh's, changed
# only in the language model and its matching adapter output width. The
# SO400M output layer and scorer match the 4B reference. The LLM and visual
# backbone remain frozen; this is adapter pretraining, with no LoRA.
#
# Keep the original two-H100 allocation and 60-hour time limit. Runtime and
# peak memory for the 14B model still need to be measured.
#
# Two training modes share this recipe:
#   * single language (de | en | zh) -- baseline_ablation over the single-
#     language data group, 80 epochs, fixed prompt only;
#   * multilingual joint (multi)     -- best_adapter_multilang's data group,
#     de+en+zh in one run at a compute-matched epoch count, fixed or diverse
#     training prompts.
# Everything below -- model, scorer, dropouts, positional embedding, schedule
# cadence -- is identical across both, so the modes stay comparable.
# --------------------------------------------------------------------------- #

# de by default: the language of the 4B checkpoint this run is compared against.
# `multi` switches to multilingual joint training on de+en+zh instead.
TARGET_LANGUAGE=de
MULTILANG=false
LANG_ARG_SEEN=false
PROMPT_CONFIG=fixed_prompt
EPOCHS_OVERRIDE=
DEBUG=false
SHARED_DATASET=false
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
  [0-9]*)
    if [[ ! "$arg" =~ ^[0-9]+$ ]] || ((10#$arg < 1)); then
      echo "epochs must be a positive integer, got: $arg" >&2
      exit 2
    fi
    EPOCHS_OVERRIDE="$arg"
    ;;
  *)
    echo "Unknown argument: $arg (supported: de, en, zh, multi, fixed, diverse, debug, share, <epochs>)" >&2
    exit 2
    ;;
  esac
done

if [[ "$MULTILANG" == true && "$LANG_ARG_SEEN" == true ]]; then
  echo "'multi' trains de+en+zh jointly; do not also pass a single language ($TARGET_LANGUAGE)." >&2
  exit 2
fi

# Prompt diversity is a multilingual-only variable here. In a single-language
# run the diverse resolver would sample paraphrases of one and the same target
# instruction, which confounds the prompt with nothing and makes the run
# incomparable to the fixed-prompt 4B reference. Refuse instead of launching it.
if [[ "$PROMPT_CONFIG" == diverse_train && "$MULTILANG" != true ]]; then
  echo "diverse prompts are only allowed for multilingual joint training; pass 'multi diverse'." >&2
  exit 2
fi

if [[ "$MULTILANG" == true ]]; then
  # PH14T multilingual holds ~3 target-language examples per source video, so a
  # compute-matched equivalent of the 80 single-language epochs below is ~27;
  # 40 gives multilingual convergence clear extra room over that.
  NUM_TRAIN_EPOCHS=40
  # Recipe with the multilingual data group already wired in; the model, scorer
  # and regularization overrides below still apply on top of it.
  TRAIN_CONFIG=train/pretrain_adapter/best_adapter_multilang
  LANG_TAG=multilang
else
  NUM_TRAIN_EPOCHS=80
  TRAIN_CONFIG=train/pretrain_adapter/baseline_ablation
  LANG_TAG="$TARGET_LANGUAGE"
fi
# A bare integer argument replaces the per-mode default; it also lands in the
# output directory and WandB tags, so two epoch counts never share a run dir.
NUM_TRAIN_EPOCHS="${EPOCHS_OVERRIDE:-$NUM_TRAIN_EPOCHS}"

if [[ "$PROMPT_CONFIG" == diverse_train ]]; then
  PROMPT_TAG=diverse
else
  PROMPT_TAG=fixed
fi

# Single-language fixed-prompt runs keep their historical tag (de / en / zh) so
# their output directories stay where earlier launches put them.
if [[ "$MULTILANG" == true ]]; then
  RUN_TAG="${LANG_TAG}-${PROMPT_TAG}"
else
  RUN_TAG="${LANG_TAG}"
fi

if [[ "$DEBUG" == true ]]; then
  echo "Debug mode: Disabling reporting to WandB, outputs go to outputs/debug-scaleup-14b-l-${RUN_TAG}."
  REPORT_TO=none
  OUTPUT_DIR="outputs/debug-scaleup-14b-l-${RUN_TAG}"
else
  export WANDB_PROJECT=sign_language_translation_v5.0-dev
  export WANDB_TAGS="next-frame,hand-roi,cls,31m,qwen3-14b,cradio-l,${PROMPT_TAG}-prompt,proj-dropout,posenc-learned,output-layer--8,ep${NUM_TRAIN_EPOCHS},scale-up,${LANG_TAG}"
  REPORT_TO=wandb
  OUTPUT_DIR="outputs/v5.0-qwen3-14b-cradio-l-nextframe-handroi-cls-31m-gate1-hardmatch-wr3-projdrop0.5-posenc-learned-ol-8-ep${NUM_TRAIN_EPOCHS}-${RUN_TAG}-0911.224x224"
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
  source "$SCRIPT_DIR/scripts/prepare_dataset.sh"
  DATASET_PATH=$(prepare_dataset \
    "$SCRIPT_DIR/dataset/phoenix-2014-T.v3.tar.gz" \
    "$HOME/localscratch/ph14t")
fi
if [[ "$MULTILANG" == true ]]; then
  echo "LANGUAGES=de+en+zh (joint)  PROMPT=$PROMPT_CONFIG  EPOCHS=$NUM_TRAIN_EPOCHS"
else
  echo "TARGET_LANGUAGE=$TARGET_LANGUAGE  PROMPT=$PROMPT_CONFIG  EPOCHS=$NUM_TRAIN_EPOCHS"
fi
echo "DATASET_PATH=$DATASET_PATH  OUTPUT_DIR=$OUTPUT_DIR"

# Same SO400M L8 scorer as the best 4B checkpoint. Must match OUTPUT_LAYER:
# the adapter raises if the scorer's recorded layer disagrees with the backbone.
OUTPUT_LAYER=-8
SCORER_PATH=outputs/hand_patch_scorer_L8
if [[ ! -f "$SCRIPT_DIR/$SCORER_PATH/config.json" ]]; then
  echo "no fitted scorer at $SCORER_PATH" >&2
  exit 3
fi

CMD_ARGS=(
  --num_processes=2
  --mixed_precision=bf16
  --debug
  -m csi_slt.commands.train
  --config-name="$TRAIN_CONFIG"
  # fixed_prompt resolves en/de/zh to their canonical IDs in every split;
  # diverse_train (multilingual only) randomizes the training prompt and keeps
  # the canonical one for val/test.
  prompt="$PROMPT_CONFIG"
  # Scale-up architecture: Qwen3-14B + C-RADIOv4-SO400M, same adapter ranks (30.85M).
  model=qwen3-14b-cradio-l-spatiotemporal-next-frame-handroi-cls-31m
  model.config.visual_backbone_config.output_layer="$OUTPUT_LAYER"
  model.config.visual_adapter_kwargs.scorer_path="$SCORER_PATH"
  # Best-checkpoint regularization. The model file already defaults to these;
  # restated so the run's recipe reads off this command in full.
  model.config.visual_adapter_kwargs.projection_dropout=0.5
  model.config.visual_adapter_kwargs.roi_projection_dropout=0.5
  model.config.visual_adapter_kwargs.cls_projection_dropout=0.5
  model.config.visual_position_embedding_type=learned
  engine.trainability.visual_position_embedding.parameter_mode=full
  +engine.optimization.visual_adapter.parameter_groups.gates.learning_rate=1e-3
  # Best-checkpoint training schedule and logging cadence.
  engine.training_args.num_train_epochs="$NUM_TRAIN_EPOCHS"
  engine.training_args.dataloader_num_workers=8
  engine.training_args.eval_steps=6000
  engine.training_args.logging_steps=15
  engine.training_args.ddp_find_unused_parameters=false
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

accelerate launch "${CMD_ARGS[@]}"
