#! /bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
DATA_ROOT="$PROJECT_ROOT/dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T"
NUM_WORKERS=32

echo "脚本绝对目录：$SCRIPT_DIR"

cd "$PROJECT_ROOT" || exit

source "$PROJECT_ROOT"/.venv/bin/activate

python3 "$PROJECT_ROOT"/preprocess/dataset_preprocess-T.py \
  --dataset-root "$DATA_ROOT" \
  --num_workers "$NUM_WORKERS"
