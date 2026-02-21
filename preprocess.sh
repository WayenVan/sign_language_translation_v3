#! /bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
DATA_ROOT="$SCRIPT_DIR/dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T"
NUM_WORKERS=32

echo "脚本绝对目录：$SCRIPT_DIR"

cd "$SCRIPT_DIR" || exit

source "$SCRIPT_DIR"/.venv/bin/activate

python3 "$SCRIPT_DIR"/preprocess/dataset_preprocess-T.py \
  --dataset-root "$DATA_ROOT" \
  --save_dir "$SCRIPT_DIR/dataset/phoenix2014-T-preprocessed" \
  -p \
  --num_workers "$NUM_WORKERS"
