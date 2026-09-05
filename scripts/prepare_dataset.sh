#! /bin/bash
#
# prepare_dataset.sh — PHOENIX-2014-T dataset preparation utilities
#
# This script defines a reusable function to prepare the PHOENIX-2014-T dataset.
# It handles copying the tar archive to local scratch, extracting it, and
# running the preprocessing step.
#
# Usage:
#   source scripts/prepare_dataset.sh
#   prepare_dataset SOURCE_ARCHIVE DATASET_PATH [LOCAL_SCRATCH]
#
# Arguments:
#   SOURCE_ARCHIVE  — path to the dataset tar.gz file
#   DATASET_PATH    — target directory for extraction & preprocessing
#   LOCAL_SCRATCH   — (optional) scratch directory (default: $HOME/localscratch)
#
# Output:
#   Prints DATASET_PATH to stdout on success.
#
# Return value:
#   0 on success, non-zero on failure.

###############################################################################
#  prepare_dataset
#
#  Copy, extract and preprocess the PHOENIX-2014-T dataset.
#
#  Globals read:
#   (none)
#
#  Globals set (via 'local'):
#   (none)
#
#  Arguments:
#   1 — SOURCE_ARCHIVE   (e.g. "dataset/phoenix-2014-T.v3.tar.gz")
#   2 — DATASET_PATH     (e.g. "$HOME/localscratch/ph14t")
#   3 — LOCAL_SCRATCH    (optional, default "$HOME/localscratch")
#
#  Output:
#   DATASET_PATH is printed to stdout.
###############################################################################
prepare_dataset() {
    local SOURCE_ARCHIVE="$1"
    local DATASET_PATH="$2"
    local LOCAL_SCRATCH="${3:-$HOME/localscratch}"

    # Derived paths
    local LOCAL_ARCHIVE="$LOCAL_SCRATCH/$(basename "$SOURCE_ARCHIVE")"
    local EXTRACT_MARKER="$DATASET_PATH/.extract_complete"
    local COMPLETE_MARKER="$DATASET_PATH/.data_complete"
    local RESIZED_FRAME_DIR="$DATASET_PATH/PHOENIX-2014-T/features/fullFrame-256x256px"

    # Set TQDM_DISABLE based on whether we have an interactive terminal
    if [[ -t 2 ]]; then
        unset TQDM_DISABLE
    else
        export TQDM_DISABLE=1
    fi

    mkdir -p "$LOCAL_SCRATCH"

    if [[ -f "$COMPLETE_MARKER" && -d "$RESIZED_FRAME_DIR" ]]; then
        echo "数据集图像已缩放为 256x256，跳过：$DATASET_PATH" >&2

    else
        if [[ -f "$EXTRACT_MARKER" ]]; then
            echo "数据集已经完成解压，直接进行预处理：$DATASET_PATH" >&2
        else
            # 本地已有 tar 就直接使用，否则从 shared scratch 复制
            if [[ -f "$LOCAL_ARCHIVE" ]]; then
                echo "发现本地 tar，跳过复制：$LOCAL_ARCHIVE" >&2
            else
                if [[ ! -f "$SOURCE_ARCHIVE" ]]; then
                    echo "错误：源 tar 不存在：$SOURCE_ARCHIVE" >&2
                    return 1
                fi

                echo "复制 tar 到 local scratch..." >&2
                rsync -ah --info=progress2 \
                    "$SOURCE_ARCHIVE" \
                    "$LOCAL_ARCHIVE" >&2
            fi

            echo "解压数据集到：$DATASET_PATH" >&2
            rm -rf "$DATASET_PATH"
            mkdir -p "$DATASET_PATH"

            tar -xf "$LOCAL_ARCHIVE" \
                -C "$DATASET_PATH" \
                --strip-components=1

            # 只有解压成功后才创建标志
            touch "$EXTRACT_MARKER"
        fi

        # 在解压后的目录上进行预处理
        echo "开始预处理..." >&2

        if ! python preprocess/dataset_preprocess-T.py \
            --dataset-root "$DATASET_PATH/PHOENIX-2014-T" \
            -m \
            -w "$(nproc)" >&2; then
            echo "错误：图像预处理失败：$DATASET_PATH" >&2
            return 1
        fi

        # 只有预处理成功后才创建标志
        touch "$COMPLETE_MARKER"
        echo "数据集准备完成：$DATASET_PATH" >&2
    fi

    # 仅在 stdout 输出 DATASET_PATH，供调用者捕获
    echo "$DATASET_PATH"
}
