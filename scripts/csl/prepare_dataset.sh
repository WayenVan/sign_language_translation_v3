#!/usr/bin/env bash
#
# prepare_dataset.sh — CSL-Daily dataset preparation utilities
#
# The archive already contains the preprocessed dataset.  This helper only:
#   1. copies the archive to local scratch;
#   2. extracts it into the requested frame data root; and
#   3. returns that directory directly on stdout.
#
# Usage:
#   source scripts/csl/prepare_dataset.sh
#   DATA_ROOT=$(prepare_dataset SOURCE_ARCHIVE DATA_ROOT [LOCAL_SCRATCH])
#
# Example:
#   DATA_ROOT=$(prepare_dataset \
#       /mnt/scratch/users/2533494w/dataset/full-trim-256x256px.tar \
#       "$HOME/localscratch/full-trim-256x256px")

prepare_dataset() {
    # Keep every operational message off stdout.  This makes command
    # substitution reliable even if more commands are added here later.
    {
    if (( $# < 2 || $# > 3 )); then
        echo "用法：prepare_dataset SOURCE_ARCHIVE DATA_ROOT [LOCAL_SCRATCH]" >&2
        return 2
    fi

    local SOURCE_ARCHIVE="$1"
    local DATA_ROOT="$2"
    local LOCAL_SCRATCH="${3:-$HOME/localscratch}"

    local LOCAL_ARCHIVE="$LOCAL_SCRATCH/$(basename "$SOURCE_ARCHIVE")"
    local LOCAL_ARCHIVE_PART="${LOCAL_ARCHIVE}.part"
    local COPY_MARKER="$LOCAL_SCRATCH/.CSL_COPY_DONE"
    local EXTRACTION_MARKER="$DATA_ROOT/.EXTRACTION_DONE"

    mkdir -p "$LOCAL_SCRATCH"

    if [[ -f "$COPY_MARKER" && -f "$LOCAL_ARCHIVE" ]]; then
        echo "CSL-Daily tar 已复制，跳过：$LOCAL_ARCHIVE" >&2
    else
        if [[ ! -f "$SOURCE_ARCHIVE" ]]; then
            echo "错误：源 tar 不存在：$SOURCE_ARCHIVE" >&2
            return 1
        fi

        echo "复制 CSL-Daily tar 到 local scratch：$LOCAL_ARCHIVE" >&2
        if ! rsync -ah --info=progress2 \
            "$SOURCE_ARCHIVE" \
            "$LOCAL_ARCHIVE_PART" >&2; then
            echo "错误：复制 CSL-Daily tar 失败" >&2
            return 1
        fi

        mv -f "$LOCAL_ARCHIVE_PART" "$LOCAL_ARCHIVE"
        touch "$COPY_MARKER"
    fi

    if [[ -f "$EXTRACTION_MARKER" ]]; then
        echo "CSL-Daily 已解压，跳过：$DATA_ROOT" >&2
    else
        echo "解压 CSL-Daily 到：$DATA_ROOT" >&2
        mkdir -p "$DATA_ROOT"

        if ! tar -xf "$LOCAL_ARCHIVE" \
            -C "$DATA_ROOT" \
            --strip-components=1; then
            echo "错误：解压 CSL-Daily 失败：$LOCAL_ARCHIVE" >&2
            return 1
        fi

        touch "$EXTRACTION_MARKER"
        echo "CSL-Daily 解压完成：$DATA_ROOT" >&2
    fi

    if [[ ! -d "$DATA_ROOT" ]]; then
        echo "错误：CSL-Daily dataroot 不存在：$DATA_ROOT" >&2
        return 1
    fi
    } >&2

    # Only the data root is written to stdout so callers can capture it.
    printf '%s\n' "$DATA_ROOT"
}
