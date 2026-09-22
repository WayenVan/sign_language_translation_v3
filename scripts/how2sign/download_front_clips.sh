#!/usr/bin/env bash

set -euo pipefail

REPO_ID="WayenVan/how2sign-front-clips"
OUTPUT_DIR="${1:-dataset/how2sign-front-clips}"

# Force huggingface_hub to use regular HTTP downloads instead of Xet.
export HF_HUB_DISABLE_XET=1

mkdir -p "$OUTPUT_DIR"

if command -v hf >/dev/null 2>&1; then
    HF_CLI=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI=(huggingface-cli)
else
    echo "错误：未找到 hf CLI。请先安装：pip install -U huggingface_hub" >&2
    exit 1
fi

echo "下载 $REPO_ID 到 $OUTPUT_DIR（workers=4，Xet 已禁用）" >&2
"${HF_CLI[@]}" download "$REPO_ID" \
    --repo-type dataset \
    --local-dir "$OUTPUT_DIR" \
    --max-workers 4

echo "下载完成：$OUTPUT_DIR" >&2
