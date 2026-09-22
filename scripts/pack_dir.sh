#!/usr/bin/env bash
# Usage:
#   scripts/pack_dir.sh [-P PREFETCH] [-f] SRC_DIR OUTPUT.tar
#
# Pack SRC_DIR into an uncompressed .tar, tuned for many small files on Lustre.
# A single tar reads one file at a time and stalls on per-file latency, so a
# background pool of PREFETCH parallel readers pulls files into the page cache
# in the same order tar archives them; tar then reads almost everything from RAM.
#
#   -P PREFETCH  parallel prefetch readers (default: 128)
#   -f           overwrite OUTPUT if it exists

set -euo pipefail

PREFETCH=128
FORCE=0

usage() { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

while getopts ":P:fh" opt; do
    case "$opt" in
        P) PREFETCH="$OPTARG" ;;
        f) FORCE=1 ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

(( $# == 2 )) || usage
[[ "$PREFETCH" =~ ^[1-9][0-9]*$ ]] || { echo "错误：PREFETCH 必须是正整数" >&2; exit 2; }

SRC_DIR="$(realpath "$1")"
OUTPUT="$(realpath -m "$2")"

[[ -d "$SRC_DIR" ]] || { echo "错误：源目录不存在：$SRC_DIR" >&2; exit 1; }
[[ "$OUTPUT" == *.tar ]] || { echo "错误：输出必须以 .tar 结尾" >&2; exit 2; }
[[ -d "$(dirname "$OUTPUT")" ]] || { echo "错误：输出目录不存在：$(dirname "$OUTPUT")" >&2; exit 1; }
if [[ -e "$OUTPUT" && $FORCE -eq 0 ]]; then
    echo "错误：$OUTPUT 已存在（使用 -f 覆盖）" >&2
    exit 1
fi

NAME="$(basename "$SRC_DIR")"
PARENT="$(dirname "$SRC_DIR")"
TMP_OUTPUT="$(dirname "$OUTPUT")/.pack.$$.${OUTPUT##*/}"
LIST_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pack_dir.XXXXXX")"
PREFETCH_PID=""

cleanup() {
    [[ -n "$PREFETCH_PID" ]] && kill "$PREFETCH_PID" 2>/dev/null || true
    rm -rf -- "$LIST_DIR"
    rm -f -- "$TMP_OUTPUT"
}
trap cleanup EXIT INT TERM HUP

SECONDS=0
echo "列出 $SRC_DIR ..." >&2
# One walk, two NUL-separated lists in the same order: every entry (for tar,
# so directories keep their metadata) and regular files only (for prefetch).
(cd "$PARENT" && find "$NAME" -fprint0 "$LIST_DIR/all" -type f -fprint0 "$LIST_DIR/files")
NFILES="$(tr -cd '\0' < "$LIST_DIR/files" | wc -c)"
echo "  $NFILES 个文件（${SECONDS}s）" >&2

echo "打包 -> $OUTPUT（预读并发 $PREFETCH）" >&2
# Prefetch: PREFETCH concurrent cats, 32 files each, discarding the bytes.
# Failures are ignored here; tar re-reads every file and reports real errors.
(cd "$PARENT" && xargs -0 -P "$PREFETCH" -n 32 cat < "$LIST_DIR/files" > /dev/null 2>&1 || true) &
PREFETCH_PID=$!

tar -C "$PARENT" -b 512 --null --no-recursion -T "$LIST_DIR/all" -cf "$TMP_OUTPUT"

wait "$PREFETCH_PID" || true
PREFETCH_PID=""
mv -f -- "$TMP_OUTPUT" "$OUTPUT"
echo "完成：${SECONDS}s，$(du -h "$OUTPUT" | cut -f1)  $OUTPUT" >&2
