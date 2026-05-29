#!/usr/bin/env bash
# run-oracle.sh — wrap llama-completion to emit a normalized token-ID stream.
#
# Usage: run-oracle.sh <model.gguf> <prompt> <n>
#
# Runs llama-completion with deterministic greedy decoding, captures the
# generated text, then re-tokenizes the prompt + generated text via the
# oracle binary and prints the n generated token IDs (one decimal per line)
# to stdout. Used by the validation harness to diff against
# `llama-cli --dump-tokens` produced by the pure-Rust port.

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <model.gguf> <prompt> <n>" >&2
    exit 2
fi

MODEL="$1"
PROMPT="$2"
N="$3"

if ! [[ "$N" =~ ^[0-9]+$ ]]; then
    echo "$0: <n> must be a non-negative integer (got '$N')" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ORACLE_BIN="${ORACLE_BIN:-$SCRIPT_DIR/build/oracle}"
LLAMA_COMPLETION="${LLAMA_COMPLETION:-$REPO_ROOT/build/bin/llama-completion}"

if [ ! -x "$ORACLE_BIN" ]; then
    echo "$0: oracle binary not found at $ORACLE_BIN (run cmake first)" >&2
    exit 1
fi
if [ ! -x "$LLAMA_COMPLETION" ]; then
    echo "$0: llama-completion not found at $LLAMA_COMPLETION" >&2
    echo "$0:   build it from repo root with:  cmake --build build --target llama-completion" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "$0: model file not found: $MODEL" >&2
    exit 1
fi

n_prompt="$("$ORACLE_BIN" tokenize "$MODEL" "$PROMPT" | wc -l)"

gen_text="$("$LLAMA_COMPLETION" \
    -m "$MODEL" \
    -p "$PROMPT" \
    -n "$N" \
    --temp 0 \
    -s 0 \
    -no-cnv \
    --simple-io \
    --no-display-prompt \
    --no-warmup 2>/dev/null)"

"$ORACLE_BIN" tokenize "$MODEL" "${PROMPT}${gen_text}" \
    | tail -n +$((n_prompt + 1)) \
    | head -n "$N"
