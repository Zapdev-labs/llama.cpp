#!/usr/bin/env bash
# Regenerate the RMSNorm fixtures used by crates/ggml/tests/rmsnorm.rs.
#
# Captures the post-block-0 hidden state ("l_out-0") and the
# corresponding attn_norm-1 output of Llama-3.2-1B-Instruct via the
# oracle binary's rmsnorm_capture subcommand (which sets a
# ggml_backend_sched eval callback during one llama_decode of "Hello").
# Also reads blk.1.attn_norm.weight directly from the GGUF file.
# Commit the three .bin files; CI does NOT rerun this script.
#
# Usage: ./regenerate.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../../../.." && pwd)"
ORACLE="$REPO_ROOT/rust/oracle/build/oracle"
MODEL="$REPO_ROOT/models/llama3-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

if [ ! -x "$ORACLE" ]; then
    echo "oracle binary missing: $ORACLE" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "model missing: $MODEL" >&2
    echo "download with: huggingface-cli download unsloth/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir \"$(dirname "$MODEL")\"" >&2
    exit 1
fi

"$ORACLE" rmsnorm_capture "$MODEL" "Hello" "$HERE" 2>/dev/null

echo "regenerated:"
ls -l "$HERE"/*.bin
