#!/usr/bin/env bash
# Regenerate the dequant fixtures used by crates/ggml/tests/dequant.rs.
#
# Inputs are produced deterministically by a small Python script with a
# fixed seed; expected outputs are produced by the oracle binary that
# links the C++ ggml dequantize_row_* routines. Commit both .bin files;
# CI does NOT rerun this script.
#
# Usage: ./regenerate.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../../../.." && pwd)"
ORACLE="$REPO_ROOT/rust/oracle/build/oracle"

if [ ! -x "$ORACLE" ]; then
    echo "oracle binary missing: $ORACLE" >&2
    exit 1
fi

python3 - "$HERE" <<'PY'
import os, struct, sys, random

out_dir = sys.argv[1]

# f16 fixture: 1024 random u16 little-endian values.
rng = random.Random(0xF16_FEED)
with open(os.path.join(out_dir, "f16_input.bin"), "wb") as f:
    f.write(struct.pack("<1024H", *(rng.randrange(0, 1 << 16) for _ in range(1024))))

# q8_0 fixture: 8 blocks; each block is { d: f16, qs: [i8; 32] }.
# 'd' values are sampled to span typical real-tensor scales (~1e-4..1e0),
# in both signs; qs values cover the full i8 range. Stored raw — block
# bytes don't need to come from quantize_row_q8_0_reference for the
# dequant correctness check, only the dequantize-byte-equality matters.
rng = random.Random(0xC8_0FEED)
def f32_to_f16_bits(x: float) -> int:
    (b32,) = struct.unpack("<I", struct.pack("<f", x))
    sign = (b32 >> 31) & 0x1
    exp  = (b32 >> 23) & 0xFF
    mant = b32 & 0x7FFFFF
    if exp == 0:
        return sign << 15
    if exp == 0xFF:
        return (sign << 15) | (0x1F << 10) | (1 if mant else 0)
    new_exp = exp - 127 + 15
    if new_exp >= 0x1F:
        return (sign << 15) | (0x1F << 10)
    if new_exp <= 0:
        return sign << 15
    return (sign << 15) | (new_exp << 10) | (mant >> 13)

with open(os.path.join(out_dir, "q8_0_input.bin"), "wb") as f:
    for blk in range(8):
        mag  = 10.0 ** rng.uniform(-4, 0)
        sign = -1.0 if rng.random() < 0.5 else 1.0
        d_bits = f32_to_f16_bits(sign * mag)
        f.write(struct.pack("<H", d_bits))
        for _ in range(32):
            f.write(struct.pack("<b", rng.randrange(-128, 128)))
PY

"$ORACLE" dequant f16  "$HERE/f16_input.bin"  "$HERE/f16_output.bin"
"$ORACLE" dequant q8_0 "$HERE/q8_0_input.bin" "$HERE/q8_0_output.bin"

echo "regenerated:"
ls -l "$HERE"/*.bin
