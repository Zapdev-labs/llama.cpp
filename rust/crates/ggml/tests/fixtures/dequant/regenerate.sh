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

QWEN_Q4KM="/run/media/dih/8CEDA5F938E73A48/AI/models/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf"

if [ ! -x "$ORACLE" ]; then
    echo "oracle binary missing: $ORACLE" >&2
    exit 1
fi

python3 - "$HERE" "$QWEN_Q4KM" <<'PY'
import os, struct, sys, random

out_dir = sys.argv[1]
qwen_path = sys.argv[2]

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

# q4_0 fixture: 8 blocks; each block is { d: f16, qs: [u8; 16] }, where each
# byte of qs packs two 4-bit nibbles. Same scale-distribution strategy as
# q8_0; nibble bytes cover the full u8 range so both nibbles span 0..16.
rng = random.Random(0xC4_0FEED)
with open(os.path.join(out_dir, "q4_0_input.bin"), "wb") as f:
    for blk in range(8):
        mag  = 10.0 ** rng.uniform(-4, 0)
        sign = -1.0 if rng.random() < 0.5 else 1.0
        d_bits = f32_to_f16_bits(sign * mag)
        f.write(struct.pack("<H", d_bits))
        for _ in range(16):
            f.write(struct.pack("<B", rng.randrange(0, 256)))

# q4_K fixture: 4 super-blocks (1024 elements). Each block is 144 bytes:
# { d: f16, dmin: f16, scales: [u8; 12] (6-bit packed), qs: [u8; 128] }.
# Block 0 is hand-crafted to make the 8 sub-block scales span the full
# 6-bit range (0..=63) and the 8 sub-block mins likewise; the remaining
# three blocks are seeded-random. See `get_scale_min_k4` in
# ggml/src/ggml-quants.c for the bit layout.
def pack_q4k_scales(d8, m8):
    q = bytearray(12)
    for i in range(4):
        q[i]     = (d8[i] & 0x3F) | (((d8[i+4] >> 4) & 0x3) << 6)
        q[i + 4] = (m8[i] & 0x3F) | (((m8[i+4] >> 4) & 0x3) << 6)
        q[i + 8] = (d8[i+4] & 0x0F) | ((m8[i+4] & 0x0F) << 4)
    return bytes(q)

rng = random.Random(0xC4_F)
with open(os.path.join(out_dir, "q4_k_input.bin"), "wb") as f:
    # Block 0 — full 6-bit scale & min coverage, qs covers full u8 range.
    d_bits    = f32_to_f16_bits(0.012)
    dmin_bits = f32_to_f16_bits(0.003)
    f.write(struct.pack("<H", d_bits))
    f.write(struct.pack("<H", dmin_bits))
    d8 = [0, 9, 18, 27, 36, 45, 54, 63]
    m8 = [63, 54, 45, 36, 27, 18, 9, 0]
    f.write(pack_q4k_scales(d8, m8))
    for j in range(128):
        f.write(struct.pack("<B", (j * 7 + 13) & 0xFF))
    # Blocks 1..4 — seeded-random.
    for blk in range(3):
        mag  = 10.0 ** rng.uniform(-4, -1)
        sign = -1.0 if rng.random() < 0.5 else 1.0
        f.write(struct.pack("<H", f32_to_f16_bits(sign * mag)))
        f.write(struct.pack("<H", f32_to_f16_bits(mag * 0.1)))
        d8 = [rng.randrange(0, 64) for _ in range(8)]
        m8 = [rng.randrange(0, 64) for _ in range(8)]
        f.write(pack_q4k_scales(d8, m8))
        for _ in range(128):
            f.write(struct.pack("<B", rng.randrange(0, 256)))

# q4_K real-tensor fixture: first 4 super-blocks (576 bytes) of the first
# Q4_K-typed tensor in qwen2.5-0.5b-instruct-q4_k_m.gguf. In this file the
# tensor named `blk.0.attn_q.weight` is actually Q5_0 — only the
# `blk.{11,12,14,...}.ffn_down.weight` tensors are stored as Q4_K — so we
# pick `blk.11.ffn_down.weight` (first Q4_K tensor by file offset). Parses
# the GGUF header just enough to find that tensor's data offset.
if not os.path.isfile(qwen_path):
    sys.stderr.write(f"warn: qwen file missing at {qwen_path}; skipping q4_k_qwen fixture\n")
else:
    SCALAR_FMT = {
        0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2), 3: ('<h', 2),
        4: ('<I', 4), 5: ('<i', 4), 6: ('<f', 4), 7: ('<B', 1),
        10: ('<Q', 8), 11: ('<q', 8), 12: ('<d', 8),
    }
    def read_string(buf, off):
        (n,) = struct.unpack_from('<Q', buf, off); off += 8
        s = buf[off:off+n].decode('utf-8'); off += n
        return s, off
    def read_value(buf, off, ty):
        if ty == 8:
            return read_string(buf, off)
        if ty == 9:
            (sub,) = struct.unpack_from('<I', buf, off); off += 4
            (n,) = struct.unpack_from('<Q', buf, off); off += 8
            vals = []
            for _ in range(n):
                v, off = read_value(buf, off, sub)
                vals.append(v)
            return vals, off
        fmt, sz = SCALAR_FMT[ty]
        return struct.unpack_from(fmt, buf, off)[0], off + sz

    with open(qwen_path, 'rb') as f:
        # Only read the metadata header — tensor info typically lives in
        # the first ~MB. Read 8 MB to be safe; we'll seek for the data.
        head = f.read(8 * 1024 * 1024)
        assert head[0:4] == b'GGUF', "not a GGUF file"
        off = 4
        (version,) = struct.unpack_from('<I', head, off); off += 4
        (n_tensors,) = struct.unpack_from('<q', head, off); off += 8
        (n_kv,) = struct.unpack_from('<q', head, off); off += 8
        alignment = 32
        for _ in range(n_kv):
            key, off = read_string(head, off)
            (ty,) = struct.unpack_from('<I', head, off); off += 4
            val, off = read_value(head, off, ty)
            if key == 'general.alignment':
                alignment = int(val)
        wanted_offset = None
        wanted_type   = None
        for _ in range(n_tensors):
            name, off = read_string(head, off)
            (n_dims,) = struct.unpack_from('<I', head, off); off += 4
            dims = []
            for _ in range(n_dims):
                (d,) = struct.unpack_from('<q', head, off); off += 8
                dims.append(d)
            (tty,) = struct.unpack_from('<i', head, off); off += 4
            (toff,) = struct.unpack_from('<Q', head, off); off += 8
            if name == 'blk.11.ffn_down.weight':
                wanted_offset = toff
                wanted_type   = tty
        data_off = (off + alignment - 1) // alignment * alignment
        if wanted_offset is None:
            sys.stderr.write("warn: blk.11.ffn_down.weight not found in qwen file\n")
        else:
            assert wanted_type == 12, f"expected Q4_K(12), got {wanted_type}"
            f.seek(data_off + wanted_offset)
            chunk = f.read(4 * 144)
            assert len(chunk) == 576
            with open(os.path.join(out_dir, "q4_k_qwen_input.bin"), "wb") as g:
                g.write(chunk)
PY

"$ORACLE" dequant f16  "$HERE/f16_input.bin"  "$HERE/f16_output.bin"
"$ORACLE" dequant q8_0 "$HERE/q8_0_input.bin" "$HERE/q8_0_output.bin"
"$ORACLE" dequant q4_0 "$HERE/q4_0_input.bin" "$HERE/q4_0_output.bin"
"$ORACLE" dequant q4_k "$HERE/q4_k_input.bin" "$HERE/q4_k_output.bin"
if [ -f "$HERE/q4_k_qwen_input.bin" ]; then
    "$ORACLE" dequant q4_k "$HERE/q4_k_qwen_input.bin" "$HERE/q4_k_qwen_output.bin"
fi

echo "regenerated:"
ls -l "$HERE"/*.bin
