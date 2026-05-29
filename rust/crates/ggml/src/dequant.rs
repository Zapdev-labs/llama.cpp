//! Dequantization kernels: pack `&[u8]` blocks into `&mut [f32]` outputs.
//!
//! Mirrors `dequantize_row_<type>` in `ggml/src/ggml-quants.c`. Each function
//! is byte-exact against llama.cpp's reference kernel for in-scope inputs;
//! tests under `crates/ggml/tests/dequant.rs` compare against fixtures
//! produced by the C++ oracle binary.

use half::f16;

/// One Q8_0 block: `{ f16 d; i8 qs[QK8_0]; }`. Matches `block_q8_0` in
/// `ggml/src/ggml-common.h` (sizeof = 34).
const Q8_0_BLOCK_BYTES: usize = 34;

/// Elements per Q8_0 block (`QK8_0`).
const Q8_0_BLOCK_SIZE: usize = 32;

/// One Q4_0 block: `{ f16 d; u8 qs[QK4_0/2]; }`. Matches `block_q4_0` in
/// `ggml/src/ggml-common.h` (sizeof = 18).
const Q4_0_BLOCK_BYTES: usize = 18;

/// Elements per Q4_0 block (`QK4_0`).
const Q4_0_BLOCK_SIZE: usize = 32;

/// One Q4_K super-block: `{ f16 d; f16 dmin; u8 scales[12]; u8 qs[128]; }`.
/// Matches `block_q4_K` in `ggml/src/ggml-common.h` (sizeof = 144).
const Q4_K_BLOCK_BYTES: usize = 144;

/// Elements per Q4_K super-block (`QK_K`).
const Q4_K_BLOCK_SIZE: usize = 256;

/// Length of the packed-scales region in a Q4_K block (`K_SCALE_SIZE`).
const Q4_K_SCALES_LEN: usize = 12;

/// Dequantize an f16 row.
///
/// `src` is interpreted as a sequence of little-endian `u16` values that
/// encode IEEE-754 binary16; each is widened to `f32` in the canonical way
/// (`half::f16::from_bits(...).to_f32()`).
///
/// # Panics
///
/// Panics if `src.len()` is not exactly `2 * dst.len()`.
pub fn dequant_f16(src: &[u8], dst: &mut [f32]) {
    assert!(
        src.len() == dst.len() * 2,
        "ggml dequant_f16: src bytes ({}) must equal 2 * dst elements ({})",
        src.len(),
        dst.len(),
    );
    for (chunk, out) in src.chunks_exact(2).zip(dst.iter_mut()) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        *out = f16::from_bits(bits).to_f32();
    }
}

/// Dequantize a Q8_0 row.
///
/// `src` must hold a whole number of 34-byte Q8_0 blocks; each block is
/// `{ d: f16 (2 bytes, little-endian), qs: [i8; 32] }`. Each output element
/// is `qs[i] as f32 * d.to_f32()`, matching `dequantize_row_q8_0` in
/// `ggml/src/ggml-quants.c`.
///
/// # Panics
///
/// Panics if `src.len()` is not a multiple of 34, or if `dst.len()` does
/// not equal `32 * (src.len() / 34)`.
pub fn dequant_q8_0(src: &[u8], dst: &mut [f32]) {
    assert!(
        src.len().is_multiple_of(Q8_0_BLOCK_BYTES),
        "ggml dequant_q8_0: src bytes ({}) is not a multiple of block size {}",
        src.len(),
        Q8_0_BLOCK_BYTES,
    );
    let n_blocks = src.len() / Q8_0_BLOCK_BYTES;
    assert!(
        dst.len() == n_blocks * Q8_0_BLOCK_SIZE,
        "ggml dequant_q8_0: dst elements ({}) must equal {} * blocks ({})",
        dst.len(),
        Q8_0_BLOCK_SIZE,
        n_blocks,
    );
    for (block, out) in src
        .chunks_exact(Q8_0_BLOCK_BYTES)
        .zip(dst.chunks_exact_mut(Q8_0_BLOCK_SIZE))
    {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        for (i, out_i) in out.iter_mut().enumerate() {
            let q = block[2 + i] as i8;
            *out_i = q as f32 * d;
        }
    }
}

/// Dequantize a Q4_0 row.
///
/// `src` must hold a whole number of 18-byte Q4_0 blocks; each block is
/// `{ d: f16 (2 bytes, little-endian), qs: [u8; 16] }`. Each `qs` byte packs
/// two 4-bit nibbles; the low nibble of `qs[j]` decodes into output element
/// `j` and the high nibble into output element `j + 16`. Each nibble is
/// shifted by `-8` and scaled by `d.to_f32()`, matching
/// `dequantize_row_q4_0` in `ggml/src/ggml-quants.c`.
///
/// # Panics
///
/// Panics if `src.len()` is not a multiple of 18, or if `dst.len()` does
/// not equal `32 * (src.len() / 18)`.
pub fn dequant_q4_0(src: &[u8], dst: &mut [f32]) {
    assert!(
        src.len().is_multiple_of(Q4_0_BLOCK_BYTES),
        "ggml dequant_q4_0: src bytes ({}) is not a multiple of block size {}",
        src.len(),
        Q4_0_BLOCK_BYTES,
    );
    let n_blocks = src.len() / Q4_0_BLOCK_BYTES;
    assert!(
        dst.len() == n_blocks * Q4_0_BLOCK_SIZE,
        "ggml dequant_q4_0: dst elements ({}) must equal {} * blocks ({})",
        dst.len(),
        Q4_0_BLOCK_SIZE,
        n_blocks,
    );
    const HALF: usize = Q4_0_BLOCK_SIZE / 2;
    for (block, out) in src
        .chunks_exact(Q4_0_BLOCK_BYTES)
        .zip(dst.chunks_exact_mut(Q4_0_BLOCK_SIZE))
    {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let (lo, hi) = out.split_at_mut(HALF);
        for j in 0..HALF {
            let byte = block[2 + j];
            let x0 = (byte & 0x0F) as i32 - 8;
            let x1 = (byte >> 4) as i32 - 8;
            lo[j] = x0 as f32 * d;
            hi[j] = x1 as f32 * d;
        }
    }
}

/// Decode one (scale, min) pair out of the 12-byte packed Q4_K `scales`
/// field. Mirrors `get_scale_min_k4` in `ggml/src/ggml-quants.c`: the
/// 8 sub-block scales and 8 sub-block mins are each 6 bits wide, laid
/// out so the low four pairs live in `scales[0..8]` directly and the
/// high four pairs reuse the upper two bits of those bytes together
/// with `scales[8..12]`.
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize a Q4_K row.
///
/// `src` must hold a whole number of 144-byte Q4_K super-blocks; each block
/// is `{ d: f16, dmin: f16, scales: [u8; 12], qs: [u8; 128] }`. Per
/// super-block, the 8 sub-block 6-bit scales and 6-bit mins are unpacked
/// via [`get_scale_min_k4`] and each sub-block's 32 elements are decoded as
/// `d * sub_scale * nibble - dmin * sub_min` — low nibbles of `qs[..32]` for
/// the first sub-block in a pair, high nibbles for the second — matching
/// `dequantize_row_q4_K` in `ggml/src/ggml-quants.c`.
///
/// # Panics
///
/// Panics if `src.len()` is not a multiple of 144, or if `dst.len()` does
/// not equal `256 * (src.len() / 144)`.
pub fn dequant_q4_k(src: &[u8], dst: &mut [f32]) {
    assert!(
        src.len().is_multiple_of(Q4_K_BLOCK_BYTES),
        "ggml dequant_q4_k: src bytes ({}) is not a multiple of block size {}",
        src.len(),
        Q4_K_BLOCK_BYTES,
    );
    let n_blocks = src.len() / Q4_K_BLOCK_BYTES;
    assert!(
        dst.len() == n_blocks * Q4_K_BLOCK_SIZE,
        "ggml dequant_q4_k: dst elements ({}) must equal {} * blocks ({})",
        dst.len(),
        Q4_K_BLOCK_SIZE,
        n_blocks,
    );
    for (block, out) in src
        .chunks_exact(Q4_K_BLOCK_BYTES)
        .zip(dst.chunks_exact_mut(Q4_K_BLOCK_SIZE))
    {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let dmin = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
        let scales = &block[4..4 + Q4_K_SCALES_LEN];
        let qs = &block[4 + Q4_K_SCALES_LEN..];

        let mut is = 0usize;
        let mut q_off = 0usize;
        let mut y_off = 0usize;
        // Eight sub-blocks of 32 are processed as four pairs (low then high
        // nibble of the same 32 qs bytes).
        for _ in 0..Q4_K_BLOCK_SIZE / 64 {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;
            for l in 0..32 {
                out[y_off + l] = d1 * (qs[q_off + l] & 0x0F) as f32 - m1;
            }
            for l in 0..32 {
                out[y_off + 32 + l] = d2 * (qs[q_off + l] >> 4) as f32 - m2;
            }
            q_off += 32;
            y_off += 64;
            is += 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_handles_zero_one_and_neg_one() {
        let src: [u8; 6] = [
            0x00, 0x00, // +0.0
            0x00, 0x3C, // 1.0
            0x00, 0xBC, // -1.0
        ];
        let mut dst = [0.0f32; 3];
        dequant_f16(&src, &mut dst);
        assert_eq!(dst[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(dst[1].to_bits(), 1.0f32.to_bits());
        assert_eq!(dst[2].to_bits(), (-1.0f32).to_bits());
    }

    #[test]
    #[should_panic(expected = "src bytes")]
    fn f16_panics_on_size_mismatch() {
        let src = [0u8; 3];
        let mut dst = [0.0f32; 2];
        dequant_f16(&src, &mut dst);
    }

    #[test]
    fn q8_0_single_block_matches_scalar_formula() {
        let mut src = [0u8; Q8_0_BLOCK_BYTES];
        let d_f16 = f16::from_f32(0.25);
        src[0..2].copy_from_slice(&d_f16.to_bits().to_le_bytes());
        for i in 0..Q8_0_BLOCK_SIZE {
            src[2 + i] = ((i as i32) - 16) as u8;
        }
        let mut dst = [0.0f32; Q8_0_BLOCK_SIZE];
        dequant_q8_0(&src, &mut dst);
        let d = d_f16.to_f32();
        for (i, &v) in dst.iter().enumerate() {
            let q = (i as i32 - 16) as i8;
            assert_eq!(v.to_bits(), (q as f32 * d).to_bits(), "elem {i}");
        }
    }

    #[test]
    #[should_panic(expected = "not a multiple of block size 34")]
    fn q8_0_panics_on_unaligned_src() {
        let src = [0u8; 33];
        let mut dst = [0.0f32; 32];
        dequant_q8_0(&src, &mut dst);
    }

    #[test]
    #[should_panic(expected = "dst elements")]
    fn q8_0_panics_on_dst_size_mismatch() {
        let src = [0u8; Q8_0_BLOCK_BYTES];
        let mut dst = [0.0f32; 16];
        dequant_q8_0(&src, &mut dst);
    }

    #[test]
    fn q4_0_single_block_matches_scalar_formula() {
        let mut src = [0u8; Q4_0_BLOCK_BYTES];
        let d_f16 = f16::from_f32(0.125);
        src[0..2].copy_from_slice(&d_f16.to_bits().to_le_bytes());
        for j in 0..16 {
            let low = (j as u8) & 0x0F;
            let high = ((j as u8).wrapping_mul(3)) & 0x0F;
            src[2 + j] = (high << 4) | low;
        }
        let mut dst = [0.0f32; Q4_0_BLOCK_SIZE];
        dequant_q4_0(&src, &mut dst);
        let d = d_f16.to_f32();
        for j in 0..16 {
            let low = (j as i32) & 0x0F;
            let high = ((j as i32) * 3) & 0x0F;
            assert_eq!(
                dst[j].to_bits(),
                ((low - 8) as f32 * d).to_bits(),
                "low nibble at elem {j}"
            );
            assert_eq!(
                dst[j + 16].to_bits(),
                ((high - 8) as f32 * d).to_bits(),
                "high nibble at elem {}",
                j + 16
            );
        }
    }

    #[test]
    fn q4_0_zero_nibbles_produce_negative_eight_times_d() {
        let mut src = [0u8; Q4_0_BLOCK_BYTES];
        let d_f16 = f16::from_f32(1.0);
        src[0..2].copy_from_slice(&d_f16.to_bits().to_le_bytes());
        let mut dst = [0.0f32; Q4_0_BLOCK_SIZE];
        dequant_q4_0(&src, &mut dst);
        for v in dst {
            assert_eq!(v.to_bits(), (-8.0f32).to_bits());
        }
    }

    #[test]
    #[should_panic(expected = "not a multiple of block size 18")]
    fn q4_0_panics_on_unaligned_src() {
        let src = [0u8; 17];
        let mut dst = [0.0f32; 32];
        dequant_q4_0(&src, &mut dst);
    }

    #[test]
    #[should_panic(expected = "dst elements")]
    fn q4_0_panics_on_dst_size_mismatch() {
        let src = [0u8; Q4_0_BLOCK_BYTES];
        let mut dst = [0.0f32; 16];
        dequant_q4_0(&src, &mut dst);
    }

    fn pack_q4_k_scales(d: [u8; 8], m: [u8; 8]) -> [u8; 12] {
        let mut q = [0u8; 12];
        for i in 0..4 {
            q[i] = (d[i] & 0x3F) | (((d[i + 4] >> 4) & 0x03) << 6);
            q[i + 4] = (m[i] & 0x3F) | (((m[i + 4] >> 4) & 0x03) << 6);
            q[i + 8] = (d[i + 4] & 0x0F) | ((m[i + 4] & 0x0F) << 4);
        }
        q
    }

    #[test]
    fn q4_k_scale_min_decode_round_trips_full_6bit_range() {
        let d = [0u8, 9, 18, 27, 36, 45, 54, 63];
        let m = [63u8, 54, 45, 36, 27, 18, 9, 0];
        let packed = pack_q4_k_scales(d, m);
        for j in 0..8 {
            let (sc, mn) = get_scale_min_k4(j, &packed);
            assert_eq!(sc, d[j], "scale mismatch at sub-block {j}");
            assert_eq!(mn, m[j], "min mismatch at sub-block {j}");
        }
    }

    #[test]
    fn q4_k_zero_super_block_yields_all_zero() {
        let src = [0u8; Q4_K_BLOCK_BYTES];
        let mut dst = [1.0f32; Q4_K_BLOCK_SIZE];
        dequant_q4_k(&src, &mut dst);
        for v in dst {
            assert_eq!(v.to_bits(), 0.0f32.to_bits());
        }
    }

    #[test]
    fn q4_k_single_block_matches_scalar_formula() {
        let d_f16 = f16::from_f32(0.5);
        let dmin_f16 = f16::from_f32(0.25);
        let d_scales = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let m_scales = [8u8, 7, 6, 5, 4, 3, 2, 1];

        let mut src = [0u8; Q4_K_BLOCK_BYTES];
        src[0..2].copy_from_slice(&d_f16.to_bits().to_le_bytes());
        src[2..4].copy_from_slice(&dmin_f16.to_bits().to_le_bytes());
        src[4..16].copy_from_slice(&pack_q4_k_scales(d_scales, m_scales));
        for j in 0..128 {
            let low = (j as u8) & 0x0F;
            let high = ((j as u8).wrapping_mul(5)) & 0x0F;
            src[16 + j] = (high << 4) | low;
        }

        let mut dst = [0.0f32; Q4_K_BLOCK_SIZE];
        dequant_q4_k(&src, &mut dst);

        let d = d_f16.to_f32();
        let dmin = dmin_f16.to_f32();
        for sb in 0..8 {
            let sc = d * d_scales[sb] as f32;
            let mn = dmin * m_scales[sb] as f32;
            let pair = sb / 2;
            let half = sb % 2;
            for l in 0..32 {
                let byte = src[16 + pair * 32 + l];
                let nibble = if half == 0 { byte & 0x0F } else { byte >> 4 };
                let expected = sc * nibble as f32 - mn;
                assert_eq!(
                    dst[sb * 32 + l].to_bits(),
                    expected.to_bits(),
                    "sub-block {sb} elem {l}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "not a multiple of block size 144")]
    fn q4_k_panics_on_unaligned_src() {
        let src = [0u8; 143];
        let mut dst = [0.0f32; 256];
        dequant_q4_k(&src, &mut dst);
    }

    #[test]
    #[should_panic(expected = "dst elements")]
    fn q4_k_panics_on_dst_size_mismatch() {
        let src = [0u8; Q4_K_BLOCK_BYTES];
        let mut dst = [0.0f32; 128];
        dequant_q4_k(&src, &mut dst);
    }
}
