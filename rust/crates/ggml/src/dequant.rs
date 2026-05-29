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
}
