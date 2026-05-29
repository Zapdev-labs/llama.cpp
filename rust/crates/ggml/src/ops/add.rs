//! Element-wise in-place addition over `f32` slices.
//!
//! Mirrors the `f32` specialization of `ggml_compute_forward_add` in
//! `ggml/src/ggml-cpu/ops.c`. The kernel is single-threaded: each element
//! is a single add, so the `rayon` per-row dispatch overhead would dwarf
//! the work.

/// Compute `dst[i] += src[i]` for every `i` in `0..dst.len()`.
///
/// # Panics
///
/// Panics if `dst.len() != src.len()`; the message names both lengths.
pub fn add_inplace(dst: &mut [f32], src: &[f32]) {
    assert!(
        dst.len() == src.len(),
        "ggml add_inplace: dst.len()={} != src.len()={}",
        dst.len(),
        src.len(),
    );
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_hand_computed_small_slice() {
        let mut dst = [1.0_f32, 2.0, 3.0, 4.0];
        let src = [10.0_f32, 20.0, 30.0, 40.0];
        add_inplace(&mut dst, &src);
        assert_eq!(dst, [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn add_includes_sign_zero_and_infinity_cases() {
        let mut dst = [0.0_f32, -0.0, 1.5, f32::INFINITY, f32::NEG_INFINITY];
        let src = [0.0_f32, 0.0, -0.5, 1.0, f32::INFINITY];
        add_inplace(&mut dst, &src);
        assert_eq!(dst[0], 0.0);
        assert_eq!(dst[1], 0.0);
        assert_eq!(dst[2], 1.0);
        assert_eq!(dst[3], f32::INFINITY);
        assert!(dst[4].is_nan());
    }

    #[test]
    fn add_empty_slices_is_noop() {
        let mut dst: [f32; 0] = [];
        let src: [f32; 0] = [];
        add_inplace(&mut dst, &src);
    }

    #[test]
    fn add_matches_scalar_reference_byte_exact() {
        let dst_init: Vec<f32> = (0..257).map(|i| (i as f32) * 0.125 - 7.5).collect();
        let src: Vec<f32> = (0..257).map(|i| (i as f32).sin()).collect();

        let mut dst = dst_init.clone();
        add_inplace(&mut dst, &src);

        for i in 0..dst.len() {
            let expected = dst_init[i] + src[i];
            assert_eq!(
                dst[i].to_bits(),
                expected.to_bits(),
                "mismatch at index {i}",
            );
        }
    }

    #[test]
    #[should_panic(expected = "dst.len()=3 != src.len()=4")]
    fn add_panics_on_length_mismatch_shorter_dst() {
        let mut dst = [0.0_f32; 3];
        let src = [1.0_f32; 4];
        add_inplace(&mut dst, &src);
    }

    #[test]
    #[should_panic(expected = "dst.len()=5 != src.len()=2")]
    fn add_panics_on_length_mismatch_longer_dst() {
        let mut dst = [0.0_f32; 5];
        let src = [1.0_f32; 2];
        add_inplace(&mut dst, &src);
    }
}
