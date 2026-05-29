//! Element-wise in-place multiplication over `f32` slices.
//!
//! Mirrors the `f32` specialization of `ggml_compute_forward_mul` in
//! `ggml/src/ggml-cpu/ops.c`. Single-threaded for the same reason as
//! [`super::add::add_inplace`].

/// Compute `dst[i] *= src[i]` for every `i` in `0..dst.len()`.
///
/// # Panics
///
/// Panics if `dst.len() != src.len()`; the message names both lengths.
pub fn mul_inplace(dst: &mut [f32], src: &[f32]) {
    assert!(
        dst.len() == src.len(),
        "ggml mul_inplace: dst.len()={} != src.len()={}",
        dst.len(),
        src.len(),
    );
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d *= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_hand_computed_small_slice() {
        let mut dst = [1.0_f32, 2.0, 3.0, 4.0];
        let src = [10.0_f32, 0.5, -1.0, 0.0];
        mul_inplace(&mut dst, &src);
        assert_eq!(dst, [10.0, 1.0, -3.0, 0.0]);
    }

    #[test]
    fn mul_includes_sign_zero_and_infinity_cases() {
        let mut dst = [-0.0_f32, 2.0, f32::INFINITY, f32::INFINITY, 1.0];
        let src = [1.0_f32, -3.0, 0.5, 0.0, f32::NAN];
        mul_inplace(&mut dst, &src);
        assert_eq!(dst[0].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(dst[1], -6.0);
        assert_eq!(dst[2], f32::INFINITY);
        assert!(dst[3].is_nan());
        assert!(dst[4].is_nan());
    }

    #[test]
    fn mul_empty_slices_is_noop() {
        let mut dst: [f32; 0] = [];
        let src: [f32; 0] = [];
        mul_inplace(&mut dst, &src);
    }

    #[test]
    fn mul_matches_scalar_reference_byte_exact() {
        let dst_init: Vec<f32> = (0..257).map(|i| (i as f32) * 0.125 - 7.5).collect();
        let src: Vec<f32> = (0..257).map(|i| (i as f32).cos()).collect();

        let mut dst = dst_init.clone();
        mul_inplace(&mut dst, &src);

        for i in 0..dst.len() {
            let expected = dst_init[i] * src[i];
            assert_eq!(
                dst[i].to_bits(),
                expected.to_bits(),
                "mismatch at index {i}",
            );
        }
    }

    #[test]
    #[should_panic(expected = "dst.len()=3 != src.len()=4")]
    fn mul_panics_on_length_mismatch_shorter_dst() {
        let mut dst = [0.0_f32; 3];
        let src = [1.0_f32; 4];
        mul_inplace(&mut dst, &src);
    }

    #[test]
    #[should_panic(expected = "dst.len()=5 != src.len()=2")]
    fn mul_panics_on_length_mismatch_longer_dst() {
        let mut dst = [0.0_f32; 5];
        let src = [1.0_f32; 2];
        mul_inplace(&mut dst, &src);
    }
}
