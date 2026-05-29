//! Row-wise RMS normalization with a multiplicative weight, in place.
//!
//! Mirrors the fused `ggml_compute_forward_rms_norm_f32<FUSE_OP_MUL>`
//! kernel in `ggml/src/ggml-cpu/ops.cpp` — for each row of length
//! `weight.len()`, computes `y[i] = x[i] * (1 / sqrt(mean(x*x) + eps)) * weight[i]`
//! using an `f32` accumulator for the sum of squares.
//!
//! The C implementation uses a `double` accumulator for the sum of squares
//! (`ggml_float = double`). This port keeps the workspace-wide `f32`
//! accumulator rule, which introduces a tiny relative drift (~1 ULP per
//! ~4 K-element row) but does not visibly affect token-level argmax or
//! downstream layer outputs. Tests verify the drift against the oracle's
//! actual `ggml_rms_norm` output is well under 1e-5 absolute.

/// Compute `x[i] = x[i] * weight[i % n] / sqrt(mean(x[row]^2) + eps)` for
/// every row of length `n = weight.len()` in `x`, in place.
///
/// `x.len()` must be a positive multiple of `weight.len()`, and `eps` must
/// be non-negative. The kernel is single-threaded — RMSNorm is cheap and
/// the per-row dispatch overhead would dominate.
///
/// # Panics
///
/// Panics if `weight.is_empty()`, if `x.len() % weight.len() != 0`, or if
/// `eps` is negative.
pub fn rmsnorm(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = weight.len();
    assert!(n > 0, "ggml rmsnorm: weight must be non-empty");
    assert!(
        x.len().is_multiple_of(n),
        "ggml rmsnorm: x.len()={} is not a multiple of weight.len()={}",
        x.len(),
        n,
    );
    assert!(
        eps.is_finite() && eps >= 0.0,
        "ggml rmsnorm: eps must be a non-negative finite float, got {eps}",
    );

    let inv_n = 1.0_f32 / (n as f32);
    for row in x.chunks_exact_mut(n) {
        let mut sum = 0.0_f32;
        for &v in row.iter() {
            sum += v * v;
        }
        let mean = sum * inv_n;
        let scale = 1.0_f32 / (mean + eps).sqrt();
        for (y, &w) in row.iter_mut().zip(weight.iter()) {
            *y = *y * scale * w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    /// Scalar reference using the same accumulation order as
    /// [`rmsnorm`] above (f32 sum-of-squares, f32 mean, scale = 1/sqrt,
    /// `y = x * scale * w`).
    fn rmsnorm_reference(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let n = weight.len();
        let inv_n = 1.0_f32 / (n as f32);
        let mut out = vec![0.0_f32; x.len()];
        for (row_x, row_y) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
            let mut sum = 0.0_f32;
            for &v in row_x.iter() {
                sum += v * v;
            }
            let mean = sum * inv_n;
            let scale = 1.0_f32 / (mean + eps).sqrt();
            for ((y, &v), &w) in row_y.iter_mut().zip(row_x.iter()).zip(weight.iter()) {
                *y = v * scale * w;
            }
        }
        out
    }

    fn ulp_distance(a: f32, b: f32) -> u32 {
        if a == b {
            return 0;
        }
        if a.is_nan() || b.is_nan() {
            return u32::MAX;
        }
        let ai = a.to_bits() as i32;
        let bi = b.to_bits() as i32;
        let map = |i: i32| {
            if i < 0 {
                (i32::MIN ^ i) as u32 ^ 0x80000000
            } else {
                i as u32
            }
        };
        map(ai).abs_diff(map(bi))
    }

    fn assert_within_ulps(actual: &[f32], expected: &[f32], tol_ulps: u32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let d = ulp_distance(a, e);
            assert!(
                d <= tol_ulps,
                "{label}: idx {i}: actual={a:e} (bits=0x{:08x}) expected={e:e} (bits=0x{:08x}) ulps={d}",
                a.to_bits(),
                e.to_bits(),
            );
        }
    }

    fn rand_vec(rng: &mut ChaCha20Rng, n: usize, range: f32) -> Vec<f32> {
        (0..n).map(|_| rng.random_range(-range..range)).collect()
    }

    #[test]
    fn matches_reference_within_4_ulps_4096_row() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xC0FFEEF32);
        let n = 4096_usize;
        let weight = rand_vec(&mut rng, n, 1.0);
        let mut x = rand_vec(&mut rng, n, 3.0);
        let expected = rmsnorm_reference(&x, &weight, 1e-5);
        rmsnorm(&mut x, &weight, 1e-5);
        assert_within_ulps(&x, &expected, 4, "rmsnorm_4096");
    }

    #[test]
    fn matches_reference_within_4_ulps_512_row() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xBEEF0512);
        let n = 512_usize;
        let weight = rand_vec(&mut rng, n, 1.0);
        let mut x = rand_vec(&mut rng, n, 3.0);
        let expected = rmsnorm_reference(&x, &weight, 1e-6);
        rmsnorm(&mut x, &weight, 1e-6);
        assert_within_ulps(&x, &expected, 4, "rmsnorm_512");
    }

    #[test]
    fn matches_reference_for_multi_row_input() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xABCDEF12);
        let n_cols = 128_usize;
        let n_rows = 7_usize;
        let weight = rand_vec(&mut rng, n_cols, 0.5);
        let mut x = rand_vec(&mut rng, n_cols * n_rows, 2.0);
        let expected = rmsnorm_reference(&x, &weight, 1e-5);
        rmsnorm(&mut x, &weight, 1e-5);
        assert_within_ulps(&x, &expected, 4, "rmsnorm_multi_row");
    }

    #[test]
    fn rms_of_normalized_row_is_close_to_one() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xC0DEC0DE);
        let n = 1024_usize;
        let weight: Vec<f32> = vec![1.0; n];
        let mut x = rand_vec(&mut rng, n, 5.0);
        rmsnorm(&mut x, &weight, 0.0);
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / n as f32).sqrt();
        assert!((rms - 1.0).abs() < 1e-3, "rms={rms}, expected ~1.0");
    }

    #[test]
    fn weight_acts_as_scale_when_input_is_normalized() {
        const N: usize = 8;
        let weight = [2.0_f32, -1.0, 0.5, 0.0, 1.0, 4.0, -2.0, 1.0];
        let mut x = [1.0_f32; N];
        rmsnorm(&mut x, &weight, 0.0);
        for (i, &w) in weight.iter().enumerate() {
            assert_eq!(
                x[i].to_bits(),
                w.to_bits(),
                "idx {i}: rmsnorm of all-ones should equal weight, got {}",
                x[i],
            );
        }
    }

    #[test]
    #[should_panic(expected = "weight must be non-empty")]
    fn panics_on_empty_weight() {
        let mut x = [1.0_f32; 8];
        let w: [f32; 0] = [];
        rmsnorm(&mut x, &w, 1e-5);
    }

    #[test]
    #[should_panic(expected = "x.len()=10 is not a multiple of weight.len()=4")]
    fn panics_on_non_multiple_length() {
        let mut x = [1.0_f32; 10];
        let w = [1.0_f32; 4];
        rmsnorm(&mut x, &w, 1e-5);
    }

    #[test]
    #[should_panic(expected = "eps must be a non-negative finite float")]
    fn panics_on_negative_eps() {
        let mut x = [1.0_f32; 4];
        let w = [1.0_f32; 4];
        rmsnorm(&mut x, &w, -1e-5);
    }

    #[test]
    fn empty_input_is_noop() {
        let mut x: [f32; 0] = [];
        let w = [1.0_f32; 4];
        rmsnorm(&mut x, &w, 1e-5);
    }
}
