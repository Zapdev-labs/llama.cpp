//! Numerically stable softmax over an `f32` slice, in place.
//!
//! Mirrors the unmasked `f32` branch of `ggml_compute_forward_soft_max_f32`
//! in `ggml/src/ggml-cpu/ops.cpp` — finds the row maximum, computes
//! `exp(x[i] - max)` into each slot while accumulating the sum, then
//! divides by the sum.
//!
//! The C kernel uses an `f64` (`ggml_float`) accumulator for the sum;
//! per the workspace rule we keep `f32` storage and instead use Kahan
//! compensated summation so the final `sum(exp(...))` matches the
//! mathematically exact value to within a few ULPs regardless of length.
//! Without compensation, naive `f32` summation at length 32 000 drifts
//! by ~5e-6, which violates the `sums-to-1.0 within 1e-6` contract.
//!
//! The op is single-threaded over its input: each call operates on one
//! row. Per-head / per-row parallelism is the caller's responsibility,
//! matching the determinism rules.

/// Compute the softmax of `x` in place using the standard max-subtraction
/// stabilization: `x[i] := exp(x[i] - max(x)) / sum_j(exp(x[j] - max(x)))`.
///
/// An empty input is a no-op. For a single-element input the output is
/// `1.0`. The kernel uses `f32` storage with Kahan-compensated `f32`
/// accumulation; the output sum equals `1.0` to within `1e-6` for any
/// length tested up to 32 000.
///
/// `x` must not contain `NaN`; if it does, `NaN` propagates into the
/// output.
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }

    let mut sum = 0.0_f32;
    let mut c = 0.0_f32;
    for v in x.iter_mut() {
        let e = (*v - max).exp();
        *v = e;
        let y = e - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    let inv = 1.0_f32 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    /// Scalar reference using the same Kahan-compensated `f32` sum as
    /// [`softmax_inplace`]. Should be byte-exact with the in-place
    /// kernel under identical inputs.
    fn softmax_reference(x: &[f32]) -> Vec<f32> {
        if x.is_empty() {
            return Vec::new();
        }
        let mut max = f32::NEG_INFINITY;
        for &v in x.iter() {
            if v > max {
                max = v;
            }
        }
        let mut out: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
        let mut sum = 0.0_f32;
        let mut c = 0.0_f32;
        for &v in out.iter() {
            let y = v - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        let inv = 1.0_f32 / sum;
        for v in out.iter_mut() {
            *v *= inv;
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
        let map = |i: i32| {
            if i < 0 {
                (i32::MIN ^ i) as u32 ^ 0x80000000
            } else {
                i as u32
            }
        };
        map(a.to_bits() as i32).abs_diff(map(b.to_bits() as i32))
    }

    fn assert_within_ulps(actual: &[f32], expected: &[f32], tol: u32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let d = ulp_distance(a, e);
            assert!(
                d <= tol,
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
    fn empty_input_is_noop() {
        let mut x: [f32; 0] = [];
        softmax_inplace(&mut x);
    }

    #[test]
    fn single_element_is_one() {
        for &v in &[-1e30_f32, -1.0, 0.0, 1.0, 1e30] {
            let mut x = [v];
            softmax_inplace(&mut x);
            assert_eq!(x[0].to_bits(), 1.0_f32.to_bits());
        }
    }

    #[test]
    fn two_element_hand_computed() {
        let mut x = [0.0_f32, 0.0];
        softmax_inplace(&mut x);
        assert_eq!(x[0].to_bits(), 0.5_f32.to_bits());
        assert_eq!(x[1].to_bits(), 0.5_f32.to_bits());
    }

    /// Kahan-compensated `f32` sum, so the verification doesn't itself
    /// accumulate ~`sqrt(n) * eps` of naive-summation error and mask the
    /// algorithm's actual sum-to-1 behavior.
    fn kahan_sum_f32(xs: &[f32]) -> f32 {
        let mut sum = 0.0_f32;
        let mut c = 0.0_f32;
        for &v in xs {
            let y = v - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sum
    }

    #[test]
    fn output_sums_to_one_within_tolerance_at_all_required_lengths() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x50F7_4A1E);
        for &n in &[1_usize, 2, 17, 1024, 32_000] {
            let mut x = rand_vec(&mut rng, n, 5.0);
            softmax_inplace(&mut x);
            let sum = kahan_sum_f32(&x);
            assert!(
                (sum - 1.0).abs() <= 1e-6,
                "len={n}: sum={sum}, error={}",
                (sum - 1.0).abs(),
            );
        }
    }

    #[test]
    fn matches_reference_within_4_ulps_for_random_inputs() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x500F_7A88);
        for &n in &[2_usize, 17, 1024, 32_000] {
            let x_in = rand_vec(&mut rng, n, 5.0);
            let expected = softmax_reference(&x_in);
            let mut x = x_in;
            softmax_inplace(&mut x);
            assert_within_ulps(&x, &expected, 4, &format!("softmax_n={n}"));
        }
    }

    #[test]
    fn stable_for_extreme_inputs_no_overflow() {
        let mut x = [1000.0_f32, 1000.0, 1000.0, 1000.0];
        softmax_inplace(&mut x);
        for &v in &x {
            assert_eq!(v.to_bits(), 0.25_f32.to_bits());
        }
    }

    #[test]
    fn dominant_element_concentrates_mass() {
        let mut x = [-1000.0_f32, 0.0, -1000.0];
        softmax_inplace(&mut x);
        assert_eq!(x[1].to_bits(), 1.0_f32.to_bits());
        assert_eq!(x[0], 0.0);
        assert_eq!(x[2], 0.0);
    }

    #[test]
    fn output_is_non_negative_for_finite_input() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xBADD_F00D);
        let mut x = rand_vec(&mut rng, 1024, 10.0);
        softmax_inplace(&mut x);
        for &v in &x {
            assert!(v >= 0.0 && v.is_finite(), "non-finite or negative: {v}");
        }
    }
}
