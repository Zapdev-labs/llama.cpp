//! SiLU (Swish-1) activation over an `f32` slice, in place.
//!
//! Mirrors the scalar branch of `ggml_vec_silu_f32` in
//! `ggml/src/ggml-cpu/vec.h`: `y = x / (1 + exp(-x))`. The kernel is
//! single-threaded — SwiGLU is invoked per-row inside the FFN block and
//! parallelism happens at that outer scope.

/// Compute `x[i] = x[i] * sigmoid(x[i])` for every `i`, in place.
///
/// Uses the standard `x / (1 + exp(-x))` form rather than the equivalent
/// `x * (1 / (1 + exp(-x)))` so the divide collapses with the multiply on
/// most CPUs and matches the C kernel's accumulation order.
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0_f32 + (-*v).exp());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    fn silu_reference(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| v / (1.0_f32 + (-v).exp())).collect()
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
        silu_inplace(&mut x);
    }

    #[test]
    fn zero_maps_to_zero() {
        let mut x = [0.0_f32, -0.0];
        silu_inplace(&mut x);
        assert_eq!(x[0].to_bits(), 0.0_f32.to_bits());
        assert_eq!(x[1].to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn large_positive_input_approaches_identity() {
        let mut x = [50.0_f32];
        silu_inplace(&mut x);
        assert!((x[0] - 50.0).abs() < 1e-5, "silu(50) ≈ 50, got {}", x[0]);
    }

    #[test]
    fn large_negative_input_approaches_zero() {
        let mut x = [-50.0_f32];
        silu_inplace(&mut x);
        assert!(x[0].abs() < 1e-15, "silu(-50) ≈ 0, got {}", x[0]);
    }

    #[test]
    fn matches_reference_within_2_ulps_random() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x5170_1234);
        let x_in = rand_vec(&mut rng, 4096, 10.0);
        let expected = silu_reference(&x_in);
        let mut x = x_in;
        silu_inplace(&mut x);
        assert_within_ulps(&x, &expected, 2, "silu_random_4096");
    }

    #[test]
    fn matches_reference_within_2_ulps_extended_range() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x0BAD_CAFE);
        let x_in = rand_vec(&mut rng, 1024, 88.0);
        let expected = silu_reference(&x_in);
        let mut x = x_in;
        silu_inplace(&mut x);
        assert_within_ulps(&x, &expected, 2, "silu_extended_range");
    }

    #[test]
    fn hand_computed_known_values() {
        let inputs = [1.0_f32, -1.0, 2.0, 0.5];
        let mut x = inputs;
        silu_inplace(&mut x);
        for (i, &v) in inputs.iter().enumerate() {
            let expected = v / (1.0 + (-v).exp());
            assert_eq!(x[i].to_bits(), expected.to_bits(), "idx {i}: input {v}");
        }
    }
}
