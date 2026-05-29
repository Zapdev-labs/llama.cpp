//! NeoX-style Rotary Position Embedding over an `f32` slice, in place.
//!
//! Mirrors the pair-rotation kernel used by `ggml_compute_forward_rope_f32`
//! in `ggml/src/ggml-cpu/ops.cpp` with the pair layout that this Rust
//! port adopts uniformly across Llama-3 and Qwen-2.5: per-head pairs
//! `(x[2i], x[2i+1])` for `i in 0..head_dim/2`. Each pair is rotated by
//! `angle = pos * freq[i]` where `freq[i] = 1 / theta^(2 * i / head_dim)`.
//!
//! The kernel is single-threaded — RoPE is invoked per query/key tensor
//! and parallelism happens at the outer (per-token / per-layer) scope.

/// Apply NeoX-style RoPE in place to `x`, a packed tensor of
/// `n_heads * head_dim` `f32` elements stored as `n_heads` contiguous
/// `head_dim`-sized heads.
///
/// For each head and each pair index `i in 0..head_dim/2`, the pair
/// `(x[2i], x[2i+1])` is rotated by `pos * freq[i]` where
/// `freq[i] = 1 / theta^(2 * i / head_dim)`:
///
/// ```text
/// x[2i]   := x[2i] * cos - x[2i+1] * sin
/// x[2i+1] := x[2i] * sin + x[2i+1] * cos
/// ```
///
/// # Panics
///
/// Panics if `head_dim` is odd, if `x.len() != head_dim * n_heads`, or
/// if `theta` is not strictly positive and finite.
pub fn rope_inplace_neox(x: &mut [f32], head_dim: usize, n_heads: usize, pos: usize, theta: f32) {
    assert!(
        head_dim.is_multiple_of(2),
        "ggml rope_inplace_neox: head_dim={head_dim} must be even",
    );
    assert!(
        x.len() == head_dim * n_heads,
        "ggml rope_inplace_neox: x.len()={} != head_dim*n_heads={}",
        x.len(),
        head_dim * n_heads,
    );
    assert!(
        theta.is_finite() && theta > 0.0,
        "ggml rope_inplace_neox: theta must be positive and finite, got {theta}",
    );

    if head_dim == 0 || n_heads == 0 {
        return;
    }

    let half = head_dim / 2;
    let inv_head_dim = 1.0_f32 / (head_dim as f32);
    let pos_f = pos as f32;

    for head in x.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let exponent = 2.0_f32 * (i as f32) * inv_head_dim;
            let freq = 1.0_f32 / theta.powf(exponent);
            let angle = pos_f * freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = head[2 * i];
            let x1 = head[2 * i + 1];
            head[2 * i] = x0 * cos - x1 * sin;
            head[2 * i + 1] = x0 * sin + x1 * cos;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    fn rope_reference(
        x: &[f32],
        head_dim: usize,
        n_heads: usize,
        pos: usize,
        theta: f32,
    ) -> Vec<f32> {
        assert_eq!(x.len(), head_dim * n_heads);
        let half = head_dim / 2;
        let inv_head_dim = 1.0_f32 / (head_dim as f32);
        let pos_f = pos as f32;
        let mut out = x.to_vec();
        for head in out.chunks_exact_mut(head_dim) {
            for i in 0..half {
                let exponent = 2.0_f32 * (i as f32) * inv_head_dim;
                let freq = 1.0_f32 / theta.powf(exponent);
                let angle = pos_f * freq;
                let (sin, cos) = angle.sin_cos();
                let x0 = head[2 * i];
                let x1 = head[2 * i + 1];
                head[2 * i] = x0 * cos - x1 * sin;
                head[2 * i + 1] = x0 * sin + x1 * cos;
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
    fn matches_reference_llama3_params_within_4_ulps() {
        let head_dim = 64_usize;
        let n_heads = 32_usize;
        let theta = 500_000.0_f32;
        let pos = 37_usize;
        let mut rng = ChaCha20Rng::seed_from_u64(0x50FE_C0FE_1311);
        let x_in = rand_vec(&mut rng, head_dim * n_heads, 3.0);
        let expected = rope_reference(&x_in, head_dim, n_heads, pos, theta);
        let mut x = x_in;
        rope_inplace_neox(&mut x, head_dim, n_heads, pos, theta);
        assert_within_ulps(&x, &expected, 4, "rope_llama3");
    }

    #[test]
    fn matches_reference_qwen25_params_within_4_ulps() {
        let head_dim = 64_usize;
        let n_heads = 32_usize;
        let theta = 1_000_000.0_f32;
        let pos = 37_usize;
        let mut rng = ChaCha20Rng::seed_from_u64(0x50FE_C0FE_0025);
        let x_in = rand_vec(&mut rng, head_dim * n_heads, 3.0);
        let expected = rope_reference(&x_in, head_dim, n_heads, pos, theta);
        let mut x = x_in;
        rope_inplace_neox(&mut x, head_dim, n_heads, pos, theta);
        assert_within_ulps(&x, &expected, 4, "rope_qwen25");
    }

    #[test]
    fn theta_changes_output_for_same_input() {
        let head_dim = 64_usize;
        let n_heads = 4_usize;
        let pos = 37_usize;
        let mut rng = ChaCha20Rng::seed_from_u64(0xD1FF_5E75_0123_4567);
        let x_in = rand_vec(&mut rng, head_dim * n_heads, 1.0);

        let mut x_a = x_in.clone();
        rope_inplace_neox(&mut x_a, head_dim, n_heads, pos, 500_000.0);

        let mut x_b = x_in;
        rope_inplace_neox(&mut x_b, head_dim, n_heads, pos, 1_000_000.0);

        let mut differs = false;
        for (a, b) in x_a.iter().zip(x_b.iter()) {
            if a.to_bits() != b.to_bits() {
                differs = true;
                break;
            }
        }
        assert!(
            differs,
            "rope output should differ when theta differs (proves theta is parameter-driven)",
        );
    }

    #[test]
    fn pos_zero_is_identity() {
        let head_dim = 16_usize;
        let n_heads = 3_usize;
        let mut rng = ChaCha20Rng::seed_from_u64(0x0050_0050_0050_0050);
        let x_in = rand_vec(&mut rng, head_dim * n_heads, 2.0);
        let mut x = x_in.clone();
        rope_inplace_neox(&mut x, head_dim, n_heads, 0, 500_000.0);
        for (i, (&a, &e)) in x.iter().zip(x_in.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "idx {i}: pos=0 should be identity"
            );
        }
    }

    #[test]
    fn first_pair_uses_angle_pos_independent_of_theta() {
        let head_dim = 8_usize;
        let n_heads = 1_usize;
        let theta = 500_000.0_f32;
        let pos = 5_usize;
        let mut x = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        rope_inplace_neox(&mut x, head_dim, n_heads, pos, theta);
        let (sin, cos) = (pos as f32).sin_cos();
        assert_eq!(x[0].to_bits(), cos.to_bits());
        assert_eq!(x[1].to_bits(), sin.to_bits());
    }

    #[test]
    fn rotation_preserves_pair_magnitude_within_tolerance() {
        let head_dim = 64_usize;
        let n_heads = 8_usize;
        let pos = 42_usize;
        let theta = 500_000.0_f32;
        let mut rng = ChaCha20Rng::seed_from_u64(0x0BAD_5E75_0011_2233);
        let x_in = rand_vec(&mut rng, head_dim * n_heads, 2.0);
        let mut x = x_in.clone();
        rope_inplace_neox(&mut x, head_dim, n_heads, pos, theta);
        for head_idx in 0..n_heads {
            let base = head_idx * head_dim;
            for i in 0..head_dim / 2 {
                let a0 = x_in[base + 2 * i];
                let a1 = x_in[base + 2 * i + 1];
                let b0 = x[base + 2 * i];
                let b1 = x[base + 2 * i + 1];
                let before = (a0 * a0 + a1 * a1).sqrt();
                let after = (b0 * b0 + b1 * b1).sqrt();
                assert!(
                    (before - after).abs() <= 1e-5 * before.max(1e-6),
                    "head {head_idx} pair {i}: before={before}, after={after}",
                );
            }
        }
    }

    #[test]
    fn empty_input_is_noop() {
        let mut x: [f32; 0] = [];
        rope_inplace_neox(&mut x, 0, 0, 0, 500_000.0);
    }

    #[test]
    #[should_panic(expected = "head_dim=63 must be even")]
    fn panics_on_odd_head_dim() {
        let mut x = vec![0.0_f32; 63];
        rope_inplace_neox(&mut x, 63, 1, 0, 500_000.0);
    }

    #[test]
    #[should_panic(expected = "x.len()=100 != head_dim*n_heads=128")]
    fn panics_on_length_mismatch() {
        let mut x = vec![0.0_f32; 100];
        rope_inplace_neox(&mut x, 64, 2, 0, 500_000.0);
    }

    #[test]
    #[should_panic(expected = "theta must be positive and finite")]
    fn panics_on_zero_theta() {
        let mut x = vec![0.0_f32; 64];
        rope_inplace_neox(&mut x, 64, 1, 0, 0.0);
    }

    #[test]
    #[should_panic(expected = "theta must be positive and finite")]
    fn panics_on_negative_theta() {
        let mut x = vec![0.0_f32; 64];
        rope_inplace_neox(&mut x, 64, 1, 0, -1.0);
    }
}
