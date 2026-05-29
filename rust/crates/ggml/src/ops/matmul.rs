//! Row-parallel `f32` matrix multiply with deterministic, byte-exact output.
//!
//! Mirrors the `f32 x f32` path of `ggml_compute_forward_mul_mat` in
//! `ggml/src/ggml-cpu/ops.cpp`. Threads partition the rows of the output;
//! each row is computed by exactly one thread with a sequential scalar
//! `f32` accumulator over the inner (`k`) dimension. There is no
//! inter-thread floating-point reduction at any granularity, so the
//! output is bit-identical regardless of how many rayon worker threads
//! the call runs under.

/// Compute `out = a · b` where `a` has shape `(M, K)` and `b` has
/// shape `(K, N)`, both stored row-major. `out` is row-major `(M, N)`.
///
/// Each output row is computed by a single thread; the reduction over
/// `K` is sequential within that thread. This is the determinism
/// contract — the output is bit-exact regardless of thread count.
///
/// # Panics
///
/// Panics if:
/// - the inner dimensions don't match (`a_shape.1 != b_shape.0`); the
///   message names both shapes.
/// - any of `a`, `b`, `out` is the wrong length for its declared shape.
pub fn matmul_f32(
    a: &[f32],
    a_shape: (usize, usize),
    b: &[f32],
    b_shape: (usize, usize),
    out: &mut [f32],
) {
    let (m, k_a) = a_shape;
    let (k_b, n) = b_shape;
    assert!(
        k_a == k_b,
        "ggml matmul_f32: shape mismatch: a is {:?} (MxK), b is {:?} (KxN), K_a={k_a} != K_b={k_b}",
        a_shape,
        b_shape,
    );
    let k = k_a;
    assert!(
        a.len() == m * k,
        "ggml matmul_f32: a.len()={} does not match shape {:?} (expected {})",
        a.len(),
        a_shape,
        m * k,
    );
    assert!(
        b.len() == k * n,
        "ggml matmul_f32: b.len()={} does not match shape {:?} (expected {})",
        b.len(),
        b_shape,
        k * n,
    );
    assert!(
        out.len() == m * n,
        "ggml matmul_f32: out.len()={} does not match output shape ({m}, {n}) (expected {})",
        out.len(),
        m * n,
    );

    let compute_row = |row_idx: usize, out_row: &mut [f32]| {
        let a_row = &a[row_idx * k..(row_idx + 1) * k];
        for (col, out_cell) in out_row.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += a_row[kk] * b[kk * n + col];
            }
            *out_cell = acc;
        }
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(n)
            .enumerate()
            .for_each(|(row_idx, out_row)| compute_row(row_idx, out_row));
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (row_idx, out_row) in out.chunks_mut(n).enumerate() {
            compute_row(row_idx, out_row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    fn rand_vec(rng: &mut ChaCha20Rng, n: usize, range: f32) -> Vec<f32> {
        (0..n).map(|_| rng.random_range(-range..range)).collect()
    }

    /// Naive triple-loop reference using exactly the same per-row, per-col,
    /// per-k accumulation order as [`matmul_f32`].
    fn matmul_reference(
        a: &[f32],
        a_shape: (usize, usize),
        b: &[f32],
        b_shape: (usize, usize),
    ) -> Vec<f32> {
        let (m, k) = a_shape;
        let (_, n) = b_shape;
        let mut out = vec![0.0_f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0_f32;
                for kk in 0..k {
                    acc += a[row * k + kk] * b[kk * n + col];
                }
                out[row * n + col] = acc;
            }
        }
        out
    }

    fn assert_byte_exact(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}: idx {i}: actual={a:e} (bits=0x{:08x}) expected={e:e} (bits=0x{:08x})",
                a.to_bits(),
                e.to_bits(),
            );
        }
    }

    #[test]
    fn bit_exact_vs_naive_37x19x41() {
        let (m, k, n) = (37, 19, 41);
        let mut rng = ChaCha20Rng::seed_from_u64(0xA11A_3719_4100);
        let a = rand_vec(&mut rng, m * k, 2.0);
        let b = rand_vec(&mut rng, k * n, 2.0);
        let expected = matmul_reference(&a, (m, k), &b, (k, n));
        let mut out = vec![0.0_f32; m * n];
        matmul_f32(&a, (m, k), &b, (k, n), &mut out);
        assert_byte_exact(&out, &expected, "matmul_37x19x41");
    }

    #[test]
    fn bit_exact_vs_naive_128x256x64() {
        let (m, k, n) = (128, 256, 64);
        let mut rng = ChaCha20Rng::seed_from_u64(0xB22B_1282_5664);
        let a = rand_vec(&mut rng, m * k, 1.5);
        let b = rand_vec(&mut rng, k * n, 1.5);
        let expected = matmul_reference(&a, (m, k), &b, (k, n));
        let mut out = vec![0.0_f32; m * n];
        matmul_f32(&a, (m, k), &b, (k, n), &mut out);
        assert_byte_exact(&out, &expected, "matmul_128x256x64");
    }

    #[test]
    fn handles_single_row_and_single_column() {
        let (m, k, n) = (1, 8, 1);
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [0.5_f32, 0.25, 0.125, 0.0625, 0.5, 0.25, 0.125, 0.0625];
        let mut out = [0.0_f32];
        matmul_f32(&a, (m, k), &b, (k, n), &mut out);
        let mut expected = 0.0_f32;
        for i in 0..k {
            expected += a[i] * b[i];
        }
        assert_eq!(out[0].to_bits(), expected.to_bits());
    }

    #[test]
    fn identity_matrix_yields_input() {
        let (m, k, n) = (4, 3, 3);
        let a = [
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let b = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut out = vec![0.0_f32; m * n];
        matmul_f32(&a, (m, k), &b, (k, n), &mut out);
        for i in 0..a.len() {
            assert_eq!(out[i].to_bits(), a[i].to_bits(), "idx {i}");
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn thread_count_invariant() {
        use std::hash::{Hash, Hasher};
        let (m, k, n) = (128, 256, 64);
        let mut rng = ChaCha20Rng::seed_from_u64(0xDE7E_C0DE);
        let a = rand_vec(&mut rng, m * k, 1.5);
        let b = rand_vec(&mut rng, k * n, 1.5);

        let mut hashes = Vec::new();
        let mut outputs: Vec<Vec<f32>> = Vec::new();
        for &nt in &[1_usize, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .expect("rayon pool build");
            let mut out = vec![0.0_f32; m * n];
            pool.install(|| matmul_f32(&a, (m, k), &b, (k, n), &mut out));
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for &v in &out {
                v.to_bits().hash(&mut h);
            }
            hashes.push(h.finish());
            outputs.push(out);
        }
        for (i, h) in hashes.iter().enumerate().skip(1) {
            assert_eq!(
                *h,
                hashes[0],
                "hash mismatch at threads={} vs threads=1",
                [1, 2, 4, 8][i],
            );
        }
        for (i, out) in outputs.iter().enumerate().skip(1) {
            assert_byte_exact(out, &outputs[0], &format!("threads={}", [1, 2, 4, 8][i]));
        }
    }

    #[test]
    #[should_panic(
        expected = "shape mismatch: a is (3, 5) (MxK), b is (4, 7) (KxN), K_a=5 != K_b=4"
    )]
    fn shape_mismatch_panics_with_both_shapes() {
        let a = vec![0.0_f32; 3 * 5];
        let b = vec![0.0_f32; 4 * 7];
        let mut out = vec![0.0_f32; 3 * 7];
        matmul_f32(&a, (3, 5), &b, (4, 7), &mut out);
    }

    #[test]
    #[should_panic(expected = "a.len()=10 does not match shape (3, 5)")]
    fn wrong_a_length_panics() {
        let a = vec![0.0_f32; 10];
        let b = vec![0.0_f32; 5 * 4];
        let mut out = vec![0.0_f32; 3 * 4];
        matmul_f32(&a, (3, 5), &b, (5, 4), &mut out);
    }

    #[test]
    #[should_panic(expected = "out.len()=11 does not match output shape (3, 4)")]
    fn wrong_out_length_panics() {
        let a = vec![0.0_f32; 3 * 5];
        let b = vec![0.0_f32; 5 * 4];
        let mut out = vec![0.0_f32; 11];
        matmul_f32(&a, (3, 5), &b, (5, 4), &mut out);
    }
}
