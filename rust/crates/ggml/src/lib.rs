//! Core `ggml` types, ported to safe Rust from `ggml/include/ggml.h` and the
//! `type_traits` table in `ggml/src/ggml.c`.
//!
//! This is part of the in-progress, bottom-up Rust port of llama.cpp. Today the
//! crate provides the tensor element-type system ([`GgmlType`]) that the GGUF
//! loader and the (future) compute backends are built on. Block/type sizes were
//! generated directly from the C headers to guarantee byte-for-byte agreement
//! with the C ABI, and are checked by the unit tests below.

#![forbid(unsafe_code)]

use core::fmt;

pub mod dequant;
pub mod layout;

pub use dequant::{dequant_f16, dequant_q8_0};
pub use layout::TensorLayout;

/// Maximum number of tensor dimensions (`GGML_MAX_DIMS`).
pub const GGML_MAX_DIMS: usize = 4;

/// Tensor element type. Mirrors `enum ggml_type` in `ggml.h`.
///
/// Discriminant values match the C ABI exactly, **including** the gaps left by
/// removed types: Q4_2/Q4_3 (4, 5), the Q4_0_4_4 family (31..=33) and the
/// IQ4_NL_4_4 family (36..=38). Casting `self as i32` therefore yields the same
/// integer that appears on disk in a GGUF tensor-info record.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(i32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
}

impl GgmlType {
    /// Every concrete (non-removed) type, in ascending discriminant order.
    pub const ALL: [GgmlType; 34] = {
        use GgmlType::*;
        [
            F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K,
            IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, I8, I16, I32, I64, F64,
            IQ1_M, BF16, TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0,
        ]
    };

    /// Decode the on-disk / C-ABI integer into a type, returning `None` for
    /// unknown or removed-type discriminants.
    pub fn from_i32(v: i32) -> Option<Self> {
        use GgmlType::*;
        Some(match v {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2_K,
            11 => Q3_K,
            12 => Q4_K,
            13 => Q5_K,
            14 => Q6_K,
            15 => Q8_K,
            16 => IQ2_XXS,
            17 => IQ2_XS,
            18 => IQ3_XXS,
            19 => IQ1_S,
            20 => IQ4_NL,
            21 => IQ3_S,
            22 => IQ2_S,
            23 => IQ4_XS,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            29 => IQ1_M,
            30 => BF16,
            34 => TQ1_0,
            35 => TQ2_0,
            39 => MXFP4,
            40 => NVFP4,
            41 => Q1_0,
            _ => return None,
        })
    }

    /// Canonical ggml name (`ggml_type_name`), e.g. `"q4_K"`.
    pub const fn name(self) -> &'static str {
        use GgmlType::*;
        match self {
            F32 => "f32",
            F16 => "f16",
            Q4_0 => "q4_0",
            Q4_1 => "q4_1",
            Q5_0 => "q5_0",
            Q5_1 => "q5_1",
            Q8_0 => "q8_0",
            Q8_1 => "q8_1",
            Q2_K => "q2_K",
            Q3_K => "q3_K",
            Q4_K => "q4_K",
            Q5_K => "q5_K",
            Q6_K => "q6_K",
            Q8_K => "q8_K",
            IQ2_XXS => "iq2_xxs",
            IQ2_XS => "iq2_xs",
            IQ3_XXS => "iq3_xxs",
            IQ1_S => "iq1_s",
            IQ4_NL => "iq4_nl",
            IQ3_S => "iq3_s",
            IQ2_S => "iq2_s",
            IQ4_XS => "iq4_xs",
            I8 => "i8",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            F64 => "f64",
            IQ1_M => "iq1_m",
            BF16 => "bf16",
            TQ1_0 => "tq1_0",
            TQ2_0 => "tq2_0",
            MXFP4 => "mxfp4",
            NVFP4 => "nvfp4",
            Q1_0 => "q1_0",
        }
    }

    /// Number of elements packed into one block (`ggml_blck_size`).
    /// Non-quantized scalar types have a block size of 1.
    pub const fn block_size(self) -> i64 {
        use GgmlType::*;
        match self {
            F32 | F16 | F64 | BF16 | I8 | I16 | I32 | I64 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 | IQ4_NL | MXFP4 => 32,
            NVFP4 => 64,
            Q1_0 => 128,
            // all K- and IQ- quants
            Q2_K | Q3_K | Q4_K | Q5_K | Q6_K | Q8_K | IQ2_XXS | IQ2_XS | IQ3_XXS | IQ1_S
            | IQ3_S | IQ2_S | IQ4_XS | IQ1_M | TQ1_0 | TQ2_0 => 256,
        }
    }

    /// Size in bytes of one block (`ggml_type_size`).
    /// Generated from `sizeof(block_*)` in the C headers.
    pub const fn type_size(self) -> usize {
        use GgmlType::*;
        match self {
            I8 => 1,
            F16 | I16 | BF16 => 2,
            F32 | I32 => 4,
            I64 | F64 => 8,
            MXFP4 => 17,
            Q4_0 | IQ4_NL | Q1_0 => 18,
            Q4_1 => 20,
            Q5_0 => 22,
            Q5_1 => 24,
            Q8_0 => 34,
            Q8_1 | NVFP4 => 36,
            IQ1_S => 50,
            TQ1_0 => 54,
            IQ1_M => 56,
            IQ2_XXS | TQ2_0 => 66,
            IQ2_XS => 74,
            IQ2_S => 82,
            Q2_K => 84,
            IQ3_XXS => 98,
            Q3_K | IQ3_S => 110,
            IQ4_XS => 136,
            Q4_K => 144,
            Q5_K => 176,
            Q6_K => 210,
            Q8_K => 292,
        }
    }

    /// True for the quantized formats (block size > 1).
    pub const fn is_quantized(self) -> bool {
        self.block_size() != 1
    }

    /// Bytes occupied by a contiguous row of `ne` elements (`ggml_row_size`).
    ///
    /// Panics if `ne` is not a multiple of [`block_size`](Self::block_size),
    /// matching the `GGML_ASSERT` in the C implementation.
    pub fn row_size(self, ne: i64) -> usize {
        let blck = self.block_size();
        assert!(
            ne % blck == 0,
            "ggml row_size: {ne} elements is not a multiple of block size {blck} for type {}",
            self.name()
        );
        self.type_size() * (ne as usize) / (blck as usize)
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_i32_round_trips_for_all_types() {
        for &t in &GgmlType::ALL {
            assert_eq!(
                GgmlType::from_i32(t as i32),
                Some(t),
                "round trip failed for {t}"
            );
        }
    }

    #[test]
    fn removed_and_unknown_discriminants_are_none() {
        for v in [4, 5, 31, 32, 33, 36, 37, 38, 42, -1, 1000] {
            assert_eq!(GgmlType::from_i32(v), None, "expected None for {v}");
        }
    }

    #[test]
    fn block_and_type_sizes_match_c_headers() {
        // (type, block_size, type_size) — generated from sizeof(block_*).
        let expect = [
            (GgmlType::F32, 1, 4),
            (GgmlType::F16, 1, 2),
            (GgmlType::BF16, 1, 2),
            (GgmlType::Q4_0, 32, 18),
            (GgmlType::Q4_1, 32, 20),
            (GgmlType::Q5_0, 32, 22),
            (GgmlType::Q5_1, 32, 24),
            (GgmlType::Q8_0, 32, 34),
            (GgmlType::Q8_1, 32, 36),
            (GgmlType::Q2_K, 256, 84),
            (GgmlType::Q3_K, 256, 110),
            (GgmlType::Q4_K, 256, 144),
            (GgmlType::Q5_K, 256, 176),
            (GgmlType::Q6_K, 256, 210),
            (GgmlType::Q8_K, 256, 292),
            (GgmlType::IQ4_XS, 256, 136),
            (GgmlType::MXFP4, 32, 17),
            (GgmlType::NVFP4, 64, 36),
            (GgmlType::Q1_0, 128, 18),
        ];
        for (t, blck, sz) in expect {
            assert_eq!(t.block_size(), blck, "block_size mismatch for {t}");
            assert_eq!(t.type_size(), sz, "type_size mismatch for {t}");
        }
    }

    #[test]
    fn row_size_matches_ggml_formula() {
        assert_eq!(GgmlType::F32.row_size(10), 40);
        assert_eq!(GgmlType::F16.row_size(10), 20);
        // 64 elements of Q8_0 = 2 blocks of 32 * 34 bytes = 68.
        assert_eq!(GgmlType::Q8_0.row_size(64), 68);
        // 256 elements of Q4_K = 1 block * 144 bytes.
        assert_eq!(GgmlType::Q4_K.row_size(256), 144);
    }

    #[test]
    #[should_panic(expected = "not a multiple of block size")]
    fn row_size_panics_on_unaligned() {
        // 30 is not a multiple of 32.
        let _ = GgmlType::Q4_0.row_size(30);
    }

    #[test]
    fn quantized_classification() {
        assert!(!GgmlType::F32.is_quantized());
        assert!(!GgmlType::I32.is_quantized());
        assert!(GgmlType::Q4_K.is_quantized());
        assert!(GgmlType::Q8_0.is_quantized());
    }
}
