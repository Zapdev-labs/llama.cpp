//! Tensor memory layout: extents (`ne`) and byte strides (`nb`).
//!
//! Mirrors the layout fields of `struct ggml_tensor` in
//! `ggml/include/ggml.h` and the contiguous packing computed by
//! `ggml_new_tensor_impl` in `ggml/src/ggml.c`.

use crate::{GgmlType, GGML_MAX_DIMS};

/// Tensor memory layout: per-dimension element extents (`ne`), per-dimension
/// byte strides (`nb`), and the element type (`ty`).
///
/// Ported from `struct ggml_tensor` in `ggml/include/ggml.h`. The semantics
/// of `ne` and `nb` match the C side exactly: `ne[0]` is the fastest-moving
/// dimension and `nb[0]` is the byte distance between consecutive elements
/// along that dimension (one *block* worth of bytes for quantized types,
/// since elements are packed into blocks).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorLayout {
    /// Element type.
    pub ty: GgmlType,
    /// Per-dimension extents (number of elements along each axis).
    pub ne: [i64; GGML_MAX_DIMS],
    /// Per-dimension byte strides.
    pub nb: [usize; GGML_MAX_DIMS],
}

impl TensorLayout {
    /// Build a contiguous (row-major, block-packed) layout for a tensor of
    /// element type `ty` and extents `ne`.
    ///
    /// The stride formula is the one used by `ggml_new_tensor_impl` in
    /// `ggml/src/ggml.c`:
    ///
    /// ```text
    /// nb[0] = type_size
    /// nb[1] = nb[0] * ne[0] / block_size
    /// nb[k] = nb[k-1] * ne[k-1]      for k >= 2
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `ne[0]` is not a multiple of `ty.block_size()`, matching
    /// the `GGML_ASSERT(ne[0] % ggml_blck_size(type) == 0)` precondition in
    /// `ggml.c`. The panic message names the offending block size.
    pub fn contiguous(ty: GgmlType, ne: [i64; GGML_MAX_DIMS]) -> Self {
        let blck = ty.block_size();
        assert!(
            ne[0] % blck == 0,
            "ggml layout: ne[0]={} is not a multiple of block size {} for type {}",
            ne[0],
            blck,
            ty.name(),
        );
        let mut nb = [0usize; GGML_MAX_DIMS];
        nb[0] = ty.type_size();
        nb[1] = nb[0] * (ne[0] as usize) / (blck as usize);
        for k in 2..GGML_MAX_DIMS {
            nb[k] = nb[k - 1] * (ne[k - 1] as usize);
        }
        Self { ty, ne, nb }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_packing(ty: GgmlType, ne: [i64; GGML_MAX_DIMS]) {
        let l = TensorLayout::contiguous(ty, ne);
        assert_eq!(l.ty, ty);
        assert_eq!(l.ne, ne);
        assert_eq!(l.nb[0], ty.type_size(), "nb[0] for {ty}");
        assert_eq!(
            l.nb[1],
            l.nb[0] * (ne[0] as usize) / (ty.block_size() as usize),
            "nb[1] for {ty}",
        );
        for k in 2..GGML_MAX_DIMS {
            assert_eq!(
                l.nb[k],
                l.nb[k - 1] * (ne[k - 1] as usize),
                "nb[{k}] for {ty}",
            );
        }
    }

    #[test]
    fn f32_contiguous_packing() {
        let l = TensorLayout::contiguous(GgmlType::F32, [10, 3, 2, 1]);
        assert_eq!(l.nb, [4, 40, 120, 240]);
    }

    #[test]
    fn f16_contiguous_packing() {
        let l = TensorLayout::contiguous(GgmlType::F16, [8, 4, 2, 1]);
        assert_eq!(l.nb, [2, 16, 64, 128]);
    }

    #[test]
    fn q8_0_contiguous_packing() {
        // 64 elements = 2 blocks of 32, each block is 34 bytes.
        let l = TensorLayout::contiguous(GgmlType::Q8_0, [64, 5, 1, 1]);
        assert_eq!(l.nb[0], 34);
        assert_eq!(l.nb[1], 68);
        assert_eq!(l.nb[2], 340);
        assert_eq!(l.nb[3], 340);
    }

    #[test]
    fn q4_0_contiguous_packing() {
        // 32 elements per block, 18 bytes per block.
        let l = TensorLayout::contiguous(GgmlType::Q4_0, [128, 7, 3, 1]);
        assert_eq!(l.nb[0], 18);
        assert_eq!(l.nb[1], 18 * (128 / 32));
        assert_eq!(l.nb[2], l.nb[1] * 7);
        assert_eq!(l.nb[3], l.nb[2] * 3);
    }

    #[test]
    fn q4_k_contiguous_packing() {
        // 256 elements per super-block, 144 bytes per block.
        let l = TensorLayout::contiguous(GgmlType::Q4_K, [512, 4, 2, 1]);
        assert_eq!(l.nb[0], 144);
        assert_eq!(l.nb[1], 144 * (512 / 256));
        assert_eq!(l.nb[2], l.nb[1] * 4);
        assert_eq!(l.nb[3], l.nb[2] * 2);
    }

    #[test]
    fn all_in_scope_types_satisfy_packing_invariant() {
        let cases: &[(GgmlType, [i64; GGML_MAX_DIMS])] = &[
            (GgmlType::F32, [17, 5, 3, 2]),
            (GgmlType::F16, [33, 4, 2, 1]),
            (GgmlType::Q8_0, [256, 8, 2, 1]),
            (GgmlType::Q4_0, [64, 6, 1, 1]),
            (GgmlType::Q4_K, [768, 3, 1, 1]),
        ];
        for &(ty, ne) in cases {
            check_packing(ty, ne);
        }
    }

    #[test]
    fn one_dimensional_packing_collapses_to_block_count_bytes() {
        // A single row of 256 Q4_K elements: nb[1] should equal one super-block's bytes.
        let l = TensorLayout::contiguous(GgmlType::Q4_K, [256, 1, 1, 1]);
        assert_eq!(l.nb[0], GgmlType::Q4_K.type_size());
        assert_eq!(l.nb[1], GgmlType::Q4_K.type_size());
        assert_eq!(l.nb[2], l.nb[1]);
        assert_eq!(l.nb[3], l.nb[2]);
    }

    #[test]
    #[should_panic(expected = "block size 256")]
    fn q4_k_unaligned_ne0_panics_naming_block_size() {
        let _ = TensorLayout::contiguous(GgmlType::Q4_K, [200, 4, 1, 1]);
    }

    #[test]
    #[should_panic(expected = "block size 32")]
    fn q4_0_unaligned_ne0_panics_naming_block_size() {
        let _ = TensorLayout::contiguous(GgmlType::Q4_0, [30, 1, 1, 1]);
    }

    #[test]
    #[should_panic(expected = "block size 32")]
    fn q8_0_unaligned_ne0_panics_naming_block_size() {
        let _ = TensorLayout::contiguous(GgmlType::Q8_0, [17, 1, 1, 1]);
    }
}
