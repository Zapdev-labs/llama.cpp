//! Pure-Rust CPU operator kernels.
//!
//! Mirrors the `ggml_compute_forward_*` family in
//! `ggml/src/ggml-cpu/ops.c`. Each kernel returns `()` and writes into a
//! caller-supplied `&mut [f32]`. Per the determinism rules in this
//! workspace's `AGENTS.md`, ops that parallelize do so over independent
//! rows; intra-row reductions stay on a single thread.

pub mod add;
pub mod matmul;
pub mod mul;
pub mod rmsnorm;
pub mod rope;
pub mod silu;
pub mod softmax;

pub use add::add_inplace;
pub use matmul::matmul_f32;
pub use mul::mul_inplace;
pub use rmsnorm::rmsnorm;
pub use rope::rope_inplace_neox;
pub use silu::silu_inplace;
pub use softmax::softmax_inplace;
