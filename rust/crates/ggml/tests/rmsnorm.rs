//! Integration test: `ggml::rmsnorm` applied to a real Llama-3.2-1B
//! block-0 hidden state must match the captured oracle output of the
//! same block within 1e-5 absolute error.
//!
//! Fixture provenance: see `tests/fixtures/rmsnorm/README.md`. The
//! oracle output is the actual `attn_norm-1` tensor produced inside
//! `libllama.so` during one decode of the prompt `"Hello"` against
//! `Llama-3.2-1B-Instruct-Q4_K_M.gguf`; the matching input
//! (`l_out-0`) and the weight (`blk.1.attn_norm.weight`) are captured
//! in the same run.

use std::path::PathBuf;

use ggml::rmsnorm;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rmsnorm")
}

fn read_f32s(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert!(
        bytes.len().is_multiple_of(4),
        "fixture {path:?} length {} not a multiple of 4",
        bytes.len(),
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn matches_oracle_for_llama3_block0_hidden_state_within_1e_5() {
    let dir = fixtures_dir();
    let mut x = read_f32s(&dir.join("input.bin"));
    let weight = read_f32s(&dir.join("weight.bin"));
    let expected = read_f32s(&dir.join("expected.bin"));

    assert_eq!(weight.len(), 2048, "weight should be n_embd=2048");
    assert_eq!(x.len(), 4096, "input should be 2 tokens * n_embd = 4096");
    assert_eq!(expected.len(), x.len(), "expected shape must match input");

    rmsnorm(&mut x, &weight, 1e-5);

    let mut max_abs = 0.0_f32;
    let mut max_idx = 0_usize;
    for (i, (&a, &e)) in x.iter().zip(expected.iter()).enumerate() {
        let d = (a - e).abs();
        if d > max_abs {
            max_abs = d;
            max_idx = i;
        }
    }
    assert!(
        max_abs < 1e-5,
        "max abs error {max_abs:e} at index {max_idx} (rust={} oracle={}); expected < 1e-5",
        x[max_idx],
        expected[max_idx],
    );
}
