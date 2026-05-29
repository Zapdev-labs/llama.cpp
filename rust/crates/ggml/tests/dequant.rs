//! Integration tests: dequant kernels must be byte-equal (per `f32::to_bits`)
//! to the C++ oracle for committed fixture inputs.

use std::path::PathBuf;

use ggml::{dequant_f16, dequant_q4_0, dequant_q8_0};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dequant")
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

fn assert_byte_equal(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch ({} vs {})",
        actual.len(),
        expected.len(),
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{label}: divergence at index {i}: rust={a:e} (bits=0x{:08x}) oracle={e:e} (bits=0x{:08x})",
            a.to_bits(),
            e.to_bits(),
        );
    }
}

#[test]
fn f16_byte_exact_vs_oracle() {
    let dir = fixtures_dir();
    let src = std::fs::read(dir.join("f16_input.bin")).expect("f16_input.bin");
    let expected = read_f32s(&dir.join("f16_output.bin"));
    assert_eq!(src.len(), 2048, "f16 fixture should be 1024 u16 values");
    assert_eq!(
        expected.len(),
        1024,
        "f16 fixture should expand to 1024 f32 values"
    );

    let mut actual = vec![0.0f32; expected.len()];
    dequant_f16(&src, &mut actual);
    assert_byte_equal(&actual, &expected, "f16");
}

#[test]
fn q8_0_byte_exact_vs_oracle() {
    let dir = fixtures_dir();
    let src = std::fs::read(dir.join("q8_0_input.bin")).expect("q8_0_input.bin");
    let expected = read_f32s(&dir.join("q8_0_output.bin"));
    assert_eq!(
        src.len(),
        8 * 34,
        "q8_0 fixture should be 8 blocks (272 bytes)"
    );
    assert_eq!(
        expected.len(),
        8 * 32,
        "q8_0 fixture should expand to 256 f32 values"
    );

    let mut actual = vec![0.0f32; expected.len()];
    dequant_q8_0(&src, &mut actual);
    assert_byte_equal(&actual, &expected, "q8_0");
}

#[test]
fn q4_0_byte_exact_vs_oracle() {
    let dir = fixtures_dir();
    let src = std::fs::read(dir.join("q4_0_input.bin")).expect("q4_0_input.bin");
    let expected = read_f32s(&dir.join("q4_0_output.bin"));
    assert_eq!(
        src.len(),
        8 * 18,
        "q4_0 fixture should be 8 blocks (144 bytes)"
    );
    assert_eq!(
        expected.len(),
        8 * 32,
        "q4_0 fixture should expand to 256 f32 values"
    );

    let mut actual = vec![0.0f32; expected.len()];
    dequant_q4_0(&src, &mut actual);
    assert_byte_equal(&actual, &expected, "q4_0");
}
