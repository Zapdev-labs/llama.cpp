//! Integration tests against the real GGUF vocab fixtures shipped in the
//! llama.cpp repo (`<repo>/models/ggml-vocab-*.gguf`).
//!
//! These files contain only metadata (0 tensors), so correct parsing of every
//! key-value pair means the read cursor must land *exactly* at end-of-file:
//! `data_offset == file length`. A single byte mis-parsed anywhere in the
//! metadata would break that invariant — a strong correctness signal.

use std::path::PathBuf;

use gguf::{Array, Gguf, Value};

fn model_path(name: &str) -> PathBuf {
    // tests run with CWD = crate dir (rust/crates/gguf); models are at repo root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../models")
        .join(name)
}

fn check_vocab_file(name: &str, expected_n_kv: usize) {
    let path = model_path(name);
    if !path.exists() {
        eprintln!("skipping {name}: fixture not present at {}", path.display());
        return;
    }

    let bytes = std::fs::read(&path).expect("read fixture");
    let g = Gguf::from_bytes(&bytes).unwrap_or_else(|e| panic!("parse {name}: {e}"));

    assert_eq!(g.version(), 3, "{name}: version");
    assert_eq!(g.tensors().len(), 0, "{name}: vocab files carry no tensors");
    assert_eq!(g.metadata().len(), expected_n_kv, "{name}: kv count");

    // First key in every llama.cpp vocab file.
    assert_eq!(
        g.metadata()[0].0,
        "general.architecture",
        "{name}: first key"
    );
    assert!(
        g.get_str("general.architecture").is_some(),
        "{name}: arch is a string"
    );

    // The tokenizer vocabulary must be a non-empty string array.
    match g.get("tokenizer.ggml.tokens").and_then(Value::as_array) {
        Some(Array::String(tokens)) => {
            assert!(!tokens.is_empty(), "{name}: token list is empty")
        }
        other => panic!("{name}: tokenizer.ggml.tokens missing or wrong type: {other:?}"),
    }

    // The decisive check: every byte of metadata accounted for.
    assert_eq!(
        g.data_offset() as usize,
        bytes.len(),
        "{name}: cursor did not consume the file exactly (metadata mis-parsed)"
    );
}

#[test]
fn gpt2_vocab() {
    check_vocab_file("ggml-vocab-gpt-2.gguf", 16);
}

#[test]
fn llama_spm_vocab() {
    check_vocab_file("ggml-vocab-llama-spm.gguf", 22);
}

#[test]
fn bert_bge_vocab() {
    check_vocab_file("ggml-vocab-bert-bge.gguf", 20);
}

/// Parse *every* vocab fixture present and assert the end-of-file invariant.
/// This exercises a wide range of metadata shapes (BPE merges, SPM scores,
/// token-type arrays, bool flags, etc.) across many tokenizers.
#[test]
fn all_vocab_fixtures_consume_exactly() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../models");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("skipping: {} not present", dir.display());
        return;
    };

    let mut checked = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        let is_vocab = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("ggml-vocab-") && n.ends_with(".gguf"));
        if !is_vocab {
            continue;
        }

        let bytes = std::fs::read(&path).expect("read fixture");
        let g =
            Gguf::from_bytes(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
        assert_eq!(
            g.data_offset() as usize,
            bytes.len(),
            "{}: metadata mis-parsed",
            path.display()
        );
        checked += 1;
    }

    eprintln!("verified {checked} vocab fixture(s)");
}
