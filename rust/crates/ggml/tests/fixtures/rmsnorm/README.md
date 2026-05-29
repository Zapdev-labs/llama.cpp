# RMSNorm fixture

A 4096-element hidden state captured from `libllama.so` after block 0 of
`Llama-3.2-1B-Instruct-Q4_K_M.gguf` (so: the input to `blk.1.attn_norm`),
plus the matching `blk.1.attn_norm.weight` and the oracle's
`ggml_rms_norm(input, eps=1e-5) * weight` output.

The input has shape `(n_embd=2048, n_tokens=2)` — the prompt `"Hello"` with
the model-default Llama-3 BOS prepended tokenizes to 2 tokens; `input.bin`
holds the 2x2048 = 4096 f32 values produced for those tokens by block 0.

`expected.bin` is the actual `attn_norm-1` tensor captured during the same
forward pass via `ggml_backend_sched_set_eval_callback`, so the test
compares against ground truth that came out of the C++ implementation of
RMSNorm directly, not a re-derivation.

| file           | bytes | format                                             |
|----------------|-------|----------------------------------------------------|
| `input.bin`    | 16384 | 4096 little-endian `f32` (`ne=[2048, 2]`)          |
| `weight.bin`   | 8192  | 2048 little-endian `f32` (`blk.1.attn_norm.weight`)|
| `expected.bin` | 16384 | 4096 little-endian `f32`                           |

Regenerate with `./regenerate.sh`. CI does NOT regenerate — the fixture
is the contract.
