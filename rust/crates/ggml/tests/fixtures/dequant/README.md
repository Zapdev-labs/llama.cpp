# Dequant fixtures

Inputs are deterministic byte streams; outputs are the f32 result of running
those inputs through llama.cpp's `dequantize_row_<type>` via the oracle
binary at `rust/oracle/build/oracle`.

| file                     | bytes | format                                                          |
|--------------------------|-------|-----------------------------------------------------------------|
| `f16_input.bin`          | 2048  | 1024 little-endian `u16` (binary16)                             |
| `f16_output.bin`         | 4096  | 1024 little-endian `f32`                                        |
| `q8_0_input.bin`         | 272   | 8 blocks of `{ d: f16, qs: [i8; 32] }`                          |
| `q8_0_output.bin`        | 1024  | 256 little-endian `f32`                                         |
| `q4_0_input.bin`         | 144   | 8 blocks of `{ d: f16, qs: [u8; 16] }`                          |
| `q4_0_output.bin`        | 1024  | 256 little-endian `f32`                                         |
| `q4_k_input.bin`         | 576   | 4 super-blocks of `{ d: f16, dmin: f16, scales: [u8; 12], qs: [u8; 128] }`; block 0 spans full 6-bit scale & min range |
| `q4_k_output.bin`        | 4096  | 1024 little-endian `f32`                                        |
| `q4_k_qwen_input.bin`    | 576   | first 4 Q4_K super-blocks of `blk.11.ffn_down.weight` from `qwen2.5-0.5b-instruct-q4_k_m.gguf` (the file's first Q4_K-typed tensor; `blk.0.attn_q.weight` is Q5_0 in this build, not Q4_K) |
| `q4_k_qwen_output.bin`   | 4096  | 1024 little-endian `f32`                                        |

Regenerate with `./regenerate.sh`. CI does NOT regenerate — the fixtures
are the contract.
