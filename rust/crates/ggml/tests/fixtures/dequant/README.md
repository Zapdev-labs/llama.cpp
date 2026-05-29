# Dequant fixtures

Inputs are deterministic byte streams; outputs are the f32 result of running
those inputs through llama.cpp's `dequantize_row_<type>` via the oracle
binary at `rust/oracle/build/oracle`.

| file                | bytes | format                                |
|---------------------|-------|---------------------------------------|
| `f16_input.bin`     | 2048  | 1024 little-endian `u16` (binary16)   |
| `f16_output.bin`    | 4096  | 1024 little-endian `f32`              |
| `q8_0_input.bin`    | 272   | 8 blocks of `{ d: f16, qs: [i8; 32] }`|
| `q8_0_output.bin`   | 1024  | 256 little-endian `f32`               |

Regenerate with `./regenerate.sh`. CI does NOT regenerate — the fixtures
are the contract.
