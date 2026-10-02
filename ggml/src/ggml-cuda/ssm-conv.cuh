#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// decode-only fusion of concat(conv_state, x) + conv_state cache update + ssm_conv (+ silu), see ggml_cuda_try_conv_state_fusion
void ggml_cuda_op_ssm_conv_state_update(ggml_backend_cuda_context & ctx, const ggml_tensor * concat,
        const ggml_tensor * cpy_dst, const ggml_tensor * conv, ggml_tensor * out, bool apply_silu,
        const ggml_tensor * conv_state_src = nullptr); // conv_state_src: f16 state read through a cast
