#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    void *    data;        // rollback slot 0
    ggml_type type;        // F32 or F16
    int64_t   slot_stride; // between rollback slots (0 when K==1), in elements
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// q/k l2_norm folded into the delta net: q and k are the l2_norm inputs, eps its epsilon
struct ggml_cuda_gated_delta_net_l2 {
    const ggml_tensor * q;
    const ggml_tensor * k;
    float               eps;
};

// true if the fused-l2 kernel can run this gated_delta_net (S_v == 128, scalar gate, >= 4 sequences)
bool ggml_cuda_gated_delta_net_can_fuse_l2(const ggml_tensor * dst);

void ggml_cuda_op_gated_delta_net_l2(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                     const ggml_cuda_gated_delta_net_fused_cache * cache, ggml_cuda_gated_delta_net_l2 l2);
