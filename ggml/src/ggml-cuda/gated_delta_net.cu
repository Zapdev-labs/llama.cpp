#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/convert.cuh"

// with 4 rows per lane, each lane owns 4 consecutive rows so state/q/k move as 16 B (f32) or 8 B (f16) vectors
template <typename T>
static __device__ __forceinline__ void gdn_load4(const T * p, float * out) {
    if constexpr (std::is_same_v<T, float>) {
        const float4 v = *(const float4 *) p;
        out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
    } else {
        static_assert(std::is_same_v<T, half>, "unsupported state type");
        const uint2  raw = *(const uint2 *) p;
        const float2 a   = __half22float2(*(const half2 *) &raw.x);
        const float2 b   = __half22float2(*(const half2 *) &raw.y);
        out[0] = a.x; out[1] = a.y; out[2] = b.x; out[3] = b.y;
    }
}

template <typename T>
static __device__ __forceinline__ void gdn_store4(T * p, const float * in) {
    if constexpr (std::is_same_v<T, float>) {
        *(float4 *) p = make_float4(in[0], in[1], in[2], in[3]);
    } else {
        static_assert(std::is_same_v<T, half>, "unsupported state type");
        const half2 a = __floats2half2_rn(in[0], in[1]);
        const half2 b = __floats2half2_rn(in[2], in[3]);
        uint2 raw;
        raw.x = *(const uint32_t *) &a;
        raw.y = *(const uint32_t *) &b;
        *(uint2 *) p = raw;
    }
}

template <int S_v, bool KDA, bool keep_rs_t, typename si_t, typename so_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const si_t *  curr_state,
                                     float *       dst,
                                     so_t *        state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    // row index of shard element r for this lane
    constexpr bool vec4 = rows_per_lane == 4;
    auto row = [lane](int r) { return vec4 ? lane * rows_per_lane + r : r * warp_size + lane; };

    ggml_cuda_pdl_sync();
    if constexpr (vec4) {
        gdn_load4(curr_state + lane * rows_per_lane, s_shard);
    } else {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] = ggml_cuda_cast<float>(curr_state[row(r)]);
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
        if constexpr (vec4) {
            gdn_load4(k_t + lane * rows_per_lane, k_reg);
            gdn_load4(q_t + lane * rows_per_lane, q_reg);
        } else {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_reg[r] = k_t[row(r)];
                q_reg[r] = q_t[row(r)];
            }
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = row(r);
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = row(r);
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                so_t * curr_state = state + target_slot * state_slot_stride;
                if constexpr (vec4) {
                    gdn_store4(curr_state + col * S_v + lane * rows_per_lane, s_shard);
                } else {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        curr_state[col * S_v + row(r)] = ggml_cuda_cast<so_t>(s_shard[r]);
                    }
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
        if constexpr (vec4) {
            gdn_store4(state + col * S_v + lane * rows_per_lane, s_shard);
        } else {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                state[col * S_v + row(r)] = ggml_cuda_cast<so_t>(s_shard[r]);
            }
        }
    }
}


// decode-oriented variant for S_v == 128, warp size 32, scalar gate: each lane owns 4 consecutive rows and each warp
// handles GDN_V4_COLS columns at once, so a warp has several independent state loads in flight
#define GDN_V4_COLS  8
#define GDN_V4_WARPS 4

template <bool keep_rs_t, typename si_t, typename so_t>
__global__ void __launch_bounds__(32 * GDN_V4_WARPS)
gated_delta_net_v4_cuda(const float * q,
                        const float * k,
                        const float * v,
                        const float * g,
                        const float * beta,
                        const si_t *  curr_state,
                        float *       dst,
                        so_t *        state,
                        int64_t       H,
                        int64_t       n_tokens,
                        int64_t       sq1,
                        int64_t       sq2,
                        int64_t       sq3,
                        int64_t       sv1,
                        int64_t       sv2,
                        int64_t       sv3,
                        int64_t       sb1,
                        int64_t       sb2,
                        int64_t       sb3,
                        const uint3   neqk1_magic,
                        const uint3   rq3_magic,
                        float         scale,
                        int64_t       state_slot_stride,
                        int           K,
                        float         l2_eps) {
    constexpr int S_v  = 128;
    constexpr int C    = GDN_V4_COLS;
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col0     = (blockIdx.z * blockDim.y + threadIdx.y) * C;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_offset = (sequence * H + h_idx) * S_v * S_v;
    state      += state_offset + col0 * S_v + lane * 4;
    curr_state += state_offset + col0 * S_v + lane * 4;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

    float s[C][4];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int c = 0; c < C; c++) {
        gdn_load4(curr_state + c * S_v, s[c]);
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float beta_val = beta[gb_offset];
        const float g_val    = expf(g[gb_offset]);

        float k_reg[4];
        float q_reg[4];
        gdn_load4(k_t + lane * 4, k_reg);
        gdn_load4(q_t + lane * 4, q_reg);

        if (l2_eps >= 0.0f) {
            // fused l2_norm of q and k (same formula as the l2_norm op)
            float ssq = q_reg[0]*q_reg[0] + q_reg[1]*q_reg[1] + q_reg[2]*q_reg[2] + q_reg[3]*q_reg[3];
            float ssk = k_reg[0]*k_reg[0] + k_reg[1]*k_reg[1] + k_reg[2]*k_reg[2] + k_reg[3]*k_reg[3];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                ssq += __shfl_xor_sync(0xffffffff, ssq, offset, 32);
                ssk += __shfl_xor_sync(0xffffffff, ssk, offset, 32);
            }
            const float sq = rsqrtf(fmaxf(ssq, l2_eps * l2_eps));
            const float sk = rsqrtf(fmaxf(ssk, l2_eps * l2_eps));
#pragma unroll
            for (int r = 0; r < 4; r++) {
                q_reg[r] *= sq;
                k_reg[r] *= sk;
            }
        }

        float v_col[C];
#pragma unroll
        for (int c = 0; c < C; c++) {
            v_col[c] = v_t[col0 + c];
        }

        // kv[col] = sum_i S[i][col] * k[i]
        float kv[C];
#pragma unroll
        for (int c = 0; c < C; c++) {
            kv[c] = s[c][0] * k_reg[0] + s[c][1] * k_reg[1] + s[c][2] * k_reg[2] + s[c][3] * k_reg[3];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
            for (int c = 0; c < C; c++) {
                kv[c] += __shfl_xor_sync(0xffffffff, kv[c], offset, 32);
            }
        }

        // S[i][col] = g * S[i][col] + k[i] * delta[col], attn[col] = sum_i S[i][col] * q[i]
        float attn[C];
#pragma unroll
        for (int c = 0; c < C; c++) {
            const float delta = (v_col[c] - g_val * kv[c]) * beta_val;
            attn[c] = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; r++) {
                s[c][r]  = g_val * s[c][r] + k_reg[r] * delta;
                attn[c] += s[c][r] * q_reg[r];
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
            for (int c = 0; c < C; c++) {
                attn[c] += __shfl_xor_sync(0xffffffff, attn[c], offset, 32);
            }
        }
#pragma unroll
        for (int c = 0; c < C; c++) {
            if (lane == c) {
                attn_data[col0 + c] = attn[c] * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
#pragma unroll
                for (int c = 0; c < C; c++) {
                    gdn_store4(state + target_slot * state_slot_stride + c * S_v, s[c]);
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < C; c++) {
            gdn_store4(state + c * S_v, s[c]);
        }
    }
}

template <bool KDA, bool keep_rs_t, typename si_t, typename so_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const si_t * s_d,
        float * dst_d, so_t * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, float l2_eps, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    static const bool v4_disabled = getenv("GGML_CUDA_GDN_NO_V4") != nullptr;
    if (!KDA && S_v == 128 && warp_size == 32 && n_seqs >= 4 && !v4_disabled) {
        const dim3 v4_grid(H, n_seqs, S_v / (GDN_V4_COLS * GDN_V4_WARPS));
        const dim3 v4_block(32, GDN_V4_WARPS, 1);
        const ggml_cuda_kernel_launch_params v4_params = ggml_cuda_kernel_launch_params(v4_grid, v4_block, 0, stream);
        ggml_cuda_kernel_launch(gated_delta_net_v4_cuda<keep_rs_t, si_t, so_t>, v4_params,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
            n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, l2_eps);
        return;
    }
    GGML_ASSERT(l2_eps < 0.0f && "fused l2 norm requires the v4 kernel");

    constexpr bool all_f32 = std::is_same_v<si_t, float> && std::is_same_v<so_t, float>;
    if constexpr (!all_f32) {
        GGML_ASSERT(S_v == 128 && "f16 recurrent state is only instantiated for S_v == 128");
        ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, si_t, so_t>, launch_params,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
        return;
    } else
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, si_t, so_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, si_t, so_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, si_t, so_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, si_t, so_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache,
        const ggml_cuda_gated_delta_net_l2 * l2 = nullptr) {
    // with l2 set, q and k are the inputs of the l2_norm nodes and are normalized in the kernel
    const ggml_tensor * src_q = l2 ? l2->q : dst->src[0];
    const ggml_tensor * src_k = l2 ? l2->k : dst->src[1];
    const float l2_eps = l2 ? l2->eps : -1.0f;
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));
    GGML_ASSERT(src_state->type == GGML_TYPE_F32 || src_state->type == GGML_TYPE_F16);

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    void *    state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    ggml_type state_type        = GGML_TYPE_F32;
    int64_t   state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_type        = cache->type;
        state_slot_stride = cache->slot_stride;
    }

    auto launch = [&](auto s_d, auto st_d) {
        if (kda) {
            if (keep_rs) {
                launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, st_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, l2_eps, stream);
            } else {
                launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, st_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, l2_eps, stream);
            }
        } else {
            if (keep_rs) {
                launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, st_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, l2_eps, stream);
            } else {
                launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, st_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, l2_eps, stream);
            }
        }
    };

    // the state may be read from (identity view) and written to (fused cache) an f16 recurrent cache
    auto launch_out = [&](auto s_d) {
        if (state_type == GGML_TYPE_F16) {
            launch(s_d, (half *) state_d);
        } else {
            GGML_ASSERT(state_type == GGML_TYPE_F32);
            launch(s_d, (float *) state_d);
        }
    };

    if (src_state->type == GGML_TYPE_F16) {
        launch_out((const half *) src_state->data);
    } else {
        launch_out((const float *) src_state->data);
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}

bool ggml_cuda_gated_delta_net_can_fuse_l2(const ggml_tensor * dst) {
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * g = dst->src[3];
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    static const bool v4_disabled = getenv("GGML_CUDA_GDN_NO_V4") != nullptr;
    return !v4_disabled && warp_size == 32 && v->ne[0] == 128 && g->ne[0] == 1 && v->ne[3] >= 4;
}

void ggml_cuda_op_gated_delta_net_l2(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                     const ggml_cuda_gated_delta_net_fused_cache * cache, ggml_cuda_gated_delta_net_l2 l2) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, cache, &l2);
}
