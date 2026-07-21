#include "models.h"

#include "llama-kv-cache.h"

#include <map>

// LongCat-Flash (Meituan), LongCat-Flash-Lite
//
// Architecture notes:
//   - each HF "layer" holds 2 MLA attention sublayers + 2 dense FFNs + 1 shortcut MoE (ScMoE);
//     mapped here to 2 gguf blocks, the MoE tensors live on the even block
//   - MoE routes over n_expert + n_expert_zero logits; the zero-computation experts are identity,
//     so their contribution reduces to sum(zero weights) * x and needs no extra ggml op
//   - token embeddings are enriched with n-gram hash embeddings computed from the previous
//     n_ngram_neighbors-1 tokens; the token history is recovered from the KV cells (cell ext)

void llama_model_longcat_flash::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX;
    }
    ml.get_key(LLM_KV_EXPERT_ZERO_COUNT,           hparams.n_expert_zero);
    ml.get_key(LLM_KV_NGRAM_NEIGHBOR_COUNT,        hparams.n_ngram_neighbors);
    ml.get_key(LLM_KV_NGRAM_SPLIT_COUNT,           hparams.n_ngram_splits);

    if (ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, false)) {
        // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX] cancel the factor from the convert script
        hparams.rope_yarn_log_mul /= 0.1f;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_longcat_flash::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_head_k_mla   = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla   = hparams.n_embd_head_v_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const int64_t n_ff_exp     = hparams.n_ff_exp;

    const int64_t n_expert_total = n_expert + hparams.n_expert_zero;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // n-gram embeddings: each table has its own (prime-ish) row count - take it from the tensor itself
    const uint32_t n_ngram_embd = hparams.n_ngram_splits * (hparams.n_ngram_neighbors - 1);
    GGML_ASSERT(n_ngram_embd > 0 && n_embd % n_ngram_embd == 0);
    const int64_t ngram_dim = n_embd / n_ngram_embd;

    ngram_embd.resize(n_ngram_embd);
    ngram_proj.resize(n_ngram_embd);
    for (uint32_t i = 0; i < n_ngram_embd; ++i) {
        // the tables are indexed via the name suffix - they are input-layer tensors, not per-block ones
        const std::string sfx_embd = std::to_string(i) + ".weight";
        const std::string sfx_proj = std::to_string(i) + ".weight";

        const std::string name = tn(LLM_TENSOR_NGRAM_EMBD, sfx_embd.c_str()).str();
        ggml_tensor * meta = ml.get_tensor_meta(name.c_str());
        if (meta == nullptr) {
            throw std::runtime_error("missing tensor: " + name);
        }
        ngram_embd[i] = create_tensor(tn(LLM_TENSOR_NGRAM_EMBD, sfx_embd.c_str()), {ngram_dim, meta->ne[1]}, 0);
        ngram_proj[i] = create_tensor(tn(LLM_TENSOR_NGRAM_PROJ, sfx_proj.c_str()), {ngram_dim, n_embd}, 0);
    }

    GGML_ASSERT(n_layer % 2 == 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank}, 0);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);

        layer.wq_a      = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,     "weight", i), {n_embd, q_lora_rank}, 0);
        layer.wq_b      = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,     "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, 0);
        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, 0);
        layer.wk_b      = create_tensor(tn(LLM_TENSOR_ATTN_K_B,     "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, 0);
        layer.wv_b      = create_tensor(tn(LLM_TENSOR_ATTN_V_B,     "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,     "weight", i), {n_head * n_embd_head_v_mla, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        // dense FFN on every block
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        // shortcut MoE on even blocks
        if (i % 2 == 0) {
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert_total}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert_total}, 0);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
        }
    }
}

// input holding the token ids and the per-embedder n-gram hash ids
class llm_graph_input_longcat_ngram : public llm_graph_input_i {
public:
    llm_graph_input_longcat_ngram(
            const llama_model_longcat_flash & model,
            const llama_kv_cache_context    * mctx,
            llama_token                       tok_eos) :
        n_neighbors(model.hparams.n_ngram_neighbors),
        n_vocab(model.tok_embd->ne[1]),
        tok_eos(tok_eos),
        mctx(mctx) {
        for (const auto * t : model.ngram_embd) {
            rows.push_back(t->ne[1]);
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        this->mctx = static_cast<const llama_kv_cache_context *>(params.mctx);
        return tokens && tokens->ne[0] == params.ubatch.n_tokens && params.ubatch.token;
    }

    void set_input(const llama_ubatch * ubatch) override {
        const uint32_t n_tokens = ubatch->n_tokens;
        const uint32_t n_embdrs = (uint32_t) rows.size();
        const uint32_t n_hist   = n_neighbors - 1;

        GGML_ASSERT(ubatch->token != nullptr && "longcat-flash requires token input");

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));

        // (seq, pos) -> token for the current ubatch
        std::map<std::pair<llama_seq_id, llama_pos>, llama_token> batch_toks;
        for (uint32_t i = 0; i < n_tokens; ++i) {
            batch_toks[{ubatch->seq_id[i][0], ubatch->pos[i]}] = ubatch->token[i];
        }

        std::vector<int32_t> ids(n_tokens*n_embdrs);
        std::vector<llama_token> prev(n_hist);  // prev[d-1] = token at distance d

        for (uint32_t i = 0; i < n_tokens; ++i) {
            const llama_seq_id seq = ubatch->seq_id[i][0];
            const llama_pos    pos = ubatch->pos[i];

            for (uint32_t d = 1; d <= n_hist; ++d) {
                llama_token t = -1;
                const auto it = batch_toks.find({seq, pos - (llama_pos) d});
                if (it != batch_toks.end()) {
                    t = it->second;
                } else if (mctx && pos >= (llama_pos) d) {
                    t = mctx->seq_token_at(seq, pos - d);
                }
                prev[d - 1] = t;
            }

            // shifted token at distance d, zeroed if unknown or if an EOS occurs in between
            // (matches the reference _shift_right_ignore_eos)
            std::vector<int64_t> sh(n_hist);
            bool eos_seen = false;
            for (uint32_t d = 1; d <= n_hist; ++d) {
                const llama_token t = prev[d - 1];
                if (t == tok_eos) {
                    eos_seen = true; // an EOS neighbor ends the segment, including itself
                }
                sh[d - 1] = (eos_seen || t < 0) ? 0 : (int64_t) t;
            }

            // polynomial rolling hash per embedder table
            for (uint32_t e = 0; e < n_embdrs; ++e) {
                const uint32_t ngram = 2 + e / (n_embdrs / n_hist); // (i-2)*K + j indexing: i = 2 + e/K
                const int64_t  D     = rows[e];

                int64_t id = ubatch->token[i];
                int64_t power_mod = 1;
                for (uint32_t k = 2; k <= ngram; ++k) {
                    power_mod = (power_mod * n_vocab) % D;
                    id += sh[k - 2] * power_mod;
                }
                ids[e*n_tokens + i] = (int32_t) (id % D);
            }
        }

        ggml_backend_tensor_set(ngram_ids, ids.data(), 0, ids.size()*sizeof(int32_t));
    }

    ggml_tensor * tokens    = nullptr; // I32 [n_tokens]
    ggml_tensor * ngram_ids = nullptr; // I32 [n_tokens, n_embedders]

    const uint32_t    n_neighbors;
    const int64_t     n_vocab;
    const llama_token tok_eos;

    std::vector<int64_t> rows;

    const llama_kv_cache_context * mctx;
};

std::unique_ptr<llm_graph_context> llama_model_longcat_flash::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_longcat_flash::graph::graph(const llama_model_longcat_flash & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head_k       = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    // YaRN mscale pre-scaling, same as deepseek2 [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    // MLA low-rank scaling factors
    const float scale_q_lora  = sqrtf((float) n_embd / hparams.n_lora_q);
    const float scale_kv_lora = sqrtf((float) n_embd / hparams.n_lora_kv);

    // token + n-gram embeddings
    ggml_tensor * inpL;
    {
        auto inp = std::make_unique<llm_graph_input_longcat_ngram>(
                model, static_cast<const llama_kv_cache_context *>(mctx), model.vocab.token_eos());

        const int64_t n_embdrs = (int64_t) model.ngram_embd.size();

        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;

        inp->ngram_ids = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_embdrs);
        ggml_set_input(inp->ngram_ids);

        inpL = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);

        for (int64_t e = 0; e < n_embdrs; ++e) {
            ggml_tensor * ids_e = ggml_view_1d(ctx0, inp->ngram_ids, n_tokens, e*inp->ngram_ids->nb[1]);
            ggml_tensor * emb   = ggml_get_rows(ctx0, model.ngram_embd[e], ids_e);
            inpL = ggml_add(ctx0, inpL, ggml_mul_mat(ctx0, model.ngram_proj[e], emb));
        }

        inpL = ggml_scale(ctx0, inpL, 1.0f / (float) (1 + n_embdrs));
        cb(inpL, "inp_embd_ngram", -1);

        res->add_input(std::move(inp));
    }

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn_k = build_attn_inp_k();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MLA attention with the absorption optimization (see deepseek2)
    auto build_mla = [&](ggml_tensor * cur, int il) -> ggml_tensor * {
        ggml_tensor * q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
        q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
        q = ggml_scale(ctx0, q, scale_q_lora);
        cb(q, "q", il);

        ggml_tensor * q_nope =
            ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                         ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
        ggml_tensor * q_pe = ggml_view_3d(
            ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
            ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));

        ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
        cb(kv_cmpr_pe, "kv_cmpr_pe", il);

        ggml_tensor * kv_cmpr =
            ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                         ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);

        ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                          ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                          ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                          ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

        q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);
        k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);

        kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
        kv_cmpr = ggml_scale(ctx0, kv_cmpr, scale_kv_lora);
        cb(kv_cmpr, "kv_cmpr", il);

        q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
        ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
        q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);

        // note: rope must go first for in-place context shifting in build_rope_shift()
        ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
        cb(Qcur, "Qcur", il);

        kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
        ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
        cb(Kcur, "Kcur", il);

        ggml_tensor * Vcur = kv_cmpr;

        return build_attn(inp_attn_k,
                model.layers[il].wo, NULL, model.layers[il].wo_s,
                Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);
    };

    auto build_dense_ffn = [&](ggml_tensor * cur, int il) -> ggml_tensor * {
        return build_ffn(cur,
            model.layers[il].ffn_up,   NULL, NULL,
            model.layers[il].ffn_gate, NULL, NULL,
            model.layers[il].ffn_down, NULL, NULL,
            NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
    };

    // MoE with zero-computation (identity) experts
    auto build_zero_moe = [&](ggml_tensor * x, int il) -> ggml_tensor * {
        const auto & layer = model.layers[il];

        const int64_t n_expert_total = n_expert + hparams.n_expert_zero;

        ggml_tensor * logits = ggml_mul_mat(ctx0, layer.ffn_gate_inp, x); // [n_expert_total, T]
        cb(logits, "ffn_moe_logits", il);

        ggml_tensor * probs = ggml_soft_max(ctx0, logits);
        ggml_tensor * sel   = ggml_add(ctx0, probs, layer.ffn_exp_probs_b);
        cb(sel, "ffn_moe_probs_biased", il);

        ggml_tensor * ids = ggml_argsort_top_k(ctx0, sel, n_expert_used); // I32 [n_used, T]
        cb(ids, "ffn_moe_topk", il);

        ggml_tensor * w = ggml_get_rows(ctx0,
                ggml_reshape_3d(ctx0, probs, 1, n_expert_total, n_tokens), ids); // [1, n_used, T]
        w = ggml_scale(ctx0, w, hparams.expert_weights_scale);
        cb(w, "ffn_moe_weights", il);

        // split identity experts (id >= n_expert) from routed ones
        ggml_tensor * idf       = ggml_cast(ctx0, ids, GGML_TYPE_F32); // [n_used, T]
        ggml_tensor * is_zero   = ggml_step(ctx0, ggml_scale_bias(ctx0, idf, 1.0f, -((float) n_expert - 0.5f)));
        ggml_tensor * is_routed = ggml_scale_bias(ctx0, is_zero, -1.0f, 1.0f);

        ggml_tensor * w2 = ggml_reshape_2d(ctx0, w, n_expert_used, n_tokens);

        ggml_tensor * w_zero_sum = ggml_sum_rows(ctx0, ggml_mul(ctx0, w2, is_zero)); // [1, T]
        cb(w_zero_sum, "ffn_moe_zero_w", il);

        ggml_tensor * w_routed = ggml_reshape_3d(ctx0, ggml_mul(ctx0, w2, is_routed), 1, n_expert_used, n_tokens);

        // zero-expert slots are clamped to a valid expert id; their weight is 0 so the result is unaffected
        // note: ggml_clamp works in-place, so it needs its own cast of the ids
        ggml_tensor * ids_routed = ggml_cast(ctx0,
                ggml_clamp(ctx0, ggml_cast(ctx0, ids, GGML_TYPE_F32), 0.0f, (float) n_expert - 1), GGML_TYPE_I32);

        ggml_tensor * xr = ggml_reshape_3d(ctx0, x, n_embd, 1, n_tokens);

        ggml_tensor * gate = ggml_mul_mat_id(ctx0, layer.ffn_gate_exps, xr, ids_routed); // [n_ff_exp, n_used, T]
        ggml_tensor * up   = ggml_mul_mat_id(ctx0, layer.ffn_up_exps,   xr, ids_routed);

        ggml_tensor * act = ggml_swiglu_split(ctx0, gate, up);

        ggml_tensor * experts = ggml_mul_mat_id(ctx0, layer.ffn_down_exps, act, ids_routed); // [n_embd, n_used, T]
        experts = ggml_mul(ctx0, experts, w_routed);
        cb(experts, "ffn_moe_weighted", il);

        ggml_build_forward_expand(gf, experts);

        ggml_tensor * moe_out = nullptr;
        for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
            ggml_tensor * v = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);
            moe_out = moe_out ? ggml_add(ctx0, moe_out, v) : v;
        }

        // identity expert contribution
        moe_out = ggml_add(ctx0, moe_out, ggml_mul(ctx0, x, w_zero_sum));
        cb(moe_out, "ffn_moe_out", il);

        return moe_out;
    };

    ggml_tensor * cur;

    for (int il = 0; il < n_layer; il += 2) {
        const int il0 = il;
        const int il1 = il + 1;

        // first attention sublayer
        cur = build_norm(inpL, model.layers[il0].attn_norm, NULL, LLM_NORM_RMS, il0);
        cur = build_mla(cur, il0);
        cb(cur, "attn_out", il0);
        ggml_tensor * x = ggml_add(ctx0, cur, inpL);

        // first FFN sublayer + shortcut MoE from the same input
        ggml_tensor * ffn_inp = build_norm(x, model.layers[il0].ffn_norm, NULL, LLM_NORM_RMS, il0);
        ggml_tensor * shortcut = build_zero_moe(ffn_inp, il0);
        cur = build_dense_ffn(ffn_inp, il0);
        cb(cur, "ffn_out", il0);
        x = ggml_add(ctx0, cur, x);

        // second attention sublayer
        cur = build_norm(x, model.layers[il1].attn_norm, NULL, LLM_NORM_RMS, il1);
        cur = build_mla(cur, il1);
        cb(cur, "attn_out", il1);
        x = ggml_add(ctx0, cur, x);

        // second FFN sublayer, then add the MoE shortcut
        cur = build_norm(x, model.layers[il1].ffn_norm, NULL, LLM_NORM_RMS, il1);
        cur = build_dense_ffn(cur, il1);
        cb(cur, "ffn_out", il1);
        x = ggml_add(ctx0, ggml_add(ctx0, cur, x), shortcut);

        x = build_cvec(x, il1);
        cb(x, "l_out", il1);

        inpL = x;
    }

    cur = inpL;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
