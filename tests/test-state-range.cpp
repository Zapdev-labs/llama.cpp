// Test for the position range covered by a saved sequence state
//
// For SWA models, a state save only serializes the cells inside the sliding
// window, so the covered range is narrower than the [pos_min, pos_max] range
// of the memory. This is what llama_memory_state_pos_min/max report.
//
// See: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <vector>
#include <cstdio>

int main(int argc, char ** argv) {
    common_params params;

    params.sampling.seed = 1234;
    params.n_parallel = 1;
    params.n_ctx = 2048;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    const llama_seq_id seq_id = 0;

    llama_memory_t mem = llama_get_memory(ctx);

    const int32_t n_swa = llama_model_n_swa(model);

    // fill the sequence past the sliding window, if any
    const int32_t n_tokens = n_swa > 0 ? n_swa + 128 : 512;

    std::vector<llama_token> tokens(n_tokens, 1);

    common_batch batch(ctx);
    for (int32_t i = 0; i < n_tokens; i++) {
        batch.add(tokens[i], i, seq_id, false);
    }

    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        fprintf(stderr, "%s : failed to decode\n", __func__);
        return 1;
    }

    const llama_pos pos_min = llama_memory_seq_pos_min(mem, seq_id);
    const llama_pos pos_max = llama_memory_seq_pos_max(mem, seq_id);

    const llama_pos s_min_partial = llama_memory_state_pos_min(mem, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    const llama_pos s_max_partial = llama_memory_state_pos_max(mem, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

    const llama_pos s_min_full = llama_memory_state_pos_min(mem, seq_id, 0);
    const llama_pos s_max_full = llama_memory_state_pos_max(mem, seq_id, 0);

    fprintf(stderr, "memory  : [%d, %d]\n", pos_min, pos_max);
    fprintf(stderr, "state p : [%d, %d]\n", s_min_partial, s_max_partial);
    fprintf(stderr, "state f : [%d, %d]\n", s_min_full, s_max_full);

    // the saved range can only be narrower than the memory range, never wider
    GGML_ASSERT(s_min_partial >= pos_min);
    GGML_ASSERT(s_max_partial <= pos_max);
    GGML_ASSERT(s_min_full >= pos_min);
    GGML_ASSERT(s_max_full <= pos_max);

    // the max saved position is always the memory max
    GGML_ASSERT(s_max_partial == pos_max);
    GGML_ASSERT(s_max_full == pos_max);

    if (n_swa == 0) {
        // non-SWA models save the whole memory range
        GGML_ASSERT(s_min_partial == pos_min);
        GGML_ASSERT(s_min_full == pos_min);
    } else {
        fprintf(stderr, "n_swa   : %d\n", n_swa);

        // for all SWA types the first saved position is within n_swa of the max
        GGML_ASSERT(s_min_partial >= pos_max - n_swa + 1);

        // the sequence is longer than the window, so the first saved position
        // is strictly inside the sequence
        GGML_ASSERT(s_min_partial > 0);
    }

    fprintf(stderr, "%s : OK\n", __func__);

    return 0;
}
