#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)  // possible loss of data
#endif

struct finetune_cb_ctx {
    struct llama_model * model   = nullptr;
    int64_t              save_every = 0; // save every N train iterations (0 = only at end)
    int64_t              iter    = 0;
    std::string          ckpt_fmt;      // printf pattern, e.g. "out/ckpt-%06d.gguf"
};

static finetune_cb_ctx g_cb;

static void finetune_epoch_callback(
        bool               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    ggml_opt_epoch_callback_progress_bar(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us);

    if (train && g_cb.save_every > 0 && ibatch % g_cb.save_every == 0) {
        g_cb.iter++;
        char fname[512];
        snprintf(fname, sizeof(fname), g_cb.ckpt_fmt.c_str(), (int) g_cb.iter);
        llama_model_save_to_file(g_cb.model, fname);
        fprintf(stderr, "\nsaved checkpoint %s\n", fname);
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.escape = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    if (params.load_mode != LLAMA_LOAD_MODE_NONE) {
        LOG_INF("%s: forcing load_mode = none to enable writable pointers to the weights\n", __func__);
        params.load_mode = LLAMA_LOAD_MODE_NONE;
    }
    if (params.cache_type_k != GGML_TYPE_F32) {
        LOG_INF("%s: force changing k cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_k = GGML_TYPE_F32;
    }
    if (params.cache_type_v != GGML_TYPE_F32) {
        LOG_INF("%s: force changing v cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_v = GGML_TYPE_F32;
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    // load the model and apply lora adapter, if any
    params.ctx_type = LLAMA_CONTEXT_TYPE_OPT; // no-cache attention for clean backward gradients
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    std::vector<llama_token> tokens  = common_tokenize(ctx, params.prompt, true);
    ggml_opt_dataset_t       dataset = common_opt_dataset_init(ctx, tokens, llama_n_ctx(ctx) / 2);

    struct lr_opt & lr = params.lr;
    LOG_INF("-optimizer %s -lr0 %.2g -wd %.2g -lr-min %.2g -min-epochs %.2g -epochs %d -period %.2g -val %.2g\n",
            ggml_opt_optimizer_name(params.optimizer), (double) lr.lr0, (double) lr.wd, (double) lr.lr_min, (double) lr.decay_epochs,
            (unsigned) lr.epochs, (double) params.n_batch / params.n_ubatch, (double) params.val_split);

    // checkpoint configuration: save every N training iterations via env vars
    // FINETUNE_CKPT_EVERY=100 FINETUNE_CKPT_FMT="out/ckpt-%06d.gguf"
    {
        const char * ckpt_every = getenv("FINETUNE_CKPT_EVERY");
        const char * ckpt_fmt   = getenv("FINETUNE_CKPT_FMT");
        g_cb.model      = model;
        g_cb.save_every = ckpt_every ? atoll(ckpt_every) : 0;
        g_cb.ckpt_fmt   = ckpt_fmt ? ckpt_fmt : (params.out_file + ".ckpt-%06d");
    }

    struct llama_opt_params lopt_params{
        /*n_ctx_train     =*/0,
        /*param_filter    =*/llama_opt_param_filter_all,
        /*param_filter_ud =*/nullptr,
        /*get_opt_pars    =*/common_opt_lr_pars,
        /*get_opt_pars_ud =*/&params.lr,
        /*optimizer_type  =*/params.optimizer,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    for (lr.epoch = 0; lr.epoch < lr.epochs; ++lr.epoch) {
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
                        finetune_epoch_callback, finetune_epoch_callback);
        fprintf(stderr, "\n");

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    llama_model_save_to_file(model, params.out_file.c_str());

    llama_backend_free();

    return 0;
}
