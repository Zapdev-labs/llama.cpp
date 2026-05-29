// oracle — ground-truth harness for the pure-Rust llama.cpp port.
//
// Subcommands:
//   oracle dequant         <type> <input.bin> <output.bin>
//   oracle logits          <model.gguf> <prompt> <n>
//   oracle tokenize        <model.gguf> <text>
//   oracle rms_norm        <input.bin> <weight.bin> <eps> <output.bin>
//   oracle rmsnorm_capture <model.gguf> <prompt> <out_dir>
//
// All numeric output is little-endian f32. The binary links against the
// pre-built libllama.so + libggml*.so under build/ and is therefore not
// portable — do not commit it.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

int print_help(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s <subcommand> [args...]\n"
        "\n"
        "subcommands:\n"
        "  dequant  <type> <input.bin> <output.bin>\n"
        "      dequantize a packed block stream into f32. type ∈ {f16, q8_0, q4_0, q4_k}.\n"
        "      <input.bin> must be a whole number of blocks for the given type.\n"
        "\n"
        "  logits   <model.gguf> <prompt> <n>\n"
        "      tokenize <prompt> (with model-default BOS), generate <n> tokens greedily,\n"
        "      then write the final-step logits as binary f32 to stdout.\n"
        "\n"
        "  tokenize <model.gguf> <text>\n"
        "      tokenize <text> (with model-default BOS) and write one decimal token ID per line to stdout.\n"
        "\n"
        "  rms_norm <input.bin> <weight.bin> <eps> <output.bin>\n"
        "      compute ggml_rms_norm(input, eps) * weight using libggml on the CPU backend.\n"
        "      input/weight are little-endian f32 streams; input length must be a multiple of weight length.\n"
        "\n"
        "  rmsnorm_capture <model.gguf> <prompt> <out_dir>\n"
        "      load the model, tokenize <prompt> (with model-default BOS), run one forward decode,\n"
        "      and dump three little-endian f32 files into <out_dir>: input.bin (post-block-0 hidden\n"
        "      state, the input to blk.1.attn_norm), weight.bin (blk.1.attn_norm.weight), and\n"
        "      expected.bin (the output of blk.1.attn_norm = rms_norm(input) * weight).\n",
        argv0);
    return 0;
}

ggml_type parse_type(const std::string & s) {
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    if (s == "q4_k" || s == "q4_K") return GGML_TYPE_Q4_K;
    return GGML_TYPE_COUNT;
}

bool read_file(const std::string & path, std::vector<uint8_t> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "oracle: cannot open %s for reading: %s\n", path.c_str(), std::strerror(errno));
        return false;
    }
    f.seekg(0, std::ios::end);
    const auto sz = f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(sz));
    if (sz > 0 && !f.read(reinterpret_cast<char *>(out.data()), sz)) {
        std::fprintf(stderr, "oracle: short read on %s\n", path.c_str());
        return false;
    }
    return true;
}

bool write_file(const std::string & path, const void * data, size_t nbytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "oracle: cannot open %s for writing: %s\n", path.c_str(), std::strerror(errno));
        return false;
    }
    if (nbytes > 0 && !f.write(static_cast<const char *>(data), static_cast<std::streamsize>(nbytes))) {
        std::fprintf(stderr, "oracle: short write on %s\n", path.c_str());
        return false;
    }
    return true;
}

int cmd_dequant(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "oracle dequant: expected <type> <input.bin> <output.bin>\n");
        return 2;
    }
    const std::string type_s = argv[2];
    const std::string in_path = argv[3];
    const std::string out_path = argv[4];

    const ggml_type type = parse_type(type_s);
    if (type == GGML_TYPE_COUNT) {
        std::fprintf(stderr, "oracle dequant: unsupported type '%s' (expected f16|q8_0|q4_0|q4_k)\n", type_s.c_str());
        return 2;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (traits == nullptr || traits->to_float == nullptr) {
        std::fprintf(stderr, "oracle dequant: ggml has no to_float for type '%s'\n", type_s.c_str());
        return 1;
    }

    const int64_t block_size = traits->blck_size;
    const size_t  type_size  = traits->type_size;

    std::vector<uint8_t> in;
    if (!read_file(in_path, in)) {
        return 1;
    }
    if (in.size() % type_size != 0) {
        std::fprintf(stderr,
            "oracle dequant: input size %zu is not a multiple of block size %zu for type '%s'\n",
            in.size(), type_size, type_s.c_str());
        return 1;
    }
    const int64_t n_blocks   = static_cast<int64_t>(in.size() / type_size);
    const int64_t n_elements = n_blocks * block_size;

    std::vector<float> out(static_cast<size_t>(n_elements));
    traits->to_float(in.data(), out.data(), n_elements);

    if (!write_file(out_path, out.data(), out.size() * sizeof(float))) {
        return 1;
    }
    return 0;
}

struct LoadedModel {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    const llama_vocab * vocab = nullptr;

    ~LoadedModel() {
        if (ctx)   llama_free(ctx);
        if (model) llama_model_free(model);
    }
};

bool load_model(const std::string & path, int n_ctx, LoadedModel & out) {
    llama_model_params mparams = llama_model_default_params();
    out.model = llama_model_load_from_file(path.c_str(), mparams);
    if (out.model == nullptr) {
        std::fprintf(stderr, "oracle: failed to load model %s\n", path.c_str());
        return false;
    }
    out.vocab = llama_model_get_vocab(out.model);
    if (out.vocab == nullptr) {
        std::fprintf(stderr, "oracle: model %s has no vocab\n", path.c_str());
        return false;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = static_cast<uint32_t>(n_ctx);
    cparams.n_batch     = static_cast<uint32_t>(n_ctx);
    cparams.n_ubatch    = static_cast<uint32_t>(n_ctx);
    cparams.no_perf     = true;
    out.ctx = llama_init_from_model(out.model, cparams);
    if (out.ctx == nullptr) {
        std::fprintf(stderr, "oracle: failed to create context for %s\n", path.c_str());
        return false;
    }
    return true;
}

bool tokenize(const llama_vocab * vocab, const std::string & text, bool add_special, std::vector<llama_token> & out) {
    int n = llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), nullptr, 0, add_special, true);
    if (n == INT32_MIN) {
        std::fprintf(stderr, "oracle: tokenize overflow\n");
        return false;
    }
    const int n_max = n < 0 ? -n : n;
    out.assign(static_cast<size_t>(n_max), 0);
    n = llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), out.data(), n_max, add_special, true);
    if (n < 0) {
        std::fprintf(stderr, "oracle: tokenize failed (returned %d)\n", n);
        return false;
    }
    out.resize(static_cast<size_t>(n));
    return true;
}

int cmd_tokenize(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "oracle tokenize: expected <model.gguf> <text>\n");
        return 2;
    }
    const std::string model_path = argv[2];
    const std::string text       = argv[3];

    llama_backend_init();
    LoadedModel lm;
    if (!load_model(model_path, 512, lm)) {
        llama_backend_free();
        return 1;
    }

    std::vector<llama_token> tokens;
    if (!tokenize(lm.vocab, text, true, tokens)) {
        llama_backend_free();
        return 1;
    }
    for (const llama_token t : tokens) {
        std::printf("%d\n", static_cast<int>(t));
    }
    std::fflush(stdout);

    llama_backend_free();
    return 0;
}

int cmd_logits(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "oracle logits: expected <model.gguf> <prompt> <n>\n");
        return 2;
    }
    const std::string model_path = argv[2];
    const std::string prompt     = argv[3];
    const int n_predict          = std::atoi(argv[4]);
    if (n_predict < 0) {
        std::fprintf(stderr, "oracle logits: <n> must be >= 0\n");
        return 2;
    }

    llama_backend_init();
    LoadedModel lm;
    const int n_ctx = 2048;
    if (!load_model(model_path, n_ctx, lm)) {
        llama_backend_free();
        return 1;
    }

    std::vector<llama_token> tokens;
    if (!tokenize(lm.vocab, prompt, true, tokens)) {
        llama_backend_free();
        return 1;
    }
    if (tokens.empty()) {
        std::fprintf(stderr, "oracle logits: prompt tokenized to zero tokens\n");
        llama_backend_free();
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(lm.vocab);
    if (n_vocab <= 0) {
        std::fprintf(stderr, "oracle logits: invalid n_vocab %d\n", n_vocab);
        llama_backend_free();
        return 1;
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    if (llama_decode(lm.ctx, batch) != 0) {
        std::fprintf(stderr, "oracle logits: prompt decode failed\n");
        llama_backend_free();
        return 1;
    }

    for (int step = 0; step < n_predict; ++step) {
        const float * logits = llama_get_logits_ith(lm.ctx, -1);
        if (logits == nullptr) {
            std::fprintf(stderr, "oracle logits: get_logits_ith returned null at step %d\n", step);
            llama_backend_free();
            return 1;
        }
        llama_token best = 0;
        float best_v = logits[0];
        for (int i = 1; i < n_vocab; ++i) {
            if (logits[i] > best_v) {
                best_v = logits[i];
                best   = static_cast<llama_token>(i);
            }
        }
        llama_batch next = llama_batch_get_one(&best, 1);
        if (llama_decode(lm.ctx, next) != 0) {
            std::fprintf(stderr, "oracle logits: decode failed at step %d\n", step);
            llama_backend_free();
            return 1;
        }
    }

    const float * final_logits = llama_get_logits_ith(lm.ctx, -1);
    if (final_logits == nullptr) {
        std::fprintf(stderr, "oracle logits: no final logits available\n");
        llama_backend_free();
        return 1;
    }
    const size_t nbytes = static_cast<size_t>(n_vocab) * sizeof(float);
    if (std::fwrite(final_logits, 1, nbytes, stdout) != nbytes) {
        std::fprintf(stderr, "oracle logits: short write to stdout\n");
        llama_backend_free();
        return 1;
    }
    std::fflush(stdout);

    llama_backend_free();
    return 0;
}

int cmd_rms_norm(int argc, char ** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "oracle rms_norm: expected <input.bin> <weight.bin> <eps> <output.bin>\n");
        return 2;
    }
    const std::string in_path  = argv[2];
    const std::string w_path   = argv[3];
    const std::string eps_s    = argv[4];
    const std::string out_path = argv[5];

    const float eps = std::strtof(eps_s.c_str(), nullptr);
    if (!std::isfinite(eps) || eps <= 0.0f) {
        std::fprintf(stderr, "oracle rms_norm: eps must be a positive finite float, got '%s'\n", eps_s.c_str());
        return 2;
    }

    std::vector<uint8_t> in_bytes;
    std::vector<uint8_t> w_bytes;
    if (!read_file(in_path, in_bytes) || !read_file(w_path, w_bytes)) {
        return 1;
    }
    if (in_bytes.size() % sizeof(float) != 0 || w_bytes.size() % sizeof(float) != 0) {
        std::fprintf(stderr, "oracle rms_norm: input/weight sizes must be a multiple of 4 bytes\n");
        return 1;
    }
    const int64_t n_in = static_cast<int64_t>(in_bytes.size() / sizeof(float));
    const int64_t n_w  = static_cast<int64_t>(w_bytes.size()  / sizeof(float));
    if (n_w == 0 || n_in % n_w != 0) {
        std::fprintf(stderr,
            "oracle rms_norm: input length %lld must be a positive multiple of weight length %lld\n",
            (long long) n_in, (long long) n_w);
        return 1;
    }
    const int64_t n_rows = n_in / n_w;

    const size_t mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    std::vector<uint8_t> mem(mem_size);
    ggml_init_params iparams = { mem_size, mem.data(), true };
    ggml_context * ctx = ggml_init(iparams);
    if (ctx == nullptr) {
        std::fprintf(stderr, "oracle rms_norm: ggml_init failed\n");
        return 1;
    }

    ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_w, n_rows);
    ggml_tensor * w   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_w);
    ggml_set_name(a, "input");
    ggml_set_name(w, "weight");
    ggml_tensor * n_t = ggml_rms_norm(ctx, a, eps);
    ggml_tensor * out = ggml_mul(ctx, n_t, w);
    ggml_set_name(out, "expected");

    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        std::fprintf(stderr, "oracle rms_norm: ggml_backend_cpu_init failed\n");
        ggml_free(ctx);
        return 1;
    }
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, g)) {
        std::fprintf(stderr, "oracle rms_norm: gallocr_alloc_graph failed\n");
        ggml_gallocr_free(alloc);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 1;
    }

    ggml_backend_tensor_set(a, in_bytes.data(), 0, in_bytes.size());
    ggml_backend_tensor_set(w, w_bytes.data(),  0, w_bytes.size());

    if (ggml_backend_graph_compute(backend, g) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "oracle rms_norm: graph compute failed\n");
        ggml_gallocr_free(alloc);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> out_data(static_cast<size_t>(n_in));
    ggml_backend_tensor_get(out, out_data.data(), 0, out_data.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_backend_free(backend);
    ggml_free(ctx);

    if (!write_file(out_path, out_data.data(), out_data.size() * sizeof(float))) {
        return 1;
    }
    return 0;
}

struct RmsNormCaptureState {
    std::vector<float> input;
    std::vector<float> expected;
    int64_t input_rows = 0;
    int64_t input_cols = 0;
    int64_t expected_rows = 0;
    int64_t expected_cols = 0;
    std::string input_name;
    std::string expected_name;
};

bool capture_eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    RmsNormCaptureState * s = static_cast<RmsNormCaptureState *>(user_data);
    const char * name = t->name;
    const bool want = (name && (s->input_name == name || s->expected_name == name));
    if (ask) {
        return want;
    }
    if (!want || t->type != GGML_TYPE_F32) {
        return true;
    }
    const int64_t n = ggml_nelements(t);
    std::vector<float> & dst = (s->input_name == name) ? s->input : s->expected;
    dst.assign(static_cast<size_t>(n), 0.0f);
    ggml_backend_tensor_get(t, dst.data(), 0, dst.size() * sizeof(float));
    if (s->input_name == name) {
        s->input_cols = t->ne[0];
        s->input_rows = t->ne[1];
    } else {
        s->expected_cols = t->ne[0];
        s->expected_rows = t->ne[1];
    }
    return true;
}

bool read_f32_weight_from_gguf(const std::string & path, const std::string & tensor_name, std::vector<float> & out) {
    gguf_init_params gparams = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gparams);
    if (gctx == nullptr) {
        std::fprintf(stderr, "oracle: gguf_init_from_file failed for %s\n", path.c_str());
        return false;
    }
    const int64_t idx = gguf_find_tensor(gctx, tensor_name.c_str());
    if (idx < 0) {
        std::fprintf(stderr, "oracle: tensor '%s' not found in %s\n", tensor_name.c_str(), path.c_str());
        gguf_free(gctx);
        return false;
    }
    const ggml_type ty   = gguf_get_tensor_type(gctx, idx);
    const size_t nbytes  = gguf_get_tensor_size(gctx, idx);
    const size_t toff    = gguf_get_tensor_offset(gctx, idx);
    const size_t doff    = gguf_get_data_offset(gctx);
    gguf_free(gctx);
    if (ty != GGML_TYPE_F32) {
        std::fprintf(stderr, "oracle: tensor '%s' is type %d, expected F32\n",
            tensor_name.c_str(), static_cast<int>(ty));
        return false;
    }
    if (nbytes % sizeof(float) != 0) {
        std::fprintf(stderr, "oracle: tensor '%s' has non-multiple-of-4 size %zu\n",
            tensor_name.c_str(), nbytes);
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "oracle: cannot reopen %s: %s\n", path.c_str(), std::strerror(errno));
        return false;
    }
    f.seekg(static_cast<std::streamoff>(doff + toff));
    out.assign(nbytes / sizeof(float), 0.0f);
    if (!f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(nbytes))) {
        std::fprintf(stderr, "oracle: short read for tensor '%s' in %s\n", tensor_name.c_str(), path.c_str());
        return false;
    }
    return true;
}

bool ensure_dir(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    std::string cmd = "mkdir -p '" + path + "'";
    return std::system(cmd.c_str()) == 0;
}

int cmd_rmsnorm_capture(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "oracle rmsnorm_capture: expected <model.gguf> <prompt> <out_dir>\n");
        return 2;
    }
    const std::string model_path = argv[2];
    const std::string prompt     = argv[3];
    const std::string out_dir    = argv[4];

    if (!ensure_dir(out_dir)) {
        std::fprintf(stderr, "oracle rmsnorm_capture: cannot create %s\n", out_dir.c_str());
        return 1;
    }

    RmsNormCaptureState state;
    state.input_name    = "l_out-0";
    state.expected_name = "attn_norm-1";

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "oracle: failed to load model %s\n", model_path.c_str());
        llama_backend_free();
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx            = 512;
    cparams.n_batch          = 512;
    cparams.n_ubatch         = 512;
    cparams.no_perf          = true;
    cparams.cb_eval          = &capture_eval_cb;
    cparams.cb_eval_user_data = &state;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        std::fprintf(stderr, "oracle: failed to create context for %s\n", model_path.c_str());
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::vector<llama_token> tokens;
    if (!tokenize(vocab, prompt, true, tokens)) {
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    if (tokens.empty()) {
        std::fprintf(stderr, "oracle rmsnorm_capture: prompt tokenized to zero tokens\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "oracle rmsnorm_capture: decode failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    if (state.input.empty() || state.expected.empty()) {
        std::fprintf(stderr,
            "oracle rmsnorm_capture: did not capture both tensors (input rows=%lld cols=%lld, expected rows=%lld cols=%lld)\n",
            (long long) state.input_rows, (long long) state.input_cols,
            (long long) state.expected_rows, (long long) state.expected_cols);
        return 1;
    }
    if (state.input.size() != state.expected.size()) {
        std::fprintf(stderr,
            "oracle rmsnorm_capture: input and expected sizes differ (%zu vs %zu)\n",
            state.input.size(), state.expected.size());
        return 1;
    }

    std::vector<float> weight;
    if (!read_f32_weight_from_gguf(model_path, "blk.1.attn_norm.weight", weight)) {
        return 1;
    }
    if (state.input_cols != static_cast<int64_t>(weight.size())) {
        std::fprintf(stderr,
            "oracle rmsnorm_capture: weight length %zu != input cols %lld\n",
            weight.size(), (long long) state.input_cols);
        return 1;
    }

    if (!write_file(out_dir + "/input.bin",    state.input.data(),    state.input.size()    * sizeof(float))) return 1;
    if (!write_file(out_dir + "/weight.bin",   weight.data(),         weight.size()         * sizeof(float))) return 1;
    if (!write_file(out_dir + "/expected.bin", state.expected.data(), state.expected.size() * sizeof(float))) return 1;

    std::fprintf(stderr,
        "oracle rmsnorm_capture: wrote input.bin (%lld rows x %lld cols), weight.bin (%zu f32), expected.bin (same shape as input).\n",
        (long long) state.input_rows, (long long) state.input_cols, weight.size());
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_help(argv[0]);
        return 2;
    }
    const std::string sub = argv[1];
    if (sub == "--help" || sub == "-h" || sub == "help") {
        return print_help(argv[0]);
    }
    if (sub == "dequant")          return cmd_dequant(argc, argv);
    if (sub == "logits")           return cmd_logits(argc, argv);
    if (sub == "tokenize")         return cmd_tokenize(argc, argv);
    if (sub == "rms_norm")         return cmd_rms_norm(argc, argv);
    if (sub == "rmsnorm_capture")  return cmd_rmsnorm_capture(argc, argv);

    std::fprintf(stderr, "oracle: unknown subcommand '%s'\n\n", sub.c_str());
    print_help(argv[0]);
    return 2;
}
