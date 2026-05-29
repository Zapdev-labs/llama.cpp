// oracle — ground-truth harness for the pure-Rust llama.cpp port.
//
// Three subcommands:
//   oracle dequant  <type> <input.bin> <output.bin>
//   oracle logits   <model.gguf> <prompt> <n>
//   oracle tokenize <model.gguf> <text>
//
// All numeric output is little-endian f32. The binary links against the
// pre-built libllama.so + libggml*.so under build/ and is therefore not
// portable — do not commit it.

#include "ggml.h"
#include "llama.h"

#include <cerrno>
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
        "      tokenize <text> (with model-default BOS) and write one decimal token ID per line to stdout.\n",
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
    if (sub == "dequant")  return cmd_dequant(argc, argv);
    if (sub == "logits")   return cmd_logits(argc, argv);
    if (sub == "tokenize") return cmd_tokenize(argc, argv);

    std::fprintf(stderr, "oracle: unknown subcommand '%s'\n\n", sub.c_str());
    print_help(argv[0]);
    return 2;
}
