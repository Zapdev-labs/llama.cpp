from __future__ import annotations

import re
from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("LongcatFlashNgramForCausalLM")
class LongcatFlashModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LONGCAT_FLASH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # each HF logical layer holds 2 attention/FFN sublayers; map to 2 gguf blocks
        self.n_logical_layers = self.hparams["num_layers"]
        self.block_count = 2 * self.n_logical_layers
        self.n_routed_experts = self.hparams["n_routed_experts"]
        self._experts: list[dict[str, Tensor]] = [{} for _ in range(self.n_logical_layers)]

    def get_vocab_base_pre(self, tokenizer) -> str:
        try:
            return super().get_vocab_base_pre(tokenizer)
        except NotImplementedError:
            logger.warning("Using longcat pre-tokenizer (hash not in table)")
            return "longcat"

    def set_vocab(self):
        self._set_vocab_gpt2()
        # tokenizer_config.json has add_eos_token=true (a training-time setting); never add EOS at inference
        self.gguf_writer.add_add_eos_token(False)

    def set_gguf_parameters(self):
        hp = self.hparams
        rope = hp["rope_scaling"]

        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(hp["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hp["hidden_size"])
        self.gguf_writer.add_feed_forward_length(hp["ffn_hidden_size"])
        self.gguf_writer.add_expert_feed_forward_length(hp["expert_ffn_hidden_size"])
        self.gguf_writer.add_head_count(hp["num_attention_heads"])
        # MLA with absorption converts into MQA
        self.gguf_writer.add_head_count_kv(1)
        self.gguf_writer.add_layer_norm_rms_eps(hp["rms_norm_eps"])
        self.gguf_writer.add_vocab_size(hp["vocab_size"])

        self.gguf_writer.add_q_lora_rank(hp["q_lora_rank"])
        self.gguf_writer.add_kv_lora_rank(hp["kv_lora_rank"])
        self.gguf_writer.add_key_length(hp["kv_lora_rank"] + hp["qk_rope_head_dim"])
        self.gguf_writer.add_value_length(hp["kv_lora_rank"])
        self.gguf_writer.add_key_length_mla(hp["qk_nope_head_dim"] + hp["qk_rope_head_dim"])
        self.gguf_writer.add_value_length_mla(hp["v_head_dim"])
        self.gguf_writer.add_rope_dimension_count(hp["qk_rope_head_dim"])

        self.gguf_writer.add_rope_freq_base(hp["rope_theta"])
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
        self.gguf_writer.add_rope_scaling_factor(rope["factor"])
        self.gguf_writer.add_rope_scaling_orig_ctx_len(rope["original_max_position_embeddings"])
        # same convention as deepseek2 [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
        self.gguf_writer.add_rope_scaling_yarn_log_mul(0.1 * rope.get("mscale_all_dim", 1))

        self.gguf_writer.add_expert_count(self.n_routed_experts)
        self.gguf_writer.add_expert_used_count(hp["moe_topk"])
        self.gguf_writer.add_expert_weights_scale(hp["routed_scaling_factor"])
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)

        arch = gguf.MODEL_ARCH_NAMES[self.model_arch]
        self.gguf_writer.add_uint32(f"{arch}.expert_zero_count", hp["zero_expert_num"])
        self.gguf_writer.add_uint32(f"{arch}.ngram.neighbor_count", hp["emb_neighbor_num"])
        self.gguf_writer.add_uint32(f"{arch}.ngram.split_count", hp["emb_split_num"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("model.mtp"):
            return

        if name == "model.embed_tokens.weight":
            yield ("token_embd.weight", data_torch)
            return
        if name == "lm_head.weight":
            yield ("output.weight", data_torch)
            return
        if name == "model.norm.weight":
            yield ("output_norm.weight", data_torch)
            return

        if (m := re.match(r"model\.ngram_embeddings\.embedders\.(\d+)\.weight", name)):
            yield (f"ngram_embd.{m.group(1)}.weight", data_torch)
            return
        if (m := re.match(r"model\.ngram_embeddings\.post_projs\.(\d+)\.weight", name)):
            yield (f"ngram_proj.{m.group(1)}.weight", data_torch)
            return

        assert bid is not None, f"unexpected tensor: {name}"
        rest = name.removeprefix(f"model.layers.{bid}.")

        # MoE (attached to the first sublayer's gguf block)
        blk_moe = 2 * bid
        if rest == "mlp.router.classifier.weight":
            yield (f"blk.{blk_moe}.ffn_gate_inp.weight", data_torch)
            return
        if rest in ("mlp.router.e_score_correction_bias", "mlp.router.e_score_correction.bias"):
            yield (f"blk.{blk_moe}.exp_probs_b.bias", data_torch)
            return
        if (m := re.match(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight", rest)):
            xid = int(m.group(1))
            if xid >= self.n_routed_experts:
                return  # zero-computation experts have no useful weights
            self._experts[bid][rest] = data_torch
            if len(self._experts[bid]) == self.n_routed_experts * 3:
                for w_name in ("gate", "up", "down"):
                    stacked = torch.stack(
                        [self._experts[bid].pop(f"mlp.experts.{x}.{w_name}_proj.weight") for x in range(self.n_routed_experts)],
                        dim=0)
                    yield (f"blk.{blk_moe}.ffn_{w_name}_exps.weight", stacked)
            return

        # sublayer-indexed tensors -> gguf block 2*bid + sub
        if (m := re.match(r"(input_layernorm|post_attention_layernorm|self_attn|mlps)\.([01])\.(.+)", rest)):
            kind, sub, tail = m.group(1), int(m.group(2)), m.group(3)
            blk = 2 * bid + sub

            if kind == "input_layernorm":
                yield (f"blk.{blk}.attn_norm.weight", data_torch)
                return
            if kind == "post_attention_layernorm":
                yield (f"blk.{blk}.ffn_norm.weight", data_torch)
                return
            if kind == "mlps":
                proj = tail.removesuffix(".weight")  # gate_proj / up_proj / down_proj
                yield (f"blk.{blk}.ffn_{proj.removesuffix('_proj')}.weight", data_torch)
                return

            # self_attn
            attn_map = {
                "q_a_proj.weight":            "attn_q_a.weight",
                "q_a_layernorm.weight":       "attn_q_a_norm.weight",
                "q_b_proj.weight":            "attn_q_b.weight",
                "kv_a_proj_with_mqa.weight":  "attn_kv_a_mqa.weight",
                "kv_a_layernorm.weight":      "attn_kv_a_norm.weight",
                "o_proj.weight":              "attn_output.weight",
            }
            if tail in attn_map:
                yield (f"blk.{blk}.{attn_map[tail]}", data_torch)
                return
            if tail == "kv_b_proj.weight":
                # split for MLA absorption, same as deepseek2
                n_head = self.hparams["num_attention_heads"]
                v_head_dim = self.hparams["v_head_dim"]
                qk_nope_head_dim = self.hparams["qk_nope_head_dim"]
                assert data_torch.shape[0] == n_head * (v_head_dim + qk_nope_head_dim)
                kv_b = data_torch.view(n_head, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
                k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
                k_b = k_b.transpose(1, 2)
                yield (f"blk.{blk}.attn_k_b.weight", k_b)
                yield (f"blk.{blk}.attn_v_b.weight", v_b)
                return

        raise ValueError(f"unhandled tensor: {name}")

    def prepare_tensors(self):
        super().prepare_tensors()
        leftover = [k for d in self._experts for k in d.keys()]
        if leftover:
            raise ValueError(f"unprocessed experts: {leftover}")
