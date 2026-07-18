#!/usr/bin/env python3
"""Standalone eager MLX implementation of GraniteMoeForCausalLM.

This intentionally favors an inspectable grouped-expert loop over optimized
dispatch. It mirrors transformers 5.3.0 GraniteMoe CPU-float32 semantics and
loads the original BF16 safetensors checkpoint directly.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class GraniteMoe:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "config.json").read_text())
        self._validate_config()

        mx.set_default_device(mx.gpu)
        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal backend is unavailable")

        weight_path = self.model_dir / "model.safetensors"
        self.weights = mx.load(str(weight_path))
        self._validate_weights()

        self.hidden_size = self.config["hidden_size"]
        self.head_dim = self.hidden_size // self.config["num_attention_heads"]
        self.num_heads = self.config["num_attention_heads"]
        self.num_kv_heads = self.config["num_key_value_heads"]
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.num_experts = self.config["num_local_experts"]
        self.top_k = self.config["num_experts_per_tok"]
        self.intermediate_size = self.config["intermediate_size"]

    def _validate_config(self):
        expected = {
            "model_type": "granitemoe",
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_local_experts": 32,
            "num_experts_per_tok": 8,
            "intermediate_size": 512,
            "hidden_act": "silu",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "embedding_multiplier": 12.0,
            "attention_multiplier": 0.015625,
            "residual_multiplier": 0.22,
            "logits_scaling": 6.0,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000,
            "rope_scaling": None,
            "tie_word_embeddings": True,
        }
        mismatches = {
            key: {"expected": value, "actual": self.config.get(key)}
            for key, value in expected.items()
            if self.config.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Unsupported GraniteMoe config: {mismatches}")

    def _validate_weights(self):
        expected_count = self.config["num_hidden_layers"] * 9 + 2
        if len(self.weights) != expected_count:
            raise ValueError(f"Expected {expected_count} tensors, found {len(self.weights)}")
        if "lm_head.weight" in self.weights:
            raise ValueError("Expected tied lm_head to be omitted from this checkpoint")

        required_shapes = {
            "model.embed_tokens.weight": (self.config["vocab_size"], self.config["hidden_size"]),
            "model.norm.weight": (self.config["hidden_size"],),
        }
        for layer_index in range(self.config["num_hidden_layers"]):
            prefix = f"model.layers.{layer_index}"
            required_shapes.update(
                {
                    f"{prefix}.input_layernorm.weight": (1024,),
                    f"{prefix}.post_attention_layernorm.weight": (1024,),
                    f"{prefix}.self_attn.q_proj.weight": (1024, 1024),
                    f"{prefix}.self_attn.k_proj.weight": (512, 1024),
                    f"{prefix}.self_attn.v_proj.weight": (512, 1024),
                    f"{prefix}.self_attn.o_proj.weight": (1024, 1024),
                    f"{prefix}.block_sparse_moe.router.layer.weight": (32, 1024),
                    f"{prefix}.block_sparse_moe.input_linear.weight": (32, 1024, 1024),
                    f"{prefix}.block_sparse_moe.output_linear.weight": (32, 1024, 512),
                }
            )
        actual_keys = set(self.weights)
        expected_keys = set(required_shapes)
        if actual_keys != expected_keys:
            raise ValueError(
                f"Weight key mismatch: missing={sorted(expected_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - expected_keys)}"
            )
        for key, shape in required_shapes.items():
            if tuple(self.weights[key].shape) != shape:
                raise ValueError(f"{key}: expected shape {shape}, found {tuple(self.weights[key].shape)}")

    @staticmethod
    def _f32(array: mx.array) -> mx.array:
        return array.astype(mx.float32)

    def _linear(self, inputs: mx.array, weight_key: str) -> mx.array:
        weight = self._f32(self.weights[weight_key])
        return mx.matmul(inputs, mx.swapaxes(weight, -1, -2))

    def _rms_norm(self, hidden_states: mx.array, weight_key: str) -> mx.array:
        hidden_states = self._f32(hidden_states)
        variance = mx.mean(hidden_states * hidden_states, axis=-1, keepdims=True)
        normalized = hidden_states * mx.rsqrt(variance + self.config["rms_norm_eps"])
        return self._f32(self.weights[weight_key]) * normalized

    @staticmethod
    def _rotate_half(inputs: mx.array) -> mx.array:
        half = inputs.shape[-1] // 2
        first = inputs[..., :half]
        second = inputs[..., half:]
        return mx.concatenate([-second, first], axis=-1)

    def _rope(self, sequence_length: int) -> tuple[mx.array, mx.array]:
        positions = mx.arange(sequence_length, dtype=mx.float32)
        dimensions = mx.arange(0, self.head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (float(self.config["rope_theta"]) ** (dimensions / self.head_dim))
        frequencies = positions[:, None] * inv_freq[None, :]
        embeddings = mx.concatenate([frequencies, frequencies], axis=-1)
        return mx.cos(embeddings)[None, None, :, :], mx.sin(embeddings)[None, None, :, :]

    def _causal_mask(self, sequence_length: int) -> mx.array:
        rows = mx.arange(sequence_length)[:, None]
        columns = mx.arange(sequence_length)[None, :]
        minimum = np.finfo(np.float32).min
        mask = mx.where(columns > rows, mx.array(minimum, dtype=mx.float32), mx.array(0.0, dtype=mx.float32))
        return mask[None, None, :, :]

    def _attention(
        self,
        hidden_states: mx.array,
        layer_index: int,
        cos: mx.array,
        sin: mx.array,
        causal_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        prefix = f"model.layers.{layer_index}.self_attn"
        batch_size, sequence_length, _ = hidden_states.shape
        query = self._linear(hidden_states, f"{prefix}.q_proj.weight")
        key = self._linear(hidden_states, f"{prefix}.k_proj.weight")
        value = self._linear(hidden_states, f"{prefix}.v_proj.weight")

        query = mx.swapaxes(query.reshape(batch_size, sequence_length, self.num_heads, self.head_dim), 1, 2)
        key = mx.swapaxes(key.reshape(batch_size, sequence_length, self.num_kv_heads, self.head_dim), 1, 2)
        value = mx.swapaxes(value.reshape(batch_size, sequence_length, self.num_kv_heads, self.head_dim), 1, 2)

        query = query * cos + self._rotate_half(query) * sin
        key = key * cos + self._rotate_half(key) * sin
        key = mx.repeat(key, self.num_kv_groups, axis=1)
        value = mx.repeat(value, self.num_kv_groups, axis=1)

        scores = mx.matmul(query, mx.swapaxes(key, -1, -2)) * self.config["attention_multiplier"]
        scores = scores + causal_mask
        attention_weights = mx.softmax(scores, axis=-1)
        attention_output = mx.matmul(attention_weights, value)
        attention_output = mx.swapaxes(attention_output, 1, 2).reshape(
            batch_size, sequence_length, self.hidden_size
        )
        attention_output = self._linear(attention_output, f"{prefix}.o_proj.weight")
        return attention_output, attention_weights

    def _moe(
        self, hidden_states: mx.array, layer_index: int
    ) -> tuple[mx.array, dict[str, mx.array]]:
        prefix = f"model.layers.{layer_index}.block_sparse_moe"
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if batch_size != 1:
            raise ValueError("This correctness-only port supports batch size 1")
        flattened = hidden_states.reshape(batch_size * sequence_length, hidden_size)

        router_logits = self._linear(flattened, f"{prefix}.router.layer.weight")
        # HF uses torch.topk(..., sorted=True). Full argsort is tiny here (32 experts)
        # and gives the same descending ordering in the absence of exact ties.
        top_indices = mx.argsort(-router_logits, axis=-1)[:, : self.top_k]
        top_logits = mx.take_along_axis(router_logits, top_indices, axis=-1)
        top_weights = mx.softmax(top_logits, axis=-1).astype(flattened.dtype)
        mx.eval(router_logits, top_indices, top_weights)

        top_indices_np = np.asarray(top_indices, dtype=np.int64)
        flat_experts = top_indices_np.reshape(-1)
        # Group assignments by expert exactly as HF GraniteMoeTopKGating does.
        assignment_order_np = np.argsort(flat_experts, kind="stable")
        batch_index_np = assignment_order_np // self.top_k
        sorted_expert_ids_np = flat_experts[assignment_order_np]
        expert_size_np = np.bincount(sorted_expert_ids_np, minlength=self.num_experts)

        assignment_order = mx.array(assignment_order_np.astype(np.int64))
        batch_index = mx.array(batch_index_np.astype(np.int64))
        sorted_expert_ids = mx.array(sorted_expert_ids_np.astype(np.int64))
        expert_size = mx.array(expert_size_np.astype(np.int64))
        batch_gates = top_weights.reshape(-1)[assignment_order]
        expert_inputs = flattened[batch_index]

        input_weight = self.weights[f"{prefix}.input_linear.weight"]
        output_weight = self.weights[f"{prefix}.output_linear.weight"]
        expert_outputs = []
        offset = 0
        for expert_index, count in enumerate(expert_size_np.tolist()):
            if count == 0:
                continue
            expert_input = expert_inputs[offset : offset + count]
            projected = mx.matmul(
                expert_input,
                mx.swapaxes(self._f32(input_weight[expert_index]), -1, -2),
            )
            gate, up = mx.split(projected, 2, axis=-1)
            activated = nn.silu(gate) * up
            output = mx.matmul(
                activated,
                mx.swapaxes(self._f32(output_weight[expert_index]), -1, -2),
            )
            expert_outputs.append(output)
            offset += count
        if offset != sequence_length * self.top_k:
            raise RuntimeError(f"Layer {layer_index}: dispatched {offset} assignments")
        expert_outputs_sorted = mx.concatenate(expert_outputs, axis=0)
        weighted_sorted = expert_outputs_sorted * batch_gates[:, None]

        token_outputs = []
        for token_index in range(sequence_length):
            assignment_positions = np.flatnonzero(batch_index_np == token_index)
            accumulated = mx.zeros((hidden_size,), dtype=mx.float32)
            for assignment_position in assignment_positions.tolist():
                accumulated = accumulated + weighted_sorted[assignment_position]
            token_outputs.append(accumulated)
        combined = mx.stack(token_outputs, axis=0).reshape(batch_size, sequence_length, hidden_size)

        details = {
            "moe.router_input": flattened,
            "moe.router_logits": router_logits,
            "moe.topk_indices": top_indices,
            "moe.topk_weights": top_weights,
            "moe.index_sorted_experts": assignment_order,
            "moe.batch_index": batch_index,
            "moe.batch_gates": batch_gates,
            "moe.sorted_expert_ids": sorted_expert_ids,
            "moe.expert_size": expert_size,
            "moe.expert_outputs_sorted": expert_outputs_sorted,
            "moe.expert_outputs_weighted_sorted": weighted_sorted,
            "moe.output": combined,
        }
        mx.eval(*details.values())
        return combined, details

    def forward(self, input_ids: mx.array, capture: bool = True) -> tuple[mx.array, dict[str, mx.array]]:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"Expected input_ids shape [1, sequence], found {input_ids.shape}")
        sequence_length = input_ids.shape[1]
        captures: dict[str, mx.array] = {"input_ids": input_ids}

        raw_embeddings = self._f32(self.weights["model.embed_tokens.weight"])[input_ids]
        hidden_states = raw_embeddings * self.config["embedding_multiplier"]
        cos, sin = self._rope(sequence_length)
        causal_mask = self._causal_mask(sequence_length)
        mx.eval(raw_embeddings, hidden_states, cos, sin, causal_mask)
        if capture:
            captures["embeddings.raw"] = raw_embeddings
            captures["embed"] = hidden_states

        for layer_index in range(self.config["num_hidden_layers"]):
            prefix = f"model.layers.{layer_index}"
            residual = hidden_states
            attention_input = self._rms_norm(hidden_states, f"{prefix}.input_layernorm.weight")
            attention_output, attention_weights = self._attention(
                attention_input, layer_index, cos, sin, causal_mask
            )
            attention_residual = residual + attention_output * self.config["residual_multiplier"]
            moe_input = self._rms_norm(attention_residual, f"{prefix}.post_attention_layernorm.weight")
            moe_output, moe_details = self._moe(moe_input, layer_index)
            hidden_states = attention_residual + moe_output * self.config["residual_multiplier"]
            mx.eval(
                attention_input,
                attention_output,
                attention_weights,
                attention_residual,
                moe_input,
                hidden_states,
            )
            if capture:
                captures[f"layer.{layer_index}.attention.input_norm"] = attention_input
                captures[f"layer.{layer_index}.attention.output"] = attention_output
                captures[f"layer.{layer_index}.attention.weights"] = attention_weights
                captures[f"layer.{layer_index}.attention.residual"] = attention_residual
                captures[f"layer.{layer_index}.moe.input_norm"] = moe_input
                for name, value in moe_details.items():
                    captures[f"layer.{layer_index}.{name}"] = value
                captures[f"layer.{layer_index}.hidden"] = hidden_states
            gc.collect()
            mx.clear_cache()

        final_norm = self._rms_norm(hidden_states, "model.norm.weight")
        logits = mx.matmul(
            final_norm,
            mx.swapaxes(self._f32(self.weights["model.embed_tokens.weight"]), -1, -2),
        )
        logits = logits / self.config["logits_scaling"]
        mx.eval(final_norm, logits)
        if capture:
            captures["final_norm"] = final_norm
            captures["logits"] = logits
        return logits, captures


def device_report() -> dict[str, Any]:
    return {
        "default_device": str(mx.default_device()),
        "metal_available": mx.metal.is_available(),
        "gpu": mx.device_info(mx.gpu),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 42, 17, 9, 314, 2718])
    args = parser.parse_args()

    model = GraniteMoe(args.model_dir)
    logits, _ = model.forward(mx.array([args.tokens], dtype=mx.int64), capture=False)
    print(json.dumps({**device_report(), "logits_shape": list(logits.shape)}, sort_keys=True))
