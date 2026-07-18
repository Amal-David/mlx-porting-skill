"""Standalone eager MLX text tower for Qwen/Qwen3.5-2B.

This is intentionally correctness-only: FP32 weights/activations, explicit
causal attention, and a readable token-by-token Gated-DeltaNet recurrence.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import mlx.core as mx


TEXT_WEIGHT_PREFIX = "model.language_model."


def load_text_weights(checkpoint: str | Path) -> dict[str, mx.array]:
    """Load only text-tower tensors and materialize them as FP32 MLX arrays."""
    raw = mx.load(str(checkpoint))
    weights = {
        source_key[len(TEXT_WEIGHT_PREFIX) :]: tensor.astype(mx.float32)
        for source_key, tensor in raw.items()
        if source_key.startswith(TEXT_WEIGHT_PREFIX)
    }
    if len(weights) != 320:
        raise ValueError(f"Expected 320 text tensors, found {len(weights)}")
    mx.eval(weights)
    del raw
    gc.collect()
    return weights


def linear(x: mx.array, weight: mx.array) -> mx.array:
    return mx.matmul(x, mx.swapaxes(weight, -1, -2))


def sigmoid(x: mx.array) -> mx.array:
    return mx.sigmoid(x)


def silu(x: mx.array) -> mx.array:
    return x * sigmoid(x)


def softplus(x: mx.array) -> mx.array:
    return mx.maximum(x, mx.zeros_like(x)) + mx.log1p(mx.exp(-mx.abs(x)))


def rms_norm_additive(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Qwen3.5 RMSNorm: normalized(x) * (1 + zero-centered weight)."""
    x = x.astype(mx.float32)
    variance = mx.mean(x * x, axis=-1, keepdims=True)
    return x * mx.rsqrt(variance + eps) * (1.0 + weight.astype(mx.float32))


def rms_norm_gated(
    x: mx.array,
    gate: mx.array,
    weight: mx.array,
    eps: float,
) -> mx.array:
    """Qwen3.5 Gated-DeltaNet norm: direct weight, then SiLU gate."""
    x = x.astype(mx.float32)
    variance = mx.mean(x * x, axis=-1, keepdims=True)
    normalized = x * mx.rsqrt(variance + eps)
    return normalized * weight.astype(mx.float32) * silu(gate.astype(mx.float32))


def l2_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)


def causal_depthwise_conv1d(x: mx.array, weight: mx.array) -> mx.array:
    """Match PyTorch Conv1d(padding=k-1) followed by [:, :, :seq_len].

    Args:
        x: [batch, sequence, channels]
        weight: [channels, 1, kernel]
    """
    batch, sequence, channels = x.shape
    del batch, channels
    kernel = weight.shape[-1]
    padded = mx.pad(x, [(0, 0), (kernel - 1, 0), (0, 0)])
    output = mx.zeros_like(x)
    channel_weight = weight[:, 0, :]
    for kernel_index in range(kernel):
        output = output + padded[:, kernel_index : kernel_index + sequence, :] * channel_weight[
            None, None, :, kernel_index
        ]
    return silu(output)


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


class Qwen35TextTower:
    def __init__(self, config_path: str | Path, weights: dict[str, mx.array]):
        top_config = json.loads(Path(config_path).read_text())
        config = top_config["text_config"]
        self.config = config
        self.weights = weights
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.num_layers = config["num_hidden_layers"]
        self.layer_types = config["layer_types"]
        self.eps = config["rms_norm_eps"]

        self.num_attention_heads = config["num_attention_heads"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_scale = self.head_dim**-0.5

        self.linear_num_key_heads = config["linear_num_key_heads"]
        self.linear_key_head_dim = config["linear_key_head_dim"]
        self.linear_num_value_heads = config["linear_num_value_heads"]
        self.linear_value_head_dim = config["linear_value_head_dim"]
        self.linear_key_dim = self.linear_num_key_heads * self.linear_key_head_dim
        self.linear_value_dim = self.linear_num_value_heads * self.linear_value_head_dim

        rope = config["rope_parameters"]
        self.rotary_dim = int(self.head_dim * rope["partial_rotary_factor"])
        self.rope_theta = rope["rope_theta"]
        self.mrope_section = rope["mrope_section"]

        expected_full = [idx for idx in range(self.num_layers) if idx % 4 == 3]
        actual_full = [idx for idx, kind in enumerate(self.layer_types) if kind == "full_attention"]
        if actual_full != expected_full:
            raise ValueError(f"Unexpected hybrid layer schedule: {actual_full}")
        if self.rotary_dim != 64 or self.mrope_section != [11, 11, 10]:
            raise ValueError(
                f"Unexpected text RoPE contract: dim={self.rotary_dim}, sections={self.mrope_section}"
            )

    def weight(self, name: str) -> mx.array:
        try:
            return self.weights[name]
        except KeyError as exc:
            raise KeyError(f"Missing text weight: {name}") from exc

    def embed(self, input_ids: mx.array) -> mx.array:
        return self.weight("embed_tokens.weight")[input_ids]

    def _mlp(self, hidden_states: mx.array, layer_idx: int) -> mx.array:
        prefix = f"layers.{layer_idx}.mlp."
        gate = linear(hidden_states, self.weight(prefix + "gate_proj.weight"))
        up = linear(hidden_states, self.weight(prefix + "up_proj.weight"))
        return linear(silu(gate) * up, self.weight(prefix + "down_proj.weight"))

    def _linear_attention(self, hidden_states: mx.array, layer_idx: int) -> mx.array:
        prefix = f"layers.{layer_idx}.linear_attn."
        batch, sequence, _ = hidden_states.shape

        mixed_qkv = linear(hidden_states, self.weight(prefix + "in_proj_qkv.weight"))
        mixed_qkv = causal_depthwise_conv1d(mixed_qkv, self.weight(prefix + "conv1d.weight"))

        query = mixed_qkv[..., : self.linear_key_dim]
        key = mixed_qkv[..., self.linear_key_dim : 2 * self.linear_key_dim]
        value = mixed_qkv[..., 2 * self.linear_key_dim :]
        query = query.reshape(batch, sequence, self.linear_num_key_heads, self.linear_key_head_dim)
        key = key.reshape(batch, sequence, self.linear_num_key_heads, self.linear_key_head_dim)
        value = value.reshape(batch, sequence, self.linear_num_value_heads, self.linear_value_head_dim)

        query = l2_norm(query)
        key = l2_norm(key)
        query = query * (self.linear_key_head_dim**-0.5)

        beta_logits = linear(hidden_states, self.weight(prefix + "in_proj_b.weight"))
        beta = sigmoid(beta_logits)
        a = linear(hidden_states, self.weight(prefix + "in_proj_a.weight"))
        a_log = self.weight(prefix + "A_log")
        dt_bias = self.weight(prefix + "dt_bias")
        decay_log = -mx.exp(a_log) * softplus(a + dt_bias)

        state = mx.zeros(
            (
                batch,
                self.linear_num_value_heads,
                self.linear_key_head_dim,
                self.linear_value_head_dim,
            ),
            dtype=mx.float32,
        )
        recurrent_outputs = []
        for token_idx in range(sequence):
            q_t = query[:, token_idx]
            k_t = key[:, token_idx]
            v_t = value[:, token_idx]
            g_t = mx.exp(decay_log[:, token_idx])[:, :, None, None]
            beta_t = beta[:, token_idx, :, None]

            state = state * g_t
            memory_value = mx.sum(state * k_t[..., None], axis=-2)
            delta = (v_t - memory_value) * beta_t
            state = state + k_t[..., None] * delta[..., None, :]
            recurrent_outputs.append(mx.sum(state * q_t[..., None], axis=-2))

        core_output = mx.stack(recurrent_outputs, axis=1)
        z = linear(hidden_states, self.weight(prefix + "in_proj_z.weight"))
        z = z.reshape(batch, sequence, self.linear_num_value_heads, self.linear_value_head_dim)
        core_output = rms_norm_gated(
            core_output,
            z,
            self.weight(prefix + "norm.weight"),
            self.eps,
        )
        core_output = core_output.reshape(batch, sequence, self.linear_value_dim)
        return linear(core_output, self.weight(prefix + "out_proj.weight"))

    def _rope(self, sequence: int) -> tuple[mx.array, mx.array]:
        positions = mx.arange(sequence, dtype=mx.float32)
        exponents = mx.arange(0, self.rotary_dim, 2, dtype=mx.float32) / self.rotary_dim
        inv_freq = 1.0 / (self.rope_theta**exponents)
        freqs = positions[:, None] * inv_freq[None, :]

        # HF interleaves temporal/height/width M-RoPE frequency sections. For
        # text-only inputs all three position-id planes are identical, so that
        # operation is exactly the ordinary frequency tensor used here.
        embedding = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(embedding)[None, None, :, :], mx.sin(embedding)[None, None, :, :]

    def _full_attention(self, hidden_states: mx.array, layer_idx: int) -> mx.array:
        prefix = f"layers.{layer_idx}.self_attn."
        batch, sequence, _ = hidden_states.shape

        q_and_gate = linear(hidden_states, self.weight(prefix + "q_proj.weight"))
        q_and_gate = q_and_gate.reshape(
            batch,
            sequence,
            self.num_attention_heads,
            self.head_dim * 2,
        )
        query = q_and_gate[..., : self.head_dim]
        output_gate = q_and_gate[..., self.head_dim :]
        key = linear(hidden_states, self.weight(prefix + "k_proj.weight")).reshape(
            batch, sequence, self.num_key_value_heads, self.head_dim
        )
        value = linear(hidden_states, self.weight(prefix + "v_proj.weight")).reshape(
            batch, sequence, self.num_key_value_heads, self.head_dim
        )

        query = rms_norm_additive(query, self.weight(prefix + "q_norm.weight"), self.eps)
        key = rms_norm_additive(key, self.weight(prefix + "k_norm.weight"), self.eps)
        query = mx.transpose(query, (0, 2, 1, 3))
        key = mx.transpose(key, (0, 2, 1, 3))
        value = mx.transpose(value, (0, 2, 1, 3))

        cos, sin = self._rope(sequence)
        query_rot, query_pass = query[..., : self.rotary_dim], query[..., self.rotary_dim :]
        key_rot, key_pass = key[..., : self.rotary_dim], key[..., self.rotary_dim :]
        query_rot = query_rot * cos + rotate_half(query_rot) * sin
        key_rot = key_rot * cos + rotate_half(key_rot) * sin
        query = mx.concatenate([query_rot, query_pass], axis=-1)
        key = mx.concatenate([key_rot, key_pass], axis=-1)

        key = mx.repeat(key, self.num_key_value_groups, axis=1)
        value = mx.repeat(value, self.num_key_value_groups, axis=1)
        scores = mx.matmul(query, mx.swapaxes(key, -1, -2)) * self.attention_scale
        positions = mx.arange(sequence)
        causal_mask = mx.where(
            positions[None, :] > positions[:, None],
            mx.array(-1e30, dtype=mx.float32),
            mx.array(0.0, dtype=mx.float32),
        )
        probabilities = mx.softmax(scores + causal_mask[None, None, :, :], axis=-1)
        attention_output = mx.matmul(probabilities, value)
        attention_output = mx.transpose(attention_output, (0, 2, 1, 3))
        attention_output = attention_output * sigmoid(output_gate)
        attention_output = attention_output.reshape(batch, sequence, self.hidden_size)
        return linear(attention_output, self.weight(prefix + "o_proj.weight"))

    def forward_layer(self, hidden_states: mx.array, layer_idx: int) -> mx.array:
        prefix = f"layers.{layer_idx}."
        residual = hidden_states
        mixed_input = rms_norm_additive(
            hidden_states,
            self.weight(prefix + "input_layernorm.weight"),
            self.eps,
        )
        if self.layer_types[layer_idx] == "linear_attention":
            mixed_output = self._linear_attention(mixed_input, layer_idx)
        elif self.layer_types[layer_idx] == "full_attention":
            mixed_output = self._full_attention(mixed_input, layer_idx)
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_types[layer_idx]}")
        hidden_states = residual + mixed_output

        residual = hidden_states
        mlp_input = rms_norm_additive(
            hidden_states,
            self.weight(prefix + "post_attention_layernorm.weight"),
            self.eps,
        )
        return residual + self._mlp(mlp_input, layer_idx)

    def final_norm(self, hidden_states: mx.array) -> mx.array:
        return rms_norm_additive(hidden_states, self.weight("norm.weight"), self.eps)

    def lm_head(self, hidden_states: mx.array) -> mx.array:
        """Tied output projection; the embedding matrix is the sole owner."""
        return linear(hidden_states, self.weight("embed_tokens.weight"))

    def __call__(self, input_ids: mx.array, capture: bool = False):
        """Full-sequence prefill over real tokens only.

        There is no attention-mask or padding support: every position updates
        the recurrent state and attends causally, so a padded batch would
        corrupt downstream hidden states. Callers must pass unpadded sequences.
        """
        hidden_states = self.embed(input_ids)
        mx.eval(hidden_states)
        captures = [hidden_states] if capture else None
        for layer_idx in range(self.num_layers):
            hidden_states = self.forward_layer(hidden_states, layer_idx)
            mx.eval(hidden_states)
            if capture:
                captures.append(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        mx.eval(hidden_states)
        if capture:
            return hidden_states, captures
        return hidden_states
