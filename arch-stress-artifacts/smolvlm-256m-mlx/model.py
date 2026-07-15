"""Standalone eager MLX port of SmolVLM-256M-Instruct's Idefics3 forward graph."""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import numpy as np


def linear(x: mx.array, weight: mx.array, bias: mx.array | None = None) -> mx.array:
    y = mx.matmul(x, mx.transpose(weight))
    return y if bias is None else y + bias


def layer_norm(
    x: mx.array, weight: mx.array, bias: mx.array, epsilon: float
) -> mx.array:
    mean = mx.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    variance = mx.mean(centered * centered, axis=-1, keepdims=True)
    return centered * mx.rsqrt(variance + epsilon) * weight + bias


def rms_norm(x: mx.array, weight: mx.array, epsilon: float) -> mx.array:
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + epsilon) * weight


def gelu_pytorch_tanh(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class WeightStore:
    """Load the real BF16 safetensors checkpoint and materialize float32 GPU weights."""

    def __init__(self, checkpoint: Path):
        loaded = mx.load(str(checkpoint))
        self.tensors = {name: value.astype(mx.float32) for name, value in loaded.items()}
        mx.eval(*self.tensors.values())

    def __getitem__(self, name: str) -> mx.array:
        return self.tensors[name]


class VisionTransformer:
    hidden_size = 768
    intermediate_size = 3072
    num_heads = 12
    head_dim = 64
    num_layers = 12
    patch_size = 16
    num_patches_per_side = 32
    layer_norm_eps = 1e-6

    def __init__(self, weights: WeightStore):
        self.weights = weights
        self.prefix = "model.vision_model"

    def _position_ids(self, patch_mask: np.ndarray) -> np.ndarray:
        patch_mask = np.asarray(patch_mask, dtype=bool)
        batch_size, max_h, max_w = patch_mask.shape
        boundaries = np.arange(
            1.0 / self.num_patches_per_side,
            1.0,
            1.0 / self.num_patches_per_side,
            dtype=np.float32,
        )
        count_h = patch_mask[:, :, 0].sum(axis=1)
        count_w = patch_mask[:, 0, :].sum(axis=1)
        fractional_h = np.arange(max_h, dtype=np.float32)[None, :] / count_h[:, None]
        fractional_w = np.arange(max_w, dtype=np.float32)[None, :] / count_w[:, None]
        fractional_h = np.minimum(fractional_h, np.float32(1.0 - 1e-6))
        fractional_w = np.minimum(fractional_w, np.float32(1.0 - 1e-6))
        bucket_h = np.searchsorted(boundaries, fractional_h, side="right")
        bucket_w = np.searchsorted(boundaries, fractional_w, side="right")
        grid = bucket_h[:, :, None] * self.num_patches_per_side + bucket_w[:, None, :]
        position_ids = np.zeros((batch_size, max_h * max_w), dtype=np.int32)
        flat_mask = patch_mask.reshape(batch_size, -1)
        flat_grid = grid.reshape(batch_size, -1)
        position_ids[flat_mask] = flat_grid[flat_mask]
        return position_ids

    def _attention(self, x: mx.array, layer: int, attention_mask: mx.array) -> mx.array:
        prefix = f"{self.prefix}.encoder.layers.{layer}.self_attn"
        batch_size, sequence_length, _ = x.shape
        q = linear(x, self.weights[f"{prefix}.q_proj.weight"], self.weights[f"{prefix}.q_proj.bias"])
        k = linear(x, self.weights[f"{prefix}.k_proj.weight"], self.weights[f"{prefix}.k_proj.bias"])
        v = linear(x, self.weights[f"{prefix}.v_proj.weight"], self.weights[f"{prefix}.v_proj.bias"])
        q = mx.transpose(q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim), (0, 2, 1, 3))
        k = mx.transpose(k.reshape(batch_size, sequence_length, self.num_heads, self.head_dim), (0, 2, 1, 3))
        v = mx.transpose(v.reshape(batch_size, sequence_length, self.num_heads, self.head_dim), (0, 2, 1, 3))
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * (self.head_dim**-0.5)
        probabilities = mx.softmax(scores + attention_mask, axis=-1)
        output = mx.matmul(probabilities, v)
        output = mx.transpose(output, (0, 2, 1, 3)).reshape(batch_size, sequence_length, self.hidden_size)
        return linear(
            output,
            self.weights[f"{prefix}.out_proj.weight"],
            self.weights[f"{prefix}.out_proj.bias"],
        )

    def _mlp(self, x: mx.array, layer: int) -> mx.array:
        prefix = f"{self.prefix}.encoder.layers.{layer}.mlp"
        x = linear(x, self.weights[f"{prefix}.fc1.weight"], self.weights[f"{prefix}.fc1.bias"])
        x = gelu_pytorch_tanh(x)
        return linear(x, self.weights[f"{prefix}.fc2.weight"], self.weights[f"{prefix}.fc2.bias"])

    def __call__(
        self, pixel_values_nchw: mx.array, patch_attention_mask: np.ndarray
    ) -> tuple[mx.array, dict[str, mx.array]]:
        captures: dict[str, mx.array] = {}
        prefix = f"{self.prefix}.embeddings.patch_embedding"
        pixels_nhwc = mx.transpose(pixel_values_nchw, (0, 2, 3, 1))
        kernel = mx.transpose(self.weights[f"{prefix}.weight"], (0, 2, 3, 1))
        patch_nhwc = mx.conv2d(pixels_nhwc, kernel, stride=self.patch_size)
        patch_nhwc = patch_nhwc + self.weights[f"{prefix}.bias"]
        captures["vision_patch_conv"] = mx.transpose(patch_nhwc, (0, 3, 1, 2))

        batch_size, patch_h, patch_w, _ = patch_nhwc.shape
        x = patch_nhwc.reshape(batch_size, patch_h * patch_w, self.hidden_size)
        position_ids = mx.array(self._position_ids(patch_attention_mask), dtype=mx.int32)
        position_weight = self.weights[f"{self.prefix}.embeddings.position_embedding.weight"]
        x = x + mx.take(position_weight, position_ids, axis=0)
        mx.eval(captures["vision_patch_conv"], x)
        captures["vision_patch_embed"] = x

        valid_keys = mx.array(patch_attention_mask.reshape(batch_size, 1, 1, -1), dtype=mx.bool_)
        attention_mask = mx.where(valid_keys, mx.array(0.0), mx.array(np.finfo(np.float32).min))

        for layer in range(self.num_layers):
            block = f"{self.prefix}.encoder.layers.{layer}"
            residual = x
            normalized = layer_norm(
                x,
                self.weights[f"{block}.layer_norm1.weight"],
                self.weights[f"{block}.layer_norm1.bias"],
                self.layer_norm_eps,
            )
            x = residual + self._attention(normalized, layer, attention_mask)
            residual = x
            normalized = layer_norm(
                x,
                self.weights[f"{block}.layer_norm2.weight"],
                self.weights[f"{block}.layer_norm2.bias"],
                self.layer_norm_eps,
            )
            x = residual + self._mlp(normalized, layer)
            mx.eval(x)
            captures[f"vision_layer_{layer}"] = x

        x = layer_norm(
            x,
            self.weights[f"{self.prefix}.post_layernorm.weight"],
            self.weights[f"{self.prefix}.post_layernorm.bias"],
            self.layer_norm_eps,
        )
        mx.eval(x)
        captures["vision_final"] = x
        return x, captures


class Connector:
    scale_factor = 4

    def __init__(self, weights: WeightStore):
        self.weights = weights

    def pixel_shuffle(self, x: mx.array) -> mx.array:
        batch_size, sequence_length, embed_dim = x.shape
        height = width = math.isqrt(sequence_length)
        if height * width != sequence_length or width % self.scale_factor:
            raise ValueError(f"Unsupported vision grid: sequence_length={sequence_length}")
        x = x.reshape(batch_size, height, width, embed_dim)
        x = x.reshape(batch_size, height, width // self.scale_factor, embed_dim * self.scale_factor)
        x = mx.transpose(x, (0, 2, 1, 3))
        x = x.reshape(
            batch_size,
            width // self.scale_factor,
            height // self.scale_factor,
            embed_dim * self.scale_factor**2,
        )
        x = mx.transpose(x, (0, 2, 1, 3))
        return x.reshape(
            batch_size,
            sequence_length // self.scale_factor**2,
            embed_dim * self.scale_factor**2,
        )

    def __call__(self, vision_hidden: mx.array) -> tuple[mx.array, dict[str, mx.array]]:
        shuffled = self.pixel_shuffle(vision_hidden)
        projected = linear(
            shuffled,
            self.weights["model.connector.modality_projection.proj.weight"],
        )
        mx.eval(shuffled, projected)
        return projected, {
            "connector_pixel_shuffle": shuffled,
            "connector_projector": projected,
        }


class LlamaTextTower:
    hidden_size = 576
    intermediate_size = 1536
    num_heads = 9
    num_kv_heads = 3
    head_dim = 64
    num_layers = 30
    rms_norm_eps = 1e-5
    rope_theta = 100000.0

    def __init__(self, weights: WeightStore):
        self.weights = weights
        self.prefix = "model.text_model"

    def fuse(
        self, input_ids: np.ndarray, image_hidden: mx.array, image_token_id: int
    ) -> mx.array:
        ids = np.asarray(input_ids, dtype=np.int64)
        embeddings = mx.take(
            self.weights[f"{self.prefix}.embed_tokens.weight"],
            mx.array(ids, dtype=mx.int32),
            axis=0,
        )
        positions = np.flatnonzero(ids.reshape(-1) == image_token_id).astype(np.int32)
        image_flat = image_hidden.reshape(-1, self.hidden_size)
        if positions.size != image_flat.shape[0]:
            raise ValueError(
                f"Image placeholder count {positions.size} != projected feature count {image_flat.shape[0]}"
            )
        flat = embeddings.reshape(-1, self.hidden_size)
        position_array = mx.array(positions, dtype=mx.int32)
        current = mx.take(flat, position_array, axis=0)
        flat = flat.at[position_array].add(image_flat - current)
        fused = flat.reshape(*embeddings.shape)
        mx.eval(fused)
        return fused

    def _rope(self, sequence_length: int) -> tuple[mx.array, mx.array]:
        dimensions = mx.arange(0, self.head_dim, 2, dtype=mx.float32)
        inv_frequency = 1.0 / mx.power(self.rope_theta, dimensions / self.head_dim)
        positions = mx.arange(sequence_length, dtype=mx.float32)
        frequencies = positions[:, None] * inv_frequency[None, :]
        embedding = mx.concatenate((frequencies, frequencies), axis=-1)
        return mx.cos(embedding)[None, None, :, :], mx.sin(embedding)[None, None, :, :]

    @staticmethod
    def _rotate_half(x: mx.array) -> mx.array:
        half = x.shape[-1] // 2
        return mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)

    def _attention(
        self,
        x: mx.array,
        layer: int,
        causal_mask: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        prefix = f"{self.prefix}.layers.{layer}.self_attn"
        batch_size, sequence_length, _ = x.shape
        q = linear(x, self.weights[f"{prefix}.q_proj.weight"])
        k = linear(x, self.weights[f"{prefix}.k_proj.weight"])
        v = linear(x, self.weights[f"{prefix}.v_proj.weight"])
        q = mx.transpose(q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim), (0, 2, 1, 3))
        k = mx.transpose(k.reshape(batch_size, sequence_length, self.num_kv_heads, self.head_dim), (0, 2, 1, 3))
        v = mx.transpose(v.reshape(batch_size, sequence_length, self.num_kv_heads, self.head_dim), (0, 2, 1, 3))
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        repeats = self.num_heads // self.num_kv_heads
        k = mx.repeat(k, repeats=repeats, axis=1)
        v = mx.repeat(v, repeats=repeats, axis=1)
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * (self.head_dim**-0.5)
        probabilities = mx.softmax(scores + causal_mask, axis=-1)
        output = mx.matmul(probabilities, v)
        output = mx.transpose(output, (0, 2, 1, 3)).reshape(batch_size, sequence_length, self.hidden_size)
        return linear(output, self.weights[f"{prefix}.o_proj.weight"])

    def _mlp(self, x: mx.array, layer: int) -> mx.array:
        prefix = f"{self.prefix}.layers.{layer}.mlp"
        gate = silu(linear(x, self.weights[f"{prefix}.gate_proj.weight"]))
        up = linear(x, self.weights[f"{prefix}.up_proj.weight"])
        return linear(gate * up, self.weights[f"{prefix}.down_proj.weight"])

    def __call__(
        self, fused_embeddings: mx.array, attention_mask: np.ndarray
    ) -> tuple[mx.array, dict[str, mx.array]]:
        captures: dict[str, mx.array] = {"fused_embeds": fused_embeddings}
        x = fused_embeddings
        batch_size, sequence_length, _ = x.shape
        minimum = np.finfo(np.float32).min
        causal = mx.triu(mx.full((sequence_length, sequence_length), minimum), k=1)
        causal = causal[None, None, :, :]
        valid_keys = mx.array(np.asarray(attention_mask).reshape(batch_size, 1, 1, -1) > 0)
        padding = mx.where(valid_keys, mx.array(0.0), mx.array(minimum))
        causal_mask = causal + padding
        cos, sin = self._rope(sequence_length)

        for layer in range(self.num_layers):
            prefix = f"{self.prefix}.layers.{layer}"
            residual = x
            normalized = rms_norm(
                x,
                self.weights[f"{prefix}.input_layernorm.weight"],
                self.rms_norm_eps,
            )
            x = residual + self._attention(normalized, layer, causal_mask, cos, sin)
            residual = x
            normalized = rms_norm(
                x,
                self.weights[f"{prefix}.post_attention_layernorm.weight"],
                self.rms_norm_eps,
            )
            x = residual + self._mlp(normalized, layer)
            mx.eval(x)
            captures[f"text_layer_{layer}"] = x

        x = rms_norm(x, self.weights[f"{self.prefix}.norm.weight"], self.rms_norm_eps)
        mx.eval(x)
        captures["text_final"] = x
        return x, captures


class Idefics3MLX:
    image_token_id = 49190

    def __init__(self, checkpoint: Path):
        self.weights = WeightStore(checkpoint)
        self.vision = VisionTransformer(self.weights)
        self.connector = Connector(self.weights)
        self.text = LlamaTextTower(self.weights)

    def __call__(
        self,
        pixel_values_nchw: mx.array,
        patch_attention_mask: np.ndarray,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> dict[str, mx.array]:
        vision_hidden, captures = self.vision(pixel_values_nchw, patch_attention_mask)
        image_hidden, connector_captures = self.connector(vision_hidden)
        captures.update(connector_captures)
        fused = self.text.fuse(input_ids, image_hidden, self.image_token_id)
        text_hidden, text_captures = self.text(fused, attention_mask)
        captures.update(text_captures)
        logits = linear(text_hidden, self.weights["lm_head.weight"])
        mx.eval(logits)
        captures["logits"] = logits
        return captures
