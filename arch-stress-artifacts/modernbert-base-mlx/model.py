from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class ModernBertConfig:
    vocab_size: int = 50368
    hidden_size: int = 768
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
    intermediate_size: int = 1152
    norm_eps: float = 1e-5
    global_attn_every_n_layers: int = 3
    local_attention: int = 128
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def local_half_window(self) -> int:
        return self.local_attention // 2

    @classmethod
    def from_json(cls, path: str | Path) -> "ModernBertConfig":
        raw = json.loads(Path(path).read_text())
        config = cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=int(raw["hidden_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            intermediate_size=int(raw["intermediate_size"]),
            norm_eps=float(raw.get("norm_eps", raw["layer_norm_eps"])),
            global_attn_every_n_layers=int(raw["global_attn_every_n_layers"]),
            local_attention=int(raw["local_attention"]),
            global_rope_theta=float(raw["global_rope_theta"]),
            local_rope_theta=float(raw["local_rope_theta"]),
        )
        expected = cls()
        if config != expected:
            raise ValueError(f"checkpoint config does not match ModernBERT-base: {config!r}")
        for false_key in ("attention_bias", "mlp_bias", "norm_bias"):
            if raw.get(false_key) is not False:
                raise ValueError(f"expected {false_key}=false")
        if raw.get("hidden_activation") != "gelu":
            raise ValueError("expected exact GELU activation")
        return config


def exact_gelu(x: mx.array) -> mx.array:
    # Keep this eager instead of calling mlx.nn.gelu, whose public wrapper is compiled.
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)


class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.norm(self.tok_embeddings(input_ids))


class ModernBertMLP(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.Wi = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        projected = self.Wi(hidden_states)
        input_part = projected[..., : self.intermediate_size]
        gate = projected[..., self.intermediate_size :]
        return self.Wo(exact_gelu(input_part) * gate)


class ModernBertAttention(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.Wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        cos: mx.array,
        sin: mx.array,
        additive_mask: mx.array,
    ) -> mx.array:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        qkv = self.Wqkv(hidden_states).reshape(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        )
        query = mx.transpose(qkv[:, :, 0], (0, 2, 1, 3))
        key = mx.transpose(qkv[:, :, 1], (0, 2, 1, 3))
        value = mx.transpose(qkv[:, :, 2], (0, 2, 1, 3))

        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin

        scores = (query @ mx.transpose(key, (0, 1, 3, 2))) * self.scale
        probabilities = mx.softmax(scores + additive_mask, axis=-1)
        context = probabilities @ value
        context = mx.transpose(context, (0, 2, 1, 3)).reshape(
            batch_size, sequence_length, hidden_size
        )
        return self.Wo(context)


class ModernBertLayer(nn.Module):
    def __init__(self, config: ModernBertConfig, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.is_global = layer_index % config.global_attn_every_n_layers == 0
        self.attn_norm = (
            None
            if layer_index == 0
            else nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
        )
        self.attn = ModernBertAttention(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
        self.mlp = ModernBertMLP(config)

    def __call__(
        self,
        hidden_states: mx.array,
        cos: mx.array,
        sin: mx.array,
        additive_mask: mx.array,
    ) -> mx.array:
        normalized = hidden_states if self.attn_norm is None else self.attn_norm(hidden_states)
        hidden_states = hidden_states + self.attn(normalized, cos, sin, additive_mask)
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class ModernBertModel(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = [ModernBertLayer(config, i) for i in range(config.num_hidden_layers)]
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)

    def _rope(self, sequence_length: int, theta: float) -> tuple[mx.array, mx.array]:
        positions = mx.arange(sequence_length, dtype=mx.float32)
        dimensions = mx.arange(0, self.config.head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (theta ** (dimensions / self.config.head_dim))
        frequencies = positions[:, None] * inv_freq[None, :]
        embedding = mx.concatenate((frequencies, frequencies), axis=-1)
        # [1, 1, sequence, head_dim], matching HF's unsqueeze_dim=1 broadcast.
        return mx.cos(embedding)[None, None, :, :], mx.sin(embedding)[None, None, :, :]

    def _attention_masks(self, attention_mask: mx.array) -> tuple[mx.array, mx.array]:
        if attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must have rank 2, got {attention_mask.shape}")
        sequence_length = attention_mask.shape[1]
        positions = mx.arange(sequence_length)
        distance = mx.abs(positions[:, None] - positions[None, :])
        valid_keys = attention_mask.astype(mx.bool_)[:, None, None, :]
        local_window = distance[None, None, :, :] <= self.config.local_half_window
        zero = mx.array(0.0, dtype=mx.float32)
        minimum = mx.array(-3.4028234663852886e38, dtype=mx.float32)
        full = mx.where(valid_keys, zero, minimum)
        local = mx.where(valid_keys & local_window, zero, minimum)
        return full, local

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        *,
        capture_hidden_states: bool = False,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError(
                f"input_ids and attention_mask must share rank-2 shape, got "
                f"{input_ids.shape} and {attention_mask.shape}"
            )
        sequence_length = input_ids.shape[1]
        global_rope = self._rope(sequence_length, self.config.global_rope_theta)
        local_rope = self._rope(sequence_length, self.config.local_rope_theta)
        full_mask, local_mask = self._attention_masks(attention_mask)

        captures: dict[str, mx.array] = {}
        hidden_states = self.embeddings(input_ids)
        mx.eval(hidden_states)
        if capture_hidden_states:
            captures["embed"] = hidden_states

        for i, layer in enumerate(self.layers):
            cos, sin = global_rope if layer.is_global else local_rope
            mask = full_mask if layer.is_global else local_mask
            hidden_states = layer(hidden_states, cos, sin, mask)
            mx.eval(hidden_states)
            if capture_hidden_states:
                captures[f"layer.{i}.hidden"] = hidden_states

        hidden_states = self.final_norm(hidden_states)
        mx.eval(hidden_states)
        if capture_hidden_states:
            captures["final_norm"] = hidden_states
        return hidden_states, captures


def expected_encoder_tensor_names(config: ModernBertConfig) -> set[str]:
    names = {
        "model.embeddings.tok_embeddings.weight",
        "model.embeddings.norm.weight",
        "model.final_norm.weight",
    }
    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}"
        names.update(
            {
                f"{prefix}.attn.Wqkv.weight",
                f"{prefix}.attn.Wo.weight",
                f"{prefix}.mlp_norm.weight",
                f"{prefix}.mlp.Wi.weight",
                f"{prefix}.mlp.Wo.weight",
            }
        )
        if i != 0:
            names.add(f"{prefix}.attn_norm.weight")
    return names


def load_safetensors(
    model: ModernBertModel, checkpoint: str | Path
) -> dict[str, Any]:
    weights = mx.load(str(checkpoint))
    actual_names = set(weights)
    required_names = expected_encoder_tensor_names(model.config)
    ignored_names = {
        "decoder.bias",
        "head.dense.weight",
        "head.norm.weight",
    }
    if len(actual_names) != 137:
        raise ValueError(f"expected 137 checkpoint tensors, got {len(actual_names)}")
    if actual_names != required_names | ignored_names:
        missing = sorted(required_names - actual_names)
        unexpected = sorted(actual_names - required_names - ignored_names)
        raise ValueError(f"checkpoint key mismatch: missing={missing}, unexpected={unexpected}")

    def assign(target: Any, attribute: str, source_name: str) -> None:
        source = weights[source_name].astype(mx.float32)
        current = getattr(target, attribute)
        if source.shape != current.shape:
            raise ValueError(
                f"shape mismatch for {source_name}: checkpoint={source.shape}, target={current.shape}"
            )
        setattr(target, attribute, source)

    assign(model.embeddings.tok_embeddings, "weight", "model.embeddings.tok_embeddings.weight")
    assign(model.embeddings.norm, "weight", "model.embeddings.norm.weight")
    assign(model.final_norm, "weight", "model.final_norm.weight")
    for i, layer in enumerate(model.layers):
        prefix = f"model.layers.{i}"
        if layer.attn_norm is not None:
            assign(layer.attn_norm, "weight", f"{prefix}.attn_norm.weight")
        assign(layer.attn.Wqkv, "weight", f"{prefix}.attn.Wqkv.weight")
        assign(layer.attn.Wo, "weight", f"{prefix}.attn.Wo.weight")
        assign(layer.mlp_norm, "weight", f"{prefix}.mlp_norm.weight")
        assign(layer.mlp.Wi, "weight", f"{prefix}.mlp.Wi.weight")
        assign(layer.mlp.Wo, "weight", f"{prefix}.mlp.Wo.weight")

    # PyTorch and MLX nn.Linear both store [out, in]. MLX applies W.T at call
    # time, so checkpoint matrices are deliberately loaded without transposing.
    mx.eval(model.parameters())
    return {
        "tensor_names_verified": True,
        "checkpoint_tensor_count": len(actual_names),
        "encoder_tensor_count": len(required_names),
        "ignored_tensor_names": sorted(ignored_names),
        "linear_checkpoint_layout": "out_in_loaded_directly_runtime_transpose",
    }


def build_from_directory(model_directory: str | Path) -> tuple[ModernBertModel, dict[str, Any]]:
    model_directory = Path(model_directory)
    config = ModernBertConfig.from_json(model_directory / "config.json")
    model = ModernBertModel(config)
    report = load_safetensors(model, model_directory / "model.safetensors")
    return model, report
