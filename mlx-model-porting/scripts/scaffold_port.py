#!/usr/bin/env python3
"""Generate a parity-first MLX package from a trusted static inspection.

This generator is dependency-free. It validates JSON and writes Python source;
MLX and NumPy are imported only by the generated package at execution time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from _common import SkillError, load_structured
from make_port_plan import verify_inspection_against_artifact
from recommend_optimizations import (
    resolve_route_families,
    trusted_inspection_sha256,
    validate_trusted_inspection,
)


GENERATOR_VERSION = "1.0.0"
SCAFFOLD_MANIFEST = "scaffold-manifest.json"
DENSE_FAMILY = "dense-decoder-transformer"
DENSE_RUNBOOK = "references/runbook-decoder-transformer.md"

# Computation-bearing keys whose semantics this generator implements. Keys that
# are not here must be known non-computational metadata or they fail closed.
DENSE_CONFIG_FEATURE_ALLOWLIST = frozenset({
    "attention_bias",
    "attention_dropout",
    "hidden_act",
    "activation_function",
    "hidden_size",
    "head_dim",
    "intermediate_size",
    "max_window_layers",
    "max_position_embeddings",
    "mlp_bias",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "pretraining_tp",
    "rms_norm_eps",
    "rope_scaling",
    "rope_theta",
    "rope_traditional",
    "sliding_window",
    "tie_word_embeddings",
    "use_sliding_window",
    "vocab_size",
})
FEATURE_ALLOWLIST = DENSE_CONFIG_FEATURE_ALLOWLIST

KNOWN_METADATA_KEYS = frozenset({
    "_commit_hash",
    "_name_or_path",
    "architectures",
    "auto_map",
    "bad_words_ids",
    "begin_suppress_tokens",
    "bos_token_id",
    "decoder_start_token_id",
    "dtype",
    "early_stopping",
    "eos_token_id",
    "exponential_decay_length_penalty",
    "finetuning_task",
    "forced_bos_token_id",
    "forced_eos_token_id",
    "id2label",
    "initializer_range",
    "is_decoder",
    "is_encoder_decoder",
    "label2id",
    "length_penalty",
    "license",
    "max_length",
    "min_length",
    "model_type",
    "no_repeat_ngram_size",
    "num_beam_groups",
    "num_beams",
    "num_return_sequences",
    "output_attentions",
    "output_hidden_states",
    "output_scores",
    "pad_token_id",
    "prefix",
    "problem_type",
    "pruned_heads",
    "remove_invalid_values",
    "repetition_penalty",
    "return_dict",
    "return_dict_in_generate",
    "sep_token_id",
    "suppress_tokens",
    "task_specific_params",
    "temperature",
    "tf_legacy_loss",
    "tie_encoder_decoder",
    "tokenizer_class",
    "top_k",
    "top_p",
    "torch_dtype",
    "transformers_version",
    "typical_p",
    "use_bfloat16",
    "use_cache",
})

MOE_KEYS = frozenset({
    "decoder_sparse_step",
    "expert_capacity",
    "expert_interval",
    "moe_intermediate_size",
    "moe_layer_freq",
    "moe_shared_expert_intermediate_size",
    "norm_topk_prob",
    "num_experts",
    "num_experts_per_tok",
    "num_local_experts",
    "num_shared_experts",
    "router_aux_loss_coef",
    "router_jitter_noise",
    "shared_expert_intermediate_size",
})
QUANTIZATION_KEYS = frozenset({
    "bits",
    "group_size",
    "quantization",
    "quantization_config",
    "quant_method",
})
ATTENTION_VARIANT_KEYS = frozenset({
    "alibi",
    "attention_chunk_size",
    "attention_sink_size",
    "attention_type",
    "attn_logit_softcapping",
    "logit_softcap",
    "partial_rotary_factor",
    "position_embedding_type",
    "qk_norm",
    "query_pre_attn_scalar",
    "sliding_window",
    "use_alibi",
    "use_qk_norm",
    "use_sliding_window",
})
SUPPORTED_ACTIVATIONS = frozenset({"gelu", "relu", "silu", "swish"})
SUPPORTED_ROPE_TYPES = frozenset({"default", "dynamic", "linear"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a runnable parity-first MLX dense-decoder package",
    )
    parser.add_argument("inspection", help="Trusted inspection.json produced by inspect_model.py")
    parser.add_argument(
        "--artifact-root",
        help="Local inspected artifact; required for byte-bound re-verification",
    )
    parser.add_argument("--output", required=True, help="New directory for generated Python files")
    return parser.parse_args(argv)


def _meaningfully_set(value: Any) -> bool:
    return value not in (None, False, 0, 0.0, "", [], {})


def _config_int(config: dict[str, Any], key: str, *, default: int | None = None) -> int:
    value = config.get(key, default)
    if type(value) is not int or value <= 0:
        raise SkillError(f"config.json {key} must be a positive integer")
    return value


def _config_float(config: dict[str, Any], key: str, *, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SkillError(f"config.json {key} must be a positive number")
    return float(value)


def _rope_type(rope_scaling: dict[str, Any]) -> str:
    first = rope_scaling.get("rope_type")
    second = rope_scaling.get("type")
    if first is not None and second is not None and first != second:
        return f"conflicting:{first!r}:{second!r}"
    value = first if first is not None else second
    return str(value or "default").lower()


def unsupported_dense_features(config: dict[str, Any]) -> list[str]:
    """Return every computation feature that this generator cannot implement."""
    errors: list[str] = []
    activation = config.get("hidden_act", config.get("activation_function", "silu"))
    if not isinstance(activation, str) or activation.lower() not in SUPPORTED_ACTIVATIONS:
        errors.append(
            f"hidden_act={activation!r} is not supported; supported activations: "
            + ", ".join(sorted(SUPPORTED_ACTIVATIONS))
        )
    dropout = config.get("attention_dropout", 0.0)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or dropout != 0.0:
        errors.append("attention_dropout must be 0.0 for eager inference")
    if config.get("pretraining_tp", 1) != 1:
        errors.append("pretraining_tp must be 1")

    for key in sorted(MOE_KEYS):
        if key in config and _meaningfully_set(config[key]):
            errors.append(f"MoE config key {key!r} is set")

    for key in sorted(ATTENTION_VARIANT_KEYS - {"sliding_window", "use_sliding_window"}):
        if key not in config or not _meaningfully_set(config[key]):
            continue
        value = config[key]
        if key == "partial_rotary_factor" and value == 1.0:
            continue
        if key == "position_embedding_type" and str(value).lower() in {"rope", "rotary"}:
            continue
        errors.append(f"{key}={value!r} is not supported")

    for key in sorted(QUANTIZATION_KEYS):
        is_set = key in config and (
            config[key] is not None if key == "quantization_config" else _meaningfully_set(config[key])
        )
        if is_set:
            label = "quantization_config is set" if key == "quantization_config" else f"{key}={config[key]!r} is not supported"
            errors.append(label)

    rope_scaling = config.get("rope_scaling")
    if rope_scaling is not None:
        if not isinstance(rope_scaling, dict):
            errors.append("rope_scaling must be null or an object")
        else:
            rope_type = _rope_type(rope_scaling)
            if rope_type not in SUPPORTED_ROPE_TYPES:
                errors.append(
                    f"rope_scaling type {rope_type!r} is not supported; supported types: "
                    + ", ".join(sorted(SUPPORTED_ROPE_TYPES))
                )
            else:
                allowed_rope_keys = {"rope_type", "type"}
                if rope_type in {"dynamic", "linear"}:
                    allowed_rope_keys.update({"factor", "original_max_position_embeddings"})
                for key in sorted(set(rope_scaling) - allowed_rope_keys):
                    errors.append(f"rope_scaling key {key!r} is not supported for type {rope_type!r}")
                if rope_type in {"dynamic", "linear"}:
                    factor = rope_scaling.get("factor")
                    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor <= 1.0:
                        errors.append(f"rope_scaling factor must be a number greater than 1 for type {rope_type!r}")

    use_sliding_window = config.get("use_sliding_window")
    if "use_sliding_window" in config and not isinstance(use_sliding_window, bool):
        errors.append("use_sliding_window must be boolean when present")
    sliding_window = config.get("sliding_window")
    if "sliding_window" in config and (
        type(sliding_window) is not int or sliding_window <= 0
    ):
        errors.append("sliding_window must be a positive integer when present")
    max_window_layers = config.get("max_window_layers")
    if "max_window_layers" in config and (
        type(max_window_layers) is not int or max_window_layers <= 0
    ):
        errors.append("max_window_layers must be a positive integer when present")
    if use_sliding_window is True:
        errors.append("use_sliding_window=True is not supported")
    elif _meaningfully_set(sliding_window) and use_sliding_window is not False:
        errors.append(
            f"sliding_window={sliding_window!r} requires explicit "
            "use_sliding_window=false for full attention"
        )

    classified = (
        DENSE_CONFIG_FEATURE_ALLOWLIST
        | KNOWN_METADATA_KEYS
        | MOE_KEYS
        | QUANTIZATION_KEYS
        | ATTENTION_VARIANT_KEYS
    )
    for key in sorted(set(config) - classified):
        errors.append(f"unrecognized computation-relevant config key {key!r}")
    return errors


def validate_dense_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise SkillError("config.json must contain an object")
    errors = unsupported_dense_features(config)
    if errors:
        raise SkillError(
            "Unsupported config features; no code was generated. "
            f"Consult {DENSE_RUNBOOK}:\n- " + "\n- ".join(errors)
        )

    hidden_size = _config_int(config, "hidden_size")
    num_heads = _config_int(config, "num_attention_heads")
    num_kv_heads = _config_int(config, "num_key_value_heads", default=num_heads)
    head_dim = _config_int(config, "head_dim", default=hidden_size // num_heads)
    _config_int(config, "num_hidden_layers")
    _config_int(config, "intermediate_size")
    _config_int(config, "vocab_size")
    _config_int(config, "max_position_embeddings", default=2048)
    _config_float(config, "rms_norm_eps", default=1e-5)
    _config_float(config, "rope_theta", default=10000.0)
    if hidden_size != num_heads * head_dim:
        raise SkillError("config.json hidden_size must equal num_attention_heads * head_dim")
    if num_heads % num_kv_heads != 0:
        raise SkillError("config.json num_attention_heads must be divisible by num_key_value_heads")
    for key in ("attention_bias", "mlp_bias", "rope_traditional", "tie_word_embeddings"):
        if key in config and not isinstance(config[key], bool):
            raise SkillError(f"config.json {key} must be boolean")
    dropout = config.get("attention_dropout", 0.0)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or dropout != 0.0:
        raise SkillError("Unsupported config features; no code was generated. "
                         f"Consult {DENSE_RUNBOOK}:\n- attention_dropout must be 0.0 for eager inference")
    if config.get("pretraining_tp", 1) != 1:
        raise SkillError("Unsupported config features; no code was generated. "
                         f"Consult {DENSE_RUNBOOK}:\n- pretraining_tp must be 1")
    rope_scaling = config.get("rope_scaling")
    if isinstance(rope_scaling, dict) and _rope_type(rope_scaling) == "dynamic" and head_dim <= 2:
        raise SkillError("config.json dynamic RoPE requires head_dim greater than 2")
    return config


CONFIG_TEMPLATE = r'''__HEADER__
"""Validated config.json parser for the generated dense decoder."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _positive_int(data: dict[str, Any], key: str, default: int | None = None) -> int:
    value = data.get(key, default)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _positive_float(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{key} must be a positive number")
    return float(value)


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict[str, Any] | None
    rope_traditional: bool
    vocab_size: int
    tie_word_embeddings: bool
    attention_bias: bool
    mlp_bias: bool
    max_position_embeddings: int
    hidden_act: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        hidden_size = _positive_int(data, "hidden_size")
        heads = _positive_int(data, "num_attention_heads")
        kv_heads = _positive_int(data, "num_key_value_heads", heads)
        head_dim = _positive_int(data, "head_dim", hidden_size // heads)
        if hidden_size != heads * head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if heads % kv_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        rope_scaling = data.get("rope_scaling")
        if rope_scaling is not None and not isinstance(rope_scaling, dict):
            raise ValueError("rope_scaling must be null or an object")
        activation = data.get("hidden_act", data.get("activation_function", "silu"))
        if not isinstance(activation, str):
            raise ValueError("hidden_act must be a string")
        return cls(
            hidden_size=hidden_size,
            num_hidden_layers=_positive_int(data, "num_hidden_layers"),
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            intermediate_size=_positive_int(data, "intermediate_size"),
            rms_norm_eps=_positive_float(data, "rms_norm_eps", 1e-5),
            rope_theta=_positive_float(data, "rope_theta", 10000.0),
            rope_scaling=dict(rope_scaling) if rope_scaling is not None else None,
            rope_traditional=bool(data.get("rope_traditional", False)),
            vocab_size=_positive_int(data, "vocab_size"),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
            attention_bias=bool(data.get("attention_bias", False)),
            mlp_bias=bool(data.get("mlp_bias", False)),
            max_position_embeddings=_positive_int(data, "max_position_embeddings", 2048),
            hidden_act=activation.lower(),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("config.json must contain an object")
        return cls.from_dict(data)
'''


MODEL_TEMPLATE = r'''__HEADER__
"""Minimal eager MLX dense decoder with GQA/MHA and growing KV caches."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from config import ModelConfig


def _require_mlx():
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as exc:
        raise RuntimeError("generated model requires MLX: python3 -m pip install mlx") from exc
    return mx, nn


def _rope_kind(config: ModelConfig) -> str:
    scaling = config.rope_scaling or {}
    return str(scaling.get("rope_type", scaling.get("type", "default"))).lower()


def build_model(config: ModelConfig):
    """Build and return an MLX nn.Module without importing MLX at module import time."""
    mx, nn = _require_mlx()

    def activate(x):
        if config.hidden_act in {"silu", "swish"}:
            return nn.silu(x)
        if config.hidden_act == "gelu":
            return nn.gelu(x)
        if config.hidden_act == "relu":
            return nn.relu(x)
        raise ValueError(f"unsupported activation {config.hidden_act!r}")

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            query_dims = config.num_attention_heads * config.head_dim
            kv_dims = config.num_key_value_heads * config.head_dim
            self.q_proj = nn.Linear(config.hidden_size, query_dims, bias=__Q_PROJ_BIAS__)
            self.k_proj = nn.Linear(config.hidden_size, kv_dims, bias=__K_PROJ_BIAS__)
            self.v_proj = nn.Linear(config.hidden_size, kv_dims, bias=__V_PROJ_BIAS__)
            self.o_proj = nn.Linear(query_dims, config.hidden_size, bias=__O_PROJ_BIAS__)

        def _apply_rope(self, value, offset: int, total_length: int):
            scaling = config.rope_scaling or {}
            kind = _rope_kind(config)
            factor = float(scaling.get("factor", 1.0))
            base = config.rope_theta
            scale = 1.0 / factor if kind == "linear" else 1.0
            original = int(scaling.get("original_max_position_embeddings", config.max_position_embeddings))
            if kind == "dynamic" and total_length > original:
                ratio = factor * total_length / original - (factor - 1.0)
                base *= ratio ** (config.head_dim / (config.head_dim - 2.0))
            return mx.fast.rope(
                value,
                config.head_dim,
                traditional=config.rope_traditional,
                base=base,
                scale=scale,
                offset=offset,
            )

        def __call__(self, x, cache=None, attention_mask=None):
            batch, length, _ = x.shape
            q = self.q_proj(x).reshape(batch, length, config.num_attention_heads, config.head_dim)
            k = self.k_proj(x).reshape(batch, length, config.num_key_value_heads, config.head_dim)
            v = self.v_proj(x).reshape(batch, length, config.num_key_value_heads, config.head_dim)
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            offset = 0 if cache is None else int(cache[0].shape[2])
            if cache is not None:
                k = mx.concatenate((cache[0], k), axis=2)
                v = mx.concatenate((cache[1], v), axis=2)
            new_cache = (k, v)
            total_length = k.shape[2]
            q = self._apply_rope(q, offset, total_length)
            rotated_k = self._apply_rope(k, 0, total_length)
            repeats = config.num_attention_heads // config.num_key_value_heads
            expanded_k = mx.repeat(rotated_k, repeats, axis=1) if repeats != 1 else rotated_k
            expanded_v = mx.repeat(v, repeats, axis=1) if repeats != 1 else v
            scores = (q @ expanded_k.transpose(0, 1, 3, 2)) * (config.head_dim ** -0.5)
            key_length = expanded_k.shape[2]
            query_positions = mx.arange(offset, offset + length)[:, None]
            key_positions = mx.arange(key_length)[None, :]
            allowed = key_positions <= query_positions
            scores = mx.where(allowed[None, None, :, :], scores, mx.array(-1e9, dtype=scores.dtype))
            if attention_mask is not None:
                if attention_mask.shape[-1] != key_length:
                    raise ValueError("attention_mask length must include the complete cached context")
                scores = mx.where(
                    attention_mask[:, None, None, :].astype(mx.bool_),
                    scores,
                    mx.array(-1e9, dtype=scores.dtype),
                )
            probabilities = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
            attended = probabilities @ expanded_v
            attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, config.hidden_size)
            return self.o_proj(attended), new_cache

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

        def __call__(self, x):
            return self.down_proj(activate(self.gate_proj(x)) * self.up_proj(x))

    class DecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()
            self.mlp = MLP()
            self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        def __call__(self, x, cache=None, attention_mask=None):
            attention, new_cache = self.self_attn(
                self.input_layernorm(x),
                cache=cache,
                attention_mask=attention_mask,
            )
            hidden = x + attention
            mlp = self.mlp(self.post_attention_layernorm(hidden))
            return hidden + mlp, new_cache, attention, mlp

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.layers = [DecoderLayer() for _ in range(config.num_hidden_layers)]
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        def __call__(self, input_ids, cache=None, attention_mask=None, capture=False):
            if cache is not None and len(cache) != len(self.layers):
                raise ValueError("cache must contain one (key, value) pair per decoder layer")
            hidden = self.embed_tokens(input_ids)
            captures = {"embed": hidden} if capture else {}
            updated = []
            for index, layer in enumerate(self.layers):
                layer_cache = None if cache is None else cache[index]
                hidden, layer_cache, attention, mlp = layer(
                    hidden,
                    cache=layer_cache,
                    attention_mask=attention_mask,
                )
                updated.append(layer_cache)
                if capture:
                    captures[f"layer.{index}.attention"] = attention
                    captures[f"layer.{index}.mlp"] = mlp
                    captures[f"layer.{index}.hidden"] = hidden
            hidden = self.norm(hidden)
            if capture:
                captures["final_norm"] = hidden
            return hidden, updated, captures

    class CausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            if not config.tie_word_embeddings:
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        def __call__(self, input_ids, *, attention_mask=None, cache=None, capture=False):
            hidden, updated, captures = self.model(
                input_ids,
                cache=cache,
                attention_mask=attention_mask,
                capture=capture,
            )
            logits = (
                self.model.embed_tokens.as_linear(hidden)
                if config.tie_word_embeddings
                else self.lm_head(hidden)
            )
            if capture:
                captures["logits"] = logits
                return logits, updated, captures
            return logits, updated

    return CausalLM()


def load_model(config_path: str | Path, weights_path: str | Path):
    mx, _ = _require_mlx()
    config = ModelConfig.from_file(config_path)
    model = build_model(config)
    weights = mx.load(str(weights_path))
    if not isinstance(weights, dict):
        raise ValueError("converted weights must load as a name-to-array mapping")
    model.load_weights(list(weights.items()), strict=True)
    getattr(mx, "eval")(model.parameters())
    return model


def greedy_generate(model, input_ids, max_new_tokens: int):
    mx, _ = _require_mlx()
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    batch = input_ids.shape[0]
    if max_new_tokens == 0:
        return mx.zeros((batch, 0), dtype=input_ids.dtype)
    logits, cache = model(input_ids)
    generated = []
    next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(input_ids.dtype)
    for index in range(max_new_tokens):
        generated.append(next_token)
        if index + 1 < max_new_tokens:
            logits, cache = model(next_token[:, None], cache=cache)
            next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(input_ids.dtype)
    result = mx.stack(generated, axis=1)
    getattr(mx, "eval")(result)
    return result
'''


GENERATE_TEMPLATE = r'''__HEADER__
"""Greedy tokenizer-free generation CLI for converted MLX weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model import _require_mlx, greedy_generate, load_model


def parse_token_ids(values: list[str]) -> list[int]:
    pieces = [piece.strip() for value in values for piece in value.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise ValueError("--token-ids requires comma- or space-separated integers")
    token_ids = [int(piece, 10) for piece in pieces]
    if any(value < 0 for value in token_ids):
        raise ValueError("--token-ids must be non-negative")
    return token_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Greedy-decode token IDs with generated MLX code")
    parser.add_argument("--weights", required=True, help="Converted .safetensors or .npz weights")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--token-ids", nargs="+", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()
    mx, _ = _require_mlx()
    input_ids = mx.array([parse_token_ids(args.token_ids)], dtype=mx.int32)
    model = load_model(args.config, args.weights)
    generated = greedy_generate(model, input_ids, args.max_new_tokens)
    print(json.dumps({"generated_token_ids": generated.tolist()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


CAPTURE_TEMPLATE = r'''__HEADER__
"""Capture target MLX tensors using the source-oracle manifest contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

from model import _require_mlx, greedy_generate, load_model


def parse_token_ids(values: list[str]) -> list[int]:
    pieces = [piece.strip() for value in values for piece in value.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise ValueError("--token-ids requires comma- or space-separated integers")
    token_ids = [int(piece, 10) for piece in pieces]
    if any(value < 0 for value in token_ids):
        raise ValueError("--token-ids must be non-negative")
    return token_ids


def file_record(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"name": path.name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def tensor_inventory(arrays) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        }
        for name, value in sorted(arrays.items())
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture generated MLX intermediate tensors")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--token-ids", nargs="+", required=True)
    parser.add_argument("--generate-steps", type=int, default=4)
    parser.add_argument("--output", required=True, help="Target .npz path")
    parser.add_argument("--manifest", help="Default: <output stem>.manifest.json")
    args = parser.parse_args()
    if args.generate_steps < 0:
        raise ValueError("--generate-steps must be non-negative")
    output = Path(args.output)
    if output.suffix.lower() != ".npz":
        raise ValueError("--output must end with .npz")
    manifest_path = Path(args.manifest) if args.manifest else output.with_suffix(".manifest.json")
    config_path = Path(args.config)
    weights_path = Path(args.weights)
    token_ids = parse_token_ids(args.token_ids)
    mx, _ = _require_mlx()
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("capture.py requires NumPy") from exc
    input_ids = mx.array([token_ids], dtype=mx.int32)
    attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
    model = load_model(config_path, weights_path)
    logits, _, captures = model(input_ids, attention_mask=attention_mask, capture=True)
    generated = greedy_generate(model, input_ids, args.generate_steps)
    tensors = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        **captures,
        "logits": logits,
        "generated_token_ids": generated,
    }
    getattr(mx, "eval")(*tensors.values())
    arrays = {}
    for name, value in sorted(tensors.items()):
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32, copy=False)
        arrays[name] = np.ascontiguousarray(array)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    config_record = file_record(config_path)
    weight_record = file_record(weights_path)
    identity = json.dumps(
        {"schema_version": 1, "config": config_record, "weights": [weight_record]},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    payload = {
        "schema_version": 1,
        "model": {
            "directory": config_path.parent.name or ".",
            "fingerprint": "sha256:" + hashlib.sha256(identity).hexdigest(),
            "config": config_record,
            "weights": {
                "algorithm": "sha256",
                "file_count": 1,
                "total_bytes": weight_record["size_bytes"],
                "max_listed_files": 256,
                "truncated": False,
                "omitted_file_count": 0,
                "files": [weight_record],
            },
        },
        "capture": {
            "prompts": None,
            "token_ids": token_ids,
            "generate_steps": args.generate_steps,
            "seed": 0,
            "dtype_policy": "float32",
        },
        "tensors": tensor_inventory(arrays),
        "libraries": {
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "torch": "not-used",
            "transformers": "not-used",
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "npz": str(output), "tensor_count": len(arrays)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


README_TEMPLATE = r'''# Generated MLX dense decoder

Generated by `scaffold_port.py` version __VERSION__ from trusted inspection
`sha256:__DIGEST__` for family `dense-decoder-transformer`.

This is a **starting implementation**, not a guaranteed port. It implements the
validated eager config path, but converted weights still require tensor and
end-to-end parity validation against `capture_oracle.py` before the result is
called correct.

## Generated files

- `config.json`: canonical copy of the inspected source config.
- `config.py`: validated config parser.
- `model.py`: lazy-imported eager MLX model, GQA/MHA attention, RoPE, growing KV cache, gated MLP, and greedy decode.
- `generate.py`: tokenizer-free greedy token-ID CLI.
- `capture.py`: target tensor capture in the source-oracle NPZ/manifest schema.
- `scaffold-manifest.json`: generator/config/code digests plus the target parameter contract used by conversion and package validation.

## Stable parameter-name scheme

Linear weights use MLX/Hugging Face layout `[output_dims, input_dims]`.

- `model.embed_tokens.weight` — `[vocab_size, hidden_size]`
- `model.layers.{i}.input_layernorm.weight` — `[hidden_size]`
- `model.layers.{i}.self_attn.q_proj.weight` — `[num_attention_heads * head_dim, hidden_size]`
- `model.layers.{i}.self_attn.k_proj.weight` — `[num_key_value_heads * head_dim, hidden_size]`
- `model.layers.{i}.self_attn.v_proj.weight` — `[num_key_value_heads * head_dim, hidden_size]`
- `model.layers.{i}.self_attn.o_proj.weight` — `[hidden_size, num_attention_heads * head_dim]`
- `model.layers.{i}.post_attention_layernorm.weight` — `[hidden_size]`
- `model.layers.{i}.mlp.gate_proj.weight` — `[intermediate_size, hidden_size]`
- `model.layers.{i}.mlp.up_proj.weight` — `[intermediate_size, hidden_size]`
- `model.layers.{i}.mlp.down_proj.weight` — `[hidden_size, intermediate_size]`
- `model.norm.weight` — `[hidden_size]`
- `lm_head.weight` — `[vocab_size, hidden_size]`, untied configs only

Attention projection biases are derived independently from the inspected source
tensor inventory: __ATTENTION_BIAS_SUMMARY__. When `mlp_bias` is true, append
`.bias` to each MLP projection name. A tied LM head has no separate parameter:
conversion must map the source tie owner to `model.embed_tokens.weight` and omit
`lm_head.weight`.

## Implemented assumptions

- decoder-only pre-RMSNorm blocks with separate Q/K/V/O projections;
- causal attention, optional padding mask, MHA or GQA with a growing cache;
- standard, linear, or dynamic-NTK RoPE selected from the inspected config;
- gated SiLU/SwiGLU, GELU/GeGLU, or ReLU/ReGLU MLP;
- no dropout, active sliding window, QK normalization, MoE, quantization, soft caps, or custom attention variants;
- `use_sliding_window=false` explicitly selects full attention even when inert sliding-window metadata remains in the source config.

## Run and verify

```bash
python3 generate.py --weights converted.safetensors --token-ids 1 42 17 --max-new-tokens 4
python3 mlx-model-porting/scripts/run_parity.py --source-model MODEL --package mlx_port --weights converted --token-ids 1 42 17 --output parity-report.json
```

Before relying on output, validate every converted parameter name and shape,
compare every stable capture key, check greedy continuation agreement, and prove
prefill-plus-step KV-cache logits match full-context logits at declared tolerances.
'''


def _header(digest: str) -> str:
    return f"# Generated by scaffold_port.py {GENERATOR_VERSION} from inspection sha256:{digest}."


ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def dense_attention_biases(
    inspection: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, bool]:
    """Derive a complete per-projection bias contract from inspected source keys."""
    layers = _config_int(config, "num_hidden_layers")
    tensor_keys = {
        str(tensor.get("key"))
        for tensor in inspection.get("tensors", [])
        if isinstance(tensor, dict) and isinstance(tensor.get("key"), str)
    }
    biases: dict[str, bool] = {}
    for projection in ATTENTION_PROJECTIONS:
        expected = {
            f"model.layers.{index}.self_attn.{projection}.bias"
            for index in range(layers)
        }
        present = expected & tensor_keys
        if present and present != expected:
            missing = sorted(expected - present)
            raise SkillError(
                f"inconsistent {projection} bias coverage across decoder layers; "
                f"missing {', '.join(missing[:5])}"
            )
        biases[projection] = bool(present)

    declared = config.get("attention_bias")
    has_any_bias = any(biases.values())
    if declared is False and has_any_bias:
        raise SkillError(
            "config.json attention_bias=false conflicts with inspected attention bias tensors"
        )
    if declared is True and not all(biases.values()):
        raise SkillError(
            "config.json attention_bias=true conflicts with incomplete inspected attention bias tensors"
        )
    return biases


def dense_target_tensors(
    config: dict[str, Any],
    attention_biases: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Return the deterministic parameter contract implemented by the generated graph."""
    hidden = _config_int(config, "hidden_size")
    heads = _config_int(config, "num_attention_heads")
    kv_heads = _config_int(config, "num_key_value_heads", default=heads)
    head_dim = _config_int(config, "head_dim", default=hidden // heads)
    intermediate = _config_int(config, "intermediate_size")
    layers = _config_int(config, "num_hidden_layers")
    vocab = _config_int(config, "vocab_size")
    if attention_biases is None:
        attention_biases = {
            projection: bool(config.get("attention_bias", False))
            for projection in ATTENTION_PROJECTIONS
        }
    tensors: dict[str, list[int]] = {
        "model.embed_tokens.weight": [vocab, hidden],
        "model.norm.weight": [hidden],
    }
    if not bool(config.get("tie_word_embeddings", False)):
        tensors["lm_head.weight"] = [vocab, hidden]
    for index in range(layers):
        prefix = f"model.layers.{index}"
        tensors.update({
            f"{prefix}.input_layernorm.weight": [hidden],
            f"{prefix}.post_attention_layernorm.weight": [hidden],
            f"{prefix}.self_attn.q_proj.weight": [heads * head_dim, hidden],
            f"{prefix}.self_attn.k_proj.weight": [kv_heads * head_dim, hidden],
            f"{prefix}.self_attn.v_proj.weight": [kv_heads * head_dim, hidden],
            f"{prefix}.self_attn.o_proj.weight": [hidden, heads * head_dim],
            f"{prefix}.mlp.gate_proj.weight": [intermediate, hidden],
            f"{prefix}.mlp.up_proj.weight": [intermediate, hidden],
            f"{prefix}.mlp.down_proj.weight": [hidden, intermediate],
        })
        attention_bias_shapes = {
            "q_proj": [heads * head_dim],
            "k_proj": [kv_heads * head_dim],
            "v_proj": [kv_heads * head_dim],
            "o_proj": [hidden],
        }
        for projection in ATTENTION_PROJECTIONS:
            if attention_biases.get(projection):
                tensors[f"{prefix}.self_attn.{projection}.bias"] = attention_bias_shapes[projection]
        if bool(config.get("mlp_bias", False)):
            tensors.update({
                f"{prefix}.mlp.gate_proj.bias": [intermediate],
                f"{prefix}.mlp.up_proj.bias": [intermediate],
                f"{prefix}.mlp.down_proj.bias": [hidden],
            })
    return [{"key": key, "shape": tensors[key]} for key in sorted(tensors)]


def _scaffold_manifest(
    files: dict[str, str],
    config: dict[str, Any],
    inspection_digest: str,
    attention_biases: dict[str, bool],
) -> str:
    execution_files = ("capture.py", "config.json", "config.py", "model.py")
    records = []
    for name in execution_files:
        raw = files[name].encode("utf-8")
        records.append({
            "path": name,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    payload = {
        "schema_version": 1,
        "generator": {"name": "scaffold_port.py", "version": GENERATOR_VERSION},
        "inspection_sha256": inspection_digest,
        "config_sha256": hashlib.sha256(files["config.json"].encode("utf-8")).hexdigest(),
        "files": records,
        "tensors": dense_target_tensors(config, attention_biases),
    }
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def generate_dense_decoder(
    inspection: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, str]:
    validate_dense_config(config)
    attention_biases = dense_attention_biases(inspection, config)
    digest = trusted_inspection_sha256(inspection)
    header = _header(digest)
    model_source = MODEL_TEMPLATE.replace("__HEADER__", header)
    for projection in ATTENTION_PROJECTIONS:
        placeholder = f"__{projection.upper()}_BIAS__"
        model_source = model_source.replace(
            placeholder,
            "True" if attention_biases[projection] else "False",
        )
    bias_summary = ", ".join(
        f"`{projection}`={'present' if attention_biases[projection] else 'absent'}"
        for projection in ATTENTION_PROJECTIONS
    )
    files = {
        "config.json": json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        "config.py": CONFIG_TEMPLATE.replace("__HEADER__", header),
        "model.py": model_source,
        "generate.py": GENERATE_TEMPLATE.replace("__HEADER__", header),
        "capture.py": CAPTURE_TEMPLATE.replace("__HEADER__", header),
        "README.md": (
            README_TEMPLATE
            .replace("__VERSION__", GENERATOR_VERSION)
            .replace("__DIGEST__", digest)
            .replace("__ATTENTION_BIAS_SUMMARY__", bias_summary)
        ),
    }
    files[SCAFFOLD_MANIFEST] = _scaffold_manifest(files, config, digest, attention_biases)
    return files


Generator = Callable[[dict[str, Any], dict[str, Any]], dict[str, str]]
FAMILY_GENERATORS: dict[str, Generator] = {
    DENSE_FAMILY: generate_dense_decoder,
}


def _candidate_runbook(inspection: dict[str, Any], family: str | None) -> str:
    if family:
        for candidate in inspection.get("architecture_candidates", []):
            if isinstance(candidate, dict) and candidate.get("family") == family:
                runbook = candidate.get("runbook")
                if isinstance(runbook, str) and runbook:
                    return runbook
    runbook = inspection.get("recommended_runbook")
    return str(runbook) if isinstance(runbook, str) and runbook else "manual selection required"


def _blocked_error(inspection: dict[str, Any]) -> SkillError:
    routing = inspection.get("routing_decision", {})
    family = routing.get("winner_family") if isinstance(routing, dict) else None
    blockers = [str(item) for item in inspection.get("recommendation_blockers", [])]
    return SkillError(
        "Inspection is blocked; no code was generated.\n"
        "Recommendation blockers:\n- " + "\n- ".join(blockers) + "\n"
        f"Runbook: {_candidate_runbook(inspection, str(family) if family else None)}\n"
        "Manual work:\n"
        "- resolve every inspection blocker\n"
        "- rerun inspect_model.py against the artifact\n"
        "- regenerate the scaffold only after the trusted route is unblocked"
    )


def _unsupported_route_error(inspection: dict[str, Any], family: str) -> SkillError:
    return SkillError(
        "Unsupported architecture route; no code was generated.\n"
        f"Family: {family}\n"
        f"Runbook: {_candidate_runbook(inspection, family)}\n"
        "Manual work:\n"
        f"- implement the model graph for {family}\n"
        "- define and validate a stable parameter-name mapping\n"
        "- capture target tensors with the source-oracle key scheme\n"
        "- validate end-to-end parity and state/cache behavior"
    )


def _write_new_directory(output: Path, files: dict[str, str]) -> None:
    if output.exists() or output.is_symlink():
        raise SkillError(f"output directory already exists; refusing to overwrite: {output}")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        for name in sorted(files):
            (temporary / name).write_text(files[name], encoding="utf-8")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        inspection = validate_trusted_inspection(load_structured(args.inspection))
        blockers = inspection["recommendation_blockers"]
        if blockers:
            raise _blocked_error(inspection)
        families = resolve_route_families(inspection)
        if not families:
            raise SkillError(
                "No detected family; no code was generated. Complete manual architecture review first"
            )
        verify_inspection_against_artifact(inspection, args.artifact_root)
        unsupported = [family for family in families if family not in FAMILY_GENERATORS]
        if unsupported:
            raise _unsupported_route_error(inspection, unsupported[0])
        if len(families) != 1:
            raise SkillError(
                "Hybrid architecture routes require a composition-specific generator; no code was generated"
            )
        artifact_root = Path(args.artifact_root).expanduser().resolve()
        config = load_structured(artifact_root / "config.json")
        files = FAMILY_GENERATORS[families[0]](inspection, config)
        _write_new_directory(Path(args.output).expanduser(), files)
        print(json.dumps({
            "family": families[0],
            "files": sorted(files),
            "inspection_sha256": trusted_inspection_sha256(inspection),
            "output": str(Path(args.output).expanduser()),
        }, sort_keys=True))
        return 0
    except (SkillError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
