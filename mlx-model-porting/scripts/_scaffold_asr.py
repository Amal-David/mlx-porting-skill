"""ASR acoustic-encoder templates used by ``scaffold_port.py``.

The generated graph intentionally starts at convolutional frontend features.
It implements the HuBERT/Wav2Vec2 feature projection and transformer encoder,
including the weight-normalized convolutional positional embedding.  Raw
waveform feature extraction and decoding remain outside this scaffold.
"""
from __future__ import annotations

import json
from typing import Any

from _common import SkillError


ASR_FAMILY = "automatic-speech-recognition"
ASR_RUNBOOK = "references/runbook-asr.md"

ASR_CONFIG_FEATURE_ALLOWLIST = frozenset({
    "conv_dim",
    "conv_kernel",
    "conv_stride",
    "do_stable_layer_norm",
    "feat_extract_activation",
    "hidden_act",
    "hidden_size",
    "intermediate_size",
    "layer_norm_eps",
    "num_attention_heads",
    "num_conv_pos_embedding_groups",
    "num_conv_pos_embeddings",
    "num_hidden_layers",
})

ASR_INERT_CONFIG_KEYS = frozenset({
    "_commit_hash", "_name_or_path", "activation_dropout", "adapter_attn_dim",
    "add_adapter", "apply_spec_augment", "architectures", "attention_dropout",
    "bos_token_id", "classifier_proj_size", "conv_bias", "conv_pos_batch_norm",
    "ctc_loss_reduction", "ctc_loss_zero_infinity", "ctc_zero_infinity", "dtype",
    "eos_token_id", "feat_extract_dropout",
    "feat_extract_norm", "feat_proj_dropout", "feat_proj_layer_norm", "final_dropout",
    "gradient_checkpointing", "hidden_dropout", "hidden_dropout_prob", "id2label",
    "initializer_range", "is_decoder", "is_encoder_decoder", "label2id", "layerdrop", "license",
    "mask_feature_length", "mask_feature_min_masks", "mask_feature_prob",
    "mask_time_length", "mask_time_min_masks", "mask_time_prob", "model_type",
    "num_adapter_layers", "num_feat_extract_layers", "output_attentions",
    "output_hidden_states", "pad_token_id", "problem_type", "return_dict",
    "tokenizer_class", "torch_dtype", "transformers_version", "use_bfloat16",
    "vocab_size",
})


def _positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise SkillError(f"config.json {key} must be a positive integer")
    return value


def _positive_float(config: dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SkillError(f"config.json {key} must be a positive number")
    return float(value)


def _positive_int_list(config: dict[str, Any], key: str) -> list[int]:
    value = config.get(key)
    if not isinstance(value, list) or not value or any(type(item) is not int or item <= 0 for item in value):
        raise SkillError(f"config.json {key} must be a non-empty array of positive integers")
    return list(value)


def validate_asr_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise SkillError("config.json must contain an object")
    model_type = str(config.get("model_type", "")).lower()
    if model_type not in {"hubert", "wav2vec2"}:
        if model_type == "whisper" or bool(config.get("is_encoder_decoder")):
            raise SkillError(
                "Whisper/sequence-to-sequence ASR is out of scope for the acoustic-encoder "
                f"scaffold; consult {ASR_RUNBOOK}"
            )
        raise SkillError(
            "ASR scaffolding currently supports only model_type 'hubert' or 'wav2vec2'; "
            "CTC, transducer, Conformer, Moonshine, and other decoding graphs are out of scope"
        )
    architectures = config.get("architectures", [])
    if architectures is not None and (
        not isinstance(architectures, list)
        or any(not isinstance(name, str) or not name for name in architectures)
    ):
        raise SkillError("config.json architectures must be an array of non-empty strings")
    if any(name.endswith("ForCTC") for name in architectures or []):
        raise SkillError(
            "ASR acoustic-encoder scaffolding rejects *ForCTC checkpoints; provide an "
            "encoder-only HubertModel or Wav2Vec2Model checkpoint"
        )
    hidden = _positive_int(config, "hidden_size")
    heads = _positive_int(config, "num_attention_heads")
    _positive_int(config, "num_hidden_layers")
    _positive_int(config, "intermediate_size")
    groups = _positive_int(config, "num_conv_pos_embedding_groups")
    _positive_int(config, "num_conv_pos_embeddings")
    _positive_float(config, "layer_norm_eps", 1e-5)
    conv_dim = _positive_int_list(config, "conv_dim")
    conv_kernel = _positive_int_list(config, "conv_kernel")
    conv_stride = _positive_int_list(config, "conv_stride")
    if len(conv_dim) != len(conv_kernel) or len(conv_dim) != len(conv_stride):
        raise SkillError("config.json conv_dim, conv_kernel, and conv_stride must have equal lengths")
    if hidden % heads:
        raise SkillError("config.json hidden_size must be divisible by num_attention_heads")
    if hidden % groups:
        raise SkillError("config.json hidden_size must be divisible by num_conv_pos_embedding_groups")
    activation = config.get("hidden_act", "gelu")
    if not isinstance(activation, str) or activation.lower() != "gelu":
        raise SkillError("ASR acoustic-encoder scaffolding currently requires hidden_act='gelu'")
    positional_activation = config.get("feat_extract_activation", "gelu")
    if not isinstance(positional_activation, str) or positional_activation.lower() != "gelu":
        raise SkillError(
            "ASR acoustic-encoder scaffolding currently requires "
            "feat_extract_activation='gelu' for positional convolution"
        )
    for key in ("do_stable_layer_norm", "feat_proj_layer_norm", "conv_pos_batch_norm"):
        if key in config and not isinstance(config[key], bool):
            raise SkillError(f"config.json {key} must be boolean when present")
    if config.get("feat_proj_layer_norm", True) is not True:
        raise SkillError("ASR acoustic-encoder scaffolding requires feature projection LayerNorm")
    if config.get("conv_pos_batch_norm", False):
        raise SkillError("conv_pos_batch_norm=True is not supported")
    if config.get("adapter_attn_dim") is not None or bool(config.get("add_adapter", False)):
        raise SkillError("ASR encoder adapters are not supported")
    classified = ASR_CONFIG_FEATURE_ALLOWLIST | ASR_INERT_CONFIG_KEYS
    unknown = sorted(set(config) - classified)
    if unknown:
        raise SkillError(
            "Unsupported ASR config features; no code was generated. "
            f"Consult {ASR_RUNBOOK}:\n- " + "\n- ".join(
                f"unrecognized computation-relevant config key {key!r}" for key in unknown
            )
        )
    return config


CONFIG_TEMPLATE = r'''__HEADER__
"""Validated configuration for a generated HuBERT/Wav2Vec2 encoder."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    layer_norm_eps: float
    hidden_act: str
    conv_dim: tuple[int, ...]
    conv_kernel: tuple[int, ...]
    conv_stride: tuple[int, ...]
    num_conv_pos_embeddings: int
    num_conv_pos_embedding_groups: int
    do_stable_layer_norm: bool

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            hidden_size=int(data["hidden_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            intermediate_size=int(data["intermediate_size"]),
            layer_norm_eps=float(data.get("layer_norm_eps", 1e-5)),
            hidden_act=str(data.get("hidden_act", "gelu")).lower(),
            conv_dim=tuple(int(value) for value in data["conv_dim"]),
            conv_kernel=tuple(int(value) for value in data["conv_kernel"]),
            conv_stride=tuple(int(value) for value in data["conv_stride"]),
            num_conv_pos_embeddings=int(data["num_conv_pos_embeddings"]),
            num_conv_pos_embedding_groups=int(data["num_conv_pos_embedding_groups"]),
            do_stable_layer_norm=bool(data.get("do_stable_layer_norm", False)),
        )
'''


MODEL_TEMPLATE = r'''__HEADER__
"""Eager MLX HuBERT/Wav2Vec2 transformer encoder over extracted features."""
from __future__ import annotations

from pathlib import Path

from config import ModelConfig


def _require_mlx():
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as exc:
        raise RuntimeError("generated model requires MLX: python3 -m pip install mlx") from exc
    return mx, nn


def build_model(config: ModelConfig):
    mx, nn = _require_mlx()
    head_dim = config.hidden_size // config.num_attention_heads

    class FeatureProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
            self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size, bias=True)

        def __call__(self, features):
            return self.projection(self.layer_norm(features))

    class WeightNormConv(nn.Module):
        def __init__(self):
            super().__init__()
            kernel = config.num_conv_pos_embeddings
            channels_per_group = config.hidden_size // config.num_conv_pos_embedding_groups
            self.weight_g = mx.zeros((1, 1, kernel), dtype=mx.float32)
            self.weight_v = mx.zeros(
                (config.hidden_size, channels_per_group, kernel), dtype=mx.float32
            )
            self.bias = mx.zeros((config.hidden_size,), dtype=mx.float32)

        def __call__(self, hidden_states):
            norm = mx.sqrt(mx.sum(self.weight_v * self.weight_v, axis=(0, 1), keepdims=True))
            source_weight = self.weight_g * self.weight_v / mx.maximum(norm, 1e-12)
            kernel = source_weight.transpose(0, 2, 1)
            value = mx.conv1d(
                hidden_states,
                kernel,
                stride=1,
                padding=config.num_conv_pos_embeddings // 2,
                groups=config.num_conv_pos_embedding_groups,
            )
            value = value + self.bias
            if config.num_conv_pos_embeddings % 2 == 0:
                value = value[:, :-1, :]
            return nn.gelu(value)

    class PositionalConvEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = WeightNormConv()

        def __call__(self, hidden_states):
            return self.conv(hidden_states)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        def __call__(self, hidden_states, attention_mask=None):
            batch, length, _ = hidden_states.shape
            shape = (batch, length, config.num_attention_heads, head_dim)
            q = self.q_proj(hidden_states).reshape(shape).transpose(0, 2, 1, 3)
            k = self.k_proj(hidden_states).reshape(shape).transpose(0, 2, 1, 3)
            v = self.v_proj(hidden_states).reshape(shape).transpose(0, 2, 1, 3)
            scores = (q @ k.transpose(0, 1, 3, 2)) * (head_dim ** -0.5)
            scores = scores.astype(mx.float32)
            if attention_mask is not None:
                key_mask = attention_mask[:, None, None, :].astype(mx.bool_)
                scores = mx.where(key_mask, scores, mx.array(-1e9, dtype=mx.float32))
            probabilities = mx.softmax(scores, axis=-1)
            attended = probabilities @ v
            attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, config.hidden_size)
            return self.out_proj(attended)

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.intermediate_dense = nn.Linear(
                config.hidden_size, config.intermediate_size, bias=True
            )
            self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

        def __call__(self, hidden_states):
            return self.output_dense(nn.gelu(self.intermediate_dense(hidden_states)))

    class EncoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.feed_forward = FeedForward()
            self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        def __call__(self, hidden_states, attention_mask=None):
            if config.do_stable_layer_norm:
                residual = hidden_states
                hidden_states = residual + self.attention(
                    self.layer_norm(hidden_states), attention_mask
                )
                return hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))
            hidden_states = hidden_states + self.attention(hidden_states, attention_mask)
            hidden_states = self.layer_norm(hidden_states)
            hidden_states = hidden_states + self.feed_forward(hidden_states)
            return self.final_layer_norm(hidden_states)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos_conv_embed = PositionalConvEmbedding()
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.layers = [EncoderLayer() for _ in range(config.num_hidden_layers)]

        def __call__(self, hidden_states, attention_mask=None, capture=False):
            if attention_mask is not None:
                hidden_states = mx.where(
                    attention_mask[:, :, None].astype(mx.bool_),
                    hidden_states,
                    mx.zeros_like(hidden_states),
                )
            hidden_states = hidden_states + self.pos_conv_embed(hidden_states)
            if not config.do_stable_layer_norm:
                hidden_states = self.layer_norm(hidden_states)
            captures = {"embed": hidden_states} if capture else None
            for index, layer in enumerate(self.layers):
                hidden_states = layer(hidden_states, attention_mask)
                if capture:
                    captures[f"layer.{index}.hidden"] = hidden_states
            if config.do_stable_layer_norm:
                hidden_states = self.layer_norm(hidden_states)
            if capture:
                captures["final_hidden"] = hidden_states
            return hidden_states, captures

    class AcousticEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_projection = FeatureProjection()
            self.encoder = Encoder()

        def __call__(self, input_features, attention_mask=None, capture=False):
            projected = self.feature_projection(input_features)
            return self.encoder(projected, attention_mask=attention_mask, capture=capture)

    return AcousticEncoder()


def load_model(config_path: str | Path, weights_path: str | Path):
    config = ModelConfig.from_file(config_path)
    model = build_model(config)
    model.load_weights(str(weights_path), strict=True)
    getattr(model, "eval")()
    return model
'''


README_TEMPLATE = r'''# Generated MLX HuBERT/Wav2Vec2 acoustic encoder

Generated by `scaffold_port.py` version __VERSION__ from trusted inspection
`sha256:__DIGEST__` for family `automatic-speech-recognition`.

This additive scaffold implements the inference-time transformer encoder core:

- extracted convolutional features (`[batch, frames, conv_dim[-1]]`) as input;
- feature LayerNorm and projection;
- weight-normalized grouped convolutional positional embedding;
- bidirectional self-attention and GELU feed-forward encoder layers;
- post-norm and stable-layer-norm variants selected by config.

The raw-waveform convolutional feature extractor is deliberately not generated.
Use `capture_oracle.py --asr-encoder` to freeze real extracted features and
`run_parity.py --asr-encoder` to feed that exact tensor to both source and MLX
encoder graphs. Whisper sequence-to-sequence decoding, CTC heads/decoding,
transducers, tokenization, WER, and timestamps fail closed as out of scope.
'''

CAPTURE_TEMPLATE = r'''__HEADER__
"""Capture entrypoint marker; use the validated top-level capture_mlx.py tool."""
'''


def asr_target_tensors(config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_asr_config(config)
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    layers = int(config["num_hidden_layers"])
    feature_dim = int(config["conv_dim"][-1])
    kernel = int(config["num_conv_pos_embeddings"])
    groups = int(config["num_conv_pos_embedding_groups"])
    tensors: dict[str, list[int]] = {
        "feature_projection.layer_norm.bias": [feature_dim],
        "feature_projection.layer_norm.weight": [feature_dim],
        "feature_projection.projection.bias": [hidden],
        "feature_projection.projection.weight": [hidden, feature_dim],
        "encoder.layer_norm.bias": [hidden],
        "encoder.layer_norm.weight": [hidden],
        "encoder.pos_conv_embed.conv.bias": [hidden],
        "encoder.pos_conv_embed.conv.weight_g": [1, 1, kernel],
        "encoder.pos_conv_embed.conv.weight_v": [hidden, hidden // groups, kernel],
    }
    for index in range(layers):
        prefix = f"encoder.layers.{index}"
        for projection in ("k_proj", "q_proj", "v_proj", "out_proj"):
            tensors[f"{prefix}.attention.{projection}.weight"] = [hidden, hidden]
            tensors[f"{prefix}.attention.{projection}.bias"] = [hidden]
        tensors[f"{prefix}.feed_forward.intermediate_dense.weight"] = [intermediate, hidden]
        tensors[f"{prefix}.feed_forward.intermediate_dense.bias"] = [intermediate]
        tensors[f"{prefix}.feed_forward.output_dense.weight"] = [hidden, intermediate]
        tensors[f"{prefix}.feed_forward.output_dense.bias"] = [hidden]
        for norm in ("layer_norm", "final_layer_norm"):
            tensors[f"{prefix}.{norm}.weight"] = [hidden]
            tensors[f"{prefix}.{norm}.bias"] = [hidden]
    return [{"key": key, "shape": tensors[key]} for key in sorted(tensors)]


def generate_asr_encoder(
    config: dict[str, Any],
    *,
    header: str,
    digest: str,
    version: str,
) -> dict[str, str]:
    validate_asr_config(config)
    return {
        "config.json": json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        "config.py": CONFIG_TEMPLATE.replace("__HEADER__", header),
        "model.py": MODEL_TEMPLATE.replace("__HEADER__", header),
        "capture.py": CAPTURE_TEMPLATE.replace("__HEADER__", header),
        "README.md": README_TEMPLATE.replace("__VERSION__", version).replace("__DIGEST__", digest),
    }
