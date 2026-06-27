#!/usr/bin/env python3
"""Deterministically (re)generate the tiny synthetic test fixtures.

Every fixture under ``tests/fixtures`` is a small, non-executable stand-in. This
script is the single source of truth for them so they are reproducible and
auditable: no committed binary blob exists without a spec here that explains it.

Usage:
    python3 tests/fixtures/generate_fixtures.py            # rewrite fixtures in place
    python3 tests/fixtures/generate_fixtures.py --dest DIR # write a copy elsewhere

Determinism: tensor values come from a SHA-256-seeded numpy RNG keyed by
``model/key`` so repeated runs are byte-identical for the raw ``.safetensors``
files (``.npz`` archives match by array content; their zip container embeds
timestamps and is therefore compared by value, not bytes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

FIXTURES_ROOT = Path(__file__).resolve().parent

DTYPE_TO_NUMPY = {"F16": np.float16, "F32": np.float32}
ZIP_DATE = (2026, 1, 1, 0, 0, 0)

# --- Declarative model specs -------------------------------------------------
# Shapes are deliberately tiny. The codec codebook is a size-reduced stand-in
# (the real EnCodec codebook would be [num_codebooks, codebook_size, dim]); a
# 2 MB blob is not needed to exercise name-based architecture routing.
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "decoder": {
        "config": {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
            "torch_dtype": "float16",
            "license": "apache-2.0",
        },
        "tensors": {
            "model.embed_tokens.weight": ("F16", [32, 8]),
            "model.layers.0.self_attn.q_proj.weight": ("F16", [8, 8]),
            "model.layers.0.mlp.gate_proj.weight": ("F16", [16, 8]),
            "lm_head.weight": ("F16", [32, 8]),
        },
    },
    "moe": {
        "config": {
            "model_type": "mixtral",
            "architectures": ["MixtralForCausalLM"],
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "vocab_size": 32,
            "license": "apache-2.0",
        },
        "tensors": {
            "model.layers.0.block_sparse_moe.gate.weight": ("F16", [4, 8]),
            "model.layers.0.block_sparse_moe.experts.0.w1.weight": ("F16", [16, 8]),
            "model.layers.0.self_attn.q_proj.weight": ("F16", [8, 8]),
        },
    },
    "codec": {
        "config": {
            "model_type": "encodec",
            "architectures": ["EncodecModel"],
            "sample_rate": 24000,
            "codebook_size": 1024,
            "num_codebooks": 8,
            "hop_length": 320,
            "audio_channels": 1,
            "license": "mit",
        },
        "tensors": {
            "encoder.layers.0.conv.weight": ("F16", [32, 1, 7]),
            # Size-reduced stand-in for the residual-VQ codebook (was [8, 1024, 128]).
            "quantizer.codebook.weight": ("F16", [8, 32, 16]),
            "decoder.layers.0.conv.weight": ("F16", [1, 32, 7]),
        },
    },
    "encoder": {
        "config": {
            "model_type": "bert",
            "architectures": ["BertModel"],
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 32,
            "license": "apache-2.0",
        },
        "tensors": {
            "embeddings.word_embeddings.weight": ("F16", [32, 8]),
            "encoder.layer.0.attention.self.query.weight": ("F16", [8, 8]),
            "pooler.dense.weight": ("F16", [8, 8]),
        },
    },
    "non_generative_cv_backbone": {
        "config": {
            "model_type": "resnet",
            "architectures": ["ResNetForImageClassification"],
            "image_size": 32,
            "num_channels": 3,
            "num_labels": 10,
            "id2label": {str(i): f"class_{i}" for i in range(10)},
            "label2id": {f"class_{i}": i for i in range(10)},
            "torch_dtype": "float16",
            "license": "apache-2.0",
        },
        "tensors": {
            "conv1.weight": ("F16", [8, 3, 3, 3]),
            "stages.0.blocks.0.conv_dw.weight": ("F16", [8, 1, 3, 3]),
            "stages.0.blocks.0.norm.weight": ("F16", [8]),
            "features.0.weight": ("F16", [8, 8, 1, 1]),
            "classifier.weight": ("F16", [10, 8]),
        },
    },
    "graph_message_passing": {
        "config": {
            "model_type": "gcn",
            "architectures": ["GCNForNodeClassification"],
            "num_node_features": 6,
            "num_edge_features": 2,
            "hidden_channels": 8,
            "num_layers": 2,
            "num_labels": 3,
            "aggregation": "add",
            "add_self_loops": True,
            "normalize_adjacency": True,
            "torch_dtype": "float16",
            "license": "apache-2.0",
        },
        "tensors": {
            "node_encoder.weight": ("F16", [8, 6]),
            "edge_encoder.weight": ("F16", [8, 2]),
            "convs.0.lin.weight": ("F16", [8, 8]),
            "convs.1.lin.weight": ("F16", [8, 8]),
            "classifier.weight": ("F16", [3, 8]),
        },
    },
    "encoder_decoder": {
        "config": {
            "model_type": "t5",
            "architectures": ["T5ForConditionalGeneration"],
            "encoder_layers": 2,
            "decoder_layers": 2,
            "is_encoder_decoder": True,
            "d_model": 8,
            "num_heads": 2,
            "vocab_size": 32,
            "license": "apache-2.0",
        },
        "tensors": {
            "encoder.layers.0.self_attn.q.weight": ("F16", [8, 8]),
            "decoder.layers.0.self_attn.q.weight": ("F16", [8, 8]),
            "decoder.layers.0.encoder_attn.k.weight": ("F16", [8, 8]),
        },
    },
    "ssm_hybrid": {
        "config": {
            "model_type": "mamba",
            "architectures": ["MambaForCausalLM"],
            "hidden_size": 8,
            "state_size": 4,
            "conv_kernel": 4,
            "time_step_rank": 2,
            "vocab_size": 32,
            "license": "apache-2.0",
        },
        "tensors": {
            "backbone.layers.0.mixer.A_log": ("F32", [4]),
            "backbone.layers.0.mixer.dt_proj.weight": ("F16", [8, 2]),
            "backbone.layers.0.mixer.x_proj.weight": ("F16", [16, 8]),
            "backbone.layers.0.mixer.conv1d.weight": ("F16", [8, 1, 4]),
            "lm_head.weight": ("F16", [32, 8]),
        },
    },
    "diffusion_flow": {
        "config": {
            "model_type": "flux",
            "architectures": ["FluxTransformer2DModel"],
            "in_channels": 4,
            "sample_size": 8,
            "num_train_timesteps": 1000,
            "hidden_size": 8,
            "license": "apache-2.0",
        },
        "tensors": {
            "time_embedding.linear_1.weight": ("F16", [8, 8]),
            "transformer_blocks.0.attn.to_q.weight": ("F16", [8, 8]),
            "down_blocks.0.resnets.0.conv1.weight": ("F16", [8, 4, 3, 3]),
            "up_blocks.0.resnets.0.conv1.weight": ("F16", [4, 8, 3, 3]),
        },
    },
    "vision_language_omni": {
        "config": {
            "model_type": "llava",
            "architectures": ["LlavaForConditionalGeneration"],
            "vision_config": {"hidden_size": 8, "num_hidden_layers": 1},
            "projector_hidden_act": "gelu",
            "image_token_id": 32000,
            "vocab_size": 32,
            "license": "apache-2.0",
        },
        "tensors": {
            "vision_tower.encoder.layers.0.self_attn.q_proj.weight": ("F16", [8, 8]),
            "multi_modal_projector.linear_1.weight": ("F16", [8, 8]),
            "language_model.model.layers.0.self_attn.q_proj.weight": ("F16", [8, 8]),
        },
    },
    "autoregressive_audio_lm": {
        "config": {
            "model_type": "bark",
            "architectures": ["BarkModel"],
            "audio_vocab_size": 64,
            "num_codebooks": 8,
            "codec_config": {"sample_rate": 24000},
            "speaker_embedding_dim": 8,
            "license": "mit",
        },
        "tensors": {
            "semantic.talker.layers.0.self_attn.q_proj.weight": ("F16", [8, 8]),
            "audio_head.weight": ("F16", [64, 8]),
            "speech_embedding.weight": ("F16", [64, 8]),
            "codec.quantizer.weight": ("F16", [8, 8]),
        },
    },
    "flow_diffusion_tts": {
        "config": {
            "model_type": "f5_tts",
            "architectures": ["F5TTSModel"],
            "mel_dim": 80,
            "n_mel_channels": 80,
            "ode_method": "euler",
            "nfe_step": 16,
            "license": "mit",
        },
        "tensors": {
            "transformer.layers.0.attn.q_proj.weight": ("F16", [8, 8]),
            "duration_predictor.proj.weight": ("F16", [8, 8]),
            "mel_projection.weight": ("F16", [80, 8]),
            "flow.estimator.weight": ("F16", [8, 8]),
        },
    },
    "vocoder": {
        "config": {
            "model_type": "hifigan",
            "architectures": ["HiFiGANGenerator"],
            "upsample_rates": [8, 8, 2, 2],
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "n_fft": 1024,
            "hop_length": 256,
            "license": "mit",
        },
        "tensors": {
            "ups.0.weight": ("F16", [8, 8, 4]),
            "resblocks.0.convs1.0.weight": ("F16", [8, 8, 3]),
            "conv_post.weight": ("F16", [1, 8, 7]),
            "istft.window": ("F32", [16]),
        },
    },
    "asr": {
        "config": {
            "model_type": "wav2vec2",
            "architectures": ["Wav2Vec2ForCTC"],
            "sampling_rate": 16000,
            "feature_size": 80,
            "ctc_loss_reduction": "mean",
            "num_mel_bins": 80,
            "license": "apache-2.0",
        },
        "tensors": {
            "feature_extractor.conv_layers.0.conv.weight": ("F16", [8, 1, 7]),
            "encoder.layers.0.attention.q_proj.weight": ("F16", [8, 8]),
            "ctc.weight": ("F16", [32, 8]),
            "joiner.linear.weight": ("F16", [8, 8]),
            "predictor.embed.weight": ("F16", [32, 8]),
        },
    },
    "streaming_speech": {
        "config": {
            "model_type": "rnnt",
            "architectures": ["StreamingConformerRNNT"],
            "chunk_size": 16,
            "left_context": 4,
            "lookahead": 2,
            "streaming": True,
            "license": "apache-2.0",
        },
        "tensors": {
            "streaming.encoder.layers.0.weight": ("F16", [8, 8]),
            "state_cache.proj.weight": ("F16", [8, 8]),
            "predictor.embed.weight": ("F16", [32, 8]),
            "codec.decoder.weight": ("F16", [8, 8]),
        },
    },
    "separation_enhancement": {
        "config": {
            "model_type": "demucs",
            "architectures": ["DemucsModel"],
            "n_fft": 1024,
            "hop_length": 256,
            "num_stems": 4,
            "chunk_size": 4096,
            "license": "mit",
        },
        "tensors": {
            "mask.estimator.weight": ("F16", [8, 8]),
            "stft.proj.weight": ("F16", [8, 8]),
            "encoder.layers.0.conv.weight": ("F16", [8, 1, 7]),
            "decoder.layers.0.conv.weight": ("F16", [1, 8, 7]),
        },
    },
}

# Unsafe fixture: a non-executable lure that must trip the remote-code/pickle gates.
UNSAFE_FILES: dict[str, bytes] = {
    "config.json": json.dumps(
        {
            "model_type": "mystery",
            "architectures": ["MysteryModel"],
            "auto_map": {"AutoModel": "modeling_mystery.MysteryModel"},
        },
        indent=2,
    ).encode()
    + b"\n",
    "modeling_mystery.py": b"class MysteryModel: pass\n",
    "setup.py": b"from setuptools import setup\n",
    "pytorch_model.bin": b"not-a-real-pickle",
}

# Tensor-oracle .npz fixtures used by compare_tensors tests.
NPZ_SHAPES = {"hidden": (3, 4), "logits": (2,)}


def _seeded(model: str, key: str, shape: list[int] | tuple[int, ...]) -> np.ndarray:
    digest = hashlib.sha256(f"{model}/{key}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed).standard_normal(tuple(shape))


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]], model: str) -> None:
    header: dict[str, Any] = {}
    buffer = bytearray()
    for key, (dtype, shape) in tensors.items():
        array = _seeded(model, key, shape).astype(DTYPE_TO_NUMPY[dtype])
        raw = array.tobytes()
        header[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [len(buffer), len(buffer) + len(raw)]}
        buffer.extend(raw)
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(bytes(buffer))


def write_npz_fixtures(root: Path) -> None:
    source = {name: _seeded("tensors-source", name, shape).astype(np.float32) for name, shape in NPZ_SHAPES.items()}
    close = {name: array.copy() for name, array in source.items()}
    close["hidden"] = (close["hidden"] + np.float32(1e-7)).astype(np.float32)  # within atol/rtol
    bad = {name: array.copy() for name, array in source.items()}
    bad["hidden"].flat[0] += np.float32(11.0)  # large divergence
    bad["logits"][0] += np.float32(0.7)
    out = root / "tensors"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in {"source": source, "close": close, "bad": bad}.items():
        np.savez(out / f"{name}.npz", **data)


def write_text(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def write_bytes(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def write_keras_archive(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"keras_version": "3.0.0", "date_saved": "2026-01-01@00:00:00"}
    config = {
        "module": "keras",
        "class_name": "Functional",
        "config": {
            "name": "tiny_keras",
            "layers": [
                {"class_name": "InputLayer", "config": {"name": "tokens"}},
                {"class_name": "Dense", "config": {"name": "projection", "units": 4}},
                {"class_name": "Lambda", "config": {"name": "custom_postprocess"}},
            ],
        },
    }
    entries = {
        "metadata.json": json.dumps(metadata, sort_keys=True).encode(),
        "config.json": json.dumps(config, sort_keys=True).encode(),
        "model.weights.h5": b"not-a-real-hdf5-weight-file",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, ZIP_DATE)
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, data)
    return path


def write_source_format_fixtures(root: Path) -> list[Path]:
    written: list[Path] = []

    flax_root = root / "source_formats" / "flax_orbax"
    written.append(write_bytes(flax_root / "flax_model.msgpack", b"\x82\xa6params\x80\xa5state\x80"))
    written.append(write_text(flax_root / "tree_metadata.json", json.dumps({
        "framework": "flax",
        "tree_paths": ["params.encoder.layer_0.kernel", "params.lm_head.kernel"],
        "shapes": {
            "params.encoder.layer_0.kernel": [8, 8],
            "params.lm_head.kernel": [8, 32],
        },
    }, indent=2) + "\n"))
    written.append(write_text(flax_root / "orbax_checkpoint" / "_CHECKPOINT_METADATA", json.dumps({
        "checkpoint_type": "orbax",
        "tree_paths": ["params.encoder.layer_0.kernel", "params.lm_head.kernel"],
        "step": 7,
    }, indent=2) + "\n"))

    saved_model_root = root / "source_formats" / "tensorflow_saved_model"
    written.append(write_text(saved_model_root / "saved_model.pbtxt", """
meta_graphs {
  graph_def {
    node { name: "matmul" op: "MatMul" }
    node { name: "nms" op: "NonMaxSuppressionV5" }
  }
  signature_def {
    key: "serving_default"
    value {
      inputs {
        key: "input_ids"
        value { name: "serving_default_input_ids:0" dtype: DT_FLOAT }
      }
      outputs {
        key: "logits"
        value { name: "StatefulPartitionedCall:0" dtype: DT_FLOAT }
      }
      method_name: "tensorflow/serving/predict"
    }
  }
}
""".lstrip()))
    written.append(write_text(saved_model_root / "variables" / "variables.index", "not-a-real-tf-index\n"))
    written.append(write_bytes(saved_model_root / "variables" / "variables.data-00000-of-00001", b"tf-variables"))

    written.append(write_keras_archive(root / "source_formats" / "keras_archive" / "tiny.keras"))

    safetensors_only_root = root / "source_formats" / "safetensors_only"
    safetensors_only_tensors = {
        "model.layers.0.self_attn.q_proj.weight": ("F16", [8, 8]),
        "model.layers.0.mlp.down_proj.weight": ("F16", [8, 16]),
        "lm_head.weight": ("F16", [32, 8]),
    }
    st_path = safetensors_only_root / "model.safetensors"
    write_safetensors(st_path, safetensors_only_tensors, "safetensors-only")
    written.append(st_path)

    coreml_root = root / "source_formats" / "coreml_package" / "Tiny.mlpackage"
    written.append(write_text(coreml_root / "Manifest.json", json.dumps({
        "fileFormatVersion": "1.0.0",
        "rootModelIdentifier": "model",
        "itemInfoEntries": {
            "model": {"path": "Data/com.apple.CoreML/model.mlmodel"},
            "weights": {"path": "Data/com.apple.CoreML/weights/weight.bin"},
        },
    }, indent=2) + "\n"))
    written.append(write_bytes(coreml_root / "Data" / "com.apple.CoreML" / "model.mlmodel", b"not-a-real-coreml-protobuf"))
    written.append(write_bytes(coreml_root / "Data" / "com.apple.CoreML" / "weights" / "weight.bin", b"coreml-weights"))
    return written


def generate(root: Path) -> list[Path]:
    written: list[Path] = []
    for model, spec in MODEL_SPECS.items():
        model_dir = root / "models" / model
        model_dir.mkdir(parents=True, exist_ok=True)
        config_path = model_dir / "config.json"
        config_path.write_text(json.dumps(spec["config"], indent=2) + "\n", encoding="utf-8")
        st_path = model_dir / "model.safetensors"
        write_safetensors(st_path, spec["tensors"], model)
        written += [config_path, st_path]
    unsafe_dir = root / "models" / "unsafe"
    unsafe_dir.mkdir(parents=True, exist_ok=True)
    for name, data in UNSAFE_FILES.items():
        (unsafe_dir / name).write_bytes(data)
        written.append(unsafe_dir / name)
    write_npz_fixtures(root)
    written += [root / "tensors" / f"{n}.npz" for n in ("source", "close", "bad")]
    written += write_source_format_fixtures(root)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate synthetic test fixtures")
    parser.add_argument("--dest", default=str(FIXTURES_ROOT), help="Destination fixtures root")
    args = parser.parse_args()
    written = generate(Path(args.dest))
    print(f"wrote {len(written)} fixture files under {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
