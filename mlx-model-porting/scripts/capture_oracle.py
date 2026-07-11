#!/usr/bin/env python3
"""Capture deterministic causal-decoder or encoder-decoder source tensors.

The stable NPZ tensor-key scheme is:

``input_ids``
    Token IDs supplied directly or produced by the local tokenizer.
``attention_mask``
    The corresponding integer attention mask.
``embed``
    Decoder embedding-stage hidden state before the first decoder block.
``layer.{i}.hidden``
    Post-block hidden state for zero-based decoder layer ``i``.
``layer.{i}.attention``
    Optional attention-module branch output when a standard submodule is found.
``layer.{i}.mlp``
    Optional MLP branch output before the block residual when hookable.
``final_norm``
    Output of the decoder's final normalization module.
``logits``
    Full prompt logits.
``generated_token_ids``
    Exactly ``--generate-steps`` greedy continuation IDs (not the prompt IDs).

PyTorch, Transformers, and NumPy are execution-only dependencies. Importing
this module, using ``--help``, and validating a manifest do not import them.
The execution path is local-only and never enables Hugging Face remote code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from _capture_common import (
    DEFAULT_GENERATE_STEPS,
    DEFAULT_MAX_OUTPUT_MB,
    MAX_MANIFEST_JSON_BYTES,
    MAX_TOKEN_IDS,
    SCHEMA_VERSION,
    absolute_lexical,
    collect_prompts as collect_shared_prompts,
    parse_token_ids as parse_shared_token_ids,
    prepare_output_parent as prepare_shared_output_parent,
    raw_npz_bound,
    read_bounded_json as read_shared_bounded_json,
    read_bounded_text as read_shared_bounded_text,
    tensor_inventory,
    validate_capture_limits,
    validate_capture_manifest,
    validate_input_mode,
    validate_output_path as validate_shared_output_path,
    write_capture_outputs,
)
from _common import SkillError, sha256_file


MAX_CONFIG_JSON_BYTES = 16 * 1024 * 1024
MAX_TOKENIZER_JSON_BYTES = 128 * 1024 * 1024
MAX_MODEL_ROOT_ENTRIES = 10_000
MAX_WEIGHT_FILES = 4_096
MAX_LISTED_WEIGHT_FILES = 256

TOKENIZER_JSON_NAMES = frozenset({
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture deterministic intermediate tensors and greedy tokens from a local "
            "PyTorch/Hugging Face causal language model"
        ),
    )
    parser.add_argument("model", nargs="?", help="Local model directory; hub IDs are not accepted")
    parser.add_argument("--output", help="Destination .npz archive")
    parser.add_argument(
        "--manifest",
        help="Destination JSON manifest (default: <output stem>.manifest.json)",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="Prompt to tokenize locally; repeat for a prompt batch",
    )
    parser.add_argument(
        "--prompts-file",
        action="append",
        help="Bounded UTF-8 file containing one prompt per non-empty line; repeatable",
    )
    parser.add_argument(
        "--token-ids",
        nargs="+",
        help="One tokenizer-free token sequence as space- or comma-separated integers",
    )
    parser.add_argument(
        "--generate-steps",
        type=int,
        default=DEFAULT_GENERATE_STEPS,
        help=f"Exact greedy continuation length (default: {DEFAULT_GENERATE_STEPS})",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic execution seed")
    parser.add_argument(
        "--keep-dtype",
        action="store_true",
        help="Keep representable source floating dtypes instead of casting floats to float32",
    )
    parser.add_argument(
        "--max-output-mb",
        type=float,
        default=DEFAULT_MAX_OUTPUT_MB,
        help=f"Maximum raw and serialized NPZ size in MiB (default: {DEFAULT_MAX_OUTPUT_MB:g})",
    )
    parser.add_argument(
        "--validate-manifest",
        help="Validate one oracle manifest without importing execution dependencies",
    )
    return parser.parse_args(argv)


def read_bounded_json(path: Path, limit: int, *, label: str) -> Any:
    return read_shared_bounded_json(path, limit, label=label)


def read_bounded_text(path: Path, limit: int, *, label: str) -> str:
    return read_shared_bounded_text(path, limit, label=label)


def _portable_name(path: Path) -> str:
    name = PurePosixPath(path.name).as_posix()
    if not name or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
        raise SkillError(f"model artifact name is not portable: {path.name!r}")
    return name


def _regular_file_record(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise SkillError(f"{label} must be a regular non-symlink file: {path.name}")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise SkillError(f"could not inspect {label} {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise SkillError(f"{label} must be a regular non-symlink file: {path.name}")
    return {
        "name": _portable_name(path),
        "size_bytes": metadata.st_size,
        "sha256": sha256_file(path),
    }


def _model_root_entries(model_dir: Path) -> list[Path]:
    try:
        iterator = os.scandir(model_dir)
    except OSError as exc:
        raise SkillError(f"could not enumerate local model directory {model_dir}: {exc}") from exc
    entries: list[Path] = []
    with iterator:
        for entry in iterator:
            entries.append(Path(entry.path))
            if len(entries) > MAX_MODEL_ROOT_ENTRIES:
                raise SkillError(
                    f"model directory exceeds root-entry limit {MAX_MODEL_ROOT_ENTRIES}"
                )
    return sorted(entries, key=lambda path: path.name)


def _is_weight_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".safetensors") or (
        name.endswith(".bin")
        and (name.startswith("pytorch_model") or name.startswith("model"))
    )


def inspect_local_model(model_value: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    model_dir = Path(os.path.abspath(os.path.expanduser(model_value)))
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise SkillError(f"model must be a local, non-symlink directory: {model_value}")
    config_path = model_dir / "config.json"
    config = read_bounded_json(config_path, MAX_CONFIG_JSON_BYTES, label="config.json")
    if not isinstance(config, dict):
        raise SkillError("config.json must contain a JSON object")
    if config.get("auto_map"):
        raise SkillError(
            "config.json declares auto_map remote model code; capture_oracle.py refuses remote code"
        )
    architectures = config.get("architectures")
    if architectures is not None and (
        not isinstance(architectures, list)
        or not all(isinstance(item, str) and item for item in architectures)
    ):
        raise SkillError("config.json architectures must be an array of non-empty strings")

    entries = _model_root_entries(model_dir)
    weight_paths = [path for path in entries if _is_weight_file(path)]
    if not weight_paths:
        raise SkillError("local model directory has no .safetensors or PyTorch .bin weight files")
    if len(weight_paths) > MAX_WEIGHT_FILES:
        raise SkillError(f"model has more than {MAX_WEIGHT_FILES} weight files")

    config_record = _regular_file_record(config_path, label="config.json")
    weight_records = [
        _regular_file_record(path, label="model weight")
        for path in weight_paths
    ]
    identity_bytes = json.dumps(
        {
            "schema_version": 1,
            "config": config_record,
            "weights": weight_records,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    listed = weight_records[:MAX_LISTED_WEIGHT_FILES]
    model_record = {
        "directory": model_dir.name or ".",
        "fingerprint": "sha256:" + hashlib.sha256(identity_bytes).hexdigest(),
        "config": config_record,
        "weights": {
            "algorithm": "sha256",
            "file_count": len(weight_records),
            "total_bytes": sum(record["size_bytes"] for record in weight_records),
            "max_listed_files": MAX_LISTED_WEIGHT_FILES,
            "truncated": len(weight_records) > len(listed),
            "omitted_file_count": len(weight_records) - len(listed),
            "files": listed,
        },
    }
    return model_dir, config, model_record


def validate_tokenizer_json_files(model_dir: Path) -> None:
    for path in _model_root_entries(model_dir):
        name = path.name.lower()
        if name not in TOKENIZER_JSON_NAMES and not (
            name.startswith("tokenizer") and name.endswith(".json")
        ):
            continue
        payload = read_bounded_json(
            path,
            MAX_TOKENIZER_JSON_BYTES,
            label=f"tokenizer JSON {path.name}",
        )
        if name == "tokenizer_config.json" and isinstance(payload, dict) and payload.get("auto_map"):
            raise SkillError(
                "tokenizer_config.json declares auto_map remote tokenizer code; "
                "capture_oracle.py refuses remote code"
            )


def parse_token_ids(values: list[str]) -> list[int]:
    return parse_shared_token_ids(values)


def collect_prompts(inline: list[str] | None, files: list[str] | None) -> list[str]:
    return collect_shared_prompts(inline, files)


def _absolute_lexical(path: Path) -> Path:
    return absolute_lexical(path)


def validate_output_path(path: Path, *, label: str) -> Path:
    return validate_shared_output_path(path, label=label)


def prepare_output_parent(path: Path, *, label: str) -> None:
    prepare_shared_output_parent(path, label=label)


def validate_manifest(payload: Any) -> dict[str, Any]:
    return validate_capture_manifest(payload)


def _find_decoder_layers(model: Any, expected_layers: int | None, torch: Any) -> tuple[str, Any]:
    candidates: list[tuple[int, str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or not module:
            continue
        if expected_layers is not None and len(module) != expected_layers:
            continue
        lowered = name.lower()
        leaf = lowered.rsplit(".", 1)[-1]
        score = 0
        if leaf == "layers":
            score += 100
        elif leaf in {"h", "blocks", "block"}:
            score += 80
        if "decoder" in lowered or lowered.startswith("model"):
            score += 20
        if "encoder" in lowered:
            score -= 100
        score += min(len(module), 20)
        candidates.append((score, name, module))
    if not candidates:
        expected = f" with {expected_layers} entries" if expected_layers is not None else ""
        raise SkillError(f"could not locate a standard decoder layer ModuleList{expected}")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    best = candidates[0]
    if len(candidates) > 1 and candidates[1][0] == best[0]:
        raise SkillError(
            "decoder layer discovery is ambiguous: "
            + ", ".join(item[1] or "<root>" for item in candidates[:2])
        )
    return best[1], best[2]


def _expected_layer_count(config: dict[str, Any]) -> int | None:
    for key in ("num_hidden_layers", "n_layer", "num_layers", "decoder_layers"):
        value = config.get(key)
        if type(value) is int and value > 0:
            return value
    return None


def _find_final_norm(model: Any, layers_name: str) -> Any:
    parent_name, _, layers_leaf = layers_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    candidates: list[tuple[int, str, Any]] = []
    for name, module in parent.named_children():
        if name == layers_leaf:
            continue
        lowered = name.lower()
        class_name = module.__class__.__name__.lower()
        if "norm" not in lowered and "norm" not in class_name and lowered != "ln_f":
            continue
        score = 100 if lowered in {"norm", "final_layer_norm", "ln_f"} else 20
        candidates.append((score, name, module))
    if not candidates:
        raise SkillError("could not locate the decoder final normalization module")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if len(candidates) > 1 and candidates[1][0] == candidates[0][0]:
        raise SkillError("decoder final normalization discovery is ambiguous")
    return candidates[0][2]


def _find_branch_module(layer: Any, branch: str) -> Any | None:
    if branch == "attention":
        preferred = {"self_attn": 100, "attn": 90, "attention": 80, "self_attention": 80}
        marker = ("attn", "attention")
    else:
        preferred = {"mlp": 100, "feed_forward": 90, "ffn": 80, "feedforward": 80}
        marker = ("mlp", "feed_forward", "ffn", "feedforward")
    candidates: list[tuple[int, str, Any]] = []
    for name, module in layer.named_children():
        lowered = name.lower()
        class_name = module.__class__.__name__.lower()
        if lowered in preferred:
            candidates.append((preferred[lowered], name, module))
        elif any(value in lowered or value in class_name for value in marker):
            candidates.append((20, name, module))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if len(candidates) > 1 and candidates[1][0] == candidates[0][0]:
        return None
    return candidates[0][2]


def _first_tensor(value: Any, torch: Any) -> Any | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item, torch)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item, torch)
            if tensor is not None:
                return tensor
    return None


def _capture_hook(captures: dict[str, Any], key: str, torch: Any) -> Callable[..., None]:
    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        if key in captures:
            return
        tensor = _first_tensor(output, torch)
        if tensor is not None:
            captures[key] = tensor.detach()

    return hook


def configure_determinism(torch: Any, np: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A hosting process may already have initialized its one inter-op pool.
        pass
    torch.use_deterministic_algorithms(True)


def build_inputs(
    model_dir: Path,
    prompts: list[str],
    token_ids: list[int] | None,
    config: dict[str, Any],
    torch: Any,
    AutoTokenizer: Any,
) -> tuple[Any, Any]:
    if token_ids is not None:
        vocab_size = config.get("vocab_size")
        if type(vocab_size) is int and any(value >= vocab_size for value in token_ids):
            raise SkillError(f"--token-ids values must be below config vocab_size={vocab_size}")
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        return input_ids, torch.ones_like(input_ids)

    validate_tokenizer_json_files(model_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise SkillError(
            "could not load a built-in local tokenizer; provide complete tokenizer files "
            f"or use --token-ids: {exc}"
        ) from exc
    if len(prompts) > 1:
        if tokenizer.pad_token_id is None:
            raise SkillError("tokenizer has no pad token; use one prompt or --token-ids")
        tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=len(prompts) > 1,
            add_special_tokens=True,
        )
    except Exception as exc:
        raise SkillError(f"could not tokenize local prompt fixture: {exc}") from exc
    input_ids = encoded.get("input_ids")
    if input_ids is None or not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise SkillError("local tokenizer did not produce rank-2 input_ids")
    if input_ids.shape[1] > MAX_TOKEN_IDS:
        raise SkillError(f"tokenized prompt exceeds the {MAX_TOKEN_IDS}-token limit")
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def _zero_padded_query_rows(value: Any, attention_mask: Any, torch: Any) -> Any:
    if not torch.is_tensor(value) or value.ndim < 2 or value.shape[:2] != attention_mask.shape:
        return value
    query_mask = attention_mask.to(device=value.device, dtype=torch.bool)
    for _ in range(value.ndim - 2):
        query_mask = query_mask.unsqueeze(-1)
    return torch.where(query_mask, value, torch.zeros_like(value))


def capture_tensors(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    config: dict[str, Any],
    generate_steps: int,
    torch: Any,
) -> dict[str, Any]:
    layers_name, layers = _find_decoder_layers(model, _expected_layer_count(config), torch)
    final_norm = _find_final_norm(model, layers_name)
    captures: dict[str, Any] = {}
    handles: list[Any] = []
    try:
        try:
            embedding = model.get_input_embeddings()
        except (AttributeError, NotImplementedError) as exc:
            raise SkillError("model does not expose a standard input embedding module") from exc
        if embedding is None:
            raise SkillError("model does not expose a standard input embedding module")
        handles.append(embedding.register_forward_hook(_capture_hook(captures, "embed", torch)))
        handles.append(final_norm.register_forward_hook(_capture_hook(captures, "final_norm", torch)))
        for index, layer in enumerate(layers):
            handles.append(
                layer.register_forward_hook(
                    _capture_hook(captures, f"layer.{index}.hidden", torch)
                )
            )
            attention = _find_branch_module(layer, "attention")
            if attention is not None:
                handles.append(
                    attention.register_forward_hook(
                        _capture_hook(captures, f"layer.{index}.attention", torch)
                    )
                )
            mlp = _find_branch_module(layer, "mlp")
            if mlp is not None:
                handles.append(
                    mlp.register_forward_hook(
                        _capture_hook(captures, f"layer.{index}.mlp", torch)
                    )
                )

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states and torch.is_tensor(hidden_states[0]):
            captures["embed"] = hidden_states[0].detach()
        logits = getattr(outputs, "logits", None)
        if logits is None or not torch.is_tensor(logits):
            raise SkillError("model forward did not return logits")
        captures["logits"] = logits.detach()
    finally:
        for handle in reversed(handles):
            handle.remove()

    required = {"embed", "final_norm", "logits"}
    required.update(f"layer.{index}.hidden" for index in range(len(layers)))
    missing = sorted(required - set(captures))
    if missing:
        raise SkillError("model hooks did not capture required tensors: " + ", ".join(missing))
    captures = {
        name: _zero_padded_query_rows(value, attention_mask, torch)
        for name, value in captures.items()
    }

    generated: list[Any] = []
    sequence = input_ids
    mask = attention_mask
    with torch.inference_mode():
        for _ in range(generate_steps):
            outputs = model(
                input_ids=sequence,
                attention_mask=mask,
                use_cache=False,
                return_dict=True,
            )
            logits = getattr(outputs, "logits", None)
            if logits is None or not torch.is_tensor(logits):
                raise SkillError("model generation forward did not return logits")
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated.append(next_token)
            sequence = torch.cat((sequence, next_token[:, None]), dim=1)
            mask = torch.cat(
                (
                    mask,
                    torch.ones(
                        (mask.shape[0], 1),
                        dtype=mask.dtype,
                        device=mask.device,
                    ),
                ),
                dim=1,
            )
    if generated:
        generated_ids = torch.stack(generated, dim=1)
    else:
        generated_ids = torch.empty(
            (input_ids.shape[0], 0),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
    return {
        "input_ids": input_ids.detach(),
        "attention_mask": attention_mask.detach(),
        **captures,
        "generated_token_ids": generated_ids.detach(),
    }


def capture_encoder_decoder_tensors(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    config: dict[str, Any],
    generate_steps: int,
    torch: Any,
) -> dict[str, Any]:
    """Capture the T5-style encoder and first decoder step plus greedy IDs."""
    start = config.get("decoder_start_token_id")
    if type(start) is not int or start < 0:
        raise SkillError("encoder-decoder config requires decoder_start_token_id")
    decoder_input_ids = torch.full(
        (input_ids.shape[0], 1), start, dtype=input_ids.dtype, device=input_ids.device
    )
    captures: dict[str, Any] = {}
    handles: list[Any] = []
    decoder = getattr(model, "decoder", None)
    blocks = getattr(decoder, "block", None)
    encoder = getattr(model, "encoder", None)
    encoder_blocks = getattr(encoder, "block", None)
    if (
        blocks is None
        or encoder_blocks is None
        or len(blocks) != _expected_layer_count(config)
        or len(encoder_blocks) != len(blocks)
    ):
        raise SkillError("could not locate the standard encoder-decoder decoder blocks")
    try:
        for index, block in enumerate(encoder_blocks):
            handles.append(
                block.register_forward_hook(
                    _capture_hook(captures, f"encoder.layer.{index}.hidden", torch)
                )
            )
        for index, block in enumerate(blocks):
            layers = getattr(block, "layer", None)
            if layers is None or len(layers) < 3:
                raise SkillError("decoder block does not expose self/cross/FF layers")
            handles.append(
                layers[1].register_forward_hook(
                    _capture_hook(captures, f"decoder.layer.{index}.cross_attention", torch)
                )
            )
            handles.append(
                block.register_forward_hook(
                    _capture_hook(captures, f"decoder.layer.{index}.hidden", torch)
                )
            )
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        for handle in reversed(handles):
            handle.remove()
    encoder_states = getattr(outputs, "encoder_hidden_states", None)
    decoder_states = getattr(outputs, "decoder_hidden_states", None)
    logits = getattr(outputs, "logits", None)
    expected = _expected_layer_count(config)
    if (
        not encoder_states
        or not decoder_states
        or expected is None
        or len(encoder_states) != expected + 1
        or len(decoder_states) != expected + 1
        or logits is None
        or not torch.is_tensor(logits)
    ):
        raise SkillError("encoder-decoder forward did not return complete hidden states and logits")
    captures["encoder.embed"] = encoder_states[0].detach()
    captures["decoder.embed"] = decoder_states[0].detach()
    required_layers = {
        *(f"encoder.layer.{index}.hidden" for index in range(expected)),
        *(f"decoder.layer.{index}.hidden" for index in range(expected)),
    }
    missing_layers = sorted(required_layers - captures.keys())
    if missing_layers:
        raise SkillError("encoder-decoder block hooks did not capture: " + ", ".join(missing_layers))
    captures["encoder.final_norm"] = encoder_states[-1].detach()
    captures["decoder.final_norm"] = decoder_states[-1].detach()
    captures["logits"] = logits.detach()
    missing_cross = sorted(
        f"decoder.layer.{index}.cross_attention"
        for index in range(expected)
        if f"decoder.layer.{index}.cross_attention" not in captures
    )
    if missing_cross:
        raise SkillError("decoder cross-attention hooks did not capture: " + ", ".join(missing_cross))
    captures = {
        name: _zero_padded_query_rows(value, attention_mask, torch)
        for name, value in captures.items()
    }

    generated: list[Any] = []
    sequence = decoder_input_ids
    with torch.inference_mode():
        for _ in range(generate_steps):
            step = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=sequence,
                use_cache=False,
                return_dict=True,
            )
            next_token = torch.argmax(step.logits[:, -1, :], dim=-1)
            generated.append(next_token)
            sequence = torch.cat((sequence, next_token[:, None]), dim=1)
    generated_ids = (
        torch.stack(generated, dim=1)
        if generated
        else torch.empty((input_ids.shape[0], 0), dtype=input_ids.dtype, device=input_ids.device)
    )
    return {
        "input_ids": input_ids.detach(),
        "attention_mask": attention_mask.detach(),
        "decoder_input_ids": decoder_input_ids.detach(),
        **captures,
        "generated_token_ids": generated_ids.detach(),
    }


def to_numpy_tensors(tensors: dict[str, Any], *, keep_dtype: bool, np: Any) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    for key in sorted(tensors):
        tensor = tensors[key].detach().cpu().contiguous()
        if tensor.is_floating_point() and not keep_dtype:
            tensor = tensor.float()
        try:
            array = tensor.numpy()
        except (TypeError, RuntimeError) as exc:
            policy = " with --keep-dtype" if keep_dtype else ""
            raise SkillError(
                f"tensor {key} dtype {tensor.dtype} cannot be represented in NumPy{policy}"
            ) from exc
        if array.dtype.hasobject:
            raise SkillError(f"tensor {key} produced an unsafe object dtype")
        arrays[key] = np.ascontiguousarray(array)
    return arrays


def _validate_capture_args(args: argparse.Namespace) -> tuple[list[str], list[int] | None]:
    validate_capture_limits(args.generate_steps, args.max_output_mb)
    if args.model is None:
        raise SkillError("model directory is required unless --validate-manifest is used")
    if args.output is None:
        raise SkillError("--output is required for oracle capture")
    return validate_input_mode(args.token_ids, args.prompt, args.prompts_file)


def _run_manifest_validation(path_value: str) -> int:
    payload = read_bounded_json(
        Path(path_value),
        MAX_MANIFEST_JSON_BYTES,
        label="oracle manifest",
    )
    manifest = validate_manifest(payload)
    print(json.dumps({"ok": True, "schema_version": manifest["schema_version"]}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate_manifest:
            conflicting = any((
                args.model is not None,
                args.output is not None,
                args.manifest is not None,
                bool(args.prompt),
                bool(args.prompts_file),
                bool(args.token_ids),
                args.keep_dtype,
                args.generate_steps != DEFAULT_GENERATE_STEPS,
                args.seed != 0,
                args.max_output_mb != DEFAULT_MAX_OUTPUT_MB,
            ))
            if conflicting:
                raise SkillError("--validate-manifest cannot be combined with capture arguments")
            return _run_manifest_validation(args.validate_manifest)

        prompts, token_ids = _validate_capture_args(args)
        model_dir, config, model_record = inspect_local_model(args.model)

        output = validate_output_path(Path(args.output), label="NPZ output")
        if output.suffix.lower() != ".npz":
            raise SkillError("--output must end with .npz")
        manifest = validate_output_path(
            Path(args.manifest) if args.manifest else output.with_suffix(".manifest.json"),
            label="manifest output",
        )
        if output == manifest:
            raise SkillError("NPZ and manifest outputs must be different paths")
        protected = {_absolute_lexical(model_dir / "config.json")}
        protected.update(
            _absolute_lexical(path)
            for path in _model_root_entries(model_dir)
            if _is_weight_file(path)
        )
        if output in protected or manifest in protected:
            raise SkillError("oracle outputs must not overwrite config.json or model weights")

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        try:
            import torch
        except ImportError as exc:
            raise SkillError(
                "capture_oracle.py requires optional package 'torch'; "
                "install it with python3 -m pip install torch"
            ) from exc
        try:
            import numpy as np
        except ImportError as exc:
            raise SkillError(
                "capture_oracle.py requires optional package 'numpy'; "
                "install it with python3 -m pip install numpy"
            ) from exc
        try:
            import transformers
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise SkillError(
                "capture_oracle.py requires optional package 'transformers'; "
                "install it with python3 -m pip install transformers"
            ) from exc
        transformers.logging.disable_progress_bar()

        configure_determinism(torch, np, args.seed)
        try:
            hf_config = AutoConfig.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:
            raise SkillError(
                "model architecture is not built into Transformers; "
                "capture_oracle.py refuses remote code"
            ) from exc
        try:
            model_class = AutoModelForSeq2SeqLM if config.get("is_encoder_decoder") is True else AutoModelForCausalLM
            model = model_class.from_pretrained(
                str(model_dir),
                config=hf_config,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:
            message = str(exc)
            if "remote code" in message.lower() or "trust_remote_code" in message.lower():
                raise SkillError(
                    "model architecture requires remote code; capture_oracle.py supports only "
                    "built-in Transformers architectures"
                ) from exc
            raise SkillError(f"could not load built-in local language model: {exc}") from exc
        model.train(False)

        input_ids, attention_mask = build_inputs(
            model_dir,
            prompts,
            token_ids,
            config,
            torch,
            AutoTokenizer,
        )
        capture_function = (
            capture_encoder_decoder_tensors
            if config.get("is_encoder_decoder") is True
            else capture_tensors
        )
        captured = capture_function(
            model, input_ids, attention_mask, config, args.generate_steps, torch
        )
        arrays = to_numpy_tensors(captured, keep_dtype=args.keep_dtype, np=np)
        cap_bytes = int(args.max_output_mb * 1024 * 1024)
        bounded_size = raw_npz_bound(arrays)
        if bounded_size > cap_bytes:
            raise SkillError(
                f"captured tensor payload exceeds --max-output-mb: "
                f"{bounded_size} > {cap_bytes} bytes"
            )

        manifest_payload = {
            "schema_version": SCHEMA_VERSION,
            "model": model_record,
            "capture": {
                "prompts": prompts if token_ids is None else None,
                "token_ids": token_ids,
                "generate_steps": args.generate_steps,
                "seed": args.seed,
                "dtype_policy": "keep" if args.keep_dtype else "float32",
            },
            "tensors": tensor_inventory(arrays),
            "libraries": {
                "python": platform.python_version(),
                "numpy": str(np.__version__),
                "torch": str(torch.__version__),
                "transformers": str(transformers.__version__),
            },
        }
        write_capture_outputs(
            output,
            manifest,
            arrays,
            manifest_payload,
            cap_bytes,
            np,
        )

        print(json.dumps({
            "manifest": str(manifest),
            "npz": str(output),
            "ok": True,
            "tensor_count": len(arrays),
        }, sort_keys=True))
        return 0
    except (SkillError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
