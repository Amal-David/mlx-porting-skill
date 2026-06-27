#!/usr/bin/env python3
"""Statically inspect a local or opt-in Hugging Face model without importing model code."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import SkillError, dump_json, load_structured, sha256_file

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
DEFAULT_REGISTRY = SKILL_ROOT / "assets" / "architectures.yaml"

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "C64": 8,
    "C128": 16,
}

ONNX_TENSOR_TYPES = {
    1: "FLOAT",
    2: "UINT8",
    3: "INT8",
    4: "UINT16",
    5: "INT16",
    6: "INT32",
    7: "INT64",
    8: "STRING",
    9: "BOOL",
    10: "FLOAT16",
    11: "DOUBLE",
    12: "UINT32",
    13: "UINT64",
    14: "COMPLEX64",
    15: "COMPLEX128",
    16: "BFLOAT16",
}

GGUF_VALUE_TYPES = {
    0: "UINT8",
    1: "INT8",
    2: "UINT16",
    3: "INT16",
    4: "UINT32",
    5: "INT32",
    6: "FLOAT32",
    7: "BOOL",
    8: "STRING",
    9: "ARRAY",
    10: "UINT64",
    11: "INT64",
    12: "FLOAT64",
}

GGUF_TENSOR_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
}

MAX_FORMAT_FILE_BYTES = 256 * 1024 * 1024
GGUF_DEFAULT_ALIGNMENT = 32
STATIC_TEXT_LIMIT = 2 * 1024 * 1024
ONNX_STATIC_COVERED_OPS = {
    "Abs", "Add", "Cast", "Concat", "Constant", "Cos", "Div", "Equal", "Erf",
    "Exp", "Flatten", "Gather", "Gemm", "LayerNormalization", "Log",
    "MatMul", "Mul", "Neg", "Pow", "ReduceMean", "Relu", "Reshape",
    "Shape", "Sigmoid", "Sin", "Slice", "Softmax", "Split", "Sqrt",
    "Sub", "Tanh", "Transpose", "Unsqueeze",
}
TENSORFLOW_STATIC_COVERED_OPS = {
    "Add", "BatchMatMulV2", "BiasAdd", "Cast", "ConcatV2", "Const",
    "Einsum", "Erf", "Exp", "GatherV2", "Identity", "MatMul", "Mul",
    "NoOp", "Pack", "Placeholder", "RealDiv", "Relu", "Reshape",
    "Rsqrt", "Shape", "Softmax", "Square", "StridedSlice", "Sub",
    "Tanh", "Transpose",
}
KERAS_STATIC_COVERED_LAYERS = {
    "Activation", "Add", "Concatenate", "Conv1D", "Conv2D", "Dense",
    "Dropout", "Embedding", "Flatten", "InputLayer", "LayerNormalization",
    "MultiHeadAttention", "Reshape", "Softmax",
}

CONFIG_NAMES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "feature_extractor_config.json",
    "model_index.json",
    "adapter_config.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statically inspect model metadata, safetensors headers, architecture signals, and risk flags."
    )
    parser.add_argument("model", help="Local model directory/file or Hugging Face repo id")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Architecture registry")
    parser.add_argument("--output", help="Write JSON report to this path; stdout if omitted")
    parser.add_argument("--markdown", help="Also write a compact Markdown report")
    parser.add_argument("--allow-network", action="store_true", help="Allow metadata-only Hugging Face download")
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="With --allow-network, include safetensors files. This can download very large artifacts.",
    )
    parser.add_argument("--revision", help="Pinned Hugging Face revision")
    parser.add_argument("--max-files", type=int, default=5000, help="Maximum files to inventory")
    parser.add_argument("--hash-small-files", action="store_true", help="SHA-256 files up to 10 MiB")
    return parser.parse_args()


def resolve_model(value: str, allow_network: bool, revision: str | None, download_weights: bool) -> tuple[Path, dict[str, Any]]:
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve(), {"kind": "local", "input": value, "revision": revision}
    if not allow_network:
        raise SkillError(
            f"{value!r} does not exist locally. Re-run with --allow-network to fetch only inspectable metadata."
        )
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:
        raise SkillError("huggingface_hub is required for --allow-network") from exc

    allow_patterns = [
        "*.json", "*.md", "*.txt", "LICENSE*", "NOTICE*", "*.model", "*.tiktoken",
        "*.jinja", "*.py", "*.yaml", "*.yml", "*.safetensors.index.json",
    ]
    if download_weights:
        allow_patterns.append("*.safetensors")
    local = snapshot_download(
        repo_id=value,
        revision=revision,
        allow_patterns=allow_patterns,
    )
    return Path(local).resolve(), {
        "kind": "huggingface",
        "input": value,
        "revision": revision,
        "downloaded_weights": download_weights,
    }


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise SkillError(f"Invalid safetensors header in {path}: missing length")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len <= 0 or header_len > 256 * 1024 * 1024:
            raise SkillError(f"Suspicious safetensors header length {header_len} in {path}")
        raw = handle.read(header_len)
        if len(raw) != header_len:
            raise SkillError(f"Truncated safetensors header in {path}")
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillError(f"Invalid safetensors JSON header in {path}: {exc}") from exc
    metadata = header.pop("__metadata__", {}) if isinstance(header, dict) else {}
    if not isinstance(header, dict):
        raise SkillError(f"Safetensors header in {path} is not a mapping")
    return header, metadata if isinstance(metadata, dict) else {}


def product(shape: list[int]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def inventory(root: Path, max_files: int, hash_small: bool) -> tuple[list[dict[str, Any]], bool]:
    paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    truncated = len(paths) > max_files
    records: list[dict[str, Any]] = []
    for path in paths[:max_files]:
        stat = path.stat()
        rec: dict[str, Any] = {
            "path": str(path.relative_to(root) if root.is_dir() else path.name),
            "size_bytes": stat.st_size,
            "suffix": path.suffix.lower(),
        }
        if hash_small and stat.st_size <= 10 * 1024 * 1024:
            rec["sha256"] = sha256_file(path)
        records.append(rec)
    return records, truncated


def load_configs(root: Path) -> dict[str, dict[str, Any]]:
    base = root if root.is_dir() else root.parent
    configs: dict[str, dict[str, Any]] = {}
    for name in CONFIG_NAMES:
        path = base / name
        if path.exists():
            value = read_json(path)
            if value is not None:
                configs[name] = value
    return configs


def inspect_tensors(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    files = [root] if root.is_file() and root.suffix == ".safetensors" else sorted(root.glob("*.safetensors")) if root.is_dir() else []
    tensors: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    errors: list[str] = []
    for file in files:
        try:
            header, file_meta = read_safetensors_header(file)
        except SkillError as exc:
            errors.append(str(exc))
            continue
        if file_meta:
            metadata[file.name] = file_meta
        for key, spec in header.items():
            if not isinstance(spec, dict):
                continue
            shape = spec.get("shape")
            dtype = spec.get("dtype")
            if not isinstance(shape, list) or not all(isinstance(x, int) for x in shape):
                continue
            count = product(shape)
            tensors.append({
                "key": key,
                "shape": shape,
                "dtype": dtype,
                "parameters": count,
                "estimated_bytes": count * DTYPE_BYTES.get(str(dtype), 0),
                "file": file.name,
            })
    tensors.sort(key=lambda x: x["key"])
    return tensors, metadata, errors


def read_varint(data: bytes, pos: int, limit: int | None = None) -> tuple[int, int]:
    limit = len(data) if limit is None else limit
    shift = 0
    value = 0
    while pos < limit:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
        if shift > 70:
            raise SkillError("protobuf varint is too long")
    raise SkillError("truncated protobuf varint")


def iter_proto_fields(data: bytes) -> Any:
    pos = 0
    limit = len(data)
    while pos < limit:
        key, pos = read_varint(data, pos, limit)
        field_no = key >> 3
        wire_type = key & 0x7
        if wire_type == 0:
            value, pos = read_varint(data, pos, limit)
            yield field_no, wire_type, value
        elif wire_type == 1:
            if pos + 8 > limit:
                raise SkillError("truncated protobuf fixed64")
            yield field_no, wire_type, data[pos:pos + 8]
            pos += 8
        elif wire_type == 2:
            length, pos = read_varint(data, pos, limit)
            end = pos + length
            if end > limit:
                raise SkillError("truncated protobuf length-delimited field")
            yield field_no, wire_type, data[pos:end]
            pos = end
        elif wire_type == 5:
            if pos + 4 > limit:
                raise SkillError("truncated protobuf fixed32")
            yield field_no, wire_type, data[pos:pos + 4]
            pos += 4
        else:
            raise SkillError(f"unsupported protobuf wire type {wire_type}")


def decode_proto_string(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def parse_onnx_value_info(data: bytes) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for field_no, wire_type, value in iter_proto_fields(data):
        if field_no == 1 and wire_type == 2:
            info["name"] = decode_proto_string(value)
        elif field_no == 2 and wire_type == 2:
            info.update(parse_onnx_type(value))
    return info


def parse_onnx_type(data: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field_no, wire_type, value in iter_proto_fields(data):
        if field_no == 1 and wire_type == 2:
            for tensor_field, tensor_wire, tensor_value in iter_proto_fields(value):
                if tensor_field == 1 and tensor_wire == 0:
                    result["dtype"] = ONNX_TENSOR_TYPES.get(tensor_value, f"UNKNOWN_{tensor_value}")
                elif tensor_field == 2 and tensor_wire == 2:
                    result["shape"] = parse_onnx_shape(tensor_value)
    return result


def parse_onnx_shape(data: bytes) -> list[Any]:
    shape: list[Any] = []
    for field_no, wire_type, value in iter_proto_fields(data):
        if field_no == 1 and wire_type == 2:
            dim_value: int | None = None
            dim_param: str | None = None
            for dim_field, dim_wire, dim_raw in iter_proto_fields(value):
                if dim_field == 1 and dim_wire == 0:
                    dim_value = int(dim_raw)
                elif dim_field == 2 and dim_wire == 2:
                    dim_param = decode_proto_string(dim_raw)
            shape.append(dim_value if dim_value is not None else dim_param)
    return shape


def parse_onnx_tensor(data: bytes) -> dict[str, Any]:
    tensor: dict[str, Any] = {"shape": [], "external_data": []}
    for field_no, wire_type, value in iter_proto_fields(data):
        if field_no == 1 and wire_type == 0:
            tensor["shape"].append(int(value))
        elif field_no == 1 and wire_type == 2:
            pos = 0
            while pos < len(value):
                dim, pos = read_varint(value, pos, len(value))
                tensor["shape"].append(int(dim))
        elif field_no == 2 and wire_type == 0:
            tensor["dtype"] = ONNX_TENSOR_TYPES.get(value, f"UNKNOWN_{value}")
            tensor["data_type"] = int(value)
        elif field_no == 8 and wire_type == 2:
            tensor["name"] = decode_proto_string(value)
        elif field_no == 9 and wire_type == 2:
            tensor["raw_data_bytes"] = len(value)
        elif field_no == 13 and wire_type == 2:
            entry: dict[str, Any] = {}
            for kv_field, kv_wire, kv_value in iter_proto_fields(value):
                if kv_field == 1 and kv_wire == 2:
                    entry["key"] = decode_proto_string(kv_value)
                elif kv_field == 2 and kv_wire == 2:
                    entry["value"] = decode_proto_string(kv_value)
            if entry:
                tensor["external_data"].append(entry)
        elif field_no == 14 and wire_type == 0:
            tensor["data_location"] = int(value)
    return tensor


def parse_onnx_node(data: bytes) -> dict[str, Any]:
    node: dict[str, Any] = {"inputs": [], "outputs": []}
    for field_no, wire_type, value in iter_proto_fields(data):
        if field_no == 1 and wire_type == 2:
            node["inputs"].append(decode_proto_string(value))
        elif field_no == 2 and wire_type == 2:
            node["outputs"].append(decode_proto_string(value))
        elif field_no == 3 and wire_type == 2:
            node["name"] = decode_proto_string(value)
        elif field_no == 4 and wire_type == 2:
            node["op_type"] = decode_proto_string(value)
        elif field_no == 7 and wire_type == 2:
            node["domain"] = decode_proto_string(value)
    return node


def parse_onnx_graph(data: bytes) -> dict[str, Any]:
    graph: dict[str, Any] = {"nodes": [], "inputs": [], "outputs": [], "initializers": []}
    op_counts: dict[str, int] = {}
    for field_no, wire_type, value in iter_proto_fields(data):
        if field_no == 1 and wire_type == 2:
            node = parse_onnx_node(value)
            graph["nodes"].append(node)
            op_type = node.get("op_type")
            if op_type:
                op_counts[op_type] = op_counts.get(op_type, 0) + 1
        elif field_no == 2 and wire_type == 2:
            graph["name"] = decode_proto_string(value)
        elif field_no == 5 and wire_type == 2:
            tensor = parse_onnx_tensor(value)
            shape = tensor.get("shape", [])
            if all(isinstance(dim, int) for dim in shape):
                parameters = product(shape)
                tensor["parameters"] = parameters
                tensor["estimated_bytes"] = parameters * DTYPE_BYTES.get(str(tensor.get("dtype")), 0)
            graph["initializers"].append(tensor)
        elif field_no == 11 and wire_type == 2:
            graph["inputs"].append(parse_onnx_value_info(value))
        elif field_no == 12 and wire_type == 2:
            graph["outputs"].append(parse_onnx_value_info(value))
    graph["node_count"] = len(graph["nodes"])
    graph["op_types"] = dict(sorted(op_counts.items()))
    graph["initializer_count"] = len(graph["initializers"])
    graph["external_initializer_count"] = sum(1 for item in graph["initializers"] if item.get("external_data") or item.get("data_location") == 1)
    return graph


def onnx_external_data_holds(graph: dict[str, Any]) -> list[str]:
    holds = []
    for tensor in graph.get("initializers", []):
        for entry in tensor.get("external_data", []):
            if entry.get("key") != "location":
                continue
            value = str(entry.get("value") or "")
            if not value:
                holds.append(f"Initializer {tensor.get('name', '<unknown>')} has empty external data location.")
            elif value.startswith(("/", "http://", "https://")) or ".." in Path(value).parts:
                holds.append(f"Initializer {tensor.get('name', '<unknown>')} has unsafe external data location {value!r}.")
            else:
                holds.append(f"Initializer {tensor.get('name', '<unknown>')} uses external data location {value!r}; do not open it during intake.")
    return holds


def static_coverage_report(
    observed_counts: dict[str, int],
    covered_names: set[str],
    *,
    subject: str,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    if unavailable_reason:
        return {
            "scope": "static-triage",
            "subject": subject,
            "coverage_state": "unavailable",
            "reason": unavailable_reason,
            "total_count": 0,
            "covered_count": 0,
            "unsupported_or_unclassified_count": 0,
            "covered": {},
            "unsupported_or_unclassified": {},
        }
    covered = {name: count for name, count in sorted(observed_counts.items()) if name in covered_names}
    unknown = {name: count for name, count in sorted(observed_counts.items()) if name not in covered_names}
    total = sum(observed_counts.values())
    return {
        "scope": "static-triage",
        "subject": subject,
        "coverage_state": "partial" if observed_counts else "empty",
        "total_count": total,
        "covered_count": sum(covered.values()),
        "unsupported_or_unclassified_count": sum(unknown.values()),
        "covered": covered,
        "unsupported_or_unclassified": unknown,
        "note": "Static coverage means the name maps to ordinary MLX primitives; it is not conversion support.",
    }


def inspect_onnx_file(path: Path, root: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_FORMAT_FILE_BYTES:
        return {
            "format": "onnx",
            "path": str(path.relative_to(root) if root.is_dir() else path.name),
            "errors": [f"file is larger than static parser limit ({MAX_FORMAT_FILE_BYTES} bytes)"],
            "limitations": ["Large ONNX files need a path-based parser before metadata can be reported safely."],
        }
    data = path.read_bytes()
    manifest: dict[str, Any] = {
        "format": "onnx",
        "path": str(path.relative_to(root) if root.is_dir() else path.name),
        "size_bytes": len(data),
        "opsets": [],
        "metadata_props": {},
        "limitations": [
            "ONNX metadata supports intake triage only; conversion support requires unsupported-op reports and MLX parity fixtures."
        ],
    }
    try:
        for field_no, wire_type, value in iter_proto_fields(data):
            if field_no == 1 and wire_type == 0:
                manifest["ir_version"] = int(value)
            elif field_no == 2 and wire_type == 2:
                manifest["producer_name"] = decode_proto_string(value)
            elif field_no == 3 and wire_type == 2:
                manifest["producer_version"] = decode_proto_string(value)
            elif field_no == 4 and wire_type == 2:
                manifest["domain"] = decode_proto_string(value)
            elif field_no == 5 and wire_type == 0:
                manifest["model_version"] = int(value)
            elif field_no == 7 and wire_type == 2:
                manifest["graph"] = parse_onnx_graph(value)
            elif field_no == 8 and wire_type == 2:
                opset: dict[str, Any] = {}
                for op_field, op_wire, op_value in iter_proto_fields(value):
                    if op_field == 1 and op_wire == 2:
                        opset["domain"] = decode_proto_string(op_value)
                    elif op_field == 2 and op_wire == 0:
                        opset["version"] = int(op_value)
                manifest["opsets"].append(opset)
            elif field_no == 14 and wire_type == 2:
                key = None
                item_value = None
                for prop_field, prop_wire, prop_value in iter_proto_fields(value):
                    if prop_field == 1 and prop_wire == 2:
                        key = decode_proto_string(prop_value)
                    elif prop_field == 2 and prop_wire == 2:
                        item_value = decode_proto_string(prop_value)
                if key:
                    manifest["metadata_props"][key] = item_value
    except SkillError as exc:
        manifest.setdefault("errors", []).append(str(exc))
    graph = manifest.get("graph", {})
    manifest["hold_conditions"] = [
        "Record unsupported opset/domain/operator coverage before porting.",
        "Require source-framework oracle and MLX parity fixtures before recommending conversion.",
    ]
    if graph and not graph.get("initializers"):
        manifest["hold_conditions"].append("No initializer metadata was found; weight coverage cannot be inferred.")
    if graph:
        coverage = static_coverage_report(graph.get("op_types", {}), ONNX_STATIC_COVERED_OPS, subject="onnx-operators")
        manifest["operator_coverage"] = coverage
        if coverage["unsupported_or_unclassified_count"]:
            names = ", ".join(coverage["unsupported_or_unclassified"])
            manifest["hold_conditions"].append(f"Unsupported or unclassified ONNX operators require review: {names}.")
        manifest["hold_conditions"].extend(onnx_external_data_holds(graph))
    return manifest


def read_u32(data: bytes, pos: int) -> tuple[int, int]:
    if pos + 4 > len(data):
        raise SkillError("truncated GGUF u32")
    return struct.unpack_from("<I", data, pos)[0], pos + 4


def read_u64(data: bytes, pos: int) -> tuple[int, int]:
    if pos + 8 > len(data):
        raise SkillError("truncated GGUF u64")
    return struct.unpack_from("<Q", data, pos)[0], pos + 8


def read_i64(data: bytes, pos: int) -> tuple[int, int]:
    if pos + 8 > len(data):
        raise SkillError("truncated GGUF i64")
    return struct.unpack_from("<q", data, pos)[0], pos + 8


def read_gguf_string(data: bytes, pos: int) -> tuple[str, int]:
    length, pos = read_u64(data, pos)
    end = pos + length
    if end > len(data):
        raise SkillError("truncated GGUF string")
    return data[pos:end].decode("utf-8", errors="replace"), end


def read_gguf_value(data: bytes, pos: int, value_type: int) -> tuple[Any, int]:
    if value_type == 0:
        if pos >= len(data):
            raise SkillError("truncated GGUF uint8")
        return data[pos], pos + 1
    if value_type == 1:
        if pos >= len(data):
            raise SkillError("truncated GGUF int8")
        return struct.unpack_from("<b", data, pos)[0], pos + 1
    if value_type in {2, 3}:
        if pos + 2 > len(data):
            raise SkillError("truncated GGUF 16-bit value")
        fmt = "<H" if value_type == 2 else "<h"
        return struct.unpack_from(fmt, data, pos)[0], pos + 2
    if value_type in {4, 5, 6}:
        if pos + 4 > len(data):
            raise SkillError("truncated GGUF 32-bit value")
        fmt = {4: "<I", 5: "<i", 6: "<f"}[value_type]
        return struct.unpack_from(fmt, data, pos)[0], pos + 4
    if value_type == 7:
        if pos >= len(data):
            raise SkillError("truncated GGUF bool")
        return bool(data[pos]), pos + 1
    if value_type == 8:
        return read_gguf_string(data, pos)
    if value_type == 9:
        array_type, pos = read_u32(data, pos)
        length, pos = read_u64(data, pos)
        values = []
        for _ in range(min(length, 128)):
            item, pos = read_gguf_value(data, pos, array_type)
            values.append(item)
        if length > 128:
            raise SkillError("GGUF arrays longer than 128 values are not expanded by the static parser")
        return {"array_type": GGUF_VALUE_TYPES.get(array_type, str(array_type)), "values": values}, pos
    if value_type == 10:
        return read_u64(data, pos)
    if value_type == 11:
        return read_i64(data, pos)
    if value_type == 12:
        if pos + 8 > len(data):
            raise SkillError("truncated GGUF float64")
        return struct.unpack_from("<d", data, pos)[0], pos + 8
    raise SkillError(f"unsupported GGUF metadata value type {value_type}")


def align_offset(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def quantized_tensor_type(value: str) -> bool:
    return value not in {"F32", "F16", "BF16", "F64", "I8", "I16", "I32", "I64"}


def inspect_gguf_file(path: Path, root: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_FORMAT_FILE_BYTES:
        return {
            "format": "gguf",
            "path": str(path.relative_to(root) if root.is_dir() else path.name),
            "errors": [f"file is larger than static parser limit ({MAX_FORMAT_FILE_BYTES} bytes)"],
            "limitations": ["Large GGUF files need streaming metadata parsing before all metadata can be reported safely."],
        }
    data = path.read_bytes()
    manifest: dict[str, Any] = {
        "format": "gguf",
        "path": str(path.relative_to(root) if root.is_dir() else path.name),
        "size_bytes": len(data),
        "metadata": {},
        "metadata_types": {},
        "tensors": [],
        "limitations": [
            "GGUF is commonly a converted inference artifact; require source provenance and quality gates before treating it as source-preserving."
        ],
    }
    try:
        if data[:4] != b"GGUF":
            raise SkillError("missing GGUF magic")
        pos = 4
        version, pos = read_u32(data, pos)
        if version not in {2, 3}:
            raise SkillError(f"unsupported GGUF version {version}")
        tensor_count, pos = read_u64(data, pos)
        metadata_count, pos = read_u64(data, pos)
        manifest.update({
            "version": version,
            "tensor_count": tensor_count,
            "metadata_kv_count": metadata_count,
            "header": {
                "magic": "GGUF",
                "version": version,
                "endian": "little",
                "tensor_count": tensor_count,
                "metadata_kv_count": metadata_count,
            },
        })
        for _ in range(metadata_count):
            key, pos = read_gguf_string(data, pos)
            value_type, pos = read_u32(data, pos)
            value, pos = read_gguf_value(data, pos, value_type)
            manifest["metadata"][key] = value
            manifest["metadata_types"][key] = GGUF_VALUE_TYPES.get(value_type, str(value_type))
        manifest["metadata_keys"] = sorted(manifest["metadata"])
        names_seen: set[str] = set()
        for _ in range(tensor_count):
            name, pos = read_gguf_string(data, pos)
            if name in names_seen:
                manifest.setdefault("errors", []).append(f"duplicate GGUF tensor name {name}")
            names_seen.add(name)
            n_dims, pos = read_u32(data, pos)
            shape = []
            for _ in range(n_dims):
                dim, pos = read_u64(data, pos)
                shape.append(dim)
            tensor_type, pos = read_u32(data, pos)
            offset, pos = read_u64(data, pos)
            type_name = GGUF_TENSOR_TYPES.get(tensor_type, f"UNKNOWN_{tensor_type}")
            manifest["tensors"].append({
                "name": name,
                "shape": shape,
                "type_id": tensor_type,
                "type": type_name,
                "element_count": product([int(dim) for dim in shape]),
                "relative_data_offset": offset,
            })
        alignment = int(manifest["metadata"].get("general.alignment") or GGUF_DEFAULT_ALIGNMENT)
        data_start = align_offset(pos, alignment)
        manifest["alignment"] = alignment
        manifest["header"]["alignment"] = alignment
        manifest["tensor_data_start_offset"] = data_start
        for tensor in manifest["tensors"]:
            tensor["absolute_data_offset"] = data_start + int(tensor["relative_data_offset"])
            if tensor["absolute_data_offset"] > len(data):
                manifest.setdefault("errors", []).append(f"tensor {tensor['name']} data offset is outside the file")
    except SkillError as exc:
        manifest.setdefault("errors", []).append(str(exc))
    metadata = manifest.get("metadata", {})
    manifest["architecture"] = metadata.get("general.architecture")
    manifest["name"] = metadata.get("general.name")
    manifest["tokenizer_model"] = metadata.get("tokenizer.ggml.model")
    manifest["file_type"] = metadata.get("general.file_type")
    manifest["quantization_version"] = metadata.get("general.quantization_version")
    source_keys = sorted(
        key for key in metadata
        if key.startswith("general.source")
        or key.startswith("general.base_model")
        or key in {"general.url", "general.repo_url", "general.license", "general.quantized_by"}
    )
    manifest["source_provenance_keys"] = source_keys
    tensor_type_counts = dict(sorted(_counts(tensor["type"] for tensor in manifest.get("tensors", [])).items()))
    manifest["quantization_summary"] = {
        "file_type": manifest.get("file_type"),
        "quantization_version": manifest.get("quantization_version"),
        "tensor_type_counts": tensor_type_counts,
        "quantized_tensor_count": sum(count for dtype, count in tensor_type_counts.items() if quantized_tensor_type(dtype)),
    }
    manifest["tokenizer_summary"] = {
        "model": manifest.get("tokenizer_model"),
        "has_chat_template": "tokenizer.chat_template" in metadata,
    }
    manifest["hold_conditions"] = [
        "Verify source/base-model provenance, tokenizer metadata, and quantization deltas before mapping weights.",
        "Build a source oracle from the pre-GGUF source when source-preserving conversion is required.",
    ]
    if not manifest.get("architecture"):
        manifest["hold_conditions"].append("Missing general.architecture metadata; architecture routing must remain manual.")
    if not source_keys:
        manifest["hold_conditions"].append("Missing source/base-model provenance metadata.")
    if any(quantized_tensor_type(tensor.get("type", "")) for tensor in manifest.get("tensors", [])) and manifest.get("quantization_version") is None:
        manifest["hold_conditions"].append("Quantized tensors are present but general.quantization_version is missing.")
    return manifest


def rel_display(path: Path, root: Path) -> str:
    return str(path.relative_to(root) if root.is_dir() else path.name)


def read_text_prefix(path: Path, limit: int = STATIC_TEXT_LIMIT) -> str:
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    return data[:limit].decode("utf-8", errors="replace")


def safe_json_file(path: Path) -> dict[str, Any] | None:
    value = read_json(path)
    return value if isinstance(value, dict) else None


def inspect_flax_orbax_dir(path: Path, root: Path) -> dict[str, Any]:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    msgpack_files = [rel_display(p, path) for p in files if p.suffix == ".msgpack"]
    metadata_files = [p for p in files if p.name in {"_CHECKPOINT_METADATA", "tree_metadata.json", "checkpoint"}]
    metadata: dict[str, Any] = {}
    for file in metadata_files:
        parsed = safe_json_file(file)
        metadata[rel_display(file, path)] = parsed if parsed is not None else {"parse_error": "not JSON metadata"}
    manifest = {
        "format": "flax-orbax",
        "path": rel_display(path, root),
        "file_count": len(files),
        "msgpack_files": msgpack_files,
        "metadata_files": [rel_display(p, path) for p in metadata_files],
        "metadata": metadata,
        "tree_paths": sorted(
            key
            for item in metadata.values()
            if isinstance(item, dict)
            for key in item.get("tree_paths", [])
            if isinstance(key, str)
        ),
        "limitations": [
            "Flax/Orbax metadata supports intake triage only; restoring checkpoints requires source runtime review."
        ],
        "hold_conditions": [
            "Record the source apply function, params tree, preprocessing, and RNG/dropout mode before mapping weights.",
            "Do not restore Orbax or Flax checkpoints during intake; build a source oracle and deterministic tree manifest first.",
        ],
    }
    if not msgpack_files and not metadata_files:
        manifest["hold_conditions"].append("No Flax/Orbax checkpoint metadata files were recognized.")
    return manifest


def inspect_saved_model_dir(path: Path, root: Path) -> dict[str, Any]:
    pbtxt = path / "saved_model.pbtxt"
    pb = path / "saved_model.pb"
    text = read_text_prefix(pbtxt) if pbtxt.exists() else ""
    signature_keys = sorted(set(re.findall(r'(?m)^\s*key:\s*"([^"]+)"', text)))
    method_names = sorted(set(re.findall(r'(?m)^\s*method_name:\s*"([^"]+)"', text)))
    tensor_names = sorted(set(re.findall(r'(?m)^\s*name:\s*"([^"]+)"', text)))
    op_counts = dict(sorted(_counts(re.findall(r'\bop:\s*"([^"]+)"', text)).items()))
    variable_dir = path / "variables"
    variable_files = sorted(rel_display(p, path) for p in variable_dir.glob("*") if p.is_file()) if variable_dir.exists() else []
    coverage = static_coverage_report(op_counts, TENSORFLOW_STATIC_COVERED_OPS, subject="tensorflow-operators")
    manifest = {
        "format": "tensorflow-savedmodel",
        "path": rel_display(path, root),
        "saved_model_files": [name for name, exists in {"saved_model.pb": pb.exists(), "saved_model.pbtxt": pbtxt.exists()}.items() if exists],
        "signature_keys": signature_keys,
        "method_names": method_names,
        "tensor_names": tensor_names[:50],
        "operator_counts": op_counts,
        "operator_coverage": coverage,
        "variables": {
            "present": bool(variable_files),
            "files": variable_files,
        },
        "assets_count": len(list((path / "assets").glob("*"))) if (path / "assets").exists() else 0,
        "limitations": [
            "SavedModel metadata supports intake triage only; protobuf graph and TensorFlow op support are not fully decoded."
        ],
        "hold_conditions": [
            "Record concrete signatures, TensorFlow ops, preprocessing, and source-framework outputs before conversion.",
            "Do not load SavedModel objects during intake; parse signatures statically or in a sandboxed source-oracle step.",
        ],
    }
    if coverage["unsupported_or_unclassified_count"]:
        names = ", ".join(coverage["unsupported_or_unclassified"])
        manifest["hold_conditions"].append(f"Unsupported or unclassified TensorFlow operators require review: {names}.")
    return manifest


def inspect_keras_archive(path: Path, root: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "format": "keras-archive",
        "path": rel_display(path, root),
        "size_bytes": path.stat().st_size,
        "entries": [],
        "metadata": {},
        "config": {},
        "limitations": [
            "Keras archive metadata supports intake triage only; custom objects and TensorFlow ops require separate review."
        ],
        "hold_conditions": [
            "Review config, custom objects, preprocessing, and TensorFlow/Keras version before mapping weights.",
            "Do not load the Keras archive during intake; inspect archive members statically.",
        ],
    }
    if path.stat().st_size > MAX_FORMAT_FILE_BYTES:
        manifest["errors"] = [f"file is larger than static parser limit ({MAX_FORMAT_FILE_BYTES} bytes)"]
        return manifest
    try:
        with zipfile.ZipFile(path) as archive:
            names = sorted(archive.namelist())
            manifest["entries"] = names
            for name, key in (("metadata.json", "metadata"), ("config.json", "config")):
                if name not in names:
                    continue
                raw = archive.read(name)
                if len(raw) > STATIC_TEXT_LIMIT:
                    manifest.setdefault("errors", []).append(f"{name} exceeds static JSON read limit")
                    continue
                parsed = json.loads(raw.decode("utf-8"))
                if isinstance(parsed, dict):
                    manifest[key] = parsed
    except (zipfile.BadZipFile, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        manifest.setdefault("errors", []).append(str(exc))
    config = manifest.get("config", {})
    model_config = config.get("config", {}) if isinstance(config, dict) else {}
    layers = model_config.get("layers", []) if isinstance(model_config, dict) else []
    manifest["class_name"] = config.get("class_name") if isinstance(config, dict) else None
    manifest["layer_count"] = len(layers) if isinstance(layers, list) else 0
    manifest["layer_class_names"] = [
        str(layer.get("class_name"))
        for layer in layers
        if isinstance(layer, dict) and layer.get("class_name")
    ][:50]
    layer_counts = dict(sorted(_counts(manifest["layer_class_names"]).items()))
    coverage = static_coverage_report(layer_counts, KERAS_STATIC_COVERED_LAYERS, subject="keras-layers")
    manifest["layer_counts"] = layer_counts
    manifest["layer_coverage"] = coverage
    if coverage["unsupported_or_unclassified_count"]:
        names = ", ".join(coverage["unsupported_or_unclassified"])
        manifest["hold_conditions"].append(f"Unsupported or unclassified Keras layers require review: {names}.")
    manifest["weight_files"] = [
        name for name in manifest.get("entries", [])
        if name.endswith((".weights.h5", ".h5"))
    ]
    return manifest


def inspect_coreml_artifact(path: Path, root: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "format": "coreml-model",
            "path": rel_display(path, root),
            "size_bytes": path.stat().st_size,
            "operator_coverage": static_coverage_report(
                {},
                set(),
                subject="coreml-operators",
                unavailable_reason="Core ML protobuf/spec decoding is not implemented in static intake.",
            ),
            "limitations": [
                "Core ML protobuf metadata is not decoded by static intake yet."
            ],
            "hold_conditions": [
                "Require source model provenance, operator coverage, preprocessing, and source-oracle parity before porting from Core ML.",
            ],
        }
    manifest_json = safe_json_file(path / "Manifest.json") or {}
    files = sorted(p for p in path.rglob("*") if p.is_file())
    model_files = [rel_display(p, path) for p in files if p.suffix in {".mlmodel", ".mlmodelc"} or "model.mlmodel" in p.name]
    weight_files = [rel_display(p, path) for p in files if p.suffix in {".bin", ".weights"}]
    item_info = manifest_json.get("itemInfoEntries", {})
    return {
        "format": "coreml-package",
        "path": rel_display(path, root),
        "file_count": len(files),
        "manifest": {
            "fileFormatVersion": manifest_json.get("fileFormatVersion"),
            "rootModelIdentifier": manifest_json.get("rootModelIdentifier"),
            "item_count": len(item_info) if isinstance(item_info, dict) else 0,
        },
        "model_files": model_files,
        "weight_files": weight_files,
        "operator_coverage": static_coverage_report(
            {},
            set(),
            subject="coreml-operators",
            unavailable_reason="Core ML protobuf/spec decoding is not implemented in static intake.",
        ),
        "limitations": [
            "Core ML package metadata supports intake triage only; compiled/runtime semantics are not source-preserving proof."
        ],
        "hold_conditions": [
            "Require source-model provenance, Core ML op coverage, preprocessing metadata, and source-oracle parity before conversion.",
            "Do not treat Core ML packages as authoritative source checkpoints unless the pre-CoreML source is unavailable and quality gates are explicit.",
        ],
    }


def discover_source_format_artifacts(root: Path) -> list[tuple[str, Path]]:
    if root.is_file():
        suffix = root.suffix.lower()
        if suffix == ".onnx":
            return [("onnx", root)]
        if suffix == ".gguf":
            return [("gguf", root)]
        if suffix == ".keras":
            return [("keras", root)]
        if suffix == ".mlmodel":
            return [("coreml", root)]
        if suffix == ".msgpack":
            return [("flax-orbax", root.parent)]
        return []

    artifacts: list[tuple[str, Path]] = []
    for file in sorted(root.rglob("*")):
        if not file.is_file():
            continue
        inside_mlpackage = any(parent.suffix == ".mlpackage" for parent in file.parents)
        suffix = file.suffix.lower()
        if suffix == ".onnx":
            artifacts.append(("onnx", file))
        elif suffix == ".gguf":
            artifacts.append(("gguf", file))
        elif suffix == ".keras":
            artifacts.append(("keras", file))
        elif suffix == ".mlmodel" and not inside_mlpackage:
            artifacts.append(("coreml", file))

    dir_artifacts: list[tuple[str, Path]] = []
    for directory in sorted([root, *[p for p in root.rglob("*") if p.is_dir()]]):
        if directory.suffix == ".mlpackage":
            dir_artifacts.append(("coreml", directory))
        if (directory / "saved_model.pb").exists() or (directory / "saved_model.pbtxt").exists():
            dir_artifacts.append(("tensorflow-savedmodel", directory))

    flax_dirs: list[Path] = []
    for file in sorted(root.rglob("*")):
        if not file.is_file():
            continue
        if file.name in {"flax_model.msgpack", "model.msgpack", "_CHECKPOINT_METADATA", "tree_metadata.json"}:
            flax_dirs.append(file.parent)
    selected_flax_dirs: list[Path] = []
    for directory in sorted(set(flax_dirs), key=lambda p: (len(p.parts), str(p))):
        if not any(parent in directory.parents for parent in selected_flax_dirs):
            selected_flax_dirs.append(directory)
    dir_artifacts.extend(("flax-orbax", directory) for directory in selected_flax_dirs)

    all_artifacts = artifacts + dir_artifacts
    unique: dict[tuple[str, Path], tuple[str, Path]] = {}
    for kind, path in all_artifacts:
        unique[(kind, path)] = (kind, path)
    return [unique[key] for key in sorted(unique, key=lambda item: (str(item[1]), item[0]))]


def inspect_source_formats(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    manifests: list[dict[str, Any]] = []
    errors: list[str] = []
    for kind, artifact in discover_source_format_artifacts(root):
        try:
            if kind == "onnx":
                manifests.append(inspect_onnx_file(artifact, root))
            elif kind == "gguf":
                manifests.append(inspect_gguf_file(artifact, root))
            elif kind == "flax-orbax":
                manifests.append(inspect_flax_orbax_dir(artifact, root))
            elif kind == "tensorflow-savedmodel":
                manifests.append(inspect_saved_model_dir(artifact, root))
            elif kind == "keras":
                manifests.append(inspect_keras_archive(artifact, root))
            elif kind == "coreml":
                manifests.append(inspect_coreml_artifact(artifact, root))
        except (OSError, SkillError) as exc:
            errors.append(f"{artifact}: {exc}")
    return manifests, errors


def inspect_safetensors_checkpoint(
    root: Path,
    configs: dict[str, dict[str, Any]],
    tensors: list[dict[str, Any]],
    file_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    safetensors_files = sorted({str(tensor.get("file")) for tensor in tensors if tensor.get("file")})
    if not safetensors_files or "config.json" in configs:
        return None
    paths = {str(rec["path"]) for rec in file_records}
    config_files = sorted(configs)
    sidecar_files = sorted(
        path for path in paths
        if Path(path).name.lower().startswith(("readme", "license", "copying", "notice"))
        or path.endswith((".json", ".txt", ".md", ".model", ".tiktoken"))
    )
    processor_configs = [
        name for name in config_files
        if name in {"tokenizer_config.json", "preprocessor_config.json", "processor_config.json", "feature_extractor_config.json"}
    ]
    manifest: dict[str, Any] = {
        "format": "safetensors-checkpoint",
        "path": "." if root.is_dir() else root.name,
        "safetensors_files": safetensors_files,
        "tensor_count": len(tensors),
        "tensor_key_samples": [str(tensor.get("key")) for tensor in tensors[:20]],
        "config_files": config_files,
        "processor_config_files": processor_configs,
        "sidecar_files": sidecar_files,
        "limitations": [
            "Safetensors headers prove tensor names and shapes only; they do not prove architecture, preprocessing, tokenizer, or source provenance."
        ],
        "hold_conditions": [
            "Missing config.json; architecture routing is weight-key-only and must remain blocked.",
            "Record source/base-model provenance, license, tokenizer or processor metadata, and source-framework oracle outputs before mapping weights.",
        ],
    }
    if not processor_configs:
        manifest["hold_conditions"].append("Missing tokenizer/processor metadata; preprocessing and postprocessing cannot be inferred.")
    if not sidecar_files:
        manifest["hold_conditions"].append("Missing model card, license, or provenance sidecar files.")
    return manifest


def extract_license(root: Path, configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values: list[dict[str, str]] = []
    for name, cfg in configs.items():
        for key in ("license", "license_name", "license_link"):
            if isinstance(cfg.get(key), str):
                values.append({"source": name, "value": cfg[key]})
    base = root if root.is_dir() else root.parent
    readme = base / "README.md"
    if readme.exists():
        try:
            head = readme.read_text(encoding="utf-8", errors="replace")[:10000]
            match = re.search(r"(?im)^license:\s*['\"]?([^\n'\"]+)", head)
            if match:
                values.append({"source": "README.md", "value": match.group(1).strip()})
        except OSError:
            pass
    license_files = [p.name for p in base.iterdir() if p.is_file() and p.name.lower().startswith(("license", "copying", "notice"))] if base.exists() else []
    return {"declared": values, "license_files": sorted(license_files), "requires_review": not bool(values or license_files)}


def detect_risks(root: Path, configs: dict[str, dict[str, Any]], file_records: list[dict[str, Any]]) -> list[dict[str, str]]:
    risks: list[dict[str, str]] = []
    config = configs.get("config.json", {})
    if config.get("auto_map"):
        risks.append({"severity": "high", "type": "remote-code", "detail": "config.json contains auto_map; do not enable trust_remote_code without review."})
    suffixes = {rec["suffix"] for rec in file_records}
    paths = {rec["path"] for rec in file_records}
    if suffixes.intersection({".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}):
        risks.append({"severity": "high", "type": "unsafe-serialization", "detail": "Pickle-capable weight artifacts are present; inspect/convert in isolation."})
    py_files = sorted(p for p in paths if p.endswith(".py"))
    if py_files:
        risks.append({"severity": "medium", "type": "custom-code", "detail": f"Custom Python files present ({len(py_files)}); review statically before import."})
    if any(Path(p).name in {"setup.py", "pyproject.toml", "requirements.txt"} for p in paths):
        risks.append({"severity": "medium", "type": "dependency-install", "detail": "Package/install metadata is present; do not install blindly."})
    if not any(p.endswith(".safetensors") for p in paths) and suffixes.intersection({".bin", ".pt", ".pth"}):
        risks.append({"severity": "high", "type": "no-safe-weights", "detail": "No safetensors file was found alongside executable-capable weight formats."})
    return risks


def architecture_scores(registry: dict[str, Any], configs: dict[str, dict[str, Any]], tensor_keys: list[str]) -> list[dict[str, Any]]:
    config = configs.get("config.json", {})
    model_type = str(config.get("model_type", "")).lower().replace("-", "_")
    arch_values = config.get("architectures", [])
    if isinstance(arch_values, str):
        arch_values = [arch_values]
    arch_text = " ".join(str(x).lower() for x in arch_values if isinstance(x, str))
    key_blob = "\n".join(tensor_keys[:20000]).lower()
    config_keys = set(config.keys())
    candidates: list[dict[str, Any]] = []
    for family in registry.get("families", []):
        score = 0.0
        evidence: list[str] = []
        aliases = [str(x).lower().replace("-", "_") for x in family.get("model_type_aliases", [])]
        if model_type and model_type in aliases:
            score += 10
            evidence.append(f"exact model_type={model_type}")
        elif model_type and any(alias in model_type or model_type in alias for alias in aliases):
            score += 5
            evidence.append(f"partial model_type={model_type}")
        for pattern in family.get("class_patterns", []):
            if str(pattern).lower() in arch_text:
                score += 4
                evidence.append(f"architecture class contains {pattern}")
        matched_config = [sig for sig in family.get("config_signals", []) if sig in config_keys]
        score += min(len(matched_config), 5) * 0.8
        if matched_config:
            evidence.append("config keys: " + ", ".join(matched_config[:5]))
        matched_weights = [sig for sig in family.get("weight_signals", []) if str(sig).lower() in key_blob]
        score += min(len(matched_weights), 5) * 1.5
        if matched_weights:
            evidence.append("weight signals: " + ", ".join(matched_weights[:5]))
        if score > 0:
            candidates.append({
                "family": family["id"],
                "score": round(score, 2),
                "runbook": family.get("runbook"),
                "targets": family.get("targets", []),
                "state": family.get("state"),
                "evidence": evidence,
                "notes": family.get("notes", ""),
            })
    candidates.sort(key=lambda x: (-x["score"], x["family"]))
    top = candidates[0]["score"] if candidates else 0
    for item in candidates:
        item["confidence"] = round(item["score"] / top, 3) if top else 0
    return candidates[:5]


def recommendation_blockers(
    risks: list[dict[str, str]],
    tensor_count: int,
    source_format_manifests: list[dict[str, Any]] | None = None,
) -> list[str]:
    blockers: list[str] = []
    high_risks = {risk["type"] for risk in risks if risk.get("severity") == "high"}
    format_names = {manifest.get("format") for manifest in (source_format_manifests or [])}
    if tensor_count == 0 and not format_names:
        blockers.append("no safetensors were inspected; architecture routing is metadata-only")
    if format_names:
        blockers.append(
            "source-format static intake is triage-only; conversion support requires unsupported-op reports and MLX parity fixtures"
        )
    if "gguf" in format_names and any(
        any("Missing source/base-model provenance metadata" in condition for condition in manifest.get("hold_conditions", []))
        for manifest in (source_format_manifests or [])
        if manifest.get("format") == "gguf"
    ):
        blockers.append("GGUF source/base-model provenance is missing or incomplete")
    if "safetensors-checkpoint" in format_names:
        blockers.append("safetensors checkpoint is missing required config or provenance metadata")
    if high_risks.intersection({"remote-code", "unsafe-serialization", "no-safe-weights"}):
        blockers.append("high-risk remote-code or unsafe-serialization flags require manual review")
    return blockers


def make_markdown(report: dict[str, Any]) -> str:
    summary = report["tensor_summary"]
    candidates = report["architecture_candidates"]
    lines = [
        "# Static model inspection",
        "",
        f"- Source: `{report['source']['input']}`",
        f"- Local path: `{report['local_path']}`",
        f"- Files: {report['file_summary']['count']}" + (" (inventory truncated)" if report['file_summary']['truncated'] else ""),
        f"- Tensors: {summary['count']}",
        f"- Parameters represented in inspected safetensors: {summary['parameters']:,}",
        f"- Estimated tensor bytes: {summary['estimated_bytes']:,}",
        "",
        "## Architecture candidates",
        "",
    ]
    if candidates:
        lines.extend(["| Family | Score | Confidence | Runbook |", "|---|---:|---:|---|"])
        for c in candidates:
            lines.append(f"| {c['family']} | {c['score']} | {c['confidence']:.3f} | `{c['runbook']}` |")
    else:
        lines.append("No architecture family reached a non-zero score. Manual routing is required.")
    source_formats = report.get("source_format_summary", {})
    if source_formats.get("count"):
        lines += ["", "## Source format manifests", ""]
        for manifest in source_formats.get("manifests", []):
            lines.append(f"- `{manifest.get('path')}`: {manifest.get('format')} static metadata")
            for condition in manifest.get("hold_conditions", [])[:3]:
                lines.append(f"  - hold: {condition}")
    if report.get("recommendation_blockers"):
        lines += ["", "## Recommendation blockers", ""]
        for blocker in report["recommendation_blockers"]:
            lines.append(f"- {blocker}")
    lines += ["", "## Risk flags", ""]
    if report["risks"]:
        for risk in report["risks"]:
            lines.append(f"- **{risk['severity']} / {risk['type']}**: {risk['detail']}")
    else:
        lines.append("- No static high-signal risk flag detected. This is not a security guarantee.")
    lines += ["", "## Next actions", "", "1. Confirm license and pinned source revision.", "2. Review any custom Python statically; do not execute it during intake.", "3. Freeze source-oracle fixtures.", "4. Generate a port plan with `make_port_plan.py`."]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        root, source = resolve_model(args.model, args.allow_network, args.revision, args.download_weights)
        registry = load_structured(args.registry)
        files, truncated = inventory(root, args.max_files, args.hash_small_files)
        configs = load_configs(root)
        tensors, tensor_metadata, tensor_errors = inspect_tensors(root)
        source_format_manifests, source_format_errors = inspect_source_formats(root)
        checkpoint_manifest = inspect_safetensors_checkpoint(root, configs, tensors, files)
        if checkpoint_manifest:
            source_format_manifests.append(checkpoint_manifest)
        tensor_keys = [t["key"] for t in tensors]
        candidates = architecture_scores(registry, configs, tensor_keys)
        risks = detect_risks(root, configs, files)
        blockers = recommendation_blockers(risks, len(tensors), source_format_manifests)
        recommended = candidates[0] if candidates and not blockers else None
        source_formats = sorted({manifest.get("format", "unknown") for manifest in source_format_manifests})
        report = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "inspection_mode": "static-no-model-import",
            "source": source,
            "local_path": str(root),
            "configs": configs,
            "file_summary": {
                "count": len(files),
                "truncated": truncated,
                "extensions": dict(sorted(_counts(rec["suffix"] or "<none>" for rec in files).items())),
                "files": files,
            },
            "tensor_summary": {
                "count": len(tensors),
                "parameters": sum(t["parameters"] for t in tensors),
                "estimated_bytes": sum(t["estimated_bytes"] for t in tensors),
                "dtypes": dict(sorted(_counts(str(t["dtype"]) for t in tensors).items())),
                "files": dict(sorted(_counts(t["file"] for t in tensors).items())),
                "metadata": tensor_metadata,
                "errors": tensor_errors,
            },
            "tensors": tensors,
            "source_format_summary": {
                "count": len(source_format_manifests),
                "formats": source_formats,
                "errors": source_format_errors,
                "manifests": source_format_manifests,
            },
            "architecture_candidates": candidates,
            "recommended_family": recommended["family"] if recommended else None,
            "recommended_runbook": recommended["runbook"] if recommended else None,
            "recommendation_blockers": blockers,
            "license": extract_license(root, configs),
            "risks": risks,
            "limitations": [
                "Static inspection does not prove model behavior or license compatibility.",
                "Parameter totals may double-count tied tensors and omit non-safetensors weights.",
                "Remote metadata-only inspection cannot see tensor keys without downloading weights.",
            ],
        }
        text = dump_json(report, args.output)
        if args.output is None:
            sys.stdout.write(text)
        if args.markdown:
            Path(args.markdown).write_text(make_markdown(report), encoding="utf-8")
        return 0
    except SkillError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
