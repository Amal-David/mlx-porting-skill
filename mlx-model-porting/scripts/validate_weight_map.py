#!/usr/bin/env python3
"""Validate schema-2 WEIGHT_MAP files with the converter's shape executor."""
from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path
from typing import Any

from _common import SkillError, dump_json, load_structured
from _weight_map_common import (
    entry_output_shapes,
    parse_conversion_map,
    parse_strict_json,
    shape,
    validate_mapping_against_source,
)


MAX_JSON_BYTES = 16 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a WEIGHT_MAP manifest against source and target tensor manifests"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output")
    parser.add_argument("--allow-unmapped", action="store_true")
    return parser.parse_args()


def tensor_map(data: Any) -> dict[str, dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("tensors"), list):
        tensors = {
            str(item["key"]): item
            for item in data["tensors"]
            if isinstance(item, dict) and "key" in item
        }
        if tensors or "source_format_summary" not in data:
            return tensors
    if isinstance(data, dict) and isinstance(data.get("tensors"), dict):
        tensors = {
            str(key): value
            for key, value in data["tensors"].items()
            if isinstance(value, dict)
        }
        if tensors or "source_format_summary" not in data:
            return tensors
    if isinstance(data, dict) and isinstance(data.get("source_format_summary"), dict):
        tensors = source_format_tensor_map(data["source_format_summary"])
        if tensors:
            return tensors
        raise SkillError(
            "Source-format manifests do not expose static tensor shapes usable for weight-map validation"
        )
    if isinstance(data, dict) and all(
        isinstance(value, dict) and "shape" in value for value in data.values()
    ):
        return {str(key): value for key, value in data.items()}
    raise SkillError(
        "Manifest must contain tensors as a list of {key, shape} or a key->spec mapping"
    )


def source_format_tensor_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tensors: dict[str, dict[str, Any]] = {}
    manifests = summary.get("manifests", [])
    if not isinstance(manifests, list):
        return tensors
    for manifest in manifests:
        if not isinstance(manifest, dict):
            continue
        fmt = manifest.get("format")
        if fmt == "onnx":
            for item in manifest.get("graph", {}).get("initializers", []):
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                item_shape = item.get("shape")
                if isinstance(item_shape, list) and all(isinstance(x, int) for x in item_shape):
                    tensors[str(item["name"])] = {
                        "key": str(item["name"]),
                        "shape": list(item_shape),
                        "dtype": item.get("dtype"),
                        "source_format": "onnx",
                    }
        elif fmt == "gguf":
            for item in manifest.get("tensors", []):
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                item_shape = item.get("shape")
                if isinstance(item_shape, list) and all(isinstance(x, int) for x in item_shape):
                    tensors[str(item["name"])] = {
                        "key": str(item["name"]),
                        "shape": list(item_shape),
                        "dtype": item.get("type"),
                        "source_format": "gguf",
                    }
    return tensors


def shape_of(spec: dict[str, Any]) -> list[int]:
    return shape(spec.get("shape"), label=f"Invalid tensor shape {spec.get('shape')!r}")


def load_mapping(path_value: str) -> Any:
    path = Path(path_value)
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise SkillError(f"could not inspect WEIGHT_MAP {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SkillError("WEIGHT_MAP must be a regular non-symlink file")
    if metadata.st_size > MAX_JSON_BYTES:
        raise SkillError(f"WEIGHT_MAP exceeds {MAX_JSON_BYTES} bytes")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SkillError(f"could not read WEIGHT_MAP {path}: {exc}") from exc
    if len(raw) != metadata.st_size:
        raise SkillError("WEIGHT_MAP changed while it was read")
    return parse_strict_json(raw, label="WEIGHT_MAP")


def _append_once(errors: list[str], message: str) -> None:
    if message not in errors:
        errors.append(message)


def main() -> int:
    args = parse_args()
    try:
        source = tensor_map(load_structured(args.source))
        target = tensor_map(load_structured(args.target))
        mapping = parse_conversion_map(load_mapping(args.mapping))
        errors: list[str] = []
        try:
            validate_mapping_against_source(
                mapping,
                source,
                allow_unmapped=args.allow_unmapped,
            )
        except SkillError as exc:
            _append_once(errors, str(exc))

        checks: list[dict[str, Any]] = []
        for index, entry in enumerate(mapping["entries"]):
            try:
                output_shapes = entry_output_shapes(entry)
            except SkillError as exc:
                _append_once(errors, f"entry {index}: {exc}")
                continue
            source_names = [record["source"] for record in entry["sources"]]
            source_field: Any = source_names[0] if len(source_names) == 1 else source_names
            source_shape: Any = (
                shape_of(source[source_names[0]])
                if len(source_names) == 1 and source_names[0] in source
                else [shape_of(source[name]) for name in source_names if name in source]
            )
            for target_record, transformed_shape in zip(
                entry["targets"], output_shapes, strict=True
            ):
                target_name = target_record["target"]
                if target_name not in target:
                    _append_once(errors, f"entry {index}: missing target tensor {target_name}")
                    continue
                target_shape = shape_of(target[target_name])
                ok = (
                    transformed_shape == target_shape
                    and target_record["shape"] == target_shape
                )
                checks.append(
                    {
                        "source": source_field,
                        "target": target_name,
                        "source_shape": source_shape,
                        "transformed_shape": transformed_shape,
                        "target_shape": target_shape,
                        "ok": ok,
                    }
                )
                if target_record["shape"] != target_shape:
                    _append_once(
                        errors,
                        f"declared target shape mismatch for {target_name}: "
                        f"mapping {target_record['shape']}, target {target_shape}",
                    )
                if transformed_shape != target_shape:
                    _append_once(
                        errors,
                        f"shape mismatch {source_field} {source_shape} -> "
                        f"{transformed_shape}, target {target_name} {target_shape}",
                    )

        ignored = {item["source"] for item in mapping["ignore"]}
        unexplained_source = sorted(set(source) - mapping["mapped_sources"] - ignored)
        unexplained_target = sorted(set(target) - mapping["mapped_targets"])
        if not args.allow_unmapped:
            if unexplained_source:
                _append_once(errors, f"unexplained source tensors: {len(unexplained_source)}")
            if unexplained_target:
                _append_once(errors, f"unexplained target tensors: {len(unexplained_target)}")

        report = {
            "schema_version": 1,
            "ok": not errors,
            "source_tensors": len(source),
            "target_tensors": len(target),
            "mapping_entries": len(mapping["entries"]),
            "checks": checks,
            "unexplained_source": unexplained_source,
            "unexplained_target": unexplained_target,
            "warnings": [],
            "errors": errors,
        }
        text = dump_json(report, args.output)
        if args.output is None:
            sys.stdout.write(text)
        return 0 if not errors else 1
    except SkillError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
