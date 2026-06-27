#!/usr/bin/env python3
"""Validate deterministic source-to-target weight shape mappings without loading tensors."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from _common import SkillError, dump_json, load_structured


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a WEIGHT_MAP manifest against source and target tensor manifests")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output")
    parser.add_argument("--allow-unmapped", action="store_true")
    return parser.parse_args()


def tensor_map(data: Any) -> dict[str, dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("tensors"), list):
        tensors = {str(x["key"]): x for x in data["tensors"] if isinstance(x, dict) and "key" in x}
        if tensors or "source_format_summary" not in data:
            return tensors
    if isinstance(data, dict) and isinstance(data.get("tensors"), dict):
        tensors = {str(k): v for k, v in data["tensors"].items() if isinstance(v, dict)}
        if tensors or "source_format_summary" not in data:
            return tensors
    if isinstance(data, dict) and isinstance(data.get("source_format_summary"), dict):
        tensors = source_format_tensor_map(data["source_format_summary"])
        if tensors:
            return tensors
        raise SkillError("Source-format manifests do not expose static tensor shapes usable for weight-map validation")
    if isinstance(data, dict) and all(isinstance(v, dict) and "shape" in v for v in data.values()):
        return {str(k): v for k, v in data.items()}
    raise SkillError("Manifest must contain tensors as a list of {key, shape} or a key->spec mapping")


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
                shape = item.get("shape")
                if isinstance(shape, list) and all(isinstance(x, int) for x in shape):
                    tensors[str(item["name"])] = {
                        "key": str(item["name"]),
                        "shape": list(shape),
                        "dtype": item.get("dtype"),
                        "source_format": "onnx",
                    }
        elif fmt == "gguf":
            for item in manifest.get("tensors", []):
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                shape = item.get("shape")
                if isinstance(shape, list) and all(isinstance(x, int) for x in shape):
                    tensors[str(item["name"])] = {
                        "key": str(item["name"]),
                        "shape": list(shape),
                        "dtype": item.get("type"),
                        "source_format": "gguf",
                    }
    return tensors


def shape_of(spec: dict[str, Any]) -> list[int]:
    shape = spec.get("shape")
    if not isinstance(shape, list) or not all(isinstance(x, int) and x >= 0 for x in shape):
        raise SkillError(f"Invalid tensor shape: {shape!r}")
    return list(shape)


def infer_reshape(old: list[int], new: list[int]) -> list[int]:
    if new.count(-1) > 1:
        raise SkillError(f"reshape has more than one -1: {new}")
    old_n = math.prod(old)
    out = list(new)
    if -1 in out:
        known = math.prod(x for x in out if x != -1)
        if known == 0 or old_n % known:
            raise SkillError(f"cannot infer reshape {old} -> {new}")
        out[out.index(-1)] = old_n // known
    if math.prod(out) != old_n:
        raise SkillError(f"reshape changes element count: {old} -> {out}")
    return out


def apply_transforms(shape: list[int], transforms: list[dict[str, Any]]) -> list[int]:
    current = list(shape)
    for transform in transforms:
        op = transform.get("op")
        if op in (None, "identity", "cast", "rename"):
            continue
        if op in {"transpose", "permute"}:
            axes = transform.get("axes")
            if axes is None and len(current) == 2:
                axes = [1, 0]
            if not isinstance(axes, list) or sorted(axes) != list(range(len(current))):
                raise SkillError(f"invalid axes {axes} for shape {current}")
            current = [current[i] for i in axes]
        elif op == "reshape":
            new_shape = transform.get("shape")
            if not isinstance(new_shape, list) or not all(isinstance(x, int) for x in new_shape):
                raise SkillError("reshape requires integer shape")
            current = infer_reshape(current, new_shape)
        elif op == "squeeze":
            axis = int(transform["axis"])
            axis %= len(current)
            if current[axis] != 1:
                raise SkillError(f"cannot squeeze non-unit axis {axis} of {current}")
            current.pop(axis)
        elif op == "unsqueeze":
            axis = int(transform["axis"])
            if axis < 0:
                axis += len(current) + 1
            if axis < 0 or axis > len(current):
                raise SkillError(f"invalid unsqueeze axis {axis} for {current}")
            current.insert(axis, 1)
        elif op == "slice":
            axis = int(transform["axis"]) % len(current)
            start = int(transform.get("start", 0))
            end = int(transform.get("end", current[axis]))
            step = int(transform.get("step", 1))
            if step <= 0:
                raise SkillError("slice step must be positive")
            length = max(0, math.ceil((min(end, current[axis]) - max(start, 0)) / step))
            current[axis] = length
        else:
            raise SkillError(f"Unsupported transform op: {op!r}")
    return current


def main() -> int:
    args = parse_args()
    try:
        source = tensor_map(load_structured(args.source))
        target = tensor_map(load_structured(args.target))
        mapping = load_structured(args.mapping)
        entries = mapping.get("entries", []) if isinstance(mapping, dict) else []
        ignored = set(mapping.get("ignored_source", [])) if isinstance(mapping, dict) else set()
        generated = set(mapping.get("generated_target", [])) if isinstance(mapping, dict) else set()
        if not isinstance(entries, list):
            raise SkillError("mapping.entries must be a list")

        errors: list[str] = []
        warnings: list[str] = []
        mapped_source: set[str] = set()
        mapped_target: set[str] = set()
        checks: list[dict[str, Any]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                errors.append(f"entry {index}: not an object")
                continue
            src = entry.get("source")
            dst = entry.get("target")
            if not isinstance(src, str) or not isinstance(dst, str):
                errors.append(f"entry {index}: source and target must be strings")
                continue
            if src in mapped_source:
                errors.append(f"source mapped more than once: {src}")
            if dst in mapped_target and not entry.get("allow_shared_target"):
                errors.append(f"target mapped more than once: {dst}")
            mapped_source.add(src)
            mapped_target.add(dst)
            if src not in source:
                errors.append(f"entry {index}: missing source tensor {src}")
                continue
            if dst not in target:
                errors.append(f"entry {index}: missing target tensor {dst}")
                continue
            src_shape = shape_of(source[src])
            dst_shape = shape_of(target[dst])
            try:
                actual = apply_transforms(src_shape, entry.get("transforms", []))
            except (SkillError, KeyError, ValueError) as exc:
                errors.append(f"entry {index} {src}->{dst}: {exc}")
                continue
            ok = actual == dst_shape
            checks.append({"source": src, "target": dst, "source_shape": src_shape, "transformed_shape": actual, "target_shape": dst_shape, "ok": ok})
            if not ok:
                errors.append(f"shape mismatch {src} {src_shape} -> {actual}, target {dst} {dst_shape}")

        unexplained_source = sorted(set(source) - mapped_source - ignored)
        unexplained_target = sorted(set(target) - mapped_target - generated)
        unknown_ignored = sorted(ignored - set(source))
        unknown_generated = sorted(generated - set(target))
        if unknown_ignored:
            errors.append(f"ignored_source keys not found: {unknown_ignored}")
        if unknown_generated:
            errors.append(f"generated_target keys not found: {unknown_generated}")
        if not args.allow_unmapped:
            if unexplained_source:
                errors.append(f"unexplained source tensors: {len(unexplained_source)}")
            if unexplained_target:
                errors.append(f"unexplained target tensors: {len(unexplained_target)}")

        report = {
            "schema_version": 1,
            "ok": not errors,
            "source_tensors": len(source),
            "target_tensors": len(target),
            "mapping_entries": len(entries),
            "checks": checks,
            "unexplained_source": unexplained_source,
            "unexplained_target": unexplained_target,
            "warnings": warnings,
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
