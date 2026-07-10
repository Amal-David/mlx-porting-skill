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


def _axis(axis: int, rank: int) -> int:
    if rank <= 0:
        raise SkillError("axis transform requires a non-scalar tensor")
    axis %= rank
    return axis


def entry_shape_pairs(
    entry: dict[str, Any],
    source: dict[str, dict[str, Any]],
) -> list[tuple[list[str], str, list[int], list[int] | None]]:
    """Return (sources, target, transformed shape, declared target shape)."""
    transforms = entry.get("transforms", [])
    if not isinstance(transforms, list) or not all(isinstance(item, dict) for item in transforms):
        raise SkillError("transforms must be a list of objects")
    if "sources" in entry:
        records = entry.get("sources")
        target = entry.get("target")
        if not isinstance(records, list) or len(records) < 2 or not isinstance(target, str):
            raise SkillError("merge entries require sources records and one target")
        source_names: list[str] = []
        shapes: list[list[int]] = []
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("source"), str):
                raise SkillError("merge source records require source strings")
            name = record["source"]
            if name not in source:
                raise SkillError(f"missing source tensor {name}")
            actual = shape_of(source[name])
            declared = record.get("shape")
            if declared is not None and shape_of({"shape": declared}) != actual:
                raise SkillError(f"declared source shape mismatch for {name}")
            source_names.append(name)
            shapes.append(actual)
        current: list[int] | None = None
        for transform in transforms:
            op = transform.get("op")
            if op == "merge":
                if current is not None:
                    raise SkillError("merge transform may appear only once")
                if any(len(shape) != len(shapes[0]) for shape in shapes):
                    raise SkillError("merge sources must have the same rank")
                axis = _axis(int(transform["axis"]), len(shapes[0]))
                for shape in shapes[1:]:
                    if any(shape[index] != shapes[0][index] for index in range(len(shape)) if index != axis):
                        raise SkillError("merge source shapes differ outside the merge axis")
                current = list(shapes[0])
                current[axis] = sum(shape[axis] for shape in shapes)
            elif current is None:
                if op not in (None, "identity", "rename", "cast"):
                    raise SkillError("only identity, rename, or cast may precede merge")
            else:
                current = apply_transforms(current, [transform])
        if current is None:
            raise SkillError("grouped sources require exactly one merge transform")
        declared_target = entry.get("target_shape")
        declared_shape = shape_of({"shape": declared_target}) if declared_target is not None else None
        return [(source_names, target, current, declared_shape)]

    source_name = entry.get("source")
    if not isinstance(source_name, str):
        raise SkillError("entry source must be a string")
    if source_name not in source:
        raise SkillError(f"missing source tensor {source_name}")
    current = shape_of(source[source_name])
    declared_source = entry.get("source_shape")
    if declared_source is not None and shape_of({"shape": declared_source}) != current:
        raise SkillError(f"declared source shape mismatch for {source_name}")
    if "targets" not in entry:
        target = entry.get("target")
        if not isinstance(target, str):
            raise SkillError("entry target must be a string")
        actual = apply_transforms(current, transforms)
        declared_target = entry.get("target_shape")
        declared_shape = shape_of({"shape": declared_target}) if declared_target is not None else None
        return [([source_name], target, actual, declared_shape)]

    targets = entry.get("targets")
    if not isinstance(targets, list) or len(targets) < 2:
        raise SkillError("split entries require at least two target records")
    output_shapes: list[list[int]] | None = None
    for transform in transforms:
        op = transform.get("op")
        if op == "split":
            if output_shapes is not None:
                raise SkillError("split transform may appear only once")
            axis = _axis(int(transform["axis"]), len(current))
            sizes = transform.get("sizes")
            if (
                not isinstance(sizes, list)
                or len(sizes) != len(targets)
                or not all(isinstance(size, int) and not isinstance(size, bool) and size >= 0 for size in sizes)
                or sum(sizes) != current[axis]
            ):
                raise SkillError("split sizes must match the source axis and target count")
            output_shapes = []
            for size in sizes:
                shape = list(current)
                shape[axis] = size
                output_shapes.append(shape)
        elif output_shapes is None:
            current = apply_transforms(current, [transform])
        elif op not in (None, "identity", "rename", "cast"):
            raise SkillError("only identity, rename, or cast may follow split")
    if output_shapes is None:
        raise SkillError("grouped targets require exactly one split transform")
    pairs: list[tuple[list[str], str, list[int], list[int] | None]] = []
    for record, actual in zip(targets, output_shapes):
        if not isinstance(record, dict) or not isinstance(record.get("target"), str):
            raise SkillError("split target records require target strings")
        declared = record.get("shape")
        declared_shape = shape_of({"shape": declared}) if declared is not None else None
        pairs.append(([source_name], record["target"], actual, declared_shape))
    return pairs


def ignored_source_names(mapping: dict[str, Any], errors: list[str]) -> set[str]:
    ignored = set(mapping.get("ignored_source", []))
    reasoned = mapping.get("ignore", [])
    if not isinstance(reasoned, list):
        errors.append("mapping.ignore must be a list")
        return ignored
    for index, record in enumerate(reasoned):
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("source"), str)
            or not isinstance(record.get("reason"), str)
            or not record["reason"].strip()
        ):
            errors.append(f"ignore {index}: source and non-empty reason are required")
            continue
        ignored.add(record["source"])
    return ignored


def main() -> int:
    args = parse_args()
    try:
        source = tensor_map(load_structured(args.source))
        target = tensor_map(load_structured(args.target))
        mapping = load_structured(args.mapping)
        entries = mapping.get("entries", []) if isinstance(mapping, dict) else []
        errors: list[str] = []
        ignored = ignored_source_names(mapping, errors) if isinstance(mapping, dict) else set()
        generated = set(mapping.get("generated_target", [])) if isinstance(mapping, dict) else set()
        if not isinstance(entries, list):
            raise SkillError("mapping.entries must be a list")

        warnings: list[str] = []
        mapped_source: set[str] = set()
        mapped_target: set[str] = set()
        checks: list[dict[str, Any]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                errors.append(f"entry {index}: not an object")
                continue
            try:
                pairs = entry_shape_pairs(entry, source)
            except (SkillError, KeyError, ValueError) as exc:
                errors.append(f"entry {index}: {exc}")
                continue
            entry_sources = sorted({name for pair in pairs for name in pair[0]})
            for src in entry_sources:
                if src in mapped_source:
                    errors.append(f"source mapped more than once: {src}")
                mapped_source.add(src)
            for sources, dst, actual, declared_target in pairs:
                if dst in mapped_target and not entry.get("allow_shared_target"):
                    errors.append(f"target mapped more than once: {dst}")
                mapped_target.add(dst)
                if dst not in target:
                    errors.append(f"entry {index}: missing target tensor {dst}")
                    continue
                dst_shape = shape_of(target[dst])
                if declared_target is not None and declared_target != dst_shape:
                    errors.append(
                        f"declared target shape mismatch for {dst}: mapping {declared_target}, target {dst_shape}"
                    )
                ok = actual == dst_shape and (declared_target is None or declared_target == dst_shape)
                source_shape: Any
                if len(sources) == 1:
                    source_shape = shape_of(source[sources[0]])
                    source_field: Any = sources[0]
                else:
                    source_shape = [shape_of(source[name]) for name in sources]
                    source_field = sources
                checks.append({
                    "source": source_field,
                    "target": dst,
                    "source_shape": source_shape,
                    "transformed_shape": actual,
                    "target_shape": dst_shape,
                    "ok": ok,
                })
                if actual != dst_shape:
                    errors.append(
                        f"shape mismatch {source_field} {source_shape} -> {actual}, target {dst} {dst_shape}"
                    )

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
