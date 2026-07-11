"""Shared schema-2 WEIGHT_MAP parsing and deterministic shape execution."""
from __future__ import annotations

import json
import math
from typing import Any

from _common import SkillError


SCHEMA_VERSION = 2
MAX_DIMENSIONS = 64
MAX_DIM = (1 << 63) - 1
DTYPE_POLICIES = {"keep", "f16", "bf16", "f32"}
POLICY_TO_DTYPE = {"f16": "F16", "bf16": "BF16", "f32": "F32"}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SkillError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def parse_strict_json(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    except SkillError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillError(f"{label} is not strict UTF-8 JSON: {exc}") from exc


def require_fields(
    value: Any,
    required: set[str],
    optional: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SkillError(f"{label} must be an object")
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        raise SkillError(
            f"{label} has invalid fields; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return value


def name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SkillError(f"{label} must be a non-empty string")
    return value


def shape(value: Any, *, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) > MAX_DIMENSIONS
        or not all(type(dim) is int and 0 <= dim <= MAX_DIM for dim in value)
    ):
        raise SkillError(f"{label} must be a bounded list of non-negative integers")
    elements = math.prod(value)
    if elements > MAX_DIM:
        raise SkillError(f"{label} exceeds the element-count limit")
    return list(value)


def policy(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or value not in DTYPE_POLICIES:
        raise SkillError(f"{label} must be one of {sorted(DTYPE_POLICIES)}")
    return value


def normalize_axis(axis: int, rank: int, *, insertion: bool = False) -> int:
    limit = rank + 1 if insertion else rank
    if limit <= 0:
        raise SkillError("axis transform requires a non-scalar tensor")
    original = axis
    if axis < 0:
        axis += limit
    if axis < 0 or axis >= limit:
        raise SkillError(f"axis {original} is out of range for rank {rank}")
    return axis


def infer_reshape(old: list[int], new: list[int]) -> list[int]:
    if new.count(-1) > 1:
        raise SkillError(f"reshape has more than one -1: {new}")
    elements = math.prod(old)
    result = list(new)
    if -1 in result:
        known = math.prod(dim for dim in result if dim != -1)
        if known == 0 or elements % known:
            raise SkillError(f"cannot infer reshape {old} -> {new}")
        result[result.index(-1)] = elements // known
    if math.prod(result) != elements:
        raise SkillError(f"reshape changes element count: {old} -> {result}")
    return result


def validate_transforms(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise SkillError(f"{label} must be a list")
    result: list[dict[str, Any]] = []
    schemas = {
        "identity": (set(), set()),
        "rename": (set(), set()),
        "transpose": ({"axes"}, set()),
        "permute": ({"axes"}, set()),
        "reshape": ({"shape"}, set()),
        "squeeze": ({"axis"}, set()),
        "unsqueeze": ({"axis"}, set()),
        "slice": ({"axis"}, {"start", "end", "step"}),
        "cast": ({"dtype"}, set()),
        "split": ({"axis", "sizes"}, set()),
        "merge": ({"axis"}, set()),
    }
    for index, transform in enumerate(value):
        if not isinstance(transform, dict) or not isinstance(transform.get("op"), str):
            raise SkillError(f"{label}[{index}] must have a string op")
        op = transform["op"]
        if op not in schemas:
            raise SkillError(f"{label}[{index}] has unsupported transform op {op!r}")
        required, optional = schemas[op]
        require_fields(
            transform,
            {"op", *required},
            optional,
            label=f"{label}[{index}]",
        )
        if op in {"transpose", "permute"}:
            axes = transform["axes"]
            if not isinstance(axes, list) or not all(type(axis) is int for axis in axes):
                raise SkillError(f"{label}[{index}].axes must be an integer list")
        elif op == "reshape":
            new_shape = transform["shape"]
            if not isinstance(new_shape, list) or not all(
                type(dim) is int and dim >= -1 for dim in new_shape
            ):
                raise SkillError(f"{label}[{index}].shape must contain integers >= -1")
        elif op in {"squeeze", "unsqueeze", "split", "merge", "slice"}:
            if type(transform["axis"]) is not int:
                raise SkillError(f"{label}[{index}].axis must be an integer")
        if op == "slice":
            for field in ("start", "end", "step"):
                if field in transform and type(transform[field]) is not int:
                    raise SkillError(f"{label}[{index}].{field} must be an integer")
            if transform.get("step", 1) <= 0:
                raise SkillError(f"{label}[{index}].step must be positive")
        elif op == "cast":
            policy(transform["dtype"], label=f"{label}[{index}].dtype")
            if transform["dtype"] == "keep":
                raise SkillError(
                    f"{label}[{index}] cast dtype must name a concrete float dtype"
                )
        elif op == "split":
            sizes = transform["sizes"]
            if (
                not isinstance(sizes, list)
                or not sizes
                or not all(type(size) is int and size >= 0 for size in sizes)
            ):
                raise SkillError(
                    f"{label}[{index}].sizes must be a non-empty non-negative integer list"
                )
        result.append(dict(transform))
    return result


def apply_unary_shape(input_shape: list[int], transform: dict[str, Any]) -> list[int]:
    current = list(input_shape)
    op = transform["op"]
    if op in {"identity", "rename", "cast"}:
        return current
    if op in {"transpose", "permute"}:
        axes = transform["axes"]
        if sorted(axes) != list(range(len(current))):
            raise SkillError(f"invalid axes {axes} for shape {current}")
        return [current[index] for index in axes]
    if op == "reshape":
        return infer_reshape(current, transform["shape"])
    if op == "squeeze":
        axis = normalize_axis(transform["axis"], len(current))
        if current[axis] != 1:
            raise SkillError(f"cannot squeeze non-unit axis {axis} of {current}")
        current.pop(axis)
        return current
    if op == "unsqueeze":
        axis = normalize_axis(transform["axis"], len(current), insertion=True)
        current.insert(axis, 1)
        return current
    if op == "slice":
        axis = normalize_axis(transform["axis"], len(current))
        start, end, step = slice(
            transform.get("start"),
            transform.get("end"),
            transform.get("step", 1),
        ).indices(current[axis])
        current[axis] = len(range(start, end, step))
        return current
    raise SkillError(f"{op!r} is not a unary transform")


def effective_policy(
    global_policy: str,
    entry: dict[str, Any],
    transforms: list[dict[str, Any]],
    target: dict[str, Any] | None = None,
) -> str:
    casts = [transform["dtype"] for transform in transforms if transform["op"] == "cast"]
    if len(set(casts)) > 1:
        raise SkillError("one mapping entry contains conflicting cast transforms")
    explicit = target.get("dtype_policy") if target is not None else None
    if explicit is None:
        explicit = entry.get("dtype_policy")
    if explicit is not None:
        explicit = policy(explicit, label="entry dtype_policy")
    if casts and explicit is not None and casts[0] != explicit:
        raise SkillError("cast transform conflicts with the entry dtype_policy")
    return str(explicit or (casts[0] if casts else global_policy))


def parse_conversion_map(value: Any) -> dict[str, Any]:
    mapping = require_fields(
        value,
        {"schema_version", "dtype_policy", "entries", "ignore", "unresolved"},
        {"draft"},
        label="WEIGHT_MAP",
    )
    if type(mapping["schema_version"]) is not int or mapping["schema_version"] != SCHEMA_VERSION:
        raise SkillError(f"WEIGHT_MAP schema_version must be integer {SCHEMA_VERSION}")
    if mapping.get("draft") is True:
        raise SkillError("refusing to convert with a draft WEIGHT_MAP")
    if "draft" in mapping and not isinstance(mapping["draft"], bool):
        raise SkillError("WEIGHT_MAP draft must be boolean")
    global_policy = policy(mapping["dtype_policy"], label="WEIGHT_MAP dtype_policy")
    if not isinstance(mapping["unresolved"], list):
        raise SkillError("WEIGHT_MAP unresolved must be a list")
    if mapping["unresolved"]:
        raise SkillError("refusing to convert with non-empty WEIGHT_MAP unresolved entries")
    if not isinstance(mapping["ignore"], list):
        raise SkillError("WEIGHT_MAP ignore must be a list")
    ignore: list[dict[str, str]] = []
    ignored_names: set[str] = set()
    for index, item in enumerate(mapping["ignore"]):
        record = require_fields(
            item,
            {"source", "reason"},
            set(),
            label=f"WEIGHT_MAP ignore[{index}]",
        )
        source = name(record["source"], label=f"WEIGHT_MAP ignore[{index}].source")
        reason = record["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise SkillError(f"WEIGHT_MAP ignore[{index}].reason must be non-empty")
        if source in ignored_names:
            raise SkillError(f"duplicate ignored source tensor: {source}")
        ignored_names.add(source)
        ignore.append({"source": source, "reason": reason.strip()})
    if not isinstance(mapping["entries"], list) or not mapping["entries"]:
        raise SkillError("WEIGHT_MAP entries must be a non-empty list")
    entries: list[dict[str, Any]] = []
    mapped_sources: set[str] = set()
    mapped_targets: set[str] = set()
    for index, raw in enumerate(mapping["entries"]):
        label = f"WEIGHT_MAP entries[{index}]"
        if not isinstance(raw, dict):
            raise SkillError(f"{label} must be an object")
        if "sources" in raw:
            entry = require_fields(
                raw,
                {"sources", "target", "target_shape", "transforms"},
                {"dtype_policy"},
                label=label,
            )
            kind = "merge"
            if not isinstance(entry["sources"], list) or len(entry["sources"]) < 2:
                raise SkillError(f"{label}.sources must contain at least two source records")
            sources = []
            for source_index, item in enumerate(entry["sources"]):
                record = require_fields(
                    item,
                    {"source", "shape"},
                    set(),
                    label=f"{label}.sources[{source_index}]",
                )
                sources.append(
                    {
                        "source": name(
                            record["source"],
                            label=f"{label}.sources[{source_index}].source",
                        ),
                        "shape": shape(
                            record["shape"],
                            label=f"{label}.sources[{source_index}].shape",
                        ),
                    }
                )
            targets = [
                {
                    "target": name(entry["target"], label=f"{label}.target"),
                    "shape": shape(entry["target_shape"], label=f"{label}.target_shape"),
                }
            ]
        elif "targets" in raw:
            entry = require_fields(
                raw,
                {"source", "source_shape", "targets", "transforms"},
                {"dtype_policy"},
                label=label,
            )
            kind = "split"
            sources = [
                {
                    "source": name(entry["source"], label=f"{label}.source"),
                    "shape": shape(entry["source_shape"], label=f"{label}.source_shape"),
                }
            ]
            if not isinstance(entry["targets"], list) or len(entry["targets"]) < 2:
                raise SkillError(f"{label}.targets must contain at least two target records")
            targets = []
            for target_index, item in enumerate(entry["targets"]):
                record = require_fields(
                    item,
                    {"target", "shape"},
                    {"dtype_policy"},
                    label=f"{label}.targets[{target_index}]",
                )
                target_record: dict[str, Any] = {
                    "target": name(
                        record["target"],
                        label=f"{label}.targets[{target_index}].target",
                    ),
                    "shape": shape(
                        record["shape"],
                        label=f"{label}.targets[{target_index}].shape",
                    ),
                }
                if "dtype_policy" in record:
                    target_record["dtype_policy"] = policy(
                        record["dtype_policy"],
                        label=f"{label}.targets[{target_index}].dtype_policy",
                    )
                targets.append(target_record)
        else:
            entry = require_fields(
                raw,
                {"source", "target", "source_shape", "target_shape", "transforms"},
                {"dtype_policy"},
                label=label,
            )
            kind = "single"
            sources = [
                {
                    "source": name(entry["source"], label=f"{label}.source"),
                    "shape": shape(entry["source_shape"], label=f"{label}.source_shape"),
                }
            ]
            targets = [
                {
                    "target": name(entry["target"], label=f"{label}.target"),
                    "shape": shape(entry["target_shape"], label=f"{label}.target_shape"),
                }
            ]
        transforms = validate_transforms(entry["transforms"], label=f"{label}.transforms")
        merge_count = sum(transform["op"] == "merge" for transform in transforms)
        split_count = sum(transform["op"] == "split" for transform in transforms)
        if kind == "merge" and (merge_count != 1 or split_count):
            raise SkillError(f"{label} grouped sources require exactly one merge transform")
        if kind == "split" and (split_count != 1 or merge_count):
            raise SkillError(f"{label} grouped targets require exactly one split transform")
        if kind == "single" and (merge_count or split_count):
            raise SkillError(f"{label} single-source entry cannot contain split or merge")
        if "dtype_policy" in entry:
            policy(entry["dtype_policy"], label=f"{label}.dtype_policy")
        for source_record in sources:
            source_name = source_record["source"]
            if source_name in mapped_sources:
                raise SkillError(f"source tensor mapped more than once: {source_name}")
            if source_name in ignored_names:
                raise SkillError(f"source tensor is both mapped and ignored: {source_name}")
            mapped_sources.add(source_name)
        for target_record in targets:
            target_name = target_record["target"]
            if target_name in mapped_targets:
                raise SkillError(f"target tensor mapped more than once: {target_name}")
            mapped_targets.add(target_name)
        normalized = {
            "kind": kind,
            "sources": sources,
            "targets": targets,
            "transforms": transforms,
        }
        if "dtype_policy" in entry:
            normalized["dtype_policy"] = entry["dtype_policy"]
        entries.append(normalized)
    return {
        "schema_version": SCHEMA_VERSION,
        "global_policy": global_policy,
        "entries": entries,
        "ignore": ignore,
        "mapped_sources": mapped_sources,
        "mapped_targets": mapped_targets,
    }


def entry_output_shapes(entry: dict[str, Any]) -> list[list[int]]:
    transforms = entry["transforms"]
    if entry["kind"] == "single":
        current = list(entry["sources"][0]["shape"])
        for transform in transforms:
            current = apply_unary_shape(current, transform)
        return [current]
    if entry["kind"] == "split":
        current = list(entry["sources"][0]["shape"])
        outputs: list[list[int]] | None = None
        for transform in transforms:
            if transform["op"] == "split":
                axis = normalize_axis(transform["axis"], len(current))
                sizes = transform["sizes"]
                if len(sizes) != len(entry["targets"]) or sum(sizes) != current[axis]:
                    raise SkillError("split sizes must match the source axis and target count")
                outputs = []
                for size in sizes:
                    output_shape = list(current)
                    output_shape[axis] = size
                    outputs.append(output_shape)
            elif outputs is None:
                current = apply_unary_shape(current, transform)
            elif transform["op"] not in {"identity", "rename", "cast"}:
                raise SkillError("only identity, rename, or cast may follow split")
        if outputs is None:
            raise SkillError("split entry is missing its split transform")
        return outputs
    shapes = [list(item["shape"]) for item in entry["sources"]]
    merged: list[int] | None = None
    for transform in transforms:
        if transform["op"] == "merge":
            if any(len(source_shape) != len(shapes[0]) for source_shape in shapes):
                raise SkillError("merge sources must have the same rank")
            axis = normalize_axis(transform["axis"], len(shapes[0]))
            for source_shape in shapes[1:]:
                if any(
                    source_shape[index] != shapes[0][index]
                    for index in range(len(source_shape))
                    if index != axis
                ):
                    raise SkillError("merge source shapes differ outside the merge axis")
            merged = list(shapes[0])
            merged[axis] = sum(source_shape[axis] for source_shape in shapes)
        elif merged is None:
            if transform["op"] not in {"identity", "rename", "cast"}:
                raise SkillError("only identity, rename, or cast may precede merge")
        else:
            merged = apply_unary_shape(merged, transform)
    if merged is None:
        raise SkillError("merge entry is missing its merge transform")
    return [merged]


def validate_mapping_against_source(
    mapping: dict[str, Any],
    source_tensors: dict[str, dict[str, Any]],
    *,
    allow_unmapped: bool = False,
) -> None:
    source_names = set(source_tensors)
    missing = sorted(mapping["mapped_sources"] - source_names)
    ignored = {item["source"] for item in mapping["ignore"]}
    unknown_ignored = sorted(ignored - source_names)
    unmapped = sorted(source_names - mapping["mapped_sources"] - ignored)
    if missing:
        raise SkillError(f"mapped source tensors are missing: {missing}")
    if unknown_ignored:
        raise SkillError(f"ignored source tensors are missing: {unknown_ignored}")
    if unmapped and not allow_unmapped:
        raise SkillError(f"source tensors are unmapped and not ignored: {unmapped}")
    for entry in mapping["entries"]:
        for source_record in entry["sources"]:
            actual = source_tensors[source_record["source"]]["shape"]
            if actual != source_record["shape"]:
                raise SkillError(
                    f"declared source shape mismatch for {source_record['source']}: "
                    f"mapping {source_record['shape']}, source {actual}"
                )
        actual_targets = entry_output_shapes(entry)
        for target_record, actual in zip(entry["targets"], actual_targets, strict=True):
            if actual != target_record["shape"]:
                raise SkillError(
                    f"post-transform shape mismatch for {target_record['target']}: "
                    f"declared {target_record['shape']}, computed {actual}"
                )
