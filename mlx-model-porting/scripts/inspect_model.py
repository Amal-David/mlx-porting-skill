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


def recommendation_blockers(risks: list[dict[str, str]], tensor_count: int) -> list[str]:
    blockers: list[str] = []
    high_risks = {risk["type"] for risk in risks if risk.get("severity") == "high"}
    if tensor_count == 0:
        blockers.append("no safetensors were inspected; architecture routing is metadata-only")
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
        tensor_keys = [t["key"] for t in tensors]
        candidates = architecture_scores(registry, configs, tensor_keys)
        risks = detect_risks(root, configs, files)
        blockers = recommendation_blockers(risks, len(tensors))
        recommended = candidates[0] if candidates and not blockers else None
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
