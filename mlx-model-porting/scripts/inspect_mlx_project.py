#!/usr/bin/env python3
"""Inspect an existing local MLX project without executing project code."""
from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import (
    SkillError,
    atomic_write_text,
    bounded_files,
    dump_json,
    load_structured,
    redact_secret_text,
    run_process_capture,
    sha256_file,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TEXT_EXTENSIONS = {
    ".py", ".pyi", ".ipynb", ".md", ".txt", ".sh", ".toml", ".yaml", ".yml",
    ".json", ".swift", ".m", ".mm", ".cpp", ".cc", ".h", ".hpp",
}
MLX_PACKAGES = ("mlx", "mlx_lm", "mlx_vlm", "mlx_audio")
MAX_TEXT_BYTES = 2 * 1024 * 1024


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a local MLX project or already-ported MLX model without running project code."
    )
    parser.add_argument("project", help="Local MLX project directory or file")
    parser.add_argument("--model", action="append", default=[], help="Optional local model/checkpoint path to inspect statically")
    parser.add_argument("--inspection", action="append", default=[], help="Existing inspect_model.py JSON report to fold in")
    parser.add_argument("--output", help="Write JSON report")
    parser.add_argument("--markdown", help="Write Markdown report")
    parser.add_argument("--max-files", type=int, default=800, help="Maximum project files to inventory")
    parser.add_argument("--hash-small-files", action="store_true", help="Hash inspected files up to 10 MiB")
    parser.add_argument(
        "--include-local-paths",
        action="store_true",
        help="Include absolute local paths in the report (off by default for portability)",
    )
    raw_args = list(sys.argv[1:] if argv is None else argv)
    for index, value in enumerate(raw_args):
        if value in {"--model", "--inspection"} and (
            index + 1 >= len(raw_args) or raw_args[index + 1].startswith("-")
        ):
            parser.error(f"{value} requires a path value that does not begin with '-'")
    args = parser.parse_args(raw_args)
    if any(str(model).startswith("-") for model in args.model):
        parser.error("--model requires a path value that does not begin with '-'")
    return args


def safe_read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def inventory(root: Path, max_files: int, hash_small: bool) -> tuple[list[dict[str, Any]], bool]:
    paths, truncated = bounded_files(root, max_files)
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


def dotted_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return dotted_name(node.func)
    return None


def python_signals(text: str) -> tuple[set[str], set[str], list[str]]:
    imports: set[str] = set()
    calls: set[str] = set()
    classes: list[str] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return imports, calls, classes

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Call):
            name = dotted_name(node.func)
            if name:
                calls.add(name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                name = dotted_name(decorator)
                if name:
                    calls.add(name)
        elif isinstance(node, ast.ClassDef):
            bases = [dotted_name(base) for base in node.bases]
            if any(base and base.endswith("Module") for base in bases):
                classes.append(node.name)
    return imports, calls, classes


def package_from_import(name: str) -> str | None:
    for package in MLX_PACKAGES:
        if name == package or name.startswith(f"{package}."):
            return package
    return None


def has_any(text: str, calls: set[str], patterns: list[str]) -> bool:
    haystack = "\n".join(sorted(calls)) + "\n" + text
    return any(pattern in haystack for pattern in patterns)


def analyze_project(root: Path, files: list[dict[str, Any]]) -> dict[str, Any]:
    packages: set[str] = set()
    mlx_files: list[str] = []
    evidence: dict[str, list[dict[str, str]]] = {}
    risks: list[dict[str, str]] = []
    module_classes: list[dict[str, str]] = []
    feature_flags = {
        "raw_mlx_core": False,
        "mlx_lm_runtime": False,
        "mlx_vlm_runtime": False,
        "mlx_audio_runtime": False,
        "module_definitions": False,
        "explicit_eval": False,
        "compiled_regions": False,
        "fast_attention": False,
        "quantization": False,
        "kv_or_prompt_cache": False,
        "custom_metal_kernel": False,
        "benchmark_evidence": False,
        "parity_evidence": False,
        "generation_runtime": False,
    }

    def mark(feature: str, path: str, detail: str) -> None:
        feature_flags[feature] = True
        evidence.setdefault(feature, [])
        if len(evidence[feature]) < 8:
            evidence[feature].append({"path": path, "detail": detail})

    base = root if root.is_dir() else root.parent
    for record in files:
        suffix = record["suffix"]
        if suffix not in TEXT_EXTENSIONS:
            continue
        path = base / record["path"] if root.is_dir() else root
        text = safe_read_text(path)
        if text is None:
            continue
        rel = record["path"]
        imports, calls, classes = python_signals(text) if suffix in {".py", ".pyi"} else (set(), set(), [])
        for name in imports:
            package = package_from_import(name)
            if package:
                packages.add(package)
                if rel not in mlx_files:
                    mlx_files.append(rel)
        for cls in classes:
            module_classes.append({"path": rel, "class": cls})
            mark("module_definitions", rel, f"class {cls}")

        if "import mlx" in text or "from mlx" in text:
            packages.add("mlx")
            if rel not in mlx_files:
                mlx_files.append(rel)
        if has_any(text, calls, ["mlx.core", "mx.array", "mx.matmul", "mx.eval", "mx.compile"]):
            mark("raw_mlx_core", rel, "uses mlx.core primitives")
        if any(name == "mlx_lm" or name.startswith("mlx_lm.") for name in imports) or "mlx_lm" in text:
            packages.add("mlx_lm")
            mark("mlx_lm_runtime", rel, "uses mlx-lm")
        if any(name == "mlx_vlm" or name.startswith("mlx_vlm.") for name in imports) or "mlx_vlm" in text:
            packages.add("mlx_vlm")
            mark("mlx_vlm_runtime", rel, "uses mlx-vlm")
        if any(name == "mlx_audio" or name.startswith("mlx_audio.") for name in imports) or "mlx_audio" in text:
            packages.add("mlx_audio")
            mark("mlx_audio_runtime", rel, "uses mlx-audio")
        if has_any(text, calls, ["mx.eval", "mlx.core.eval"]):
            mark("explicit_eval", rel, "explicit eval boundary")
        if has_any(text, calls, ["mx.compile", "mlx.core.compile", "@mx.compile"]):
            mark("compiled_regions", rel, "compiled region")
        if has_any(text, calls, ["scaled_dot_product_attention", "fast.scaled_dot_product_attention"]):
            mark("fast_attention", rel, "fast attention call")
        if has_any(text, calls, ["quantize", "Quantized", "quantized"]):
            mark("quantization", rel, "quantization path")
        if re.search(r"\b(KVCache|kv_cache|prompt_cache|cache_prompt|cache\.make_prompt_cache)\b", text):
            mark("kv_or_prompt_cache", rel, "cache-related path")
        if re.search(r"\b(metal_kernel|custom_kernel|MetalKernel|MLX_USE_METAL)\b", text):
            mark("custom_metal_kernel", rel, "custom Metal/kernel signal")
        if re.search(r"\b(benchmark|tok/s|tokens/s|time\.perf_counter|peak memory|throughput)\b", text, re.IGNORECASE):
            mark("benchmark_evidence", rel, "benchmark or timing evidence")
        if re.search(r"\b(allclose|assert_close|compare_tensors|cosine|parity|oracle)\b", text, re.IGNORECASE):
            mark("parity_evidence", rel, "parity/oracle evidence")
        if re.search(r"\b(generate|stream_generate|generate_step|load\()\b", text):
            mark("generation_runtime", rel, "generation/load path")

        for risk_type, pattern, detail in (
            ("remote-code", r"trust_remote_code\s*=\s*True", "trust_remote_code=True appears in project code"),
            ("pickle-capable-load", r"\b(pickle\.load|torch\.load)\s*\(", "pickle-capable model loading appears in project code"),
            ("dynamic-exec", r"(?<!mx\.)\b(exec|eval)\s*\(", "dynamic exec/eval appears in project code"),
        ):
            if re.search(pattern, text):
                risks.append({"severity": "high", "type": risk_type, "path": rel, "detail": detail})

    return {
        "packages": sorted(packages),
        "mlx_files": mlx_files[:80],
        "module_classes": module_classes[:80],
        "features": feature_flags,
        "evidence": evidence,
        "risks": risks,
    }


def portable_local_reference(value: str | Path, include_local_paths: bool) -> str:
    path = Path(value).expanduser()
    return str(path.resolve()) if include_local_paths else (path.name or ".")


def summarize_model_inspection(
    report: dict[str, Any],
    source: str,
    include_local_paths: bool = False,
) -> dict[str, Any]:
    return {
        "source": portable_local_reference(source, include_local_paths),
        "ok": True,
        "inspection_mode": report.get("inspection_mode"),
        "recommended_family": report.get("recommended_family"),
        "recommended_runbook": report.get("recommended_runbook"),
        "recommendation_blockers": report.get("recommendation_blockers", []),
        "tensor_summary": {
            "count": (report.get("tensor_summary") or {}).get("count", 0),
            "parameters": (report.get("tensor_summary") or {}).get("parameters", 0),
            "estimated_bytes": (report.get("tensor_summary") or {}).get("estimated_bytes", 0),
        },
        "architecture_candidates": (report.get("architecture_candidates") or [])[:3],
        "risk_count": len(report.get("risks", [])),
        "source_format_summary": report.get("source_format_summary", {}),
    }


def run_model_inspection(model: str, include_local_paths: bool = False) -> dict[str, Any]:
    if not isinstance(model, str) or not model or model.startswith("-"):
        raise SkillError("model path must be non-empty and must not begin with '-'")
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "inspection.json"
        command = [sys.executable, str(SCRIPT_DIR / "inspect_model.py"), model, "--output", str(output)]
        completed, timed_out = run_process_capture(command, timeout=120.0)
        if timed_out:
            return {
                "source": portable_local_reference(model, include_local_paths),
                "ok": False,
                "error": "inspect_model.py timed out after 120s",
            }
        if completed.returncode != 0:
            return {
                "source": portable_local_reference(model, include_local_paths),
                "ok": False,
                "error": redact_secret_text(
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or f"inspect_model.py exited {completed.returncode}"
                ),
            }
        if not output.is_file():
            raise SkillError("inspect_model.py exited successfully without writing its inspection output")
        try:
            report = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SkillError(f"inspect_model.py wrote an invalid inspection report: {exc}") from exc
        if not isinstance(report, dict):
            raise SkillError("inspect_model.py inspection output must be a JSON object")
        return summarize_model_inspection(report, model, include_local_paths)


def build_health(
    surface: dict[str, Any],
    models: list[dict[str, Any]],
    *,
    inventory_truncated: bool = False,
) -> dict[str, Any]:
    features = surface["features"]
    risks = surface["risks"]
    strengths: list[str] = []
    if surface["packages"]:
        strengths.append("MLX packages detected: " + ", ".join(surface["packages"]))
    if features["parity_evidence"]:
        strengths.append("Parity or oracle evidence appears in project files.")
    if features["benchmark_evidence"]:
        strengths.append("Benchmark or timing evidence appears in project files.")
    if features["compiled_regions"] or features["quantization"] or features["kv_or_prompt_cache"]:
        strengths.append("At least one MLX optimization path is already present.")
    if any(model.get("recommended_family") for model in models if model.get("ok")):
        families = sorted({str(model["recommended_family"]) for model in models if model.get("recommended_family")})
        strengths.append("Model inspection routed: " + ", ".join(families))

    blockers: list[str] = []
    if inventory_truncated:
        blockers.append(
            "Project inventory was truncated; clean status and optimization recommendations are blocked until a complete inventory is inspected."
        )
        status = "inventory-truncated"
        summary = "Static inspection reached the file limit, so absence of risks or proof gaps cannot be established."
    elif risks:
        status = "review-needed"
        summary = "Static inspection found high-signal risk flags before optimization advice should be trusted."
        blockers.append("Resolve the reported high-signal project risks before trusting optimization advice.")
    elif not surface["packages"]:
        status = "no-mlx-surface-detected"
        summary = "No MLX imports or package usage were detected in the inspected project files."
    elif not features["parity_evidence"] or not features["benchmark_evidence"]:
        status = "proof-gaps"
        summary = "The project appears to run MLX, but parity and/or benchmark evidence is missing from the static surface."
    else:
        status = "looks-good"
        summary = "The project has MLX code plus parity and benchmark evidence; inspect model-specific gates before changing performance paths."
    return {
        "status": status,
        "summary": summary,
        "strengths": strengths,
        "recommendation_blockers": blockers,
    }


def build_opportunities(
    surface: dict[str, Any],
    models: list[dict[str, Any]],
    *,
    inventory_truncated: bool = False,
) -> list[dict[str, str]]:
    features = surface["features"]
    opportunities: list[dict[str, str]] = []

    def add(identifier: str, title: str, detail: str, gate: str) -> None:
        opportunities.append({"id": identifier, "title": title, "detail": detail, "validation_gate": gate})

    if inventory_truncated:
        add(
            "complete-project-inventory",
            "Complete the project inventory",
            "The configured file limit truncated static inspection, so clean status and optimization recommendations are held.",
            "Rerun with a sufficient --max-files value and review every included file before clearing this blocker.",
        )

    if not features["parity_evidence"]:
        add(
            "add-source-oracle",
            "Add a source-oracle/parity harness",
            "No allclose, compare_tensors, oracle, or parity signal was found. Treat the running project as unproven until selected tensors or task outputs match a source.",
            "Capture deterministic source outputs and compare MLX tensors before changing implementation.",
        )
    if not features["benchmark_evidence"]:
        add(
            "add-benchmark-receipt",
            "Add benchmark metadata",
            "No benchmark, throughput, tok/s, or timing signal was found. Performance advice should remain qualitative.",
            "Record hardware, software versions, workload, baseline, quality gate, and rollback condition.",
        )
    if features["raw_mlx_core"] and not features["explicit_eval"]:
        add(
            "review-eval-boundaries",
            "Review MLX lazy-evaluation boundaries",
            "The code uses MLX primitives but no explicit mx.eval signal was found.",
            "Profile first, then add eval boundaries only where they reduce synchronization or measurement ambiguity.",
        )
    if features["generation_runtime"] and not features["kv_or_prompt_cache"]:
        add(
            "consider-cache-policy",
            "Consider prompt/KV cache policy",
            "Generation code was detected without an obvious prompt or KV cache path.",
            "Only add caching after logits/cache parity and a workload with repeated context proves benefit.",
        )
    if features["generation_runtime"] and not features["quantization"]:
        add(
            "consider-weight-quantization",
            "Consider weight quantization",
            "Generation code was detected without an obvious quantization path.",
            "Measure quality and decode/memory impact against the current baseline on the target Mac.",
        )
    blocked_models = [model for model in models if model.get("recommendation_blockers")]
    if blocked_models:
        add(
            "resolve-model-intake-blockers",
            "Resolve model intake blockers",
            "At least one inspected model has blockers; hold optimization recommendations until those are cleared.",
            "Review the inspect_model.py blockers and add missing config, provenance, safe weights, or parity fixtures.",
        )
    return opportunities


def build_contribution_candidates(surface: dict[str, Any], models: list[dict[str, Any]]) -> list[dict[str, str]]:
    features = surface["features"]
    candidates: list[dict[str, str]] = []

    def add(identifier: str, title: str, detail: str, evidence_needed: str) -> None:
        candidates.append({"id": identifier, "title": title, "detail": detail, "evidence_needed": evidence_needed})

    if features["custom_metal_kernel"]:
        add(
            "custom-metal-kernel-pattern",
            "Contribute a custom kernel pattern",
            "The project appears to contain a custom Metal/MLX kernel path. If it is reproducible, it may extend the optimization corpus.",
            "Minimal source fixture, MLX fallback, parity test, benchmark receipt, supported shapes/dtypes, and rollback condition.",
        )
    if features["compiled_regions"] and features["benchmark_evidence"] and features["parity_evidence"]:
        add(
            "compiled-region-receipt",
            "Contribute a compile-region receipt",
            "The project appears to combine compiled MLX regions with parity and benchmark evidence.",
            "Before/after benchmark, compile/cache behavior, quality gate, and a short note on why the region is stable.",
        )
    if features["mlx_audio_runtime"] or features["mlx_vlm_runtime"]:
        add(
            "modality-port-receipt",
            "Contribute a modality-specific inspection note",
            "The project uses MLX audio or VLM packages, where preprocessing and quality gates often contain reusable lessons.",
            "Processor config, task-quality metric, failure mode, and the specific runbook section affected.",
        )
    if any(model.get("ok") and not model.get("recommended_family") for model in models):
        add(
            "new-routing-signal",
            "Contribute a new architecture routing signal",
            "A model inspection completed but did not route to a known family.",
            "Pinned config, representative tensor keys, safe source reference, and the expected runbook or new-family proposal.",
        )
    if not candidates:
        add(
            "no-new-learning-yet",
            "No unique contribution candidate yet",
            "The inspected surface did not expose a novel kernel, routed gap, or measured optimization receipt.",
            "Add a small reproducible fixture plus parity and benchmark receipts before proposing new knowledge.",
        )
    return candidates


def build_next_actions(args: argparse.Namespace, project: Path) -> list[str]:
    include_local_paths = bool(getattr(args, "include_local_paths", False))
    inspect_command = [
        "python3",
        "mlx-model-porting/scripts/inspect_mlx_project.py",
        portable_local_reference(project, include_local_paths),
    ]
    for model in args.model:
        inspect_command.extend([
            "--model",
            portable_local_reference(model, include_local_paths),
        ])
    if include_local_paths:
        inspect_command.append("--include-local-paths")
    inspect_command.extend(["--markdown", "MLX_INSPECTION.md"])
    return [
        shlex.join(inspect_command),
        "Resolve any intake blockers before changing code.",
        "If a candidate is genuinely new, add a minimal fixture, parity receipt, benchmark metadata, and a proposed runbook change.",
    ]


def make_markdown(report: dict[str, Any]) -> str:
    health = report["health"]
    surface = report["code_surface"]
    lines = [
        "# MLX project inspection",
        "",
        f"- Project: `{report['project']['path']}`",
        f"- Status: `{health['status']}`",
        f"- Summary: {health['summary']}",
        f"- MLX packages: {', '.join(surface['packages']) or 'none detected'}",
        "",
    ]
    if report.get("recommendation_blockers"):
        lines += ["## Recommendation blockers", ""]
        lines.extend(f"- {item}" for item in report["recommendation_blockers"])
        lines.append("")
    lines += ["## What looks good", ""]
    if health["strengths"]:
        lines.extend(f"- {item}" for item in health["strengths"])
    else:
        lines.append("- No positive MLX evidence beyond the file inventory yet.")

    lines += ["", "## Potential improvements", ""]
    for item in report["improvement_opportunities"]:
        lines.append(f"- **{item['title']}** (`{item['id']}`): {item['detail']} Gate: {item['validation_gate']}")
    if not report["improvement_opportunities"]:
        lines.append("- No immediate static improvement gaps found. Keep measuring before changing performance paths.")

    lines += ["", "## Contribution candidates", ""]
    for item in report["contribution_candidates"]:
        lines.append(f"- **{item['title']}** (`{item['id']}`): {item['detail']} Evidence needed: {item['evidence_needed']}")

    if report["model_inspections"]:
        lines += ["", "## Model inspections", ""]
        for model in report["model_inspections"]:
            if model.get("ok"):
                lines.append(
                    f"- `{model['source']}`: family `{model.get('recommended_family')}`; "
                    f"blockers: {', '.join(model.get('recommendation_blockers') or []) or 'none'}"
                )
            else:
                lines.append(f"- `{model['source']}`: inspection failed: {model.get('error')}")

    lines += ["", "## Next actions", ""]
    lines.extend(f"{index}. {action}" for index, action in enumerate(report["next_actions"], start=1))
    lines += ["", "## Limitations", ""]
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        project = Path(args.project).expanduser()
        if not project.exists():
            raise SkillError(f"Project path does not exist: {project}")
        project = project.resolve()
        files, truncated = inventory(project, args.max_files, args.hash_small_files)
        surface = analyze_project(project, files)
        model_inspections = [
            run_model_inspection(model, args.include_local_paths) for model in args.model
        ]
        for inspection in args.inspection:
            loaded = load_structured(inspection)
            model_inspections.append(
                summarize_model_inspection(loaded, inspection, args.include_local_paths)
            )
        health = build_health(surface, model_inspections, inventory_truncated=truncated)
        report = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "inspection_mode": "local-mlx-project-inspector",
            "project": {
                "path": portable_local_reference(project, args.include_local_paths),
                "file_count": len(files),
                "inventory_truncated": truncated,
                "extensions": dict(sorted(_counts(record["suffix"] or "<none>" for record in files).items())),
            },
            "code_surface": surface,
            "model_inspections": model_inspections,
            "health": health,
            "recommendation_blockers": health["recommendation_blockers"],
            "improvement_opportunities": build_opportunities(
                surface,
                model_inspections,
                inventory_truncated=truncated,
            ),
            "contribution_candidates": build_contribution_candidates(surface, model_inspections),
            "next_actions": build_next_actions(args, project),
            "limitations": [
                "This inspector is static-first and does not prove the project is currently running.",
                "It does not execute local project code, import local modules, or validate numerical correctness by itself.",
                "Optimization suggestions are hypotheses until parity, quality, benchmark metadata, and rollback evidence exist.",
                "Contribution candidates are review leads; they must not auto-modify recommendation assets.",
                "A truncated inventory blocks clean status because uninspected files may contain risks or missing proof.",
            ],
        }
        text = dump_json(report, args.output)
        if args.output is None:
            sys.stdout.write(text)
        if args.markdown:
            atomic_write_text(args.markdown, make_markdown(report))
        return 0
    except SkillError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
