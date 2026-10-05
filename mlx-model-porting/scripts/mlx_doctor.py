#!/usr/bin/env python3
"""Static, offline MLX model/project diagnosis. Never executes target code."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shlex
import sys
import tempfile
from typing import Any

from _common import SkillError, dump_json, load_structured, redact_secret_text, run_process_capture
from scaffold_port import FAMILY_GENERATORS
from inference_engine_advisor import WORKLOADS, load_registry, shortlist

SCRIPT_DIR = Path(__file__).resolve().parent
BACKENDS = {"mlx-lm": "mlx-lm", "mlx-vlm": "mlx-vlm", "mlx-audio": "mlx-audio", "standalone-mlx": None}
PACKAGES = ("mlx", "mlx-lm", "mlx-vlm", "mlx-audio", "numpy", "torch", "transformers", "safetensors", "auto-mlx")
SCHEMA_VERSION = 1


class DoctorArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise SkillError(message)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = DoctorArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("path", nargs="?", help="local model or project; omit for host metadata only")
    parser.add_argument("--kind", choices=("auto", "model", "project"), default="auto")
    parser.add_argument("--model", help="optional local checkpoint when inspecting a project")
    parser.add_argument("--backend", choices=tuple(BACKENDS), help="check metadata for this target backend")
    parser.add_argument("--workload", choices=("interactive", "coding-agent", "multi-user", "multimodal", "audio", "embeddings", "distributed", "specialized"), help="attach an offline serving-engine research shortlist, not compatibility approval")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", help="create a new JSON report outside the inspected input")
    parser.add_argument("--markdown", help="create a new Markdown report outside the inspected input")
    parser.add_argument("--include-local-paths", action="store_true", help="include absolute paths and directly usable local command argv")
    parser.add_argument("--max-files", type=int, default=2000)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--memory-budget-gib", type=int)
    parser.add_argument("--require-runtime", action="store_true", help="exit 1 if Apple Silicon host/package metadata prerequisites are missing; does not probe Metal")
    args = parser.parse_args(argv)
    for key, low, high in (("max_files", 1, 10000), ("timeout_seconds", 1, 300), ("memory_budget_gib", 1, 1024)):
        value = getattr(args, key)
        if value is not None and not low <= value <= high:
            parser.error(f"--{key.replace(chr(95), chr(45))} must be an integer in [{low}, {high}]")
    return args


def local_path(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise SkillError(f"{label} must be an existing local path; network identifiers are not accepted")
    if path.is_symlink():
        raise SkillError(f"{label} must not be a symlink; choose and review its real local path")
    return path.resolve()


def kind_for(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if path.is_file():
        return "project" if path.suffix in {".py", ".swift", ".toml"} else "model"
    if (path / "config.json").is_file():
        return "model"
    with os.scandir(path) as entries:
        for index, entry in enumerate(entries):
            if index >= 2000:
                raise SkillError("automatic kind detection reached its file limit; specify --kind explicitly")
            if entry.name.endswith((".safetensors", ".gguf", ".onnx", ".keras", ".mlpackage")):
                return "model"
    return "project"


def host_metadata(*, include_paths: bool = False) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    system, machine = platform.system(), platform.machine()
    host: dict[str, Any] = {"system": system, "machine": machine, "python": platform.python_version(), "python_executable": str(Path(sys.executable).resolve()) if include_paths else Path(sys.executable).name, "apple_silicon": system == "Darwin" and machine == "arm64", "chip": None, "physical_memory_bytes": None, "packages": packages, "metal_execution": "not-probed", "check_scope": "metadata-only-no-ML-framework-import"}
    if system == "Darwin":
        for name, key in (("machdep.cpu.brand_string", "chip"), ("hw.memsize", "physical_memory_bytes")):
            try:
                completed, timed_out = run_process_capture(["/usr/sbin/sysctl", "-n", name], timeout=2)
                text = completed.stdout.strip()
                if not timed_out and completed.returncode == 0:
                    host[key] = int(text) if key == "physical_memory_bytes" else text[:256]
            except (OSError, ValueError, SkillError):
                pass  # An unavailable hardware field is unknown, never invented.
    return host


def inspect_input(path: Path, kind: str, args: argparse.Namespace, model: Path | None) -> dict[str, Any]:
    script = "inspect_model.py" if kind == "model" else "inspect_mlx_project.py"
    with tempfile.TemporaryDirectory(prefix="mlx-doctor-") as temporary:
        output = Path(temporary) / "inspection.json"
        command = [sys.executable, str(SCRIPT_DIR / script), str(path), "--output", str(output), "--max-files", str(args.max_files)]
        if args.include_local_paths:
            command.append("--include-local-paths")
        if model is not None:
            command.extend(("--model", str(model)))
        completed, timed_out = run_process_capture(command, timeout=args.timeout_seconds)
        if timed_out:
            raise SkillError("canonical inspector exceeded the requested timeout; no clean diagnosis is available")
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
            if not args.include_local_paths:
                detail = detail.replace(str(path), path.name)
                if model is not None:
                    detail = detail.replace(str(model), model.name)
            raise SkillError(f"{script} failed: {redact_secret_text(detail[-2000:])}")
        report = load_structured(output)
        if not isinstance(report, dict) or report.get("schema_version") != 1:
            raise SkillError("canonical inspector produced an invalid report")
        return report


def action(identifier: str, title: str, argv: list[str] | None = None, *, validation: str) -> dict[str, Any]:
    placeholders = sorted({placeholder for arg in (argv or []) for placeholder in re.findall(r"<[A-Z_]+>", arg)})
    return {"id": identifier, "title": title, "argv": argv, "command": shlex.join(argv) if argv else None, "requires_values": placeholders, "executed": False, "validation_gate": validation}


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    path = local_path(args.path, "input") if args.path else None
    model = local_path(args.model, "--model") if args.model else None
    kind = kind_for(path, args.kind) if path is not None else "host"
    if model is not None and kind != "project":
        raise SkillError("--model is only valid with a project input")
    if kind == "host" and args.kind != "auto":
        raise SkillError("--kind requires a local input path")
    host = host_metadata(include_paths=args.include_local_paths)
    raw = inspect_input(path, kind, args, model) if path is not None else {}
    blockers = list(raw.get("recommendation_blockers", []))
    warnings: list[str] = []
    findings: list[dict[str, Any]] = []
    families: list[str] = []
    runbooks: list[str] = []
    targets: list[str] = []
    weight_bytes: int | None = None
    static_identity = None
    project_health = None
    if kind == "model":
        families = raw.get("recommended_families") or ([raw["recommended_family"]] if raw.get("recommended_family") else [])
        runbooks = raw.get("recommended_runbooks") or ([raw["recommended_runbook"]] if raw.get("recommended_runbook") else [])
        for candidate in raw.get("architecture_candidates", []):
            if candidate.get("family") in families:
                targets.extend(candidate.get("targets", []))
        static_identity = raw.get("artifact_identity", {}).get("fingerprint")
        weight_bytes = raw.get("tensor_summary", {}).get("estimated_bytes")
        findings = raw.get("risks", [])
        license_summary = raw.get("license", {})
        if not blockers:
            warnings.append("Static intake passed; generator profile, complete weight mapping, source parity and task quality are still unverified.")
    else:
        license_summary = None
    if kind == "project":
        project_health = raw.get("health", {})
        findings = raw.get("code_surface", {}).get("risks", [])
        if project_health.get("status") in {"proof-gaps", "no-mlx-surface-detected"}:
            warnings.append(project_health.get("summary", "Project evidence is incomplete."))
        for entry in raw.get("model_inspections", []):
            if not entry.get("ok"):
                blockers.append("A project model inspection failed: " + str(entry.get("error", "unknown failure")))
            blockers.extend(entry.get("recommendation_blockers", []))
            if entry.get("recommended_family"):
                families.append(entry["recommended_family"])
            if entry.get("recommended_runbook"):
                runbooks.append(entry["recommended_runbook"])
            for candidate in entry.get("architecture_candidates", []):
                if candidate.get("family") == entry.get("recommended_family"):
                    targets.extend(candidate.get("targets", []))
    families, runbooks, targets = list(dict.fromkeys(families)), list(dict.fromkeys(runbooks)), list(dict.fromkeys(targets))
    backend = args.backend or next((target for target in targets if target in BACKENDS), None)
    runtime_blockers: list[str] = []
    if not host["apple_silicon"]:
        runtime_blockers.append("This workflow targets Apple Silicon execution; this host can still perform static inspection.")
    required = ["mlx"] + ([BACKENDS[backend]] if backend and BACKENDS[backend] else [])
    for package in required:
        if not host["packages"].get(package):
            runtime_blockers.append(f"Required runtime package is not installed in this interpreter: {package}")
    if args.backend and targets and args.backend not in targets:
        runtime_blockers.append("Requested backend is not a target hint for the selected architecture route; review the runbook.")
    budget = args.memory_budget_gib * 1024**3 if args.memory_budget_gib is not None else None
    if budget is not None and weight_bytes is not None and weight_bytes > budget:
        runtime_blockers.append("Declared tensor storage alone exceeds the requested memory budget; runtime fit is not established.")
    if len(families) == 1 and families[0] in FAMILY_GENERATORS and not blockers:
        scaffold = "family-available-profile-unverified"
    elif len(families) > 1 and not blockers:
        scaffold = "manual-hybrid-composition"
    else:
        scaffold = "blocked" if blockers else "runbook-guided-or-not-assessed"
    python = sys.executable if args.include_local_paths else "<PYTHON>"
    actions: list[dict[str, Any]] = []
    script_prefix = str(SCRIPT_DIR) if args.include_local_paths else "<SKILL_SCRIPTS>"
    def script(name: str) -> str:
        return str(Path(script_prefix) / name)
    target_arg = str(path) if args.include_local_paths and path else ("<PROJECT_PATH>" if kind == "project" else "<MODEL_PATH>")
    if blockers:
        actions.append(action("resolve-blockers", "Resolve every canonical inspection blocker before implementation or optimization.", validation="A fresh canonical inspection has no blockers; do not edit the report to turn it green."))
    if path is not None:
        inspector = "inspect_model.py" if kind == "model" else "inspect_mlx_project.py"
        command = [python, script(inspector), target_arg, "--output", "inspection.json"]
        if model is not None:
            command.extend(("--model", str(model) if args.include_local_paths else "<MODEL_PATH>"))
        actions.append(action("inspect", "Capture a fresh canonical inspection for downstream tools.", command, validation="The inspector re-reads local artifacts; no remote code or network flags are enabled."))
    if kind == "model":
        actions.append(action("plan", "Write a port plan, or a remediation-only plan while blocked.", [python, script("make_port_plan.py"), "inspection.json", "--artifact-root", target_arg, "--output", "PORT_PLAN.md"], validation="The planner independently re-inspects the artifact bytes."))
        if scaffold == "family-available-profile-unverified":
            actions.append(action("scaffold", "Run the generator's configuration gate; family availability is not profile acceptance.", [python, script("scaffold_port.py"), "inspection.json", "--artifact-root", target_arg, "--output", "mlx_port"], validation="The existing generator must accept the full configuration; then review the weight map and pass source parity."))
    if runtime_blockers:
        actions.append(action("runtime-prerequisites", "Resolve the reported host/backend prerequisites in a separate environment.", validation="Re-run Doctor in that interpreter, then run a real device smoke/parity check; package metadata alone is insufficient."))
    if not actions:
        actions.append(action("choose-input", "Supply a local model or project to inspect.", validation="A model-specific report requires actual local artifacts."))
    blocked = bool(blockers) or (args.require_runtime and bool(runtime_blockers))
    status = "blocked" if blockers else ("runtime-prerequisites-missing" if runtime_blockers else ("host-metadata-only" if kind == "host" else "static-review-complete"))
    serving = shortlist(load_registry(), args.workload, blockers=list(dict.fromkeys(blockers + runtime_blockers))) if getattr(args, "workload", None) else None
    return {"serving_candidates": serving, "schema_version": SCHEMA_VERSION, "tool": "mlx-doctor", "inspection_mode": "offline-static-no-target-code", "status": status, "ok": not blocked, "input": {"kind": kind, "path": str(path) if path and args.include_local_paths else (path.name if path else None), "model": str(model) if model and args.include_local_paths else (model.name if model else None)}, "host": host, "architecture": {"families": families, "runbooks": runbooks, "target_hints": targets, "routing_decision": raw.get("routing_decision"), "scaffold": scaffold, "native_execution_verified": False}, "artifact_fingerprint": static_identity, "license": license_summary, "project_health": project_health, "memory": {"declared_tensor_storage_bytes": weight_bytes, "requested_budget_bytes": budget, "estimated_runtime_bytes": None, "fit_verified": False, "scope": "tensor-storage-only; excludes caches, activations, runtime and other processes"}, "selected_backend": backend, "blockers": list(dict.fromkeys(blockers)), "runtime_blockers": runtime_blockers, "warnings": warnings, "findings": findings, "next_actions": actions, "limitations": ["Recognition, installed packages and static proof-file names do not establish native execution, source parity or task quality.", "Doctor never imports or executes the target project or ML framework and never enables network intake.", "Scaffold family availability is separate from configuration acceptance, weight coverage and end-to-end conversion.", "No model-fit, throughput, quality or optimization claim is promoted by this diagnostic.", "Commands are suggestions, not executed actions. Placeholder arguments must be supplied when local paths are omitted."]}


def make_markdown(report: dict[str, Any]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    architecture, host = report["architecture"], report["host"]
    lines = ["# MLX Doctor", "", f"**Status:** {cell(report['status'])}", "", "| Check | Finding |", "| --- | --- |", f"| Input | {cell(report['input']['kind'])}: {cell(report['input']['path'] or 'not supplied')} |", f"| Host | {cell(host['system'])} / {cell(host['machine'])}, Python {cell(host['python'])} |", f"| Architecture | {cell(', '.join(architecture['families']) or 'not established')} |", f"| Scaffold | {cell(architecture['scaffold'])} |", f"| Backend hint | {cell(report['selected_backend'] or 'not selected')} |", "| Native execution / parity | Not verified |", "| Runtime memory fit | Not verified |", ""]
    for title, values in (("Inspection blockers", report["blockers"]), ("Runtime prerequisites", report["runtime_blockers"]), ("Warnings", report["warnings"])):
        if values:
            lines.extend(["## " + title, ""])
            lines.extend("- " + cell(value) for value in values)
            lines.append("")
    serving = report.get("serving_candidates")
    if serving is not None:
        lines.extend(["## Serving engine research shortlist", "", "Workload fit only, not model compatibility or execution approval. Alphabetical, not performance ranked.", ""])
        for candidate in serving["candidates"]:
            lines.extend(["### " + cell(candidate["name"]), "", cell(candidate["rationale"]), "", "Gate: " + cell(candidate["eligibility"]) + ". " + cell(candidate["limitations"]), ""])
    lines.extend(["## Next steps", ""])
    for step in report["next_actions"]:
        lines.extend(["### " + cell(step["title"]), ""])
        if step["command"]:
            # Do not let target path contents close a Markdown code block.
            command = step["command"].replace("```", "` ` `")
            lines.extend(["```sh", command, "```", ""])
        lines.extend([cell(step["validation_gate"]), ""])
    lines.extend(["## Limits", "", *["- " + value for value in report["limitations"]], ""])
    return "\n".join(lines)


def publish_new(path_value: str, text: str) -> None:
    """Atomic create-only report publication through no-follow parent handles."""
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
        raise SkillError("atomic report files require POSIX no-follow handles; use stdout on this host")
    path = Path(path_value).expanduser().absolute()
    if ".." in path.parts:
        raise SkillError("report paths must not contain '..'")
    parent_fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    temporary = ".mlx-doctor-" + secrets.token_hex(12)
    created = False
    try:
        for component in path.parent.parts[1:]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd)
        created = True
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
    finally:
        if created:
            os.unlink(temporary, dir_fd=parent_fd)
        os.close(parent_fd)


def check_output_paths(args: argparse.Namespace) -> None:
    inputs = [local_path(value, "input") for value in (args.path, args.model) if value]
    outputs = [Path(value).expanduser().absolute() for value in (args.output, args.markdown) if value]
    if len(outputs) != len(set(outputs)):
        raise SkillError("JSON and Markdown outputs must be different new files")
    for output in outputs:
        if output.exists() or output.is_symlink():
            raise SkillError("report output already exists; choose a new file")
        for target in inputs:
            artifact_root = target if target.is_dir() else target.parent
            if output.resolve().is_relative_to(artifact_root):
                raise SkillError("report output must be outside the inspected model/project to keep it unchanged")


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        check_output_paths(args)
        report = diagnose(args)
        json_text = dump_json(report)
        safe_report = json.loads(json_text)
        markdown = make_markdown(safe_report)
        if args.output:
            publish_new(args.output, json_text)
        if args.markdown:
            publish_new(args.markdown, markdown)
        sys.stdout.write(json_text if args.format == "json" else markdown)
        return 0 if report["ok"] else 1
    except (SkillError, OSError, ValueError) as exc:
        diagnostic = {"schema_version": SCHEMA_VERSION, "tool": "mlx-doctor", "ok": False, "status": "inspection-error", "error": redact_secret_text(str(exc))}
        sys.stderr.write(json.dumps(diagnostic, ensure_ascii=True) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
