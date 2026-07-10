#!/usr/bin/env python3
"""Generate a runbook-grounded MLX port plan from an inspection report."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from _common import SkillError, load_structured, run_process_capture
from recommend_optimizations import (
    resolve_route_families,
    trusted_inspection_sha256,
    validate_trusted_inspection,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_BOUND_INSPECTION_FIELDS = (
    "artifact_identity",
    "configs",
    "file_summary",
    "tensor_summary",
    "tensors",
    "source_format_summary",
    "architecture_candidates",
    "architecture_profile",
    "architecture_traits",
    "routing_decision",
    "recommended_family",
    "recommended_runbook",
    "recommended_families",
    "recommended_runbooks",
    "recommendation_blockers",
    "license",
    "risks",
)

ADVISOR_BUCKET_TITLES = {
    "validated-locally": "Validated locally",
    "validated-source-theory": "Validated by source or theory",
    "benchmark-required": "Benchmark required",
    "experimental-approach": "Experimental approaches",
}
ALL_ADVISOR_BUCKETS = {*ADVISOR_BUCKET_TITLES, "rejected-do-not-use"}
TRUSTED_RECOMMENDATION_FIELDS = {
    "schema_version",
    "ok",
    "inspection_sha256",
    "family",
    "families",
    "objectives",
    "target_profile_status",
    "target_profile",
    "receipt_assessments_status",
    "verified_receipt_count",
    "effective_claim_catalog_status",
    "candidate_limit",
    "match_context",
    "blocked",
    "blocked_advice_visible",
    "blockers",
    "advisor_buckets",
    "ready_candidates",
    "research_candidates",
    "held_candidates",
    "notable_exclusions",
    "guidance_reviewed",
    "taxonomy_reviewed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PORT_PLAN.md from inspect_model.py JSON")
    parser.add_argument("inspection")
    parser.add_argument(
        "--recommendations",
        help="Canonical schema-2 output from recommend_optimizations.py; no advice is embedded without it",
    )
    parser.add_argument("--output", default="PORT_PLAN.md")
    parser.add_argument("--family", help="Override detected architecture family")
    parser.add_argument(
        "--artifact-root",
        help=(
            "Required for an actionable plan: rerun static inspection against this local artifact "
            "and verify the bound inspection before emitting implementation steps"
        ),
    )
    return parser.parse_args()


def verify_inspection_against_artifact(
    inspection: dict[str, Any],
    artifact_root: str | None,
) -> None:
    """Reinspect local bytes and fail closed when safety-critical derivation drifts."""
    if artifact_root is None:
        raise SkillError(
            "Actionable port plans require --artifact-root so the inspection can be reverified"
        )
    root = Path(artifact_root).expanduser().resolve()
    file_summary = inspection.get("file_summary", {})
    records = file_summary.get("files", []) if isinstance(file_summary, dict) else []
    max_files = max(5000, len(records) + 1 if isinstance(records, list) else 5000)
    hash_small_files = isinstance(records, list) and any(
        isinstance(record, dict) and "sha256" in record for record in records
    )
    with tempfile.TemporaryDirectory(prefix="mlx-port-plan-verify-") as raw_tmp:
        verified_path = Path(raw_tmp) / "inspection.json"
        command = [
            sys.executable,
            str(SCRIPT_DIR / "inspect_model.py"),
            str(root),
            "--output",
            str(verified_path),
            "--max-files",
            str(max_files),
        ]
        if hash_small_files:
            command.append("--hash-small-files")
        result, timed_out = run_process_capture(command, timeout=120)
        if timed_out:
            raise SkillError("Artifact reinspection timed out after 120 seconds")
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown inspection failure"
            raise SkillError(f"Artifact reinspection failed: {detail}")
        verified = validate_trusted_inspection(load_structured(verified_path))

    mismatches = [
        field
        for field in ARTIFACT_BOUND_INSPECTION_FIELDS
        if inspection.get(field) != verified.get(field)
    ]
    if mismatches:
        raise SkillError(
            "Inspection does not match --artifact-root for safety-critical fields: "
            + ", ".join(mismatches)
        )


def route_metadata(
    inspection: dict[str, Any],
    families: list[str],
) -> tuple[list[str], list[str]]:
    """Return family-aligned runbooks and preserve traits for the same inspected route."""
    runbooks_by_family: dict[str, str] = {}
    for family, runbook in zip(
        inspection.get("recommended_families", []),
        inspection.get("recommended_runbooks", []),
    ):
        if isinstance(family, str) and isinstance(runbook, str):
            runbooks_by_family[family] = runbook
    profile = inspection.get("architecture_profile")
    if isinstance(profile, dict):
        for component in profile.get("components", []):
            if not isinstance(component, dict):
                continue
            family = component.get("family")
            runbook = component.get("runbook")
            if isinstance(family, str) and isinstance(runbook, str):
                runbooks_by_family[family] = runbook
    for candidate in inspection.get("architecture_candidates", []):
        if not isinstance(candidate, dict):
            continue
        family = candidate.get("family")
        runbook = candidate.get("runbook")
        if isinstance(family, str) and isinstance(runbook, str):
            runbooks_by_family.setdefault(family, runbook)

    runbooks = [runbooks_by_family.get(family, "manual selection required") for family in families]
    inspected_route = [str(value) for value in inspection.get("recommended_families", [])]
    if not inspected_route and isinstance(profile, dict):
        inspected_route = [
            str(component.get("family"))
            for component in profile.get("components", [])
            if isinstance(component, dict) and isinstance(component.get("family"), str)
        ]
    traits = (
        [str(value) for value in inspection.get("architecture_traits", [])]
        if len(families) == len(inspected_route) and set(families) == set(inspected_route)
        else []
    )
    return runbooks, traits


def load_recommendation_report(
    path: str | None,
    inspection: dict[str, Any],
    families: list[str],
) -> dict[str, Any] | None:
    if path is None:
        return None
    report = load_structured(path)
    if not isinstance(report, dict) or report.get("schema_version") != 2 or report.get("ok") is not True:
        raise SkillError("Recommendation report must be successful schema_version 2 output")
    missing_fields = sorted(TRUSTED_RECOMMENDATION_FIELDS - set(report))
    if missing_fields:
        raise SkillError(
            "Recommendation report is missing canonical fields: " + ", ".join(missing_fields)
        )
    if report.get("effective_claim_catalog_status") != "current":
        raise SkillError("Recommendation report must use the current effective-claim catalog")
    if report.get("inspection_sha256") != trusted_inspection_sha256(inspection):
        raise SkillError("Recommendation report is not bound to this exact inspection")
    if report.get("families") != families:
        raise SkillError("Recommendation report families do not match the port-plan route")
    if not families or report.get("family") != families[0]:
        raise SkillError("Recommendation report primary family does not match the port-plan route")
    for field in (
        "objectives",
        "ready_candidates",
        "research_candidates",
        "held_candidates",
        "notable_exclusions",
    ):
        if not isinstance(report.get(field), list):
            raise SkillError(f"Recommendation report {field} must be a list")
    if not all(isinstance(value, str) for value in report["objectives"]):
        raise SkillError("Recommendation report objectives must contain only strings")
    if report.get("target_profile_status") not in {"missing", "provided"}:
        raise SkillError("Recommendation report target_profile_status is invalid")
    if report.get("receipt_assessments_status") not in {"checked-in", "provided"}:
        raise SkillError("Recommendation report receipt assessments were not verified")
    receipt_count = report.get("verified_receipt_count")
    if isinstance(receipt_count, bool) or not isinstance(receipt_count, int) or receipt_count < 0:
        raise SkillError("Recommendation report verified_receipt_count is invalid")
    candidate_limit = report.get("candidate_limit")
    if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int) or candidate_limit < 1:
        raise SkillError("Recommendation report candidate_limit is invalid")
    match_context = report.get("match_context")
    if not isinstance(match_context, dict):
        raise SkillError("Recommendation report match_context must be an object")
    if match_context.get("families") != sorted(families):
        raise SkillError("Recommendation report match_context does not match the port-plan route")
    blockers = inspection["recommendation_blockers"]
    if not isinstance(report.get("blocked"), bool) or not isinstance(
        report.get("blocked_advice_visible"),
        bool,
    ):
        raise SkillError("Recommendation report blocker flags must be boolean")
    if report.get("blocked") is not bool(blockers) or report.get("blockers") != blockers:
        raise SkillError("Recommendation report blocker state does not match the inspection")
    buckets = report.get("advisor_buckets")
    if not isinstance(buckets, dict):
        raise SkillError("Recommendation report advisor_buckets must be an object")
    seen_method_ids: set[str] = set()
    for bucket_id in ALL_ADVISOR_BUCKETS:
        if not isinstance(buckets.get(bucket_id), list):
            raise SkillError(f"Recommendation report is missing advisor bucket {bucket_id}")
        for index, item in enumerate(buckets[bucket_id]):
            if not isinstance(item, dict):
                raise SkillError(
                    f"Recommendation report advisor bucket {bucket_id}[{index}] must be an object"
                )
            required = {
                "id", "status", "advisor_bucket", "expected_effect", "validation_gates",
                "requires_user_opt_in", "opt_in_prompt", "execution_allowed",
            }
            missing = sorted(required - set(item))
            if missing:
                raise SkillError(
                    f"Recommendation report candidate in {bucket_id} is missing fields: "
                    + ", ".join(missing)
                )
            method_id = item.get("id")
            if not isinstance(method_id, str) or not method_id:
                raise SkillError(f"Recommendation report candidate in {bucket_id} has an invalid id")
            if method_id in seen_method_ids:
                raise SkillError(f"Recommendation report duplicates candidate {method_id!r}")
            seen_method_ids.add(method_id)
            if item.get("advisor_bucket") != bucket_id:
                raise SkillError(
                    f"Recommendation report candidate {method_id!r} is in the wrong advisor bucket"
                )
            if not isinstance(item.get("validation_gates"), list) or not all(
                isinstance(gate, str) for gate in item["validation_gates"]
            ):
                raise SkillError(
                    f"Recommendation report candidate {method_id!r} has invalid validation gates"
                )
            if not isinstance(item.get("requires_user_opt_in"), bool) or not isinstance(
                item.get("execution_allowed"),
                bool,
            ):
                raise SkillError(
                    f"Recommendation report candidate {method_id!r} has invalid execution flags"
                )
            if bucket_id == "experimental-approach" and (
                item["requires_user_opt_in"] is not True
                or item["execution_allowed"] is not False
                or not isinstance(item.get("opt_in_prompt"), str)
                or not item["opt_in_prompt"]
            ):
                raise SkillError(
                    f"Recommendation report experimental candidate {method_id!r} violates opt-in policy"
                )
    if report.get("receipt_assessments_status") != "checked-in":
        raise SkillError(
            "Port plans accept only recommendations recomputable from checked-in receipt assessments"
        )

    with tempfile.TemporaryDirectory(prefix="mlx-port-plan-recommendations-") as raw_tmp:
        tmp = Path(raw_tmp)
        canonical_inspection = tmp / "inspection.json"
        canonical_output = tmp / "recommendations.json"
        canonical_inspection.write_text(
            json.dumps(inspection, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(SCRIPT_DIR / "recommend_optimizations.py"),
            str(canonical_inspection),
            "--family",
            str(report["family"]),
            "--limit",
            str(candidate_limit),
            "--output",
            str(canonical_output),
        ]
        for objective in report["objectives"]:
            command.extend(["--objective", objective])
        if report["target_profile_status"] == "provided":
            if not isinstance(report.get("target_profile"), dict):
                raise SkillError("Recommendation report provided TargetProfile must be an object")
            target_profile = tmp / "target-profile.json"
            target_profile.write_text(
                json.dumps(report["target_profile"], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            command.extend(["--target-profile", str(target_profile)])
        if report["blocked_advice_visible"]:
            command.append("--allow-blocked")
        result, timed_out = run_process_capture(command, timeout=60)
        if timed_out:
            raise SkillError("Canonical recommendation recomputation timed out after 60 seconds")
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown recommender failure"
            raise SkillError(f"Canonical recommendation recomputation failed: {detail}")
        canonical_report = load_structured(canonical_output)
    if report != canonical_report:
        raise SkillError(
            "Recommendation report does not exactly match canonical recomputation; regenerate it"
        )
    return report


def recommendation_lines(report: dict[str, Any] | None) -> list[str]:
    lines = ["## Optimization advice", ""]
    if report is None:
        return [
            *lines,
            "No optimization candidates are embedded without canonical recommender output.",
            "Generate it, then rebuild this plan:",
            "",
            "```bash",
            "python3 scripts/recommend_optimizations.py inspection.json --output recommendations.json",
            "python3 scripts/make_port_plan.py inspection.json --artifact-root /path/to/model "
            "--recommendations recommendations.json --output PORT_PLAN.md",
            "```",
        ]
    stack = report.get("planning_stack")
    lines.extend(["- Canonical source: `recommend_optimizations.py` schema-2 output", ""])
    if isinstance(stack, dict) and stack.get("id"):
        lines.extend([f"- Canonical planning stack: `{stack['id']}`", ""])
    buckets = report["advisor_buckets"]
    for bucket_id, title in ADVISOR_BUCKET_TITLES.items():
        items = buckets[bucket_id]
        lines.extend([f"### {title}", ""])
        if not items:
            lines.append("None.")
            lines.append("")
            continue
        for item in items:
            gates = item.get("validation_gates")
            first_gate = gates[0] if isinstance(gates, list) and gates else "No validation gate supplied"
            line = (
                f"- `{item.get('id')}` (`{item.get('status')}`): {item.get('expected_effect', '')} "
                f"Gate: {first_gate}"
            )
            if item.get("requires_user_opt_in"):
                line += (
                    f" Opt-in required: {item.get('opt_in_prompt')} "
                    "Execution remains held until explicit consent."
                )
            lines.append(line)
        lines.append("")
    return lines


def write_blocked_plan(
    args: argparse.Namespace,
    inspection: dict[str, Any],
    family: str | None,
) -> int:
    blockers = inspection["recommendation_blockers"]
    source = inspection["source"]
    artifact_identity = inspection["artifact_identity"]
    license_info = inspection["license"]
    lines = [
        "# MLX port plan — blocked",
        "",
        "- Status: **non-actionable remediation only**",
        f"- Source: `{source.get('input')}`",
        f"- Pinned revision: `{source.get('revision') or 'missing'}`",
        f"- Artifact identity: `{artifact_identity.get('fingerprint') or 'incomplete'}`",
        f"- Architecture route: `{family or 'unresolved'}`",
        "- License evidence: "
        + str(license_info.get("declared") or license_info.get("license_files") or "missing"),
        "",
        "A family override does not clear intake blockers. Do not implement, convert weights, benchmark, or optimize from this report.",
        "",
        "## Blocking findings",
        "",
        *(f"- {blocker}" for blocker in blockers),
        "",
        "## Remediation only",
        "",
        "1. Repair every integrity, provenance, license, safety, and routing issue listed above.",
        "2. Re-run `inspect_model.py`; do not hand-edit the inspection report.",
        "3. Continue only when the regenerated report has no recommendation blockers.",
        "4. Generate canonical optimization advice with `recommend_optimizations.py` after intake is clean.",
    ]
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)
    return 0


def main() -> int:
    args = parse_args()
    try:
        inspection = validate_trusted_inspection(load_structured(args.inspection))
        candidates = inspection.get("architecture_candidates", [])
        families = resolve_route_families(
            inspection,
            str(args.family) if args.family else None,
        )
        family = families[0] if families else inspection.get("routing_decision", {}).get("winner_family")
        blockers = inspection["recommendation_blockers"]
        if blockers:
            if args.recommendations:
                load_recommendation_report(args.recommendations, inspection, families)
            return write_blocked_plan(args, inspection, str(family) if family else None)
        verify_inspection_against_artifact(inspection, args.artifact_root)
        if not family:
            raise SkillError("No detected family; pass --family after manual architecture review")
        selected = next((candidate for candidate in candidates if candidate.get("family") == family), None)
        if selected is None:
            raise SkillError(f"Resolved family {family!r} is not an inspected architecture candidate")
        runbooks, traits = route_metadata(inspection, families)
        secondary_families = [value for value in families if value != family]
        candidates_by_family = {
            str(candidate.get("family")): candidate
            for candidate in candidates
            if candidate.get("family")
        }
        state_contracts = [
            (
                value,
                str(candidates_by_family.get(value, {}).get("state") or "must be documented"),
            )
            for value in families
        ]
        recommendations = load_recommendation_report(args.recommendations, inspection, families)
        risks = inspection.get("risks", [])
        license_info = inspection.get("license", {})
        source = inspection.get("source", {})
        artifact_identity = inspection.get("artifact_identity", {})
        tensors = inspection.get("tensor_summary", {})

        lines = [
            "# MLX port plan",
            "",
            "## Source and target",
            "",
            f"- Source: `{source.get('input', inspection.get('local_path', 'unknown'))}`",
            f"- Requested/pinned revision: `{source.get('revision') or 'REQUIRED BEFORE IMPLEMENTATION'}`",
            f"- Artifact identity: `{artifact_identity.get('fingerprint')}`",
            f"- Local inspection path: `{inspection.get('local_path', '')}`",
            f"- Static tensor count: {tensors.get('count', 0)}",
            f"- Static parameter count: {int(tensors.get('parameters', 0)):,}",
            f"- License evidence: {license_info.get('declared') or license_info.get('license_files') or 'MISSING — STOP FOR REVIEW'}",
            "- Target package: decide among MLX-LM, MLX-VLM, MLX-Audio, or standalone MLX after source comparison.",
            "- Target Mac/workload: **fill in chip, memory, OS, MLX version, input lengths, batch/concurrency, latency/memory/quality objective**.",
            "",
            "## Architecture route",
            "",
            f"- Primary family: `{family}`",
            "- Secondary families: "
            + (", ".join(f"`{value}`" for value in secondary_families) if secondary_families else "none"),
            "- Required runbooks: " + ", ".join(f"`{value}`" for value in runbooks),
            "- Architecture traits: "
            + (", ".join(f"`{value}`" for value in traits) if traits else "not declared; verify manually"),
            f"- Detection evidence: {', '.join((selected or {}).get('evidence', [])) or 'manual override'}",
            "- State/cache contracts:",
            *(f"  - `{value}`: {state}" for value, state in state_contracts),
            "",
            "## Static risk gates",
            "",
        ]
        if risks:
            lines.extend(f"- **{r.get('severity')} / {r.get('type')}**: {r.get('detail')}" for r in risks)
        else:
            lines.append("- No high-signal static risk was found; still review code, license, and provenance.")

        lines += [
            "",
            "## Source oracle",
            "",
            "- [ ] Pin source repository, model, tokenizer/processor, codec/vocoder, and generation revisions.",
            "- [ ] Save post-preprocessing minimal, ordinary, and boundary fixtures.",
            "- [ ] Capture semantic intermediate checkpoints described by the selected runbook.",
            "- [ ] Capture deterministic end-to-end output and task-quality baseline.",
            "- [ ] Capture full versus incremental/chunked state behavior.",
            "",
            "## Weight conversion",
            "",
            "- [ ] Export source tensor manifest without importing untrusted model code.",
            "- [ ] Create explicit `WEIGHT_MAP.json` entries for every source/target tensor.",
            "- [ ] Record fused/split QKV, gate/up, convolution layout, codebook, and tied/shared transforms.",
            "- [ ] Explain every ignored source and generated target tensor.",
            "- [ ] Validate shape coverage with `validate_weight_map.py`.",
            "",
            "## Implementation phases",
            "",
            "1. Implement config and preprocessing contract.",
            "2. Implement a readable primitive/block oracle in eager floating point.",
            "3. Achieve block and staged intermediate parity.",
            "4. Assemble end-to-end model and deterministic output.",
            "5. Add exact cache/recurrent/streaming state and compare against full recomputation.",
            "6. Save/reload and package a smoke-test fixture.",
            "7. Establish a baseline profile and benchmark report.",
            "8. Run one optimization experiment at a time.",
            "9. Quantize only after unquantized quality and performance are recorded.",
            "10. Publish only after license/provenance and clean-environment load tests.",
            "",
            *recommendation_lines(recommendations),
        ]
        lines += [
            "",
            "## Validation matrix",
            "",
            "| Stage | Fixture | Metric/tolerance | Status |",
            "|---|---|---|---|",
            "| Preprocessing | minimal/ordinary/boundary | exact or declared | pending |",
            "| Weight coverage | all tensors | 100% explained | pending |",
            "| Primitive/block | deterministic tensors | per-op tolerance | pending |",
            "| End-to-end | deterministic input | logits/latent/output | pending |",
            "| Cache/state | full vs incremental/chunks | declared tolerance | pending |",
            "| Task quality | representative set | baseline-relative | pending |",
            "| Performance | target workload | raw baseline/candidate | pending |",
            "",
            "## Stop conditions",
            "",
            "- Missing or incompatible license/provenance.",
            "- Required behavior exists only in unreviewed remote code.",
            "- First divergence cannot be localized through staged checkpoints.",
            "- Candidate optimization fails correctness/quality or lacks end-to-end benefit.",
            "- Custom kernel lacks a reference fallback and shape/dtype coverage.",
        ]
        Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(args.output)
        return 0
    except SkillError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
