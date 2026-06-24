#!/usr/bin/env python3
"""Validate source provenance and technique evidence without executing remote code."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from _common import SkillError, dump_json, load_structured

IMPLEMENTATION_KINDS = {"official-doc", "repository", "source-code", "release"}
SOURCE_KINDS = IMPLEMENTATION_KINDS | {"paper", "technical-blog", "benchmark-artifact", "issue-report"}
SUPPORTED_STATUSES = {"native-mlx", "official-mlx-project", "proven-mlx-port"}
REVIEW_DEPTHS = {"synthesized", "screened", "indexed"}
EVIDENCE_CLASSES = {
    "official_api_doc",
    "primary_source_code",
    "pinned_mlx_repo",
    "release_note",
    "primary_paper",
    "technical_blog",
    "local_benchmark_artifact",
    "issue_report",
}
SUPPORT_SCOPES = {
    "official_mlx",
    "official_mlx_project",
    "third_party_pinned",
    "paper_only",
    "context_only",
    "local_reproduced",
}
CLAIM_TYPES = {
    "api_support",
    "mlx_implementation",
    "algorithm_candidate",
    "performance",
    "memory",
    "quality",
    "serving_semantics",
    "audio_quality",
    "risk_or_negative",
}
KIND_TO_EVIDENCE_CLASS = {
    "official-doc": {"official_api_doc"},
    "source-code": {"primary_source_code"},
    "repository": {"primary_source_code", "pinned_mlx_repo"},
    "release": {"release_note"},
    "paper": {"primary_paper"},
    "technical-blog": {"technical_blog"},
    "benchmark-artifact": {"local_benchmark_artifact"},
    "issue-report": {"issue_report"},
}
KNOWN_ARXIV_TITLES = {
    "paper-2206-02658": "Longitudinal Analysis of Privacy Labels in the Apple App Store",
    "paper-2309-15531": "Rethinking Channel Dimensions to Isolate Outliers for Low-bit Weight Quantization of Large Language Models",
    "paper-2404-19124": "Accelerating Production LLMs with Combined Token/Embedding Speculators",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MLX skill source provenance")
    parser.add_argument("skill", nargs="?", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--check-urls", action="store_true", help="Perform non-executing HTTPS URL checks")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", help="Write a JSON report")
    return parser.parse_args()


def add_error(errors: list[str], condition: bool, message: str) -> None:
    if condition:
        errors.append(message)


def check_url(source: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = str(source.get("url", ""))
    headers = {"User-Agent": "mlx-model-porting-skill/0.1"}
    result: dict[str, Any] = {"id": source.get("id"), "url": url, "ok": False}
    try:
        request = urllib.request.Request(url, method="HEAD", headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result.update({"ok": 200 <= response.status < 400, "status": response.status})
    except urllib.error.HTTPError as exc:
        if exc.code not in {403, 405, 429}:
            result.update({"status": exc.code, "error": str(exc)})
            return result
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response.read(1024)
                result.update({"ok": 200 <= response.status < 400, "status": response.status})
        except Exception as retry_exc:  # noqa: BLE001 - report external check failure precisely
            result.update({"status": getattr(retry_exc, "code", None), "error": str(retry_exc)})
    except Exception as exc:  # noqa: BLE001 - report external check failure precisely
        result["error"] = str(exc)
    return result


def validate(skill: Path, check_urls: bool, timeout: float, workers: int) -> tuple[dict[str, Any], list[str]]:
    assets = skill / "assets"
    sources = load_structured(assets / "sources.yaml")
    techniques = load_structured(assets / "techniques.yaml")
    optimization_guidance_path = assets / "optimization_guidance.yaml"
    optimization_guidance = load_structured(optimization_guidance_path) if optimization_guidance_path.exists() else {}
    taxonomy_path = assets / "recommendation-taxonomy.yaml"
    taxonomy = load_structured(taxonomy_path) if taxonomy_path.exists() else {}
    errors: list[str] = []
    warnings: list[str] = []

    source_items = sources.get("sources", [])
    add_error(errors, sources.get("count") != len(source_items), "sources.yaml count does not match source list length")
    source_ids = [s.get("id") for s in source_items if isinstance(s, dict)]
    add_error(errors, len(source_ids) != len(set(source_ids)), "sources.yaml contains duplicate source IDs")

    source_by_id = {s.get("id"): s for s in source_items if isinstance(s, dict)}
    for source in source_items:
        sid = source.get("id")
        add_error(errors, not sid, "source missing id")
        add_error(errors, source.get("kind") not in SOURCE_KINDS, f"invalid source kind for {sid}: {source.get('kind')}")
        add_error(errors, source.get("review_depth") not in REVIEW_DEPTHS, f"invalid review_depth for source {sid}")
        add_error(errors, not str(source.get("url", "")).startswith("https://"), f"source URL is not https: {sid}")
        evidence_class = source.get("evidence_class")
        if evidence_class is not None:
            add_error(errors, evidence_class not in EVIDENCE_CLASSES, f"invalid evidence_class for source {sid}: {evidence_class}")
            valid_classes = KIND_TO_EVIDENCE_CLASS.get(str(source.get("kind")), set())
            add_error(errors, evidence_class not in valid_classes, f"evidence_class {evidence_class} does not match kind {source.get('kind')} for source {sid}")
        support_scope = source.get("support_scope")
        if support_scope is not None:
            add_error(errors, support_scope not in SUPPORT_SCOPES, f"invalid support_scope for source {sid}: {support_scope}")
        claim_types = source.get("claim_types", [])
        add_error(errors, claim_types is not None and not isinstance(claim_types, list), f"claim_types must be a list for source {sid}")
        for claim in claim_types or []:
            add_error(errors, claim not in CLAIM_TYPES, f"invalid claim_type for source {sid}: {claim}")
        add_error(errors, bool(source.get("benchmark_refs")) and source.get("evidence_class") != "local_benchmark_artifact", f"benchmark_refs require local_benchmark_artifact evidence class: {sid}")
        if source.get("review_depth") == "synthesized" and source.get("kind") in IMPLEMENTATION_KINDS:
            add_error(errors, not source.get("snapshot"), f"synthesized moving source lacks snapshot: {sid}")
        if source.get("kind") == "paper" and str(sid).startswith("paper-"):
            expected = str(sid).replace("paper-", "").replace("-", ".")
            snapshot = str(source.get("snapshot", ""))
            if snapshot and snapshot != expected:
                warnings.append(f"paper {sid} snapshot {snapshot!r} does not match id-derived {expected!r}")
            known_title = KNOWN_ARXIV_TITLES.get(str(sid))
            if known_title:
                add_error(
                    errors,
                    source.get("title") != known_title,
                    f"paper {sid} title does not match verified arXiv identity",
                )

    for technique in techniques.get("techniques", []):
        tid = technique.get("id")
        evidence = technique.get("evidence", [])
        for evidence_id in evidence:
            add_error(errors, evidence_id not in source_by_id, f"technique {tid} references missing evidence {evidence_id}")
        status = technique.get("status")
        if status in SUPPORTED_STATUSES:
            implementation_evidence = [
                source_by_id[evidence_id]
                for evidence_id in evidence
                if evidence_id in source_by_id and source_by_id[evidence_id].get("kind") in IMPLEMENTATION_KINDS
            ]
            add_error(
                errors,
                not implementation_evidence,
                f"supported technique {tid} has no implementation evidence source",
            )
        for evidence_class in technique.get("required_evidence_classes", []):
            add_error(errors, evidence_class not in EVIDENCE_CLASSES, f"technique {tid} has invalid required_evidence_class {evidence_class}")
        if "rollback_condition" in technique:
            add_error(errors, not technique.get("rollback_condition"), f"technique {tid} has empty rollback_condition")

    technique_ids = {t.get("id") for t in techniques.get("techniques", []) if isinstance(t, dict)}
    for method in optimization_guidance.get("methods", []):
        mid = method.get("id")
        technique_id = method.get("technique_id")
        add_error(errors, not mid, "optimization guidance method missing id")
        add_error(errors, technique_id not in technique_ids, f"optimization guidance method {mid} references missing technique {technique_id}")
        add_error(errors, method.get("status") not in SUPPORTED_STATUSES | {"research-candidate", "rejected-or-superseded"}, f"optimization guidance method {mid} has invalid status {method.get('status')}")
        add_error(errors, not method.get("expected_effect"), f"optimization guidance method {mid} lacks expected_effect")
        add_error(errors, not method.get("tradeoffs"), f"optimization guidance method {mid} lacks tradeoffs")
        add_error(errors, not method.get("validation_gates"), f"optimization guidance method {mid} lacks validation_gates")
        add_error(errors, not method.get("rollback_conditions"), f"optimization guidance method {mid} lacks rollback_conditions")
        evidence_refs = method.get("evidence_refs", {})
        add_error(errors, not isinstance(evidence_refs, dict), f"optimization guidance method {mid} evidence_refs must be a mapping")
        for role, refs in (evidence_refs or {}).items():
            add_error(errors, role not in {"official_docs", "repositories", "papers", "technical_blogs", "issues", "benchmark_artifacts"}, f"optimization guidance method {mid} has invalid evidence role {role}")
            add_error(errors, not isinstance(refs, list), f"optimization guidance method {mid} evidence role {role} must be a list")
            for source_id in refs or []:
                add_error(errors, source_id not in source_by_id, f"optimization guidance method {mid} references missing evidence {source_id}")

    if taxonomy:
        taxonomy_objectives = taxonomy.get("objective_tags", [])
        add_error(errors, not isinstance(taxonomy_objectives, list) or not taxonomy_objectives, "recommendation taxonomy lacks objective_tags")
        objective_ids = {str(item.get("id")) for item in taxonomy_objectives if isinstance(item, dict)}
        for method in optimization_guidance.get("methods", []):
            for objective in method.get("objectives", []):
                add_error(errors, objective_ids and str(objective) not in objective_ids, f"optimization guidance method {method.get('id')} uses objective outside taxonomy: {objective}")
        for evidence_class in taxonomy.get("evidence_classes", []):
            add_error(errors, evidence_class not in EVIDENCE_CLASSES, f"recommendation taxonomy has invalid evidence_class {evidence_class}")
        for scope in taxonomy.get("support_scopes", []):
            add_error(errors, scope not in SUPPORT_SCOPES, f"recommendation taxonomy has invalid support_scope {scope}")
        for claim_type in taxonomy.get("claim_types", []):
            add_error(errors, claim_type not in CLAIM_TYPES, f"recommendation taxonomy has invalid claim_type {claim_type}")

    url_results: list[dict[str, Any]] = []
    if check_urls:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(check_url, source, timeout) for source in source_items]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                url_results.append(result)
                if not result.get("ok"):
                    errors.append(f"URL check failed for {result.get('id')}: {result.get('status')} {result.get('error', '')}".strip())
        url_results.sort(key=lambda item: str(item.get("id", "")))

    report = {
        "ok": not errors,
        "skill": str(skill),
        "sources": len(source_items),
        "techniques": len(techniques.get("techniques", [])),
        "optimization_methods": len(optimization_guidance.get("methods", [])),
        "recommendation_taxonomy": bool(taxonomy),
        "warnings": warnings,
        "errors": errors,
        "url_checks": url_results,
    }
    return report, errors


def main() -> int:
    args = parse_args()
    try:
        skill = Path(args.skill).resolve()
        report, errors = validate(skill, args.check_urls, args.timeout, args.workers)
        if args.output:
            dump_json(report, args.output)
        print(json.dumps({k: v for k, v in report.items() if k != "url_checks"}, indent=2))
        return 1 if errors else 0
    except (SkillError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
