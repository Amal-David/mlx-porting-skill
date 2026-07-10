#!/usr/bin/env python3
"""Validate source provenance and technique evidence without executing remote code."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import ipaddress
import json
import re
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from _common import SkillError, dump_json, load_structured, parse_band
from generate_claim_catalog import build_claim_catalog
from validate_benchmarks import build_assessment_report, build_receipts_index, render_benchmark_report

IMPLEMENTATION_KINDS = {"official-doc", "repository", "source-code", "release"}
SOURCE_KINDS = IMPLEMENTATION_KINDS | {"paper", "technical-blog", "benchmark-artifact", "issue-report"}
SUPPORTED_STATUSES = {"native-mlx", "official-mlx-project", "proven-mlx-port"}
METHOD_STATUS_SUPPORT_SCOPES = {
    "native-mlx": {"official_mlx"},
    "official-mlx-project": {"official_mlx_project"},
    "proven-mlx-port": {"third_party_pinned", "local_reproduced"},
}
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
STACK_LOSSINESS = {"lossless", "conditionally-lossy"}
STACK_PAIR_VALIDITY = {"validated-composable", "unknown", "known-conflicting", "mutually-exclusive"}
OUTCOME_PROVENANCE = {
    "profile_required",
    "source_reported",
    "performance_observation",
    "local_reproduced",
    "not_a_speedup",
}
NUMERIC_RANGE_RE = re.compile(r"\b~?\d+(?:\.\d+)?x(?:\s*-\s*\d+(?:\.\d+)?x)?\b")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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


def find_duplicate_json_keys(path: Path) -> list[str]:
    """Return duplicate mapping keys before normal JSON loading discards them."""
    duplicates: set[str] = set()

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicates.add(key)
            result[key] = value
        return result

    try:
        json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs_hook)
    except (OSError, json.JSONDecodeError):
        return []
    return sorted(duplicates)


def has_any_evidence_ref(evidence_refs: Any) -> bool:
    if not isinstance(evidence_refs, dict):
        return False
    return any(isinstance(refs, list) and bool(refs) for refs in evidence_refs.values())


def contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(contains_key(item, key) for item in value)
    return False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def compound_has_numeric_range(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"range", "compound_range"}:
                return True
            if compound_has_numeric_range(item):
                return True
    elif isinstance(value, list):
        return any(compound_has_numeric_range(item) for item in value)
    elif isinstance(value, str):
        return bool(NUMERIC_RANGE_RE.search(value))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    return False


def https_url_structure_error(url: str) -> str | None:
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        return f"malformed URL: {exc}"
    if parsed.scheme.lower() != "https":
        return "URL scheme must be https"
    if not parsed.hostname:
        return "URL must include a hostname"
    if parsed.username is not None or parsed.password is not None:
        return "URL must not contain userinfo credentials"
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return "URL hostname must not be localhost"
    try:
        literal_address = ipaddress.ip_address(hostname.split("%", 1)[0])
    except ValueError:
        literal_address = None
    if literal_address is not None and not literal_address.is_global:
        return "URL literal address must be globally routable"
    if port is not None and not 1 <= port <= 65535:
        return "URL port is outside 1..65535"
    return None


def require_public_https_url(url: str) -> None:
    structure_error = https_url_structure_error(url)
    if structure_error:
        raise urllib.error.URLError(structure_error)
    parsed = urllib.parse.urlsplit(url)
    hostname = str(parsed.hostname)
    port = parsed.port or 443
    try:
        resolved = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise urllib.error.URLError(f"could not resolve URL hostname: {exc}") from exc
    addresses = {
        str(item[4][0]).split("%", 1)[0]
        for item in resolved
        if isinstance(item[4], tuple) and item[4]
    }
    if not addresses:
        raise urllib.error.URLError("URL hostname resolved to no addresses")
    unsafe = []
    for address in sorted(addresses):
        try:
            parsed_address = ipaddress.ip_address(address)
        except ValueError as exc:
            raise urllib.error.URLError(f"resolver returned malformed address {address!r}") from exc
        if not parsed_address.is_global:
            unsafe.append(address)
    if unsafe:
        raise urllib.error.URLError(
            "URL hostname resolves to non-public address(es): " + ", ".join(unsafe)
        )


class PublicHTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        require_public_https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _checked_open(
    opener: urllib.request.OpenerDirector,
    request: urllib.request.Request,
    timeout: float,
) -> Any:
    require_public_https_url(request.full_url)
    response = opener.open(request, timeout=timeout)
    try:
        require_public_https_url(response.geturl())
        return response
    except BaseException:
        response.close()
        raise


def check_url(source: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = str(source.get("url", ""))
    headers = {"User-Agent": "mlx-model-porting-skill/0.4.0"}
    result: dict[str, Any] = {"id": source.get("id"), "url": url, "ok": False}
    opener = urllib.request.build_opener(PublicHTTPSRedirectHandler())
    try:
        request = urllib.request.Request(url, method="HEAD", headers=headers)
        with _checked_open(opener, request, timeout) as response:
            result.update({"ok": 200 <= response.status < 400, "status": response.status})
    except urllib.error.HTTPError as exc:
        if exc.code not in {403, 405, 429}:
            result.update({"status": exc.code, "error": str(exc)})
            return result
        try:
            request = urllib.request.Request(url, headers=headers)
            with _checked_open(opener, request, timeout) as response:
                response.read(1024)
                result.update({"ok": 200 <= response.status < 400, "status": response.status})
        except Exception as retry_exc:  # noqa: BLE001 - report external check failure precisely
            result.update({"status": getattr(retry_exc, "code", None), "error": str(retry_exc)})
    except Exception as exc:  # noqa: BLE001 - report external check failure precisely
        result["error"] = str(exc)
    return result


def validate(skill: Path, check_urls: bool, timeout: float, workers: int) -> tuple[dict[str, Any], list[str]]:
    assets = skill / "assets"
    benchmark_root = (assets / "benchmarks").resolve()
    sources = load_structured(assets / "sources.yaml")
    techniques = load_structured(assets / "techniques.yaml")
    optimization_guidance_path = assets / "optimization_guidance.yaml"
    optimization_guidance = load_structured(optimization_guidance_path) if optimization_guidance_path.exists() else {}
    optimization_stacks_path = assets / "optimization_stacks.yaml"
    optimization_stacks = load_structured(optimization_stacks_path) if optimization_stacks_path.exists() else {}
    taxonomy_path = assets / "recommendation-taxonomy.yaml"
    taxonomy = load_structured(taxonomy_path) if taxonomy_path.exists() else {}
    architectures_path = assets / "architectures.yaml"
    architectures = load_structured(architectures_path) if architectures_path.exists() else {}
    model_outcomes_path = assets / "model_outcomes.json"
    model_outcomes = load_structured(model_outcomes_path) if model_outcomes_path.exists() else {}
    receipt_assessments_path = assets / "benchmarks" / "receipt_assessments.json"
    receipt_assessments = (
        load_structured(receipt_assessments_path) if receipt_assessments_path.exists() else {}
    )
    effective_claims_path = assets / "effective_claims.json"
    effective_claims = load_structured(effective_claims_path) if effective_claims_path.exists() else {}
    errors: list[str] = []
    warnings: list[str] = []

    for registry_path in (
        assets / "sources.yaml",
        assets / "techniques.yaml",
        optimization_guidance_path,
        optimization_stacks_path,
        taxonomy_path,
        architectures_path,
        model_outcomes_path,
        receipt_assessments_path,
        effective_claims_path,
    ):
        if not registry_path.exists():
            continue
        for duplicate_key in find_duplicate_json_keys(registry_path):
            errors.append(f"{registry_path.name} contains duplicate JSON key {duplicate_key!r}")

    assessment_rows = receipt_assessments.get("assessments", []) if isinstance(receipt_assessments, dict) else []
    add_error(
        errors,
        receipt_assessments.get("schema_version") != 1 if isinstance(receipt_assessments, dict) else True,
        "benchmark receipt assessments must use schema_version 1",
    )
    add_error(errors, not isinstance(assessment_rows, list), "benchmark receipt assessments must contain an assessments list")
    assessments_by_receipt: dict[str, dict[str, Any]] = {}
    for row in assessment_rows if isinstance(assessment_rows, list) else []:
        receipt_ref = row.get("receipt") if isinstance(row, dict) else None
        add_error(errors, not isinstance(receipt_ref, str) or not receipt_ref, "benchmark assessment missing receipt")
        if not isinstance(receipt_ref, str) or not receipt_ref:
            continue
        add_error(errors, receipt_ref in assessments_by_receipt, f"duplicate benchmark assessment for {receipt_ref}")
        assessments_by_receipt[receipt_ref] = row
        add_error(
            errors,
            row.get("classification") not in {"performance_observation", "promotion_ready", "rejected"},
            f"benchmark assessment {receipt_ref} has invalid classification {row.get('classification')}",
        )
        expected_digest = row.get("receipt_sha256")
        add_error(
            errors,
            not isinstance(expected_digest, str) or SHA256_RE.fullmatch(expected_digest) is None,
            f"benchmark assessment {receipt_ref} has invalid receipt_sha256",
        )
        receipt_path = (assets / "benchmarks" / receipt_ref).resolve()
        benchmark_root_resolved = (assets / "benchmarks").resolve()
        try:
            receipt_path.relative_to(benchmark_root_resolved)
        except ValueError:
            errors.append(f"benchmark assessment receipt escapes assets/benchmarks: {receipt_ref}")
            continue
        if not receipt_path.is_file():
            errors.append(f"benchmark assessment receipt not found: {receipt_ref}")
        elif isinstance(expected_digest, str) and SHA256_RE.fullmatch(expected_digest):
            add_error(
                errors,
                file_sha256(receipt_path) != expected_digest,
                f"benchmark assessment digest mismatch for {receipt_ref}",
            )

    effective_rows = effective_claims.get("claims", []) if isinstance(effective_claims, dict) else []
    add_error(
        errors,
        effective_claims.get("schema_version") != 1 if isinstance(effective_claims, dict) else True,
        "effective claim catalog must use schema_version 1",
    )
    add_error(errors, not isinstance(effective_rows, list), "effective claim catalog must contain a claims list")
    effective_by_method: dict[str, dict[str, Any]] = {}
    for row in effective_rows if isinstance(effective_rows, list) else []:
        method_id = row.get("method_id") if isinstance(row, dict) else None
        add_error(errors, not isinstance(method_id, str) or not method_id, "effective claim missing method_id")
        if not isinstance(method_id, str) or not method_id:
            continue
        add_error(errors, method_id in effective_by_method, f"duplicate effective claim for {method_id}")
        effective_by_method[method_id] = row

    try:
        recomputed_assessments = build_assessment_report(benchmark_root)
        add_error(
            errors,
            recomputed_assessments != receipt_assessments,
            "benchmark receipt assessments are stale or not reproducible",
        )
        receipts_index_path = benchmark_root / "receipts_index.json"
        committed_index = load_structured(receipts_index_path) if receipts_index_path.exists() else None
        add_error(
            errors,
            committed_index != build_receipts_index(recomputed_assessments, benchmark_root),
            "benchmark receipts index is stale or not reproducible",
        )
        benchmark_report_path = assets / "BENCHMARK_REPORT.md"
        committed_report = (
            benchmark_report_path.read_text(encoding="utf-8") if benchmark_report_path.exists() else None
        )
        add_error(
            errors,
            committed_report != render_benchmark_report(recomputed_assessments),
            "benchmark report is stale or not reproducible",
        )
        recomputed_claims = build_claim_catalog(
            optimization_guidance,
            sources,
            recomputed_assessments,
            assessments_available=True,
        )
        add_error(
            errors,
            recomputed_claims != effective_claims,
            "effective claim catalog is stale or not reproducible",
        )
    except (SkillError, OSError, ValueError) as exc:
        errors.append(f"could not recompute benchmark/effective evidence: {exc}")

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
        source_url = str(source.get("url", ""))
        structure_error = https_url_structure_error(source_url)
        add_error(errors, structure_error is not None, f"invalid source URL for {sid}: {structure_error}")
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
            github_ref = re.search(
                r"^https://github\.com/[^/]+/[^/]+/(?:blob|tree)/([^/]+)",
                source_url,
            )
            if github_ref:
                pinned_ref = github_ref.group(1)
                add_error(
                    errors,
                    re.fullmatch(r"[0-9a-f]{40}", pinned_ref) is None,
                    f"synthesized GitHub source {sid} must use a full 40-hex commit URL",
                )
                add_error(
                    errors,
                    source.get("snapshot") != pinned_ref,
                    f"synthesized GitHub source {sid} snapshot must match its commit URL",
                )
        if source.get("kind") == "paper" and str(sid).startswith("paper-"):
            expected = str(sid).replace("paper-", "").replace("-", ".")
            snapshot = str(source.get("snapshot", ""))
            expected_snapshot = re.compile(rf"{re.escape(expected)}(?:v[1-9]\d*)?")
            if snapshot and expected_snapshot.fullmatch(snapshot) is None:
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

    improvement_band_tiers = set(taxonomy.get("improvement_band_policy", {})) if isinstance(taxonomy, dict) else set()
    technique_ids = {t.get("id") for t in techniques.get("techniques", []) if isinstance(t, dict)}
    method_ids = {m.get("id") for m in optimization_guidance.get("methods", []) if isinstance(m, dict)}
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
        method_status = method.get("status")
        if method_status in METHOD_STATUS_SUPPORT_SCOPES and isinstance(evidence_refs, dict):
            cited_source_ids = {
                str(source_id)
                for refs in evidence_refs.values()
                if isinstance(refs, list)
                for source_id in refs
            }
            allowed_scopes = METHOD_STATUS_SUPPORT_SCOPES[str(method_status)]
            supporting_sources = [
                source_by_id[source_id]
                for source_id in cited_source_ids
                if source_id in source_by_id
                and source_by_id[source_id].get("review_depth") == "synthesized"
                and source_by_id[source_id].get("support_scope") in allowed_scopes
                and source_by_id[source_id].get("evidence_class") in EVIDENCE_CLASSES
                and source_by_id[source_id].get("snapshot")
                and bool(
                    {"api_support", "mlx_implementation"}
                    & set(source_by_id[source_id].get("claim_types", []))
                )
            ]
            add_error(
                errors,
                not supporting_sources,
                f"optimization guidance method {mid} status {method_status} lacks synthesized pinned "
                f"supporting evidence with scope {sorted(allowed_scopes)}",
            )
        improvement_band = method.get("improvement_band")
        if improvement_band is not None:
            add_error(errors, not isinstance(improvement_band, dict), f"optimization guidance method {mid} improvement_band must be a mapping")
            if isinstance(improvement_band, dict):
                provenance = improvement_band.get("provenance")
                add_error(errors, provenance not in improvement_band_tiers, f"optimization guidance method {mid} has invalid improvement_band provenance {provenance}")
                band_range = improvement_band.get("range")
                if provenance == "profile_required":
                    add_error(errors, band_range is not None, f"profile_required improvement_band for {mid} must not contain a numeric range")
                else:
                    add_error(errors, not isinstance(band_range, str) or not band_range, f"optimization guidance method {mid} improvement_band lacks range")
                receipts = improvement_band.get("receipts")
                has_local_receipts = provenance == "local_reproduced" and isinstance(receipts, list) and bool(receipts)
                requires_1x_floor = provenance != "local_reproduced" or not has_local_receipts
                add_error(
                    errors,
                    isinstance(band_range, str) and requires_1x_floor and not band_range.startswith("1.0x-"),
                    f"optimization guidance method {mid} improvement_band range must start at 1.0x unless local_reproduced has receipts",
                )
                lineages = improvement_band.get("evidence_lineage_ids")
                if isinstance(band_range, str):
                    add_error(errors, not isinstance(lineages, list) or not lineages, f"numeric improvement_band for {mid} lacks evidence_lineage_ids")
                claim_status = improvement_band.get("claim_status")
                add_error(
                    errors,
                    claim_status not in {"held", "eligible-with-profile", "superseded"},
                    f"optimization guidance method {mid} has invalid claim_status {claim_status}",
                )
                add_error(errors, not improvement_band.get("metric"), f"optimization guidance method {mid} improvement_band lacks metric")
                add_error(errors, not improvement_band.get("basis"), f"optimization guidance method {mid} improvement_band lacks basis")
                add_error(errors, not improvement_band.get("applies_when"), f"optimization guidance method {mid} improvement_band lacks applies_when")
                catalog_claim = effective_by_method.get(str(mid))
                add_error(errors, catalog_claim is None, f"optimization guidance method {mid} lacks an effective claim record")
                if isinstance(catalog_claim, dict):
                    expected_observed_range = (
                        band_range
                        if isinstance(band_range, str)
                        else improvement_band.get("observed_source_range")
                    )
                    add_error(
                        errors,
                        catalog_claim.get("provenance") != provenance,
                        f"effective claim for {mid} has stale provenance",
                    )
                    add_error(
                        errors,
                        catalog_claim.get("observed_range") != expected_observed_range,
                        f"effective claim for {mid} has stale observed_range",
                    )
                    add_error(
                        errors,
                        catalog_claim.get("effective_range") is not None
                        and provenance != "local_reproduced",
                        f"non-local effective claim for {mid} must not expose effective_range",
                    )
                if provenance == "source_reported":
                    add_error(errors, not has_any_evidence_ref(evidence_refs), f"source_reported improvement_band for {mid} lacks evidence_refs")
                    add_error(
                        errors,
                        claim_status != "held",
                        f"source_reported improvement_band for {mid} must remain held until locally reproduced",
                    )
                    if isinstance(catalog_claim, dict):
                        add_error(
                            errors,
                            catalog_claim.get("promotion_state") != "withheld",
                            f"source_reported improvement_band for {mid} must remain withheld",
                        )
                        add_error(
                            errors,
                            catalog_claim.get("profile_eligible_range") is not None,
                            f"source_reported improvement_band for {mid} must not expose profile_eligible_range",
                        )
                if provenance == "performance_observation":
                    add_error(
                        errors,
                        claim_status != "held",
                        f"performance_observation improvement_band for {mid} must be held",
                    )
                    measured_on = improvement_band.get("measured_on")
                    add_error(
                        errors,
                        not isinstance(measured_on, dict) or not measured_on,
                        f"performance_observation improvement_band for {mid} lacks measured_on metadata",
                    )
                    add_error(
                        errors,
                        not isinstance(receipts, list) or not receipts,
                        f"performance_observation improvement_band for {mid} requires receipts",
                    )
                    if isinstance(catalog_claim, dict):
                        add_error(
                            errors,
                            catalog_claim.get("promotion_state") != "withheld"
                            or catalog_claim.get("effective_range") is not None,
                            f"performance_observation improvement_band for {mid} must remain withheld",
                        )
                if provenance == "local_reproduced":
                    measured_on = improvement_band.get("measured_on")
                    add_error(errors, not isinstance(measured_on, dict) or not measured_on, f"local_reproduced improvement_band for {mid} lacks measured_on metadata")
                    add_error(errors, not isinstance(receipts, list), f"local_reproduced improvement_band for {mid} receipts must be a list")
                    add_error(errors, claim_status != "eligible-with-profile", f"local_reproduced improvement_band for {mid} must be eligible-with-profile")
                    add_error(errors, not receipts, f"promotable local_reproduced improvement_band for {mid} requires receipts")
                    add_error(
                        errors,
                        not isinstance(catalog_claim, dict)
                        or catalog_claim.get("promotion_state") != "local-promotion"
                        or catalog_claim.get("effective_range") != band_range,
                        f"local_reproduced improvement_band for {mid} is not backed by a local-promotion effective claim",
                    )
                if provenance in {"performance_observation", "local_reproduced"}:
                    for receipt in (receipts if isinstance(receipts, list) else []):
                        raw_receipt = Path(str(receipt))
                        receipt_path = raw_receipt if raw_receipt.is_absolute() else benchmark_root / raw_receipt
                        resolved = receipt_path.resolve()
                        try:
                            resolved.relative_to(benchmark_root)
                        except ValueError:
                            errors.append(f"{provenance} improvement_band for {mid} receipt escapes assets/benchmarks: {receipt}")
                            continue
                        add_error(errors, not resolved.is_file(), f"{provenance} improvement_band for {mid} receipt not found: {receipt}")
                        assessment = assessments_by_receipt.get(str(receipt))
                        add_error(
                            errors,
                            assessment is None,
                            f"{provenance} improvement_band for {mid} receipt lacks assessment: {receipt}",
                        )
                        if isinstance(assessment, dict):
                            add_error(
                                errors,
                                provenance == "performance_observation"
                                and assessment.get("classification") == "rejected",
                                f"performance_observation improvement_band for {mid} references rejected receipt: {receipt}",
                            )
                if isinstance(band_range, str):
                    target_constraints = method.get("target_constraints")
                    add_error(errors, not isinstance(target_constraints, dict), f"numeric improvement_band for {mid} lacks target_constraints")
                    if isinstance(target_constraints, dict):
                        for source_id in target_constraints.get("source_ids", []):
                            add_error(errors, source_id not in source_by_id, f"target_constraints for {mid} references missing source {source_id}")
        for role, refs in (evidence_refs or {}).items():
            add_error(errors, role not in {"official_docs", "repositories", "papers", "technical_blogs", "issues", "benchmark_artifacts"}, f"optimization guidance method {mid} has invalid evidence role {role}")
            add_error(errors, not isinstance(refs, list), f"optimization guidance method {mid} evidence role {role} must be a list")
            for source_id in refs or []:
                add_error(errors, source_id not in source_by_id, f"optimization guidance method {mid} references missing evidence {source_id}")

    if optimization_stacks:
        controlled_workloads = {
            str(value) for value in taxonomy.get("workload_ids", [])
        } if isinstance(taxonomy, dict) else set()
        controlled_objectives = {
            str(item.get("id"))
            for item in taxonomy.get("objective_tags", [])
            if isinstance(item, dict) and item.get("id")
        } if isinstance(taxonomy, dict) else set()
        add_error(errors, contains_key(optimization_stacks, "compound_range"), "optimization_stacks contains forbidden compound_range key")
        stacks = optimization_stacks.get("stacks", [])
        add_error(errors, not isinstance(stacks, list), "optimization_stacks stacks must be a list")
        for stack in (stacks if isinstance(stacks, list) else []):
            sid = stack.get("id") if isinstance(stack, dict) else None
            add_error(errors, not sid, "optimization stack missing id")
            add_error(errors, stack.get("status") != "planning-only", f"optimization stack {sid} must be planning-only")
            primary_metric = stack.get("primary_metric") if isinstance(stack, dict) else None
            add_error(errors, not isinstance(primary_metric, str) or not primary_metric, f"optimization stack {sid} primary_metric must be a non-empty string")
            for workload in stack.get("workloads_any", []) if isinstance(stack, dict) else []:
                add_error(
                    errors,
                    workload not in controlled_workloads,
                    f"optimization stack {sid} references unknown workload {workload}",
                )
            for objective in stack.get("objectives_any", []) if isinstance(stack, dict) else []:
                add_error(
                    errors,
                    objective not in controlled_objectives,
                    f"optimization stack {sid} references unknown objective {objective}",
                )
            steps = stack.get("steps", []) if isinstance(stack, dict) else []
            add_error(errors, not isinstance(steps, list) or not steps, f"optimization stack {sid} must have steps")
            step_method_ids: set[str] = set()
            for step in (steps if isinstance(steps, list) else []):
                method_id = step.get("method") if isinstance(step, dict) else None
                add_error(errors, method_id not in method_ids, f"optimization stack {sid} step references missing method {method_id}")
                if method_id:
                    add_error(
                        errors,
                        str(method_id) in step_method_ids,
                        f"optimization stack {sid} has duplicate step method {method_id}",
                    )
                    step_method_ids.add(str(method_id))
                add_error(errors, not isinstance(step, dict) or step.get("lossiness") not in STACK_LOSSINESS, f"optimization stack {sid} step {method_id} has invalid lossiness")
                add_error(errors, not isinstance(step, dict) or not step.get("gate"), f"optimization stack {sid} step {method_id} lacks gate")
                add_error(errors, not isinstance(step, dict) or not step.get("rollback"), f"optimization stack {sid} step {method_id} lacks rollback")
            notes = stack.get("composition_notes", []) if isinstance(stack, dict) else []
            add_error(errors, not isinstance(notes, list), f"optimization stack {sid} composition_notes must be a list")
            normalized_pairs: set[tuple[str, str]] = set()
            for note in (notes if isinstance(notes, list) else []):
                pair = note.get("pair") if isinstance(note, dict) else None
                add_error(errors, not isinstance(pair, list) or len(pair) != 2, f"optimization stack {sid} composition_note pair must list two step methods")
                for method_id in (pair if isinstance(pair, list) else []):
                    add_error(errors, str(method_id) not in step_method_ids, f"optimization stack {sid} composition pair references method outside stack: {method_id}")
                if isinstance(pair, list) and len(pair) == 2:
                    normalized_pair = tuple(sorted((str(pair[0]), str(pair[1]))))
                    add_error(
                        errors,
                        normalized_pair[0] == normalized_pair[1],
                        f"optimization stack {sid} has self composition pair {normalized_pair[0]}",
                    )
                    add_error(
                        errors,
                        normalized_pair in normalized_pairs,
                        f"optimization stack {sid} has duplicate composition pair "
                        f"{normalized_pair[0]} + {normalized_pair[1]}",
                    )
                    normalized_pairs.add(normalized_pair)
                validity = note.get("validity") if isinstance(note, dict) else None
                add_error(errors, validity not in STACK_PAIR_VALIDITY, f"optimization stack {sid} composition pair has invalid validity {validity}")
                add_error(errors, not isinstance(note, dict) or not note.get("why"), f"optimization stack {sid} composition pair lacks why")
            compound = stack.get("compound") if isinstance(stack, dict) else None
            add_error(errors, not isinstance(compound, dict), f"optimization stack {sid} compound must be a mapping")
            if isinstance(compound, dict):
                measured_together = compound.get("measured_together")
                receipts = compound.get("receipts")
                add_error(errors, measured_together not in {True, False}, f"optimization stack {sid} compound measured_together must be a boolean")
                add_error(errors, not isinstance(receipts, list), f"optimization stack {sid} compound receipts must be a list")
                if measured_together is True:
                    add_error(errors, not receipts, f"optimization stack {sid} compound measured_together requires receipts")
                    for receipt in receipts if isinstance(receipts, list) else []:
                        if isinstance(receipt, dict):
                            receipt_ref = receipt.get("file") or (f"{receipt.get('label')}.json" if receipt.get("label") else None)
                        elif isinstance(receipt, str):
                            receipt_ref = receipt
                        else:
                            receipt_ref = None
                        add_error(errors, not receipt_ref, f"optimization stack {sid} compound receipt must name a file or label")
                        if not receipt_ref:
                            continue
                        raw_receipt = Path(str(receipt_ref))
                        receipt_path = raw_receipt if raw_receipt.is_absolute() else benchmark_root / raw_receipt
                        resolved = receipt_path.resolve()
                        try:
                            resolved.relative_to(benchmark_root)
                        except ValueError:
                            errors.append(f"optimization stack {sid} compound receipt escapes assets/benchmarks: {receipt_ref}")
                            continue
                        add_error(errors, not resolved.is_file(), f"optimization stack {sid} compound receipt not found: {receipt_ref}")
                compound_shape = {key: value for key, value in compound.items() if key != "receipts"}
                add_error(errors, compound_has_numeric_range(compound_shape), f"optimization stack {sid} compound stores a numeric range")
            observations = stack.get("observed_configurations", []) if isinstance(stack, dict) else []
            add_error(errors, not isinstance(observations, list), f"optimization stack {sid} observed_configurations must be a list")
            for observation in observations if isinstance(observations, list) else []:
                oid = observation.get("id") if isinstance(observation, dict) else None
                add_error(errors, not oid, f"optimization stack {sid} observed configuration lacks id")
                add_error(errors, observation.get("decision") != "rejected", f"optimization stack {sid} observation {oid} must be rejected")
                add_error(errors, not isinstance(observation.get("measured_ratio"), str), f"optimization stack {sid} observation {oid} lacks measured_ratio")
                add_error(errors, not observation.get("evidence_lineage_ids"), f"optimization stack {sid} observation {oid} lacks evidence_lineage_ids")
                add_error(errors, not observation.get("rollback"), f"optimization stack {sid} observation {oid} lacks rollback")
                receipt_ref = observation.get("receipt")
                if receipt_ref:
                    receipt_path = (benchmark_root / str(receipt_ref)).resolve()
                    try:
                        receipt_path.relative_to(benchmark_root)
                    except ValueError:
                        errors.append(f"optimization stack {sid} observation receipt escapes assets/benchmarks: {receipt_ref}")
                    else:
                        add_error(errors, not receipt_path.is_file(), f"optimization stack {sid} observation receipt not found: {receipt_ref}")

    if taxonomy:
        taxonomy_objectives = taxonomy.get("objective_tags", [])
        add_error(errors, not isinstance(taxonomy_objectives, list) or not taxonomy_objectives, "recommendation taxonomy lacks objective_tags")
        objective_ids = {str(item.get("id")) for item in taxonomy_objectives if isinstance(item, dict)}
        family_ids = {
            str(item.get("id"))
            for item in architectures.get("families", [])
            if isinstance(item, dict) and item.get("id")
        }
        family_groups = taxonomy.get("family_groups", {})
        add_error(errors, not isinstance(family_groups, dict) or not family_groups, "recommendation taxonomy lacks family_groups")
        for group_id, group_families in family_groups.items() if isinstance(family_groups, dict) else []:
            add_error(errors, not isinstance(group_families, list) or not group_families, f"family group {group_id} must be a non-empty list")
            for family_id in group_families if isinstance(group_families, list) else []:
                add_error(errors, str(family_id) not in family_ids, f"family group {group_id} references unknown family {family_id}")
        capability_ids = {str(value) for value in taxonomy.get("capability_ids", [])}
        workload_ids = {str(value) for value in taxonomy.get("workload_ids", [])}
        controlled_tags = family_ids | set(family_groups) | capability_ids | workload_ids
        for method in optimization_guidance.get("methods", []):
            for objective in method.get("objectives", []):
                add_error(errors, objective_ids and str(objective) not in objective_ids, f"optimization guidance method {method.get('id')} uses objective outside taxonomy: {objective}")
            for tag in method.get("applies_to", []):
                add_error(errors, str(tag) not in controlled_tags, f"optimization guidance method {method.get('id')} uses uncontrolled applies_to id {tag}")
            match = method.get("match")
            if match is not None:
                add_error(errors, not isinstance(match, dict), f"optimization guidance method {method.get('id')} match must be a mapping")
                if isinstance(match, dict):
                    for family_id in match.get("families_any", []):
                        add_error(errors, str(family_id) not in family_ids and str(family_id) not in family_groups, f"optimization guidance method {method.get('id')} match references unknown family {family_id}")
                    for capability in match.get("capabilities_all", []):
                        add_error(errors, str(capability) not in capability_ids, f"optimization guidance method {method.get('id')} match references unknown capability {capability}")
                    for workload in match.get("workloads_any", []):
                        add_error(errors, str(workload) not in workload_ids, f"optimization guidance method {method.get('id')} match references unknown workload {workload}")
                    add_error(errors, "model_types" in match and not all(isinstance(value, str) and value for value in match.get("model_types", [])), f"optimization guidance method {method.get('id')} has invalid model_types")
            constraints = method.get("target_constraints")
            if constraints is not None:
                add_error(errors, not isinstance(constraints, dict), f"optimization guidance method {method.get('id')} target_constraints must be a mapping")
                if isinstance(constraints, dict):
                    for workload in constraints.get("workloads_any", []):
                        add_error(errors, str(workload) not in workload_ids, f"optimization guidance method {method.get('id')} target constraint references unknown workload {workload}")
                    allowed_software = set(taxonomy.get("target_profile_contract", {}).get("software_keys", []))
                    for software_key in constraints.get("exact_software", {}):
                        add_error(errors, str(software_key) not in allowed_software, f"optimization guidance method {method.get('id')} target constraint references unknown software key {software_key}")
        for evidence_class in taxonomy.get("evidence_classes", []):
            add_error(errors, evidence_class not in EVIDENCE_CLASSES, f"recommendation taxonomy has invalid evidence_class {evidence_class}")
        for scope in taxonomy.get("support_scopes", []):
            add_error(errors, scope not in SUPPORT_SCOPES, f"recommendation taxonomy has invalid support_scope {scope}")
        for claim_type in taxonomy.get("claim_types", []):
            add_error(errors, claim_type not in CLAIM_TYPES, f"recommendation taxonomy has invalid claim_type {claim_type}")
        buckets = {str(item.get("id")): item for item in taxonomy.get("advisor_buckets", []) if isinstance(item, dict)}
        experimental = buckets.get("experimental-approach", {})
        add_error(errors, experimental.get("requires_user_opt_in") is not True, "experimental advisor bucket must require user opt-in")
        add_error(errors, experimental.get("prompt") != "This is an experimental approach. Do you want to try it?", "experimental advisor bucket has the wrong opt-in prompt")

    for outcome in model_outcomes.get("records", []):
        outcome_id = outcome.get("id") if isinstance(outcome, dict) else None
        potential = outcome.get("potential_speedup", {}) if isinstance(outcome, dict) else {}
        for branch_name in ("overall", "speculative_decoding"):
            branch = potential.get(branch_name, {}) if isinstance(potential, dict) else {}
            add_error(errors, not isinstance(branch, dict), f"model outcome {outcome_id} {branch_name} must be a mapping")
            if not isinstance(branch, dict):
                continue
            band_range = branch.get("range")
            provenance = branch.get("provenance")
            add_error(
                errors,
                provenance not in OUTCOME_PROVENANCE,
                f"model outcome {outcome_id} {branch_name} has invalid provenance {provenance}",
            )
            if band_range is None:
                add_error(
                    errors,
                    provenance not in {"profile_required", "performance_observation"},
                    f"model outcome {outcome_id} {branch_name} without a range must be profile_required or performance_observation",
                )
                if provenance == "performance_observation":
                    observed_range = branch.get("observed_source_range")
                    add_error(
                        errors,
                        not isinstance(observed_range, str),
                        f"model outcome {outcome_id} {branch_name} performance_observation lacks observed_source_range",
                    )
                    if isinstance(observed_range, str):
                        try:
                            parse_band(observed_range)
                        except ValueError as exc:
                            errors.append(
                                f"model outcome {outcome_id} {branch_name} has invalid observed_source_range: {exc}"
                            )
                    add_error(
                        errors,
                        not branch.get("evidence_lineage_ids"),
                        f"model outcome {outcome_id} {branch_name} performance_observation lacks evidence_lineage_ids",
                    )
            elif isinstance(band_range, str):
                try:
                    floor, ceiling = parse_band(band_range)
                except ValueError as exc:
                    errors.append(f"model outcome {outcome_id} {branch_name} has invalid range: {exc}")
                else:
                    add_error(
                        errors,
                        provenance in {"profile_required", "performance_observation"},
                        f"model outcome {outcome_id} {branch_name} {provenance} must not expose a numeric range",
                    )
                    if provenance == "not_a_speedup":
                        add_error(
                            errors,
                            floor != 1.0 or ceiling != 1.0,
                            f"model outcome {outcome_id} {branch_name} not_a_speedup range must be 1.0x-1.0x",
                        )
                    if ceiling > 1.0:
                        add_error(errors, not branch.get("evidence_lineage_ids"), f"model outcome {outcome_id} {branch_name} numeric upside lacks evidence_lineage_ids")
            else:
                errors.append(f"model outcome {outcome_id} {branch_name} range must be a string or null")

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
        "optimization_stacks": len(optimization_stacks.get("stacks", [])) if isinstance(optimization_stacks.get("stacks", []), list) else 0,
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
