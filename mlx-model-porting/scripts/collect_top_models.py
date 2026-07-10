#!/usr/bin/env python3
"""Collect a Hugging Face top-model snapshot for outcome coverage review.

The script only reads public Hub metadata. It does not download model files,
import repository code, or execute remote model code.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import SkillError, dump_json, load_structured
from validate_sources import (
    PublicHTTPSRedirectHandler,
    build_public_https_opener,
    open_public_https,
    require_public_https_url,
)


OPEN_LICENSE_HINTS = {
    "apache-2.0",
    "mit",
    "bsd",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc0-1.0",
    "openrail",
    "bigscience-openrail-m",
    "creativeml-openrail-m",
    "gemma",
    "llama3",
    "llama3.1",
    "llama3.2",
    "llama3.3",
}
RESTRICTED_LICENSE_HINTS = {
    "other",
    "unknown",
    "proprietary",
    "research-only",
    "non-commercial",
    "cc-by-nc",
}
MAX_NETWORK_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_MODEL_LIMIT = 5_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=250, help="number of public model records to collect")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("mlx-model-porting/assets/top_models_snapshot.json"),
        help="output JSON path",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=Path("mlx-model-porting/assets/model_outcomes.json"),
        help="model outcome registry used for coverage annotations",
    )
    parser.add_argument("--hf-api-base", default="https://huggingface.co", help="Hugging Face base URL")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    try:
        return load_structured(path)
    except SkillError as exc:
        raise SystemExit(str(exc)) from exc


def fetch_models(base: str, limit: int) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"sort": "downloads", "direction": "-1", "limit": str(limit), "full": "true"})
    url = f"{base.rstrip('/')}/api/models?{query}"
    request = urllib.request.Request(url, headers={"accept": "application/json", "user-agent": "mlx-porting-skill-top-model-collector/0.1"})
    opener = build_public_https_opener()
    try:
        with open_public_https(opener, request, 60) as response:
            raw = response.read(MAX_NETWORK_RESPONSE_BYTES + 1)
            if len(raw) > MAX_NETWORK_RESPONSE_BYTES:
                raise SystemExit(
                    f"Hugging Face response exceeds {MAX_NETWORK_RESPONSE_BYTES} bytes"
                )
            payload = json.loads(raw.decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"failed to fetch Hugging Face models: {error}") from error
    if not isinstance(payload, list):
        raise SystemExit("Hugging Face API returned a non-list payload")
    return payload


def license_value(model: dict[str, Any]) -> str:
    card = model.get("cardData") if isinstance(model.get("cardData"), dict) else {}
    value = model.get("license") or card.get("license") or ""
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    return str(value).strip()


def license_class(value: str) -> str:
    normalized = value.lower().strip()
    if not normalized:
        return "unknown"
    if any(token in normalized for token in RESTRICTED_LICENSE_HINTS):
        return "restricted-or-review"
    if normalized in OPEN_LICENSE_HINTS or any(token in normalized for token in OPEN_LICENSE_HINTS):
        return "open-or-open-weight"
    return "review"


def text_blob(model: dict[str, Any]) -> str:
    card = model.get("cardData") if isinstance(model.get("cardData"), dict) else {}
    values: list[Any] = [
        model.get("id") or model.get("modelId"),
        model.get("pipeline_tag"),
        model.get("library_name"),
        model.get("tags"),
        card.get("pipeline_tag"),
        card.get("library_name"),
        card.get("base_model"),
        card.get("tags"),
    ]
    flat: list[str] = []
    for value in values:
        if isinstance(value, list):
            flat.extend(str(item) for item in value)
        elif value:
            flat.append(str(value))
    return " ".join(flat).lower().replace("_", "-")


def matches(value: str, pattern: str) -> bool:
    normalized = pattern.lower().replace("_", "-")
    if len(normalized) <= 3:
        return re.search(rf"(^|[^a-z0-9]){re.escape(normalized)}([^a-z0-9]|$)", value) is not None
    return normalized in value


def outcome_matches(model: dict[str, Any], outcome: dict[str, Any]) -> bool:
    match = outcome.get("match") if isinstance(outcome.get("match"), dict) else {}
    blob = text_blob(model)
    pipeline = str(model.get("pipeline_tag") or "").lower()
    library = str(model.get("library_name") or "").lower()
    if pipeline and pipeline in {str(item).lower() for item in match.get("pipeline_tags", [])}:
        return True
    if library and library in {str(item).lower() for item in match.get("library_names", [])}:
        return True
    return any(matches(blob, str(pattern)) for pattern in match.get("id_patterns", []))


def compact_model(rank: int, model: dict[str, Any], outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    license_text = license_value(model)
    matched = [outcome["id"] for outcome in outcomes if outcome_matches(model, outcome)]
    return {
        "rank": rank,
        "id": model.get("id") or model.get("modelId") or "",
        "downloads": model.get("downloads") or 0,
        "likes": model.get("likes") or 0,
        "pipeline_tag": model.get("pipeline_tag") or "",
        "library_name": model.get("library_name") or "",
        "license": license_text,
        "license_class": license_class(license_text),
        "gated": bool(model.get("gated")),
        "private": bool(model.get("private")),
        "last_modified": model.get("lastModified") or "",
        "matched_outcome_ids": matched,
        "coverage_state": "covered" if matched else "unknown",
    }


def main() -> int:
    args = parse_args()
    if not 1 <= args.limit <= MAX_MODEL_LIMIT:
        raise SystemExit(f"--limit must be between 1 and {MAX_MODEL_LIMIT}")
    try:
        require_public_https_url(args.hf_api_base)
    except urllib.error.URLError as error:
        raise SystemExit(f"--hf-api-base must be public HTTPS: {error.reason}") from error
    outcomes_payload = read_json(args.outcomes)
    outcomes = outcomes_payload.get("records", [])
    if not isinstance(outcomes, list):
        raise SystemExit("model outcomes registry has no records list")
    models = fetch_models(args.hf_api_base, args.limit)
    compact = [compact_model(index + 1, model, outcomes) for index, model in enumerate(models[: args.limit])]
    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": f"{args.hf_api_base.rstrip('/')}/api/models?sort=downloads&direction=-1&limit={args.limit}&full=true",
        "selection": "Top downloaded public Hugging Face model records. License fields are retained for review; missing license metadata is not treated as permissive OSS.",
        "limit": args.limit,
        "model_count": len(compact),
        "covered_count": sum(1 for model in compact if model["coverage_state"] == "covered"),
        "unknown_count": sum(1 for model in compact if model["coverage_state"] == "unknown"),
        "models": compact,
    }
    dump_json(output, args.out)
    print(f"wrote {args.out} ({len(compact)} models, {output['covered_count']} with outcome matches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
