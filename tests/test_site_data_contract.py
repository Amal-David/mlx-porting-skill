"""Contracts for the deterministic, offline-loadable public site data artifact."""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "mlx-model-porting"
ASSETS = SKILL / "assets"
SCRIPT = SKILL / "scripts" / "generate_site_data.py"
SITE_DATA = ROOT / "site" / "data.js"
SITE_PAGES = (ROOT / "site" / "index.html", ROOT / "site" / "docs" / "index.html")
GLOBAL_PREFIX = "window.MLX_PORTING_SITE_DATA = "
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"fixture is not a mapping: {path}")
    return value


def parse_site_data(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith(GLOBAL_PREFIX) or not text.endswith(";\n"):
        raise AssertionError("site data must be one window-global assignment")
    value = json.loads(text[len(GLOBAL_PREFIX):-2])
    if not isinstance(value, dict):
        raise AssertionError("site data payload is not a mapping")
    return value


def counts(rows: list[object], field: str, missing: str) -> dict[str, int]:
    values = (
        str(row.get(field)) if isinstance(row, dict) and row.get(field) is not None else missing
        for row in rows
    )
    return dict(sorted(Counter(values).items()))


def nested_value(mapping: dict[str, object], dotted_path: str) -> object:
    value: object = mapping
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise AssertionError(f"missing site-data path: {dotted_path}")
        value = value[key]
    return value


class SiteDataContractTests(unittest.TestCase):
    def run_generator(
        self,
        *args: object,
        expected: int = 0,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), *(str(arg) for arg in args)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, expected, completed.stdout + completed.stderr)
        return completed

    def test_committed_site_data_is_current_and_file_protocol_safe(self) -> None:
        self.run_generator("--check")
        text = SITE_DATA.read_text(encoding="utf-8")
        self.assertEqual(text.count(GLOBAL_PREFIX), 1)
        self.assertNotIn("fetch(", text)
        self.assertNotIn("import ", text)
        self.assertNotIn("http://", text)
        self.assertNotIn("https://", text)
        self.assertEqual(json.loads(json.dumps(parse_site_data(SITE_DATA))), parse_site_data(SITE_DATA))

    def test_site_data_values_are_derived_from_canonical_assets(self) -> None:
        data = parse_site_data(SITE_DATA)
        architectures = load_json(ASSETS / "architectures.yaml")
        sources = load_json(ASSETS / "sources.yaml")
        techniques = load_json(ASSETS / "techniques.yaml")
        guidance = load_json(ASSETS / "optimization_guidance.yaml")
        assessments = load_json(ASSETS / "benchmarks" / "receipt_assessments.json")
        claims = load_json(ASSETS / "effective_claims.json")

        self.assertEqual(data["version"], (ROOT / "VERSION").read_text(encoding="utf-8").strip())

        family_rows = architectures["families"]
        self.assertEqual(data["architectures"]["total"], len(family_rows))
        self.assertEqual(
            {row["id"]: row["runbook"] for row in data["architectures"]["families"]},
            {row["id"]: row["runbook"] for row in family_rows},
        )
        expected_labels = {}
        for row in family_rows:
            heading = (SKILL / row["runbook"]).read_text(encoding="utf-8").splitlines()[0]
            label = heading.removeprefix("# Runbook:").strip()
            expected_labels[row["id"]] = label[:1].upper() + label[1:]
        self.assertEqual(
            {row["id"]: row["label"] for row in data["architectures"]["families"]},
            expected_labels,
        )

        source_rows = sources["sources"]
        self.assertEqual(data["sources"]["total"], len(source_rows))
        self.assertEqual(data["sources"]["by_kind"], counts(source_rows, "kind", "unclassified"))
        self.assertEqual(
            data["sources"]["by_review_depth"],
            counts(source_rows, "review_depth", "unclassified"),
        )
        self.assertEqual(
            data["sources"]["by_classification"],
            counts(source_rows, "evidence_class", "unclassified"),
        )
        self.assertEqual(
            data["sources"]["by_support_scope"],
            counts(source_rows, "support_scope", "unspecified"),
        )

        technique_rows = techniques["techniques"]
        guidance_rows = guidance["methods"]
        self.assertEqual(data["techniques"], {
            "by_status": counts(technique_rows, "status", "unclassified"),
            "total": len(technique_rows),
        })
        self.assertEqual(data["guidance"], {
            "by_status": counts(guidance_rows, "status", "unclassified"),
            "total": len(guidance_rows),
        })

        assessment_rows = assessments["assessments"]
        self.assertEqual(data["benchmarks"], {
            "by_classification": counts(assessment_rows, "classification", "unclassified"),
            "promotion_ready": sum(row.get("promotion_ready") is True for row in assessment_rows),
            "total": len(assessment_rows),
        })
        claim_rows = claims["claims"]
        self.assertEqual(data["effective_claims"], {
            "by_state": counts(claim_rows, "promotion_state", "unclassified"),
            "total": len(claim_rows),
        })

        references = sorted(
            path for path in (SKILL / "references").iterdir()
            if path.is_file() and path.suffix == ".md"
        )
        self.assertEqual(data["local_docs"], {
            "references": len(references),
            "runbooks": sum(path.name.startswith("runbook-") for path in references),
        })

    def test_site_summary_counts_match_live_asset_counts(self) -> None:
        data = parse_site_data(SITE_DATA)
        architectures = load_json(ASSETS / "architectures.yaml")
        sources = load_json(ASSETS / "sources.yaml")
        techniques = load_json(ASSETS / "techniques.yaml")
        guidance = load_json(ASSETS / "optimization_guidance.yaml")
        receipts = load_json(ASSETS / "benchmarks" / "receipts_index.json")
        claims = load_json(ASSETS / "effective_claims.json")

        live_counts = {
            "architecture_families": len(architectures["families"]),
            "sources": len(sources["sources"]),
            "techniques": len(techniques["techniques"]),
            "optimization_guidance_methods": len(guidance["methods"]),
            "benchmark_receipts": len(receipts["receipts"]),
            "withheld_effective_claims": sum(
                claim.get("promotion_state") == "withheld"
                for claim in claims["claims"]
            ),
        }
        published_counts = {
            "architecture_families": nested_value(data, "architectures.total"),
            "sources": nested_value(data, "sources.total"),
            "techniques": nested_value(data, "techniques.total"),
            "optimization_guidance_methods": nested_value(data, "guidance.total"),
            "benchmark_receipts": nested_value(data, "benchmarks.total"),
            "withheld_effective_claims": nested_value(
                data,
                "effective_claims.by_state.withheld",
            ),
        }
        self.assertEqual(published_counts, live_counts)

    def test_knowledge_graph_counts_and_edge_endpoints_are_current(self) -> None:
        graph = load_json(ASSETS / "knowledge_graph.json")
        nodes = graph["nodes"]
        edges = graph["edges"]

        self.assertEqual(graph["node_count"], len(nodes))
        self.assertEqual(graph["edge_count"], len(edges))
        node_ids = {node["id"] for node in nodes}
        dangling = [
            (index, endpoint, edge.get(endpoint))
            for index, edge in enumerate(edges)
            for endpoint in ("source", "target")
            if edge.get(endpoint) not in node_ids
        ]
        self.assertEqual(dangling, [])

    def test_html_fallback_values_match_generated_data(self) -> None:
        data = parse_site_data(SITE_DATA)
        pattern = re.compile(
            r'<(?:span|strong)\b[^>]*\bdata-value="([^"]+)"[^>]*>([^<]+)</(?:span|strong)>',
        )
        found = 0
        for page in SITE_PAGES:
            html = page.read_text(encoding="utf-8")
            matches = pattern.findall(html)
            self.assertTrue(matches, f"no fallback values found in {page}")
            self.assertIn("generated-fallback", html)
            for dotted_path, fallback in matches:
                found += 1
                with self.subTest(page=page.name, path=dotted_path):
                    self.assertEqual(fallback.strip(), str(nested_value(data, dotted_path)))
        self.assertGreaterEqual(found, 15)

    def test_generation_is_deterministic_and_check_detects_drift_without_rewriting(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            output = Path(raw_tmp) / "data.js"
            self.run_generator("--output", output)
            first = output.read_bytes()
            self.run_generator("--output", output)
            self.assertEqual(output.read_bytes(), first)

            output.write_bytes(first + b"// drift\n")
            drifted = output.read_bytes()
            self.run_generator("--output", output, "--check", expected=1)
            self.assertEqual(output.read_bytes(), drifted)

            self.run_generator("--output", output)
            self.run_generator("--output", output, "--check")
            self.assertEqual(output.read_bytes(), SITE_DATA.read_bytes())

    def test_structured_loads_are_size_bounded(self) -> None:
        spec = importlib.util.spec_from_file_location("generate_site_data", SCRIPT)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as raw_tmp:
            oversized = Path(raw_tmp) / "oversized.json"
            oversized.write_bytes(b"{" + b" " * module.MAX_STRUCTURED_BYTES + b"}")
            with self.assertRaisesRegex(module.SkillError, "size limit"):
                module.load_mapping(oversized, "oversized fixture")


if __name__ == "__main__":
    unittest.main()
