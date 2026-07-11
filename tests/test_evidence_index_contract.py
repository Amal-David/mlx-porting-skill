from __future__ import annotations

import collections
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "mlx-model-porting"
SOURCES = SKILL / "assets" / "sources.yaml"
GENERATOR = SKILL / "scripts" / "generate_evidence_index.py"
INDEX = ROOT / "EVIDENCE_INDEX.md"


def run_generator(*args: object, expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(GENERATOR), *(str(arg) for arg in args)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != expected:
        raise AssertionError(
            f"generator exited {result.returncode}, expected {expected}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def markdown_links(text: str) -> list[tuple[str, str]]:
    return re.findall(r"(?<!\\)\[((?:\\.|[^\]])*)\]\(([^)\s]+)\)", text)


def markdown_code_spans(text: str) -> tuple[list[str], str]:
    spans: list[str] = []
    remainder: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "`":
            remainder.append(text[index])
            index += 1
            continue
        fence_end = index
        while fence_end < len(text) and text[fence_end] == "`":
            fence_end += 1
        fence = text[index:fence_end]
        close = text.find(fence, fence_end)
        if close == -1:
            remainder.append(fence)
            index = fence_end
            continue
        content = text[fence_end:close]
        if content.startswith(" ") and content.endswith(" ") and content.strip(" "):
            content = content[1:-1]
        spans.append(content)
        index = close + len(fence)
    return spans, "".join(remainder)


class EvidenceIndexContractTests(unittest.TestCase):
    def test_generated_index_is_deterministic_complete_and_registry_order_is_unchanged(self) -> None:
        registry_before = json.loads(SOURCES.read_text(encoding="utf-8"))
        ids_before = [source["id"] for source in registry_before["sources"]]

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.md"
            second = Path(tmp) / "second.md"
            run_generator("--sources", SOURCES, "--output", first)
            run_generator("--sources", SOURCES, "--output", second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            rendered = first.read_text(encoding="utf-8")

        registry_after = json.loads(SOURCES.read_text(encoding="utf-8"))
        self.assertEqual(
            [source["id"] for source in registry_after["sources"]],
            ids_before,
            "rendering must not reorder the canonical source registry",
        )
        full_list = rendered.split("## Full source list", 1)[1]
        rendered_ids = re.findall(r"^\| `([^`]+)` \|", full_list, flags=re.MULTILINE)
        self.assertEqual(rendered_ids, sorted(ids_before))
        self.assertEqual(collections.Counter(rendered_ids), collections.Counter(ids_before))
        self.assertIn(f"**Total records:** {len(ids_before)}", rendered)

        source_kinds = collections.Counter(source["kind"] for source in registry_before["sources"])
        for kind, count in source_kinds.items():
            self.assertIn(f"| `{kind}` | {count} |", rendered)

        generator_text = GENERATOR.read_text(encoding="utf-8")
        self.assertNotIn(str(len(ids_before)), generator_text, "record counts must be computed from sources.yaml")

    def test_check_detects_drift_and_committed_index_is_current(self) -> None:
        run_generator("--check")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "index.md"
            run_generator("--sources", SOURCES, "--output", output)
            output.write_text(output.read_text(encoding="utf-8") + "\nmanual drift\n", encoding="utf-8")
            result = run_generator("--sources", SOURCES, "--output", output, "--check", expected=1)
            self.assertIn("drift", result.stderr.lower())

    def test_index_explains_claim_boundaries_and_links_benchmark_assessment(self) -> None:
        text = INDEX.read_text(encoding="utf-8")
        for phrase in (
            "Review depth",
            "Support scope",
            "Claim boundary",
            "does not prove target-workload performance",
        ):
            self.assertIn(phrase, text)
        self.assertIn(
            "[Benchmark assessment](mlx-model-porting/assets/BENCHMARK_REPORT.md)",
            text,
        )

    def test_generated_index_escapes_hostile_links_and_html_cells(self) -> None:
        registry = json.loads(SOURCES.read_text(encoding="utf-8"))
        registry["benchmark_assessment"] = "mlx-model-porting/assets/report](javascript:alert(3)).md"
        registry["sources"] = [
            {
                "id": "hostile-close",
                "title": "Hostile <script>alert(1)</script> title",
                "url": "https://example.com/already%20encoded/close)break",
                "kind": "official-doc",
                "owner": "example",
                "topics": ["test"],
                "review_depth": "indexed",
                "snapshot": "2026-07-10",
                "note": "safe",
            },
            {
                "id": "hostile-breakout",
                "title": "Hostile breakout",
                "url": "https://example.com/report](javascript:alert(1))",
                "kind": "official-doc",
                "owner": "example",
                "topics": ["test"],
                "review_depth": "indexed",
                "snapshot": "2026-07-10",
                "note": "Hostile <script>alert(2)</script> note",
            },
        ]
        registry["count"] = len(registry["sources"])

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "sources.json"
            output = Path(tmp) / "index.md"
            source_path.write_text(json.dumps(registry), encoding="utf-8")
            run_generator("--sources", source_path, "--output", output)
            rendered = output.read_text(encoding="utf-8")

        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script>", rendered)
        self.assertIn("already%20encoded/close%29break", rendered)
        self.assertNotIn("%2520", rendered)
        self.assertIn("report%5D%28javascript:alert%281%29%29", rendered)
        self.assertNotIn("](javascript:", rendered)

    def test_rendered_link_label_cannot_hijack_destination_and_code_list_stays_one_span(self) -> None:
        registry = json.loads(SOURCES.read_text(encoding="utf-8"))
        intended_url = "https://example.com/intended-source"
        hostile_topic = "safe` fake `escape"
        registry["sources"] = [
            {
                "id": "hostile-label",
                "title": "x](https://attacker.example) [disguise",
                "url": intended_url,
                "kind": "official-doc",
                "owner": "example",
                "topics": [hostile_topic],
                "review_depth": "indexed",
                "snapshot": "2026-07-10",
                "note": "literal `code` and *emphasis* must stay inert",
            }
        ]
        registry["count"] = 1

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "sources.json"
            output = Path(tmp) / "index.md"
            source_path.write_text(json.dumps(registry), encoding="utf-8")
            run_generator("--sources", source_path, "--output", output)
            rendered = output.read_text(encoding="utf-8")

        destinations = [destination for _, destination in markdown_links(rendered)]
        self.assertIn(intended_url, destinations)
        self.assertNotIn("https://attacker.example", destinations)
        source_row = next(line for line in rendered.splitlines() if "hostile-label" in line)
        spans, outside = markdown_code_spans(source_row)
        self.assertIn(hostile_topic, spans)
        self.assertNotIn("`", outside, "hostile list backticks escaped their enclosing code span")

    def test_generated_index_rejects_non_https_source_links(self) -> None:
        registry = json.loads(SOURCES.read_text(encoding="utf-8"))
        registry["sources"] = [
            {
                "id": "non-https",
                "title": "Non-HTTPS source",
                "kind": "official-doc",
                "owner": "example",
                "topics": ["test"],
                "review_depth": "indexed",
                "snapshot": "2026-07-10",
                "note": "",
            }
        ]
        registry["count"] = len(registry["sources"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for index, url in enumerate(("http://example.com/source", "javascript:alert(1)")):
                registry["sources"][0]["url"] = url
                source_path = tmp / f"sources-{index}.json"
                output = tmp / f"index-{index}.md"
                source_path.write_text(json.dumps(registry), encoding="utf-8")
                result = run_generator(
                    "--sources", source_path,
                    "--output", output,
                    expected=2,
                )
                self.assertIn("must use https", result.stderr.lower())
                self.assertFalse(output.exists())

    def test_check_rejects_synthesized_moving_github_urls(self) -> None:
        moving_urls = (
            "https://github.com/example/project",
            "https://github.com/example/project/blob/main/path.py",
            "https://github.com/example/project/tree/main/src",
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for index, url in enumerate(moving_urls):
                registry = {
                    "schema_version": 1,
                    "reviewed": "2026-07-09",
                    "review_depth_definitions": {
                        "synthesized": "used",
                        "screened": "reviewed",
                        "indexed": "catalogued",
                    },
                    "count": 1,
                    "sources": [{
                        "id": f"moving-{index}",
                        "title": "Moving source",
                        "url": url,
                        "kind": "source-code" if "/blob/" in url else "repository",
                        "owner": "example",
                        "topics": ["test"],
                        "review_depth": "synthesized",
                        "snapshot": "0123456789abcdef0123456789abcdef01234567",
                        "note": "",
                    }],
                }
                source_path = tmp / f"sources-{index}.json"
                output = tmp / f"index-{index}.md"
                source_path.write_text(json.dumps(registry), encoding="utf-8")
                result = run_generator(
                    "--sources", source_path,
                    "--output", output,
                    "--check",
                    expected=2,
                )
                self.assertIn("moving github url", result.stderr.lower())

    def test_synthesized_github_sources_are_pinned_and_current_mlx_release_is_primary(self) -> None:
        registry = json.loads(SOURCES.read_text(encoding="utf-8"))
        by_id = {source["id"]: source for source in registry["sources"]}
        self.assertIn("mlx-lm-release-0312", by_id)

        release = by_id["mlx-release-0320"]
        self.assertEqual(release["url"], "https://github.com/ml-explore/mlx/releases/tag/v0.32.0")
        self.assertEqual(release["snapshot"], "v0.32.0")
        self.assertEqual(release["review_depth"], "synthesized")
        self.assertEqual(release["evidence_class"], "release_note")
        self.assertEqual(release["support_scope"], "official_mlx")
        self.assertIn("api_support", release["claim_types"])

        run_generator("--check")


if __name__ == "__main__":
    unittest.main()
