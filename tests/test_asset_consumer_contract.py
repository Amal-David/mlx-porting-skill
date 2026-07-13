"""Caller-graph contract for checked-in research and registry assets."""
from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "mlx-model-porting" / "assets"
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"

# Benchmark JSON files are execution receipts and attestations with their own
# recursive validator. This contract covers the authored research/registry
# assets whose accidental write-only state would otherwise be silent.
EXCLUDED_PREFIXES = {"benchmarks"}

# No intentional write-only archive exists today. New entries require a short
# reason and should be exceptional; ordinary assets must gain a real consumer.
INTENTIONAL_WRITE_ONLY_ARCHIVES: dict[str, str] = {}

# Explicit reader declarations make reviews inspectable and catch new assets.
# A declared reader must name the asset and contain a structured read path.
ASSET_CONSUMERS: dict[str, tuple[str, ...]] = {
    "WEIGHT_MAP.json": ("audit_skill.py",),
    "architectures.yaml": ("audit_skill.py",),
    "contributor-refresh.json": ("knowledge_curator.py",),
    "contributor_learnings.json": ("knowledge_curator.py",),
    "effective_claims.json": ("recommend_optimizations.py",),
    "knowledge_graph.json": ("recommend_optimizations.py",),
    "learning_paths.json": ("generate_site_data.py",),
    "model_outcomes.json": ("knowledge_curator.py",),
    "optimization_guidance.yaml": ("recommend_optimizations.py",),
    "optimization_stacks.yaml": ("recommend_optimizations.py",),
    "recommendation-taxonomy.yaml": ("recommend_optimizations.py",),
    "research_backlog.json": ("validate_sources.py",),
    "research_loop_config.json": ("research_loop.py",),
    "sources.yaml": ("audit_skill.py",),
    "techniques.yaml": ("audit_skill.py",),
    "update-candidates.json": ("knowledge_curator.py",),
    "update-watchlist.json": ("update_sources.py",),
}

# Generated assets must be consumed by a different script than their writer.
ASSET_WRITERS = {
    "contributor-refresh.json": "collect_contributors.py",
    "effective_claims.json": "generate_claim_catalog.py",
    "knowledge_graph.json": "knowledge_curator.py",
    "research_backlog.json": "knowledge_curator.py",
    "update-candidates.json": "update_sources.py",
}

READ_MARKERS = (
    "load_structured(",
    "read_text(",
    "read_bytes(",
    ".open(\"rb\")",
    ".open('rb')",
)


class AssetConsumerContractTests(unittest.TestCase):
    def test_every_authored_asset_has_a_declared_consumer(self) -> None:
        discovered = {
            path.relative_to(ASSETS).as_posix()
            for path in ASSETS.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".json", ".yaml", ".yml"}
            and path.relative_to(ASSETS).parts[0] not in EXCLUDED_PREFIXES
        }
        declared = set(ASSET_CONSUMERS) | set(INTENTIONAL_WRITE_ONLY_ARCHIVES)
        self.assertEqual(discovered, declared)

    def test_declared_consumers_are_real_readers_and_not_only_writers(self) -> None:
        for asset, consumers in ASSET_CONSUMERS.items():
            with self.subTest(asset=asset):
                self.assertTrue(consumers, f"{asset} has no consumer")
                writer = ASSET_WRITERS.get(asset)
                self.assertTrue(
                    any(consumer != writer for consumer in consumers),
                    f"{asset} is read only by its writer {writer}",
                )
                for consumer in consumers:
                    script = SCRIPTS / consumer
                    self.assertTrue(script.is_file(), f"missing consumer script {consumer}")
                    source = script.read_text(encoding="utf-8")
                    self.assertIn(asset, source, f"{consumer} does not name {asset}")
                    self.assertTrue(
                        any(marker in source for marker in READ_MARKERS),
                        f"{consumer} has no structured file read path",
                    )


if __name__ == "__main__":
    unittest.main()
