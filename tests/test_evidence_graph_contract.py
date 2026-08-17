"""Contract tests for the schema-v2 evidence graph.

These assert the properties that make the graph safe to merge across
contributors and safe to consume from another repository: the copied schema
still matches the validator, the shards satisfy every structural rule, the
generated artifacts are byte-reproducible, and the pack gate actually refuses
the things it claims to refuse.

Standard library only, matching the rest of the offline suite.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_ROOT = REPO_ROOT / "mlx-model-porting" / "graph"
TOOLS = GRAPH_ROOT / "tools"

sys.path.insert(0, str(TOOLS))

import _graphlib as gl  # noqa: E402


def run_tool(name, *args):
    return subprocess.run(
        [sys.executable, str(TOOLS / name), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


class SchemaAgreementTests(unittest.TestCase):
    def test_copied_schema_matches_the_validator(self):
        """The validator hard-codes the contract; drift must fail loudly, not silently."""
        self.assertEqual([], gl.check_schema_agreement())

    def test_schema_copy_is_present_and_is_version_two(self):
        schema = gl.load_strict(gl.SCHEMA_PATH)
        self.assertEqual(2, schema["properties"]["schema_version"]["const"])


class GraphValidityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = gl.load_shards()
        cls.compiled = gl.load_strict(gl.COMPILED_PATH)

    def test_every_shard_is_a_valid_graph_document_on_its_own(self):
        for label, document in self.documents:
            with self.subTest(shard=label):
                self.assertEqual([], gl.validate_document(document, label))

    def test_corpus_structure_holds_across_shards(self):
        self.assertEqual([], gl.validate_corpus(self.documents))

    def test_compiled_graph_is_valid(self):
        self.assertEqual(
            [], gl.validate_document(self.compiled, "compiled/evidence-graph.json")
        )
        self.assertEqual([], gl.validate_corpus([("compiled", self.compiled)]))

    def test_no_float_survives_anywhere_in_the_corpus(self):
        """Effects are integer basis points; a float would desynchronise two tools."""

        def walk(value, where):
            if isinstance(value, float):
                self.fail("float at %s" % where)
            if isinstance(value, dict):
                for key, item in value.items():
                    walk(item, "%s.%s" % (where, key))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, "%s[%d]" % (where, index))

        walk(self.compiled, "$")

    def test_every_applied_result_is_a_complete_hyperedge(self):
        anchors = {}
        for edge in self.compiled["edges"]:
            if edge["relation"] in ("instantiates", "applied_on", "measured_on", "under_workload"):
                anchors.setdefault(edge["from"], set()).add(edge["relation"])
        for node in self.compiled["nodes"]:
            if node["kind"] != "applied_result":
                continue
            with self.subTest(result=node["id"]):
                self.assertIn("effect", node)
                self.assertEqual(
                    {"instantiates", "applied_on", "measured_on", "under_workload"},
                    anchors.get(node["id"], set()),
                )

    def test_compilation_is_lossless(self):
        """Every shard node and edge must survive compilation byte-identically.

        This is what lets one tool-side validation of the compiled graph stand
        in for validating all sixteen shards. The auto-mlx validator accepts
        only one self-contained document, so shards holding cross-shard edge
        references cannot be checked there individually; if compilation is
        lossless, checking the compiled graph checks every one of them.
        """
        nodes = {item["id"]: item for item in self.compiled["nodes"]}
        edges = {
            (item["from"], item["relation"], item["to"]): item
            for item in self.compiled["edges"]
        }
        for label, document in self.documents:
            for node in document["nodes"]:
                with self.subTest(shard=label, node=node["id"]):
                    self.assertEqual(node, nodes.get(node["id"]))
            for edge in document["edges"]:
                key = (edge["from"], edge["relation"], edge["to"])
                with self.subTest(shard=label, edge=key):
                    self.assertEqual(edge, edges.get(key))

    def test_array_caps_match_the_tool_side_validator(self):
        """Caps the JSON Schema does not express, so drift detection cannot see them."""
        for node in self.compiled["nodes"]:
            with self.subTest(node=node["id"]):
                self.assertLessEqual(len(node["evidence"]), gl.MAX_EVIDENCE)
                self.assertLessEqual(len(node["tags"]), gl.MAX_TAGS)
                self.assertLessEqual(
                    len(node.get("identity", {})), gl.MAX_IDENTITY_PROPERTIES
                )
                self.assertTrue(gl.ID_MIN_LENGTH <= len(node["id"]) <= gl.ID_MAX_LENGTH)

    def test_no_absolute_private_paths_or_secrets(self):
        self.assertEqual([], gl.scan_private_paths(self.compiled))
        self.assertEqual([], gl.scan_secrets(self.compiled))

    def test_verdicts_never_overstate_a_delta_inside_the_rerun_spread(self):
        """A single-run delta inside the observed evaluator spread is inconclusive.

        `finding:official-rerun-variation` records a 130 bp spread between
        byte-identical official runs. A campaign result claiming improved or
        regressed inside that band would be reading noise as signal.
        """
        spread = 130
        for node in self.compiled["nodes"]:
            if node["kind"] != "applied_result":
                continue
            if "official-run" not in node["tags"]:
                continue
            low, high = node["effect"]["delta_bp_ci"]
            if abs(low) <= spread and abs(high) <= spread:
                with self.subTest(result=node["id"]):
                    self.assertEqual("inconclusive", node["effect"]["verdict"])

    def test_measurement_provenance_matches_the_hardware_it_names(self):
        """An `official_verified` effect must be anchored to the official runner."""
        measured_on = {
            edge["from"]: edge["to"]
            for edge in self.compiled["edges"]
            if edge["relation"] == "measured_on"
        }
        for node in self.compiled["nodes"]:
            if node["kind"] != "applied_result":
                continue
            if node["provenance"] != "official_verified":
                continue
            with self.subTest(result=node["id"]):
                self.assertEqual(
                    "hardware:yukon-mlxfast-runner", measured_on.get(node["id"])
                )


class DeterminismTests(unittest.TestCase):
    def test_validator_passes_from_the_command_line(self):
        result = run_tool("validate_graph.py", "--quiet")
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn("OK", result.stdout)

    def test_compiled_graph_is_not_stale(self):
        result = run_tool("compile_graph.py", "--check")
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_mechanism_index_is_not_stale(self):
        result = run_tool("render_graph_summary.py", "--check")
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_migration_output_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a"
            second = Path(tmp) / "b"
            for target in (first, second):
                result = run_tool("migrate_v1.py", "--out", str(target))
                self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            names = sorted(path.name for path in first.glob("*.json"))
            self.assertTrue(names)
            for name in names:
                with self.subTest(shard=name):
                    self.assertEqual(
                        (first / name).read_bytes(), (second / name).read_bytes()
                    )

    def test_checked_in_migration_matches_a_fresh_run(self):
        """A stale migrated shard means the v1 assets moved without regeneration."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "migrated"
            result = run_tool("migrate_v1.py", "--out", str(target))
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            for path in sorted(target.glob("*.json")):
                checked_in = gl.SHARDS_DIR / "migrated" / path.name
                with self.subTest(shard=path.name):
                    self.assertTrue(checked_in.exists())
                    self.assertEqual(checked_in.read_bytes(), path.read_bytes())

    def test_compiler_output_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.json"
            second = Path(tmp) / "b.json"
            for target in (first, second):
                result = run_tool("compile_graph.py", "--out", str(target))
                self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertEqual(first.read_bytes(), second.read_bytes())


class PackGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = gl.load_strict(gl.COMPILED_PATH)
        cls.example = gl.PACKS_DIR / "example-kvq8-m3max"

    def _mutated(self, tmp, mutate_manifest=None, mutate_delta=None):
        target = Path(tmp) / "pack"
        shutil.copytree(self.example, target)
        if mutate_manifest is not None:
            manifest = json.loads((target / "manifest.json").read_text())
            mutate_manifest(manifest)
            (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if mutate_delta is not None:
            delta = json.loads((target / "graph-delta.json").read_text())
            mutate_delta(delta)
            (target / "graph-delta.json").write_text(json.dumps(delta, indent=2))
        return target

    def test_example_pack_is_valid(self):
        import validate_pack

        self.assertEqual([], validate_pack.validate_pack(self.example, self.compiled))

    def test_pack_may_not_claim_a_verified_measurement(self):
        import validate_pack

        def promote(delta):
            for node in delta["nodes"]:
                if node["kind"] == "applied_result":
                    node["provenance"] = "official_verified"

        with tempfile.TemporaryDirectory() as tmp:
            target = self._mutated(tmp, mutate_delta=promote)
            problems = validate_pack.validate_pack(target, self.compiled)
            self.assertTrue(
                any("contributor_claim" in item for item in problems), problems
            )

    def test_pack_may_not_redefine_an_existing_node(self):
        import validate_pack

        def redefine(delta):
            delta["nodes"].append(
                {
                    "id": "mechanism:uniform-kv-quantization",
                    "kind": "mechanism",
                    "title": "redefined",
                    "status": "x",
                    "confidence": "high",
                    "provenance": "research_inference",
                    "summary": "attempted redefinition",
                    "evidence": ["packs/x"],
                    "tags": [],
                    "observed_at": "2026-08-14",
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            target = self._mutated(tmp, mutate_delta=redefine)
            problems = validate_pack.validate_pack(target, self.compiled)
            self.assertTrue(any("already exists" in item for item in problems), problems)

    def test_pack_scrub_rejects_credentials_and_private_paths(self):
        import validate_pack

        def leak(manifest):
            # Built from parts so this file itself stays portable-path clean.
            manifest["notes"] = (
                "run from /" + "Users/someone/bench with token " + "ghp_" + "A" * 36
            )

        with tempfile.TemporaryDirectory() as tmp:
            target = self._mutated(tmp, mutate_manifest=leak)
            problems = validate_pack.validate_pack(target, self.compiled)
            self.assertTrue(any("github-token" in item for item in problems), problems)
            self.assertTrue(
                any("absolute private path" in item for item in problems), problems
            )

    def test_pack_requires_a_handle_not_an_identity(self):
        import validate_pack

        def deanonymise(manifest):
            manifest["contributor"] = "someone@example.com"

        with tempfile.TemporaryDirectory() as tmp:
            target = self._mutated(tmp, mutate_manifest=deanonymise)
            problems = validate_pack.validate_pack(target, self.compiled)
            self.assertTrue(any("plain handle" in item for item in problems), problems)

    def test_pack_receipt_digest_must_match_the_shipped_bytes(self):
        import validate_pack

        with tempfile.TemporaryDirectory() as tmp:
            target = self._mutated(tmp)
            receipt = target / "receipts" / "kvq8-vs-fp16.json"
            receipt.write_text(receipt.read_text().replace('"median_decode_tps": 44', '"median_decode_tps": 99'))
            problems = validate_pack.validate_pack(target, self.compiled)
            self.assertTrue(any("digest mismatch" in item for item in problems), problems)

    def test_float_in_a_pack_is_rejected_at_parse_time(self):
        import validate_pack

        def add_float(delta):
            for node in delta["nodes"]:
                if node["kind"] == "applied_result":
                    node["effect"]["delta_bp_ci"] = [10.5, 20]

        with tempfile.TemporaryDirectory() as tmp:
            target = self._mutated(tmp, mutate_delta=add_float)
            problems = validate_pack.validate_pack(target, self.compiled)
            self.assertTrue(any("float literal" in item for item in problems), problems)


class StrictLoaderTests(unittest.TestCase):
    def _load(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.json"
            path.write_text(text, encoding="utf-8")
            return gl.load_strict(path)

    def test_float_is_rejected(self):
        with self.assertRaises(gl.GraphError):
            self._load('{"a": 1.5}')

    def test_exponent_without_a_point_is_still_a_float(self):
        with self.assertRaises(gl.GraphError):
            self._load('{"a": 1e3}')

    def test_nan_is_rejected(self):
        with self.assertRaises(gl.GraphError):
            self._load('{"a": NaN}')

    def test_duplicate_keys_are_rejected(self):
        with self.assertRaises(gl.GraphError):
            self._load('{"a": 1, "a": 2}')

    def test_integers_survive(self):
        self.assertEqual({"a": -257}, self._load('{"a": -257}'))


if __name__ == "__main__":
    unittest.main()
