"""Dependency-free interaction contracts for the public learning atlas."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE_DATA = ROOT / "site" / "data.js"
ATLAS = ROOT / "site" / "atlas.js"
GLOBAL_PREFIX = "window.MLX_PORTING_SITE_DATA = "

JOURNEY_IDS = (
    "qwen25-dense-decoder",
    "whisper-style-asr",
    "flux-style-diffusion",
    "llava-style-vlm",
)
CHECKPOINT_IDS = (
    "inspect",
    "oracle",
    "implement",
    "map",
    "parity",
    "profile",
    "optimize",
    "publish",
)
API_FUNCTIONS = (
    "validateAtlasLearning",
    "parseAtlasState",
    "serializeAtlasState",
    "moveCheckpointFocus",
    "stepCheckpoint",
    "exportTextPlan",
)


def load_generated_learning() -> dict[str, object]:
    text = SITE_DATA.read_text(encoding="utf-8")
    if not text.startswith(GLOBAL_PREFIX) or not text.endswith(";\n"):
        raise AssertionError("site data must remain one deterministic global assignment")
    payload = json.loads(text[len(GLOBAL_PREFIX):-2])
    learning = payload.get("learning")
    if not isinstance(learning, dict):
        raise AssertionError("generated site data is missing the learning payload")
    return learning


class AtlasInteractionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.learning = load_generated_learning()
        cls.node = shutil.which("node")

    def run_node(self, body: str) -> subprocess.CompletedProcess[str]:
        self.assertIsNotNone(self.node, "Node.js is required by the repository validation contract")
        harness = r"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = process.cwd();
const sandbox = { window: {} };
vm.runInNewContext(
  fs.readFileSync(path.join(root, "site", "data.js"), "utf8"),
  sandbox,
  { filename: "site/data.js" },
);
const learning = sandbox.window.MLX_PORTING_SITE_DATA.learning;
const atlas = require(path.join(root, "site", "atlas.js"));
""" + body
        return subprocess.run(
            [self.node or "node", "-e", harness],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def assert_node_ok(self, body: str) -> None:
        result = self.run_node(body)
        self.assertEqual(
            result.returncode,
            0,
            f"Node atlas contract failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

    def test_generated_payload_exposes_exact_journeys_and_checkpoint_order(self) -> None:
        journey_ids = tuple(row["id"] for row in self.learning["journeys"])
        checkpoint_ids = tuple(self.learning["checkpoint_order"])
        self.assertEqual(journey_ids, JOURNEY_IDS)
        self.assertEqual(checkpoint_ids, CHECKPOINT_IDS)
        self.assertEqual(
            {row["status"] for row in self.learning["journeys"]},
            {"proven", "simulation"},
        )
        self.assertTrue(all(row["proof_boundary"].strip() for row in self.learning["journeys"]))

    def test_invalid_learning_payloads_fail_closed_before_enhancement(self) -> None:
        self.assert_node_ok(
            r"""
assert.equal(atlas.validateAtlasLearning(learning), true);
for (const mutate of [
  (copy) => { copy.journeys[0].component_path = null; },
  (copy) => { delete copy.journeys[0].proof_boundary; },
  (copy) => { delete copy.journeys[0].checkpoint_notes.inspect; },
  (copy) => { copy.checkpoint_nodes[0].proof = ""; },
]) {
  const copy = JSON.parse(JSON.stringify(learning));
  mutate(copy);
  assert.equal(atlas.validateAtlasLearning(copy), false);
}
""",
        )

    def test_atlas_script_is_dependency_free_syntax_valid_and_dom_free_on_load(self) -> None:
        self.assertTrue(ATLAS.is_file(), "site/atlas.js must implement the atlas interaction API")
        source = ATLAS.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"\brequire\s*\(", source), "browser code must not import packages")
        self.assertIsNone(re.search(r"\bimport\s+(?:[^.(]|$)", source), "browser code must stay classic-script safe")

        result = subprocess.run(
            [self.node or "node", "--check", str(ATLAS)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        expected = json.dumps(API_FUNCTIONS)
        self.assert_node_ok(
            f"""
for (const name of {expected}) {{
  assert.equal(typeof atlas[name], "function", `${{name}} must be exported`);
}}
""",
        )

    def test_all_32_model_and_checkpoint_states_round_trip_through_the_url(self) -> None:
        self.assert_node_ok(
            r"""
const journeys = learning.journeys.map((journey) => journey.id);
const checkpoints = Array.from(learning.checkpoint_order);
let checked = 0;
for (const journeyId of journeys) {
  for (const checkpointId of checkpoints) {
    const state = { journeyId, checkpointId };
    const serialized = atlas.serializeAtlasState(
      "https://example.test/site/docs/index.html?mode=study#porting-atlas",
      state,
    );
    const url = new URL(serialized);
    assert.equal(url.searchParams.get("mode"), "study");
    assert.equal(url.searchParams.get("atlas-model"), journeyId);
    assert.equal(url.searchParams.get("atlas-node"), checkpointId);
    assert.equal(url.hash, "#porting-atlas");
    const restored = atlas.parseAtlasState(learning, serialized);
    assert.equal(restored.journeyId, journeyId);
    assert.equal(restored.checkpointId, checkpointId);
    checked += 1;
  }
}
assert.equal(checked, 32);
""",
        )

    def test_invalid_or_missing_url_state_falls_back_independently(self) -> None:
        self.assert_node_ok(
            r"""
const missing = atlas.parseAtlasState(
  learning,
  "https://example.test/site/docs/index.html#porting-atlas",
);
assert.equal(missing.journeyId, "qwen25-dense-decoder");
assert.equal(missing.checkpointId, "inspect");

const invalid = atlas.parseAtlasState(
  learning,
  "https://example.test/site/docs/index.html?atlas-model=unknown&atlas-node=unknown#porting-atlas",
);
assert.equal(invalid.journeyId, "qwen25-dense-decoder");
assert.equal(invalid.checkpointId, "inspect");

const validJourney = atlas.parseAtlasState(
  learning,
  "https://example.test/site/docs/index.html?atlas-model=whisper-style-asr&atlas-node=unknown#porting-atlas",
);
assert.equal(validJourney.journeyId, "whisper-style-asr");
assert.equal(validJourney.checkpointId, "inspect");

const validNode = atlas.parseAtlasState(
  learning,
  "https://example.test/site/docs/index.html?atlas-model=unknown&atlas-node=parity#porting-atlas",
);
assert.equal(validNode.journeyId, "qwen25-dense-decoder");
assert.equal(validNode.checkpointId, "parity");
""",
        )

    def test_keyboard_focus_and_guided_traversal_transitions_are_bounded(self) -> None:
        self.assert_node_ok(
            r"""
const order = Array.from(learning.checkpoint_order);
assert.equal(atlas.moveCheckpointFocus(order, "implement", "ArrowRight"), "map");
assert.equal(atlas.moveCheckpointFocus(order, "implement", "ArrowDown"), "map");
assert.equal(atlas.moveCheckpointFocus(order, "implement", "ArrowLeft"), "oracle");
assert.equal(atlas.moveCheckpointFocus(order, "implement", "ArrowUp"), "oracle");
assert.equal(atlas.moveCheckpointFocus(order, "implement", "Home"), "inspect");
assert.equal(atlas.moveCheckpointFocus(order, "implement", "End"), "publish");
assert.equal(atlas.moveCheckpointFocus(order, "implement", "Escape"), "implement");

assert.equal(atlas.stepCheckpoint(order, "inspect", "previous"), "inspect");
assert.equal(atlas.stepCheckpoint(order, "inspect", "next"), "oracle");
assert.equal(atlas.stepCheckpoint(order, "parity", "previous"), "map");
assert.equal(atlas.stepCheckpoint(order, "parity", "next"), "profile");
assert.equal(atlas.stepCheckpoint(order, "publish", "previous"), "optimize");
assert.equal(atlas.stepCheckpoint(order, "publish", "next"), "publish");
""",
        )

    def test_text_exports_are_readable_complete_and_preserve_status_disclaimers(self) -> None:
        self.assert_node_ok(
            r"""
const checkpoints = new Map(learning.checkpoint_nodes.map((node) => [node.id, node]));
for (const journey of learning.journeys) {
  const plan = atlas.exportTextPlan(learning, {
    journeyId: journey.id,
    checkpointId: "inspect",
  });
  assert.equal(typeof plan, "string");
  assert.match(plan, /MLX port plan/i);
  assert.ok(plan.includes(journey.title));
  assert.ok(plan.includes(journey.modality));
  assert.ok(plan.includes(learning.journey_statuses[journey.status]));
  assert.ok(plan.includes(journey.proof_boundary));
  assert.ok(!plan.includes("[object Object]"));
  assert.ok(!plan.trimStart().startsWith("{"), "export must be prose, not JSON");
  for (const checkpointId of learning.checkpoint_order) {
    assert.ok(plan.includes(checkpoints.get(checkpointId).title));
    assert.ok(plan.includes(journey.checkpoint_notes[checkpointId]));
  }
  if (journey.status === "simulation") {
    assert.match(plan, /not a completed checkpoint port/i);
  } else {
    assert.equal(journey.id, "qwen25-dense-decoder");
    assert.match(plan, /checked-in, reproducible MLX proof packet/i);
  }
  assert.ok(plan.split(/\r?\n/).filter(Boolean).length >= 12);
}
""",
        )


if __name__ == "__main__":
    unittest.main()
