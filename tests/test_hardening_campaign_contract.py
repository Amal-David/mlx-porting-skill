"""Security contracts for generated and executed research campaigns."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "mlx-model-porting"
SCRIPTS = SKILL / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import research_loop  # noqa: E402
import run_research_campaign  # noqa: E402
from _common import SkillError  # noqa: E402


def campaign_agent(
    persona_id: str = "researcher-a",
    *,
    index: int = 0,
    prefix: str = "wave",
) -> dict[str, object]:
    return {
        "assignment_index": index,
        "persona_id": persona_id,
        "title": "Researcher A",
        "source_lanes": ["official_docs"],
        "assignment_path": f"{prefix}/agents/{persona_id}.assignment.json",
        "prompt_path": f"{prefix}/agents/{persona_id}.prompt.md",
        "result_path": f"{prefix}/agents/{persona_id}.result.json",
        "blog_path": f"{prefix}/blogs/{persona_id}.md",
        "blog_source": "pending",
        "execution_kind": "offline-scaffold",
        "execution_state": "scaffolded_not_run",
        "review_only": True,
    }


def campaign_value(*agents: dict[str, object], output_dir: str = "wave") -> dict[str, object]:
    selected = list(agents) or [campaign_agent(prefix=output_dir)]
    return {
        "schema_version": 1,
        "run_id": "campaign-a",
        "objective": "Review evidence safely",
        "review_only": True,
        "campaign_root": ".",
        "wave_count": 1,
        "iteration_cap": 1,
        "until_review_gate": False,
        "stop_reason": "single-wave",
        "waves": [{
            "iteration": 1,
            "run_id": "campaign-a-wave-1",
            "output_dir": output_dir,
            "subagent_manifest_path": f"{output_dir}/subagents.json",
            "assignment_mode": "config-order",
            "gap_hints": [],
            "next_gap_hints": [],
            "agent_count": len(selected),
            "agents": selected,
            "ingest": {"command_args": [sys.executable, "scripts/research_loop.py"]},
            "wave_dependency": {"reason": "single-wave"},
        }],
        "orchestration_contract": {
            "dispatch_model": "one review-only subagent per campaign wave agent",
            "parallelism": "bounded",
            "promotion_rule": "review-only",
        },
        "instructions": [],
    }


def write_campaign(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class CampaignHardeningContractTests(unittest.TestCase):
    def test_campaign_runner_quarantines_stale_result_before_noop_agent(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            agent = campaign_agent(prefix=".")
            result = root / str(agent["result_path"])
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text(json.dumps({"persona_id": agent["persona_id"], "findings": []}), encoding="utf-8")

            record, failure = run_research_campaign.execute_agent(
                root,
                root / "campaign.json",
                {"run_id": "campaign-a"},
                {"iteration": 1, "run_id": "wave-a", "output_dir": "."},
                agent,
                [sys.executable, "-c", "pass"],
                f"{sys.executable} -c pass",
                10,
                1,
                1,
                root / "logs",
            )

            self.assertEqual(record["state"], "agent_failed")
            self.assertIn("did not write expected result", failure)
            self.assertFalse(result.exists())
            self.assertTrue(list((root / ".stale-results").rglob("*.stale")))

    def test_local_executor_quarantines_stale_result_before_noop_command(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            output_dir = Path(raw_tmp)
            assignment = {
                "persona_id": "researcher-a",
                "title": "Researcher A",
                "prompt": "Review evidence.",
            }
            result = research_loop.agent_output_paths(output_dir, assignment)["result"]
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text(json.dumps({"persona_id": "researcher-a", "findings": []}), encoding="utf-8")

            with self.assertRaisesRegex(SkillError, "executor did not write result JSON"):
                research_loop.execute_assignments(
                    [assignment],
                    f"{sys.executable} -c pass",
                    output_dir,
                    "run-a",
                    10,
                    1,
                )
            self.assertFalse(result.exists())
            self.assertTrue(list((output_dir / ".stale-results").rglob("*.stale")))

    def test_generated_scaffold_quarantines_preexisting_results(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            campaign = campaign_value(campaign_agent())
            result = root / str(campaign["waves"][0]["agents"][0]["result_path"])
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text('{"stale": true}\n', encoding="utf-8")

            research_loop.write_campaign_receipts(root, campaign)

            self.assertFalse(result.exists())
            self.assertTrue((root / "campaign.json").is_file())
            self.assertTrue(list((root / ".stale-results").rglob("*.stale")))

    def test_campaign_schema_rejects_unbounded_or_ambiguous_agents(self) -> None:
        cases: list[tuple[str, dict[str, object], str]] = []

        unknown_key = campaign_value(campaign_agent())
        unknown_key["waves"][0]["agents"][0]["unexpected"] = True
        cases.append(("unknown-key", unknown_key, "unsupported keys"))

        duplicate_id = campaign_value(campaign_agent(), campaign_agent(index=1))
        duplicate_id["waves"][0]["agent_count"] = 2
        cases.append(("duplicate-id", duplicate_id, "duplicate persona_id"))

        duplicate_log_identity = campaign_value(
            campaign_agent("researcher!", index=0),
            campaign_agent("researcher@", index=1),
        )
        duplicate_log_identity["waves"][0]["agent_count"] = 2
        cases.append(("duplicate-log-identity", duplicate_log_identity, "duplicate persona log identity"))

        duplicate_result = campaign_value(
            campaign_agent("researcher-a", index=0),
            campaign_agent("researcher-b", index=1),
        )
        duplicate_result["waves"][0]["agent_count"] = 2
        duplicate_result["waves"][0]["agents"][1]["result_path"] = duplicate_result["waves"][0]["agents"][0]["result_path"]
        cases.append(("duplicate-result", duplicate_result, "duplicate result_path"))

        sibling_result = campaign_value(campaign_agent(prefix="sibling"), output_dir="wave")
        cases.append(("sibling-result", sibling_result, "outside wave output_dir"))

        too_many = campaign_value(*[
            campaign_agent(f"researcher-{index}", index=index)
            for index in range(run_research_campaign.MAX_AGENTS_PER_WAVE + 1)
        ])
        too_many["waves"][0]["agent_count"] = len(too_many["waves"][0]["agents"])
        cases.append(("too-many-agents", too_many, "agent limit"))

        for name, value, expected in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as raw_tmp:
                path = Path(raw_tmp) / "campaign.json"
                write_campaign(path, value)
                with self.assertRaisesRegex(SkillError, expected):
                    run_research_campaign.load_campaign(path)

    def test_campaign_schema_rejects_symlink_loop_paths_during_load(self) -> None:
        for field in ("output_dir", "result_path"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as raw_tmp:
                root = Path(raw_tmp)
                (root / "wave").mkdir()
                (root / "wave" / "loop").symlink_to("loop")
                value = campaign_value(campaign_agent())
                if field == "output_dir":
                    value["waves"][0]["output_dir"] = "wave/loop"
                else:
                    value["waves"][0]["agents"][0]["result_path"] = "wave/loop/result.json"
                path = root / "campaign.json"
                write_campaign(path, value)
                with self.assertRaisesRegex(SkillError, "symlink cycle"):
                    run_research_campaign.load_campaign(path)

    def test_campaign_schema_bounds_waves_and_requires_unique_run_ids(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            value = campaign_value(campaign_agent())
            wave = value["waves"][0]
            value["waves"] = [dict(wave) for _ in range(run_research_campaign.MAX_CAMPAIGN_WAVES + 1)]
            value["wave_count"] = len(value["waves"])
            path = root / "campaign.json"
            write_campaign(path, value)
            with self.assertRaisesRegex(SkillError, "wave limit"):
                run_research_campaign.load_campaign(path)

            value = campaign_value(campaign_agent())
            duplicate = dict(value["waves"][0])
            duplicate["iteration"] = 2
            duplicate["output_dir"] = "wave-2"
            duplicate["agents"] = [campaign_agent(prefix="wave-2")]
            value["waves"].append(duplicate)
            value["wave_count"] = 2
            write_campaign(path, value)
            with self.assertRaisesRegex(SkillError, "duplicate wave run_id"):
                run_research_campaign.load_campaign(path)


if __name__ == "__main__":
    unittest.main()
