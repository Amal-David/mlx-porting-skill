"""Contracts for the structured local MLX port optimization loop."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from tests.mlx_keystone import require_mlx_keystone


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
SCRIPT = SCRIPTS / "optimize_port.py"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import optimize_port as optimizer  # noqa: E402


try:
    import mlx.core as mx
    from mlx_lm.models.llama import Model, ModelArgs
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    MLX_AVAILABLE = True
    MLX_REASON = ""
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - platform dependent
    MLX_AVAILABLE = False
    MLX_REASON = str(exc)


class OptimizePortDependencyFreeContractTests(unittest.TestCase):
    def fake_source(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        (source / "config.json").write_text(
            json.dumps({"model_type": "llama"}),
            encoding="utf-8",
        )
        (source / "model.safetensors").write_bytes(b"fixture")
        return source

    def candidate(
        self,
        config_id: str,
        *,
        passed: bool,
        tps: float,
        peak: int,
        ratio: float,
        exact: float = 1.0,
        firsttoken: float = 1.0,
        degenerate: float = 0.0,
    ) -> dict[str, object]:
        return {
            "config": {"id": config_id},
            "performance": {
                "decode_tokens_per_second_median": tps,
                "peak_memory_bytes": peak,
            },
            "quality": {
                "passed": passed,
                "perplexity_ratio": ratio,
                "exact_match_rate": exact,
                "firsttoken_agreement_rate": firsttoken,
                "candidate_only_degenerate_rate": degenerate,
            },
        }

    def test_help_and_module_import_do_not_require_optional_packages(self) -> None:
        result = subprocess.run(
            [sys.executable, "-S", str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--configs", result.stdout)
        self.assertIn("--reuse-existing", result.stdout)
        self.assertIn("diagnostics only", result.stdout)

        source = SCRIPT.read_text(encoding="utf-8")
        load_runtime_prefix = source.split("def load_runtime", 1)[0]
        self.assertNotIn("import mlx.core", load_runtime_prefix)
        self.assertNotIn("from mlx_lm", load_runtime_prefix)

    def test_config_sweep_parsing_is_ordered_and_fail_closed(self) -> None:
        parsed = optimizer.parse_config_sweep("4bit-g32,8bit")
        self.assertEqual([config.config_id for config in parsed], ["4bit-g32", "8bit"])
        self.assertEqual(parsed[0].group_size, 32)
        self.assertEqual(parsed[1].bits, 8)
        with self.assertRaisesRegex(optimizer.SkillError, "unknown --configs"):
            optimizer.parse_config_sweep("3bit")
        with self.assertRaisesRegex(optimizer.SkillError, "duplicate"):
            optimizer.parse_config_sweep("8bit,8bit")
        with self.assertRaisesRegex(optimizer.SkillError, "non-empty"):
            optimizer.parse_config_sweep("8bit,")

    def test_argument_validation_precedes_optional_imports(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(SCRIPT),
                "--model",
                "missing",
                "--work-dir",
                "missing-work",
                "--max-perplexity-ratio",
                "nan",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "error")
        self.assertIn("must be finite", payload["error"]["message"])
        self.assertNotIn("requires optional packages", payload["error"]["message"])

    def test_missing_runtime_fails_closed_with_install_guidance_and_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fake_source(root)
            work = root / "work"
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(SCRIPT),
                    "--model",
                    str(source),
                    "--work-dir",
                    str(work),
                    "--configs",
                    "8bit",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(
            set(payload),
            {"schema_version", "kind", "boundary", "status", "error"},
        )
        self.assertIn("python3 -m pip install mlx mlx-lm", payload["error"]["message"])
        self.assertFalse(payload["boundary"]["promotable_claim"])
        self.assertFalse(payload["boundary"]["writes_sealed_evidence"])
        self.assertFalse(work.exists(), "missing optional packages must not create the work directory")

    def test_recommendation_excludes_quality_failures(self) -> None:
        passing = self.candidate(
            "8bit",
            passed=True,
            tps=100.0,
            peak=1_000,
            ratio=1.02,
        )
        faster_failure = self.candidate(
            "4bit-g32",
            passed=False,
            tps=500.0,
            peak=500,
            ratio=1.20,
        )
        recommendation = optimizer.recommend_candidate([passing, faster_failure])
        self.assertEqual(recommendation["recommended_config_id"], "8bit")
        self.assertTrue(recommendation["quality_held"])
        self.assertIsNone(recommendation["fallback_config_id"])

    def test_no_pass_has_no_recommendation_and_labels_least_degrading_fallback(self) -> None:
        less_degrading = self.candidate(
            "8bit",
            passed=False,
            tps=100.0,
            peak=900,
            ratio=1.11,
            exact=0.75,
        )
        more_degrading = self.candidate(
            "4bit-g64",
            passed=False,
            tps=200.0,
            peak=500,
            ratio=1.30,
            exact=0.0,
        )
        recommendation = optimizer.recommend_candidate([more_degrading, less_degrading])
        self.assertIsNone(recommendation["recommended_config_id"])
        self.assertFalse(recommendation["quality_held"])
        self.assertEqual(recommendation["fallback_config_id"], "8bit")
        self.assertIn("not a quality-held recommendation", recommendation["warning"])

    def test_fallback_ranking_includes_first_token_divergence(self) -> None:
        first_token_failure = self.candidate(
            "4bit-g64",
            passed=False,
            tps=200.0,
            peak=500,
            ratio=1.10,
            exact=0.5,
            firsttoken=0.5,
        )
        less_divergent = self.candidate(
            "8bit",
            passed=False,
            tps=100.0,
            peak=900,
            ratio=1.10,
            exact=0.5,
            firsttoken=1.0,
        )

        recommendation = optimizer.recommend_candidate(
            [first_token_failure, less_divergent]
        )

        self.assertEqual(recommendation["fallback_config_id"], "8bit")

    def test_strict_json_schema_boundary_and_table_contract(self) -> None:
        with self.assertRaisesRegex(optimizer.SkillError, "strict JSON"):
            optimizer.strict_json({"value": math.nan})
        report = {
            "schema_version": 1,
            "kind": optimizer.DIAGNOSTIC_KIND,
            "boundary": dict(optimizer.BOUNDARY),
            "baseline": {
                "performance": {
                    "decode_tokens_per_second_median": 100.0,
                    "decode_tokens_per_second_cv": 0.02,
                    "peak_memory_gib": 1.0,
                },
                "quality_reference": {"greedy_outputs": [{}, {}]},
            },
            "candidates": [{
                "config": {"id": "8bit"},
                "performance": {
                    "decode_tokens_per_second_median": 110.0,
                    "decode_tokens_per_second_cv": 0.03,
                    "peak_memory_gib": 0.7,
                },
                "comparison_vs_bf16": {
                    "speedup_vs_bf16": 1.1,
                    "memory_reduction_fraction_vs_bf16": 0.3,
                },
                "quality": {
                    "perplexity_ratio": 1.02,
                    "exact_match_count": 1,
                    "prompt_count": 2,
                    "verdict": "pass",
                },
            }],
            "recommendation": {
                "recommended_config_id": "8bit",
                "fallback_config_id": None,
            },
        }
        encoded = json.loads(optimizer.strict_json(report))
        self.assertFalse(encoded["boundary"]["promotable_claim"])
        self.assertFalse(encoded["boundary"]["writes_sealed_evidence"])
        table = optimizer.human_table(report)
        self.assertIn("Local optimization observations", table)
        self.assertIn("8bit", table)
        self.assertIn("quality held", table)

    def test_source_rejects_remote_code_and_quantized_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fake_source(root)
            config_path = source / "config.json"
            config_path.write_text(
                json.dumps({"model_type": "llama", "auto_map": {"AutoModel": "x.Y"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(optimizer.SkillError, "custom model code"):
                optimizer.validate_source_model(str(source))
            config_path.write_text(
                json.dumps({
                    "model_type": "llama",
                    "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(optimizer.SkillError, "unquantized"):
                optimizer.validate_source_model(str(source))


@require_mlx_keystone(
    MLX_AVAILABLE,
    f"tiny mlx_lm optimization fixture is unavailable: {MLX_REASON}",
)
class OptimizePortMLXTests(unittest.TestCase):
    def write_tokenizer(self, path: Path) -> None:
        vocabulary = {
            "<unk>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "alpha": 3,
            "beta": 4,
            "gamma": 5,
            "delta": 6,
            "epsilon": 7,
            "zeta": 8,
            "eta": 9,
            "theta": 10,
            "iota": 11,
            "kappa": 12,
            "lambda": 13,
            "mu": 14,
            ".": 15,
        }
        backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            bos_token="<bos>",
            eos_token="<eos>",
        )
        tokenizer.save_pretrained(path)

    def build_source(self, root: Path) -> Path:
        model_args = ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=1,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=16,
            max_position_embeddings=128,
            tie_word_embeddings=False,
        )
        source = root / "source"
        source.mkdir()
        mx.random.seed(7)
        model = Model(model_args)
        model.save_weights(str(source / "model.safetensors"))
        (source / "config.json").write_text(
            json.dumps(asdict(model_args), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.write_tokenizer(source)
        return source

    def test_tiny_model_runs_baseline_quantization_measurement_gate_and_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.build_source(root)
            args = optimizer.parse_args([
                "--model",
                str(source),
                "--work-dir",
                str(root / "work"),
                "--configs",
                "4bit-g32",
                "--warmup-runs",
                "0",
                "--runs",
                "1",
                "--decode-tokens",
                "4",
                "--quality-max-tokens",
                "4",
                "--max-perplexity-ratio",
                "10",
                "--min-firsttoken-agreement",
                "0",
                "--max-degenerate-output-rate",
                "1",
            ])
            with mock.patch.object(
                optimizer,
                "run_quality_gate",
                wraps=optimizer.run_quality_gate,
            ) as invoked:
                report = optimizer.run_optimization(args)

        invoked.assert_called_once()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["environment"]["software"]["mlx"], "0.32.0")
        baseline = report["baseline"]["performance"]
        candidate = report["candidates"][0]
        self.assertGreater(baseline["decode_tokens_per_second_median"], 0)
        self.assertGreater(baseline["peak_memory_bytes"], 0)
        self.assertGreater(candidate["performance"]["decode_tokens_per_second_median"], 0)
        self.assertGreater(candidate["performance"]["peak_memory_bytes"], 0)
        self.assertEqual(candidate["quality"]["verdict"], "pass")
        self.assertEqual(len(candidate["quality"]["exact_output_divergence"]), 4)
        self.assertIn(
            "first_divergence_token_index",
            candidate["quality"]["exact_output_divergence"][0],
        )
        self.assertEqual(report["recommendation"]["recommended_config_id"], "4bit-g32")
        self.assertFalse(report["boundary"]["promotable_claim"])
        self.assertFalse(report["boundary"]["writes_sealed_evidence"])


if __name__ == "__main__":
    unittest.main()
