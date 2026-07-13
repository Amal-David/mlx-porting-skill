"""Contracts for the user-only MLX quantization quality diagnostic."""
from __future__ import annotations

import argparse
import json
import math
import shutil
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
SCRIPT = SCRIPTS / "quant_quality_gate.py"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import quant_quality_gate as gate  # noqa: E402


try:
    import mlx.core as mx
    from mlx_lm.models.llama import Model, ModelArgs
    from mlx_lm.utils import quantize_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    MLX_AVAILABLE = True
    MLX_REASON = ""
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - platform dependent
    MLX_AVAILABLE = False
    MLX_REASON = str(exc)


class QuantQualityGateContractTests(unittest.TestCase):
    def fake_model_dir(self, root: Path, name: str, *, quantized: bool = False) -> Path:
        path = root / name
        path.mkdir()
        config: dict[str, object] = {"model_type": "llama"}
        if quantized:
            config["quantization"] = {"group_size": 32, "bits": 4, "mode": "affine"}
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"fixture")
        return path

    def test_help_and_module_import_do_not_require_optional_mlx_packages(self) -> None:
        result = subprocess.run(
            [sys.executable, "-S", str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--max-perplexity-ratio", result.stdout)
        self.assertIn("--min-firsttoken-agreement", result.stdout)
        self.assertIn("user diagnostic only", result.stdout)

        source = SCRIPT.read_text(encoding="utf-8")
        load_runtime_prefix = source.split("def load_runtime", 1)[0]
        self.assertNotIn("import mlx.core", load_runtime_prefix)
        self.assertNotIn("from mlx_lm", load_runtime_prefix)

    def test_argument_validation_happens_before_optional_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = self.fake_model_dir(root, "reference")
            candidate = self.fake_model_dir(root, "candidate", quantized=True)
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(SCRIPT),
                    "--reference",
                    str(reference),
                    "--candidate",
                    str(candidate),
                    "--max-perplexity-ratio",
                    "nan",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["verdict"], "error")
        self.assertIn("must be finite", payload["error"]["message"])
        self.assertNotIn("requires optional packages", payload["error"]["message"])

    def test_missing_optional_runtime_fails_closed_with_install_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = self.fake_model_dir(root, "reference")
            candidate = self.fake_model_dir(root, "candidate", quantized=True)
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(SCRIPT),
                    "--reference",
                    str(reference),
                    "--candidate",
                    str(candidate),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(set(payload), {"schema_version", "kind", "verdict", "error"})
        self.assertEqual(payload["verdict"], "error")
        self.assertIn("python3 -m pip install mlx mlx-lm", payload["error"]["message"])

    def test_threshold_logic_passes_and_fails_on_each_configured_signal(self) -> None:
        baseline = dict(
            reference_perplexity=10.0,
            candidate_perplexity=10.5,
            firsttoken_agreement_count=3,
            prompt_count=4,
            candidate_only_degenerate_count=0,
            max_perplexity_ratio=1.10,
            min_firsttoken_agreement=0.75,
            max_degenerate_output_rate=0.0,
        )
        result = gate.evaluate_thresholds(**baseline)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["perplexity_ratio"], 1.05)
        self.assertAlmostEqual(result["firsttoken_agreement"], 0.75)
        self.assertEqual(
            set(result["checks"]),
            {
                "perplexity_ratio",
                "firsttoken_agreement",
                "candidate_only_degenerate_output_rate",
            },
        )

        cases = {
            "perplexity": {"candidate_perplexity": 11.1},
            "first-token": {"firsttoken_agreement_count": 2},
            "degenerate": {"candidate_only_degenerate_count": 1},
            "non-finite": {"candidate_perplexity": None},
        }
        for name, replacement in cases.items():
            with self.subTest(name=name):
                self.assertFalse(gate.evaluate_thresholds(**(baseline | replacement))["passed"])

    def test_prompt_contract_and_degenerate_detector_are_dependency_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.json"
            path.write_text(
                json.dumps({
                    "held_text": "A fixed held sentence long enough for scoring.",
                    "prompts": ["One prompt", "A second prompt"],
                }),
                encoding="utf-8",
            )
            workload = gate.load_workload(str(path))
        self.assertEqual(workload["prompts"], ["One prompt", "A second prompt"])
        looping = gate.detect_degenerate_output(
            list(range(14)),
            "ionoxffionoxff",
        )
        self.assertTrue(looping["detected"])
        self.assertIn("repeated-character-cycle", looping["reasons"])
        self.assertFalse(gate.detect_degenerate_output([1, 2, 3, 4], "A normal answer.")["detected"])
        # A short single-token loop (six copies + EOS) must be flagged, not slip
        # through a length gate (regression for the escaped 7-token loop).
        short_loop = gate.detect_degenerate_output([7, 7, 7, 7, 7, 7, 2], "aaaaaa")
        self.assertTrue(short_loop["detected"])
        self.assertIn("single-token-run", short_loop["reasons"])

    def test_strict_json_and_user_only_boundary_contract(self) -> None:
        with self.assertRaisesRegex(gate.SkillError, "strict JSON"):
            gate.strict_json({"value": math.nan})
        parsed = gate.parse_args(["--reference", "reference", "--candidate", "candidate"])
        self.assertFalse(hasattr(parsed, "output"))
        source = SCRIPT.read_text(encoding="utf-8")
        for sealed_consumer in (
            "generate_claim_catalog",
            "generate_evidence_index",
            "validate_benchmarks",
            "receipt_assessments.json",
            "effective_claims.json",
        ):
            self.assertNotIn(sealed_consumer, source)

    def test_model_directory_contract_rejects_wrong_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = self.fake_model_dir(root, "reference", quantized=True)
            candidate = self.fake_model_dir(root, "candidate")
            args = argparse.Namespace(
                reference=str(reference),
                candidate=str(candidate),
                prompts_file=None,
                max_perplexity_ratio=1.1,
                min_firsttoken_agreement=0.75,
                max_degenerate_output_rate=0.0,
                max_tokens=8,
            )
            with mock.patch.object(gate, "load_runtime") as load_runtime:
                with self.assertRaisesRegex(gate.SkillError, "--reference must be an unquantized"):
                    gate.run_gate(args)
            load_runtime.assert_not_called()


@require_mlx_keystone(
    MLX_AVAILABLE,
    f"tiny mlx_lm quantization quality fixture is unavailable: {MLX_REASON}",
)
class QuantQualityGateMLXTests(unittest.TestCase):
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

    def write_config(self, path: Path, config: dict[str, object]) -> None:
        (path / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def build_models(self, root: Path) -> tuple[Path, Path, Path]:
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
        config = asdict(model_args)

        reference = root / "reference"
        reference.mkdir()
        mx.random.seed(7)
        reference_model = Model(model_args)
        reference_model.save_weights(str(reference / "model.safetensors"))
        self.write_config(reference, config)
        self.write_tokenizer(reference)

        candidate = root / "candidate"
        candidate.mkdir()
        quantized_model = Model(model_args)
        quantized_model.load_weights(str(reference / "model.safetensors"))
        quantized_model, quantized_config = quantize_model(
            quantized_model,
            config,
            group_size=32,
            bits=4,
        )
        quantized_model.save_weights(str(candidate / "model.safetensors"))
        self.write_config(candidate, quantized_config)
        self.write_tokenizer(candidate)

        corrupted = root / "corrupted"
        shutil.copytree(candidate, corrupted)
        weights = mx.load(str(corrupted / "model.safetensors"))
        scale_keys = [key for key in weights if key.endswith(".scales")]
        self.assertTrue(scale_keys)
        for key in scale_keys:
            weights[key] = weights[key] * 1000.0
        mx.save_safetensors(str(corrupted / "model.safetensors"), weights)
        return reference, candidate, corrupted

    def run_fixture(
        self,
        reference: Path,
        candidate: Path,
        prompts_file: Path,
        max_ratio: float,
    ) -> dict[str, object]:
        args = gate.parse_args([
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--prompts-file",
            str(prompts_file),
            "--max-perplexity-ratio",
            str(max_ratio),
            "--min-firsttoken-agreement",
            "0",
            "--max-degenerate-output-rate",
            "1",
            "--max-tokens",
            "8",
        ])
        return gate.run_gate(args)

    def test_tiny_loadable_quantization_passes_and_corruption_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, candidate, corrupted = self.build_models(root)
            prompts_file = root / "prompts.json"
            prompts_file.write_text(
                json.dumps({
                    "held_text": (
                        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
                        "alpha beta gamma delta."
                    ),
                    "prompts": ["alpha beta gamma", "delta epsilon zeta"],
                }),
                encoding="utf-8",
            )
            clean = self.run_fixture(reference, candidate, prompts_file, max_ratio=1.1)
            broken = self.run_fixture(reference, corrupted, prompts_file, max_ratio=1.1)

        self.assertEqual(clean["verdict"], "pass")
        self.assertTrue(clean["passed"])
        clean_perplexity = clean["metrics"]["perplexity"]
        self.assertGreaterEqual(clean_perplexity["reference"], 1.0)
        self.assertLess(clean_perplexity["reference"], gate.MAX_SANE_REFERENCE_PERPLEXITY)
        self.assertLessEqual(clean_perplexity["ratio"], 1.1)
        self.assertEqual(clean["boundary"]["classification"], "user_diagnostic_only")
        self.assertFalse(clean["boundary"]["promotable_claim"])

        self.assertEqual(broken["verdict"], "fail")
        self.assertFalse(broken["passed"])
        self.assertFalse(broken["checks"]["perplexity_ratio"]["passed"])


if __name__ == "__main__":
    unittest.main()
