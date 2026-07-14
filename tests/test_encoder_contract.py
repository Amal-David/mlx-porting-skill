"""Contracts and real offline keystone for the additive BERT encoder port."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
SNAPSHOTS = (
    Path.home()
    / ".cache/huggingface/hub/models--BAAI--bge-base-en/snapshots"
)
REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import run_parity  # noqa: E402
import scaffold_port  # noqa: E402
import capture_oracle  # noqa: E402
from mlx_keystone import require_mlx_keystone  # noqa: E402


def _snapshot() -> Path | None:
    if not SNAPSHOTS.is_dir():
        return None
    candidates = sorted(path for path in SNAPSHOTS.iterdir() if path.is_dir())
    for candidate in candidates:
        if all((candidate / name).is_file() for name in REQUIRED_SNAPSHOT_FILES):
            return candidate
    return None


def _mlx_runtime_available() -> bool:
    if importlib.util.find_spec("mlx") is None:
        return False
    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core as mx; x=mx.array([1]); mx.eval(x)"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


SOURCE_SNAPSHOT = _snapshot()
HAS_EXECUTION_STACK = (
    SOURCE_SNAPSHOT is not None
    and _mlx_runtime_available()
    and all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch", "transformers"))
)


def _config(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "architectures": ["BertModel"],
        "hidden_act": "gelu",
        "hidden_size": 8,
        "intermediate_size": 16,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 32,
        "model_type": "bert",
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "type_vocab_size": 2,
        "vocab_size": 16,
    }
    value.update(overrides)
    return value


class EncoderFailClosedContractTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is required")
    def test_masked_nonzero_integer_ids_are_never_zeroed(self) -> None:
        import torch

        input_ids = torch.tensor([[101, 999]], dtype=torch.int64)
        attention_mask = torch.tensor([[1, 0]], dtype=torch.int64)
        preserved = capture_oracle._zero_padded_query_rows(
            input_ids, attention_mask, torch
        )

        torch.testing.assert_close(preserved, input_ids, rtol=0, atol=0)

    def test_encoder_ladder_has_mask_no_autoregressive_rung(self) -> None:
        keys = {
            "input_ids", "attention_mask", "embed", "layer.0.hidden",
            "layer.1.hidden", "final_hidden", "pooled",
        }
        ladder = run_parity.build_parity_ladder(keys, keys, "encoder")
        self.assertEqual(
            [rung["name"] for rung in ladder],
            [
                "input_ids", "attention_mask", "embed", "layer.0.hidden",
                "layer.1.hidden", "final_hidden", "pooled",
            ],
        )
        self.assertTrue(ladder[1]["exact"])
        self.assertNotIn("generated_token_ids", {rung["name"] for rung in ladder})

    def test_relative_position_config_fails_closed(self) -> None:
        with self.assertRaisesRegex(scaffold_port.SkillError, "only 'absolute' is accepted"):
            scaffold_port.validate_encoder_config(
                _config(position_embedding_type="relative_key")
            )

    def test_roberta_identity_fails_closed_without_position_offset_support(self) -> None:
        with self.assertRaisesRegex(scaffold_port.SkillError, "only 'bert' is accepted"):
            scaffold_port.validate_encoder_config(
                _config(model_type="roberta", architectures=["RobertaModel"])
            )

    def test_non_bert_architecture_identity_fails_closed(self) -> None:
        with self.assertRaisesRegex(scaffold_port.SkillError, "supported BERT identities"):
            scaffold_port.validate_encoder_config(_config(architectures=["RobertaModel"]))

    def test_decoder_semantic_flags_fail_closed_independently(self) -> None:
        for flag in ("is_decoder", "is_encoder_decoder", "add_cross_attention"):
            with self.subTest(flag=flag), self.assertRaisesRegex(
                scaffold_port.SkillError,
                rf"{flag}=true is not supported",
            ):
                scaffold_port.validate_encoder_config(_config(**{flag: True}))

    def test_bert_task_head_identity_fails_closed(self) -> None:
        self.assertEqual(scaffold_port.SUPPORTED_BERT_ARCHITECTURES, {"BertModel"})
        with self.assertRaisesRegex(scaffold_port.SkillError, "BertModel"):
            scaffold_port.validate_encoder_config(
                _config(architectures=["BertForMaskedLM"])
            )

    def test_falsey_unknown_computation_value_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            scaffold_port.SkillError,
            "unrecognized computation-relevant config key 'mystery_attention_scale'",
        ):
            scaffold_port.validate_encoder_config(
                _config(mystery_attention_scale=0)
            )

    def test_decoder_tensor_inventory_fails_closed(self) -> None:
        config = _config()
        tensors = [
            {"key": key, "shape": shape}
            for key, shape in scaffold_port.encoder_target_tensor_dict(
                config, pooler=True
            ).items()
        ]
        tensors.append({"key": "decoder.layer.0.self_attn.q_proj.weight", "shape": [8, 8]})
        with self.assertRaisesRegex(
            scaffold_port.SkillError,
            "decoder/cross-attention/expert tensors",
        ):
            scaffold_port.validate_encoder_topology({"tensors": tensors}, config)


@require_mlx_keystone(
    HAS_EXECUTION_STACK,
    "cached BAAI/bge-base-en plus MLX, NumPy, PyTorch, and Transformers are required",
    needs_real_model=True,
)
class RealBgeEncoderParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not HAS_EXECUTION_STACK or SOURCE_SNAPSHOT is None:
            return
        cls._temporary = tempfile.TemporaryDirectory(
            prefix=".mlx-bge-encoder-test-", dir=ROOT
        )
        cls.work = Path(cls._temporary.name)
        cls.model = cls.work / "bge-base-en"
        cls.model.mkdir()
        for name in REQUIRED_SNAPSHOT_FILES:
            shutil.copy2(SOURCE_SNAPSHOT / name, cls.model / name, follow_symlinks=True)
        config = json.loads((cls.model / "config.json").read_text(encoding="utf-8"))
        config["license"] = "mit"
        (cls.model / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        cls.inspection = cls.work / "inspection.json"
        cls.package = cls.work / "mlx-port"
        cls.weights = cls.work / "converted"
        cls.mapping = cls.work / "WEIGHT_MAP.json"
        cls._run(
            SCRIPTS / "inspect_model.py", cls.model, "--output", cls.inspection
        )
        cls._run(
            SCRIPTS / "scaffold_port.py", cls.inspection,
            "--artifact-root", cls.model, "--output", cls.package,
        )
        draft = cls.work / "draft.json"
        cls._run(
            SCRIPTS / "convert_checkpoint.py", "--source", cls.inspection,
            "--scaffold-manifest", cls.package / "scaffold-manifest.json",
            "--emit-draft-map", draft,
        )
        mapping = json.loads(draft.read_text(encoding="utf-8"))
        mapping.update({"draft": False, "dtype_policy": "f32", "unresolved": []})
        mapping["ignore"] = [{
            "source": "embeddings.position_ids",
            "reason": "deterministic non-parameter absolute-position buffer regenerated by MLX",
        }]
        cls.mapping.write_text(
            json.dumps(mapping, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        cls._run(
            SCRIPTS / "convert_checkpoint.py", "--source", cls.model,
            "--mapping", cls.mapping, "--output", cls.weights,
        )
        cls._run(
            SCRIPTS / "validate_weight_map.py", "--source", cls.inspection,
            "--target", cls.package / "scaffold-manifest.json",
            "--mapping", cls.mapping,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        temporary = getattr(cls, "_temporary", None)
        if temporary is not None:
            temporary.cleanup()

    @classmethod
    def _run(cls, script: Path, *args: object, expected: int = 0) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        completed = subprocess.run(
            [sys.executable, str(script), *(str(value) for value in args)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != expected:
            raise AssertionError(completed.stdout + completed.stderr)
        return completed

    @classmethod
    def _parity_args(cls, weights: Path, output: Path) -> tuple[object, ...]:
        return (
            "--source-model", cls.model,
            "--package", cls.package,
            "--weights", weights,
            "--mode", "encoder",
            "--token-ids", "101", "7592", "2088", "999",
            "--attention-mask", "1", "1", "1", "0",
            "--generate-steps", "0",
            "--atol", "5e-5",
            "--rtol", "1e-4",
            "--output", output,
        )

    @classmethod
    def _batch_parity_args(cls, output: Path) -> tuple[object, ...]:
        return (
            "--source-model", cls.model,
            "--package", cls.package,
            "--weights", cls.weights,
            "--mode", "encoder",
            "--prompt", "hello",
            "--prompt", "hello world",
            "--generate-steps", "0",
            "--atol", "5e-5",
            "--rtol", "1e-4",
            "--output", output,
        )

    def test_real_bge_masked_parity_ladder(self) -> None:
        report_path = self.work / "parity.json"
        self._run(SCRIPTS / "run_parity.py", *self._parity_args(self.weights, report_path))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertTrue(report["ok"])
        self.assertEqual(report["rungs"][1]["name"], "attention_mask")
        self.assertTrue(report["rungs"][1]["exact"])
        self.assertEqual(report["rungs"][-2]["name"], "final_hidden")
        self.assertEqual(report["rungs"][-1]["name"], "pooled")
        self.assertEqual(report["inputs"]["mode"], "encoder")
        self.assertEqual(report["inputs"]["input_mode"], "token_ids")
        self.assertEqual(report["inputs"]["attention_mask"], [[1, 1, 1, 0]])

    def test_real_bge_padded_unequal_length_prompt_batch(self) -> None:
        report_path = self.work / "batch-parity.json"
        self._run(SCRIPTS / "run_parity.py", *self._batch_parity_args(report_path))
        report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["inputs"]["mode"], "encoder")
        self.assertEqual(report["inputs"]["input_mode"], "prompt")
        self.assertEqual(report["inputs"]["prompts"], ["hello", "hello world"])
        self.assertEqual(report["inputs"]["attention_mask"], [[1, 1, 1, 0], [1, 1, 1, 1]])

    def test_seeded_weight_corruption_stops_at_affected_layer(self) -> None:
        report_path = self.work / "corrupt-parity.json"
        self._run(
            SCRIPTS / "run_parity.py",
            *self._parity_args(self.weights, report_path),
            "--fault-inject-target", "layer.4.hidden",
            expected=1,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertFalse(report["ok"])
        self.assertEqual(report["summary"]["stopped_at"], "layer.4.hidden")
        self.assertEqual(report["summary"]["debug_target"], "layer 4")


if __name__ == "__main__":
    unittest.main()
