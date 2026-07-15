"""Contracts for the local PyTorch/Hugging Face source-oracle capture tool."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
SCRIPT = SCRIPTS / "capture_oracle.py"
FIXTURE_MANIFEST = ROOT / "tests" / "fixtures" / "manifests" / "oracle.json"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import capture_oracle  # noqa: E402


HAS_TORCH_STACK = all(
    importlib.util.find_spec(package) is not None
    for package in ("torch", "transformers")
)


def write_static_model(root: Path, *, auto_map: bool = False) -> None:
    root.mkdir()
    config: dict[str, object] = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "vocab_size": 16,
    }
    if auto_map:
        config["auto_map"] = {"AutoModelForCausalLM": "modeling_custom.CustomModel"}
    (root / "config.json").write_text(json.dumps(config) + "\n", encoding="utf-8")
    (root / "pytorch_model.bin").write_bytes(b"fixture")


def run_tool(*args: object, no_site: bool = False) -> subprocess.CompletedProcess[str]:
    command = [sys.executable]
    if no_site:
        command.append("-S")
    command.extend([str(SCRIPT), *(str(value) for value in args)])
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class CaptureOracleDependencyFreeContractTests(unittest.TestCase):
    def test_module_import_does_not_import_torch_or_transformers(self) -> None:
        self.assertNotIn("torch", capture_oracle.__dict__)
        self.assertNotIn("transformers", capture_oracle.__dict__)

    def test_help_works_without_site_packages(self) -> None:
        completed = run_tool("--help", no_site=True)

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("--token-ids", completed.stdout)
        self.assertIn("--validate-manifest", completed.stdout)
        self.assertEqual(completed.stderr, "")

    def test_missing_torch_fails_closed_with_exact_install_message(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp).resolve()
            model = root / "model"
            write_static_model(model)

            completed = run_tool(
                model,
                "--token-ids",
                "1",
                "2",
                "--output",
                root / "oracle.npz",
                no_site=True,
            )

        self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(
            completed.stderr,
            "error: capture_oracle.py requires optional package 'torch'; "
            "install it with python3 -m pip install torch\n",
        )

    def test_argument_validation_fails_before_optional_imports(self) -> None:
        cases = (
            (
                ("model", "--output", "oracle.npz"),
                "error: provide --prompt, --prompts-file, or --token-ids\n",
            ),
            (
                (
                    "model",
                    "--prompt",
                    "fixture",
                    "--token-ids",
                    "1",
                    "--output",
                    "oracle.npz",
                ),
                "error: --token-ids cannot be combined with --prompt or --prompts-file\n",
            ),
            (
                (
                    "model",
                    "--token-ids",
                    "1",
                    "--generate-steps",
                    "-1",
                    "--output",
                    "oracle.npz",
                ),
                "error: --generate-steps must be a non-negative integer\n",
            ),
            (
                (
                    "model",
                    "--token-ids",
                    "1",
                    "--max-output-mb",
                    "nan",
                    "--output",
                    "oracle.npz",
                ),
                "error: --max-output-mb must be a finite non-negative number\n",
            ),
        )
        for args, message in cases:
            with self.subTest(message=message):
                completed = run_tool(*args, no_site=True)
                self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
                self.assertEqual(completed.stderr, message)

    def test_remote_code_config_is_rejected_before_optional_imports(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp).resolve()
            model = root / "model"
            write_static_model(model, auto_map=True)

            completed = run_tool(
                model,
                "--token-ids",
                "1",
                "--output",
                root / "oracle.npz",
                no_site=True,
            )

        self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
        self.assertEqual(
            completed.stderr,
            "error: config.json declares auto_map remote model code; "
            "capture_oracle.py refuses remote code\n",
        )

    @unittest.skipIf(os.name == "nt", "symlink semantics differ on Windows")
    def test_output_path_rejects_symlink_components_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(raw_tmp).resolve()
            model = root / "model"
            write_static_model(model)
            linked = root / "linked"
            outside = Path(outside_tmp).resolve()
            linked.symlink_to(outside, target_is_directory=True)

            completed = run_tool(
                model,
                "--token-ids",
                "1",
                "--output",
                linked / "oracle.npz",
                no_site=True,
            )

        self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
        self.assertIn("NPZ output path contains symlink component", completed.stderr)
        self.assertFalse((outside / "oracle.npz").exists())

    def test_manifest_fixture_validates_without_site_packages(self) -> None:
        completed = run_tool("--validate-manifest", FIXTURE_MANIFEST, no_site=True)

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(completed.stdout, '{"ok": true, "schema_version": 1}\n')
        self.assertEqual(completed.stderr, "")

        payload = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
        del payload["libraries"]
        with self.assertRaisesRegex(
            capture_oracle.SkillError,
            "oracle manifest has an invalid field set",
        ):
            capture_oracle.validate_manifest(payload)


@unittest.skipUnless(HAS_TORCH_STACK, "torch and transformers are required for execution tests")
class CaptureOracleTorchContractTests(unittest.TestCase):
    @staticmethod
    def write_tiny_llama(root: Path) -> None:
        import torch

        root.mkdir()
        hidden = 8
        intermediate = 16
        layers = 2
        vocab = 16
        config = {
            "architectures": ["LlamaForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "head_dim": 4,
            "hidden_act": "silu",
            "hidden_size": hidden,
            "initializer_range": 0.02,
            "intermediate_size": intermediate,
            "max_position_embeddings": 32,
            "mlp_bias": False,
            "model_type": "llama",
            "num_attention_heads": 2,
            "num_hidden_layers": layers,
            "num_key_value_heads": 2,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "torch_dtype": "float32",
            "transformers_version": "5.3.0",
            "use_cache": True,
            "vocab_size": vocab,
        }
        (root / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        counter = 0

        def values(shape: tuple[int, ...], *, norm: bool = False) -> object:
            nonlocal counter
            size = 1
            for dimension in shape:
                size *= dimension
            if norm:
                tensor = torch.ones(shape, dtype=torch.float32)
            else:
                tensor = (
                    torch.arange(counter, counter + size, dtype=torch.float32)
                    .reshape(shape)
                    .remainder(29)
                    .sub(14)
                    .mul(0.01)
                )
                counter += size
            return tensor

        state: dict[str, object] = {
            "model.embed_tokens.weight": values((vocab, hidden)),
            "model.norm.weight": values((hidden,), norm=True),
            "lm_head.weight": values((vocab, hidden)),
        }
        for index in range(layers):
            prefix = f"model.layers.{index}"
            state.update({
                f"{prefix}.self_attn.q_proj.weight": values((hidden, hidden)),
                f"{prefix}.self_attn.k_proj.weight": values((hidden, hidden)),
                f"{prefix}.self_attn.v_proj.weight": values((hidden, hidden)),
                f"{prefix}.self_attn.o_proj.weight": values((hidden, hidden)),
                f"{prefix}.mlp.gate_proj.weight": values((intermediate, hidden)),
                f"{prefix}.mlp.up_proj.weight": values((intermediate, hidden)),
                f"{prefix}.mlp.down_proj.weight": values((hidden, intermediate)),
                f"{prefix}.input_layernorm.weight": values((hidden,), norm=True),
                f"{prefix}.post_attention_layernorm.weight": values((hidden,), norm=True),
            })
        torch.save(state, root / "pytorch_model.bin")

    def run_capture(self, model: Path, output: Path, *extra: object) -> subprocess.CompletedProcess[str]:
        return run_tool(
            model,
            "--token-ids",
            "1",
            "5",
            "7",
            "--generate-steps",
            "3",
            "--output",
            output,
            *extra,
        )

    def test_float16_left_padding_is_finite_and_matches_unpadded_capture(self) -> None:
        import numpy as np
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        config = {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "head_dim": 4,
            "hidden_act": "silu",
            "hidden_size": 8,
            "intermediate_size": 16,
            "max_position_embeddings": 32,
            "mlp_bias": False,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": 16,
        }
        torch.manual_seed(8004)
        model = LlamaForCausalLM(LlamaConfig(**config)).to(dtype=torch.float16)
        model.train(False)
        batch_ids = torch.tensor([[1, 2, 3], [0, 4, 5]], dtype=torch.long)
        batch_mask = torch.tensor([[1, 1, 1], [0, 1, 1]], dtype=torch.long)
        batch = capture_oracle.capture_tensors(
            model,
            batch_ids,
            batch_mask,
            config,
            0,
            torch,
        )
        singles = [
            capture_oracle.capture_tensors(
                model,
                ids,
                torch.ones_like(ids),
                config,
                0,
                torch,
            )
            for ids in (
                torch.tensor([[1, 2, 3]], dtype=torch.long),
                torch.tensor([[4, 5]], dtype=torch.long),
            )
        ]

        floating = [
            name for name, value in batch.items()
            if torch.is_tensor(value) and value.is_floating_point() and value.ndim >= 2
        ]
        self.assertIn("logits", floating)
        for name in floating:
            with self.subTest(tensor=name):
                self.assertTrue(torch.isfinite(batch[name]).all().item())
                self.assertTrue(torch.all(batch[name][1, 0] == 0).item())
                np.testing.assert_allclose(
                    batch[name][0].detach().float().numpy(),
                    singles[0][name][0].detach().float().numpy(),
                    rtol=1e-2,
                    atol=1e-2,
                )
                np.testing.assert_allclose(
                    batch[name][1, 1:].detach().float().numpy(),
                    singles[1][name][0].detach().float().numpy(),
                    rtol=1e-2,
                    atol=1e-2,
                )

    def test_tiny_local_model_capture_has_stable_keys_shapes_and_digests(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp).resolve()
            model = root / "model"
            self.write_tiny_llama(model)
            output = root / "oracle.npz"

            completed = self.run_capture(model, output)

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertTrue(summary["ok"])
            self.assertEqual(summary["tensor_count"], 12)
            manifest_path = root / "oracle.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            capture_oracle.validate_manifest(manifest)

            expected_keys = {
                "attention_mask",
                "embed",
                "final_norm",
                "generated_token_ids",
                "input_ids",
                "layer.0.attention",
                "layer.0.hidden",
                "layer.0.mlp",
                "layer.1.attention",
                "layer.1.hidden",
                "layer.1.mlp",
                "logits",
            }
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), expected_keys)
                self.assertEqual(archive["input_ids"].shape, (1, 3))
                self.assertEqual(archive["embed"].shape, (1, 3, 8))
                self.assertEqual(archive["layer.0.hidden"].shape, (1, 3, 8))
                self.assertEqual(archive["final_norm"].shape, (1, 3, 8))
                self.assertEqual(archive["logits"].shape, (1, 3, 16))
                self.assertEqual(archive["generated_token_ids"].shape, (1, 3))
                inventory = {record["name"]: record for record in manifest["tensors"]}
                self.assertEqual(set(inventory), expected_keys)
                for key in archive.files:
                    value = archive[key]
                    self.assertEqual(inventory[key]["shape"], list(value.shape))
                    self.assertEqual(inventory[key]["dtype"], str(value.dtype))
                    self.assertEqual(
                        inventory[key]["sha256"],
                        hashlib.sha256(value.tobytes(order="C")).hexdigest(),
                    )

            self.assertEqual(
                manifest["model"]["config"]["sha256"],
                hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["model"]["weights"]["files"][0]["sha256"],
                hashlib.sha256((model / "pytorch_model.bin").read_bytes()).hexdigest(),
            )
            self.assertEqual(manifest["capture"]["dtype_policy"], "float32")
            self.assertTrue(all(
                record["dtype"] == "float32"
                for record in manifest["tensors"]
                if record["name"] not in {"attention_mask", "generated_token_ids", "input_ids"}
            ))

    def test_homogeneous_f32_safetensors_override_stale_bf16_config_dtype(self) -> None:
        import numpy as np
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp).resolve()
            model = root / "model"
            self.write_tiny_llama(model)
            state = torch.load(model / "pytorch_model.bin", weights_only=True)
            save_file(state, model / "model.safetensors", metadata={"format": "pt"})
            (model / "pytorch_model.bin").unlink()
            config_path = model / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["torch_dtype"] = "bfloat16"
            config_path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            output = root / "oracle.npz"

            completed = run_tool(
                model,
                "--token-ids",
                "1",
                "5",
                "7",
                "--generate-steps",
                "0",
                "--keep-dtype",
                "--output",
                output,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            with np.load(output, allow_pickle=False) as archive:
                floating = [
                    value.dtype
                    for value in archive.values()
                    if np.issubdtype(value.dtype, np.floating)
                ]
            self.assertTrue(floating)
            self.assertEqual(set(floating), {np.dtype(np.float32)})

    def test_repeated_capture_is_bitwise_deterministic(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp).resolve()
            model = root / "model"
            self.write_tiny_llama(model)
            first = root / "first.npz"
            second = root / "second.npz"

            first_run = self.run_capture(model, first)
            second_run = self.run_capture(model, second)

            self.assertEqual(first_run.returncode, 0, first_run.stdout + first_run.stderr)
            self.assertEqual(second_run.returncode, 0, second_run.stdout + second_run.stderr)
            first_manifest = json.loads((root / "first.manifest.json").read_text(encoding="utf-8"))
            second_manifest = json.loads((root / "second.manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(first_manifest, second_manifest)
            with np.load(first, allow_pickle=False) as left, np.load(second, allow_pickle=False) as right:
                self.assertEqual(left.files, right.files)
                for key in left.files:
                    np.testing.assert_array_equal(left[key], right[key])

    def test_near_zero_output_cap_fails_closed_without_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp).resolve()
            model = root / "model"
            self.write_tiny_llama(model)
            output = root / "oracle.npz"

            completed = self.run_capture(model, output, "--max-output-mb", "0")

            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertRegex(
                completed.stderr,
                r"\Aerror: captured tensor payload exceeds --max-output-mb: \d+ > 0 bytes\n\Z",
            )
            self.assertFalse(output.exists())
            self.assertFalse((root / "oracle.manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
