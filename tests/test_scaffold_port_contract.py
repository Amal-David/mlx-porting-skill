"""Contracts for the trusted dense-decoder MLX scaffold generator."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import py_compile
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
SCRIPT = SCRIPTS / "scaffold_port.py"
DECODER_FIXTURE = ROOT / "tests" / "fixtures" / "models" / "decoder"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))


HAS_MLX = importlib.util.find_spec("mlx") is not None


def tiny_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": 4,
        "hidden_act": "silu",
        "hidden_size": 8,
        "initializer_range": 0.02,
        "intermediate_size": 16,
        "license": "apache-2.0",
        "max_position_embeddings": 32,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-5,
        "rope_scaling": None,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "transformers_version": "5.3.0",
        "use_cache": True,
        "vocab_size": 16,
    }
    config.update(overrides)
    return config


def write_model(root: Path, config: dict[str, object]) -> None:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(DECODER_FIXTURE / "model.safetensors", root / "model.safetensors")


def run_script(script: Path, *args: object, no_site: bool = False) -> subprocess.CompletedProcess[str]:
    command = [sys.executable]
    if no_site:
        command.append("-S")
    command.extend([str(script), *(str(value) for value in args)])
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def inspect_model(model: Path, inspection: Path, *, no_site: bool = False) -> None:
    result = run_script(
        SCRIPTS / "inspect_model.py",
        model,
        "--output",
        inspection,
        no_site=no_site,
    )
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)


def scaffold(model: Path, inspection: Path, output: Path, *, no_site: bool = False) -> subprocess.CompletedProcess[str]:
    return run_script(
        SCRIPT,
        inspection,
        "--artifact-root",
        model,
        "--output",
        output,
        no_site=no_site,
    )


class ScaffoldPortDependencyFreeContractTests(unittest.TestCase):
    def test_generator_registry_is_explicit_and_imports_no_ml_framework(self) -> None:
        import scaffold_port

        self.assertEqual(
            set(scaffold_port.FAMILY_GENERATORS),
            {"dense-decoder-transformer"},
        )
        self.assertIs(scaffold_port.FEATURE_ALLOWLIST, scaffold_port.DENSE_CONFIG_FEATURE_ALLOWLIST)
        self.assertNotIn("mlx", scaffold_port.__dict__)
        self.assertNotIn("torch", scaffold_port.__dict__)

    def test_help_is_dependency_free(self) -> None:
        result = run_script(SCRIPT, "--help", no_site=True)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--artifact-root", result.stdout)
        self.assertIn("--output", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_blocked_inspection_fails_closed_without_output(self) -> None:
        from test_tooling import trusted_inspection_fixture

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            inspection = root / "inspection.json"
            output = root / "generated"
            payload = trusted_inspection_fixture(
                ["dense-decoder-transformer"],
                blockers=["architecture routing requires manual review"],
                runbooks=["references/runbook-decoder-transformer.md"],
            )
            inspection.write_text(json.dumps(payload), encoding="utf-8")

            result = run_script(
                SCRIPT,
                inspection,
                "--output",
                output,
                no_site=True,
            )

        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "error: Inspection is blocked; no code was generated.\n"
            "Recommendation blockers:\n"
            "- architecture routing requires manual review\n"
            "Runbook: references/runbook-decoder-transformer.md\n"
            "Manual work:\n"
            "- resolve every inspection blocker\n"
            "- rerun inspect_model.py against the artifact\n"
            "- regenerate the scaffold only after the trusted route is unblocked\n",
        )
        self.assertFalse(output.exists())

    def test_unsupported_family_fails_closed_with_runbook_and_manual_work(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            model = ROOT / "tests" / "fixtures" / "models" / "diffusion_flow"
            inspection = root / "inspection.json"
            output = root / "generated"
            inspect_model(model, inspection, no_site=True)

            result = scaffold(model, inspection, output, no_site=True)

        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(
            result.stderr,
            "error: Unsupported architecture route; no code was generated.\n"
            "Family: diffusion-flow\n"
            "Runbook: references/runbook-diffusion-flow.md\n"
            "Manual work:\n"
            "- implement the model graph for diffusion-flow\n"
            "- define and validate a stable parameter-name mapping\n"
            "- capture target tensors with the source-oracle key scheme\n"
            "- validate end-to-end parity and state/cache behavior\n",
        )
        self.assertFalse(output.exists())

    def test_unsupported_features_are_all_reported_with_exact_message(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            model = root / "model"
            inspection = root / "inspection.json"
            output = root / "generated"
            write_model(model, tiny_config(
                hidden_act="mish",
                rope_scaling={"rope_type": "yarn", "factor": 4.0},
                sliding_window=4096,
                use_sliding_window=True,
                num_local_experts=8,
                quantization_config={"bits": 4},
                qk_norm=True,
                mystery_attention_scale=0.5,
            ))
            inspect_model(model, inspection, no_site=True)

            result = scaffold(model, inspection, output, no_site=True)

        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(
            result.stderr,
            "error: Unsupported config features; no code was generated. "
            "Consult references/runbook-decoder-transformer.md:\n"
            "- hidden_act='mish' is not supported; supported activations: gelu, relu, silu, swish\n"
            "- MoE config key 'num_local_experts' is set\n"
            "- qk_norm=True is not supported\n"
            "- quantization_config is set\n"
            "- rope_scaling type 'yarn' is not supported; supported types: default, dynamic, linear\n"
            "- use_sliding_window=True is not supported\n"
            "- unrecognized computation-relevant config key 'mystery_attention_scale'\n",
        )
        self.assertFalse(output.exists())

    def test_explicitly_disabled_sliding_window_metadata_uses_full_attention(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            model = root / "model"
            inspection = root / "inspection.json"
            output = root / "generated"
            write_model(model, tiny_config(
                sliding_window=4096,
                use_sliding_window=False,
                max_window_layers=2,
            ))
            inspect_model(model, inspection, no_site=True)

            result = scaffold(model, inspection, output, no_site=True)
            readme = (output / "README.md").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("`use_sliding_window=false` explicitly selects full attention", readme)

    def test_attention_bias_contract_is_inferred_per_projection(self) -> None:
        import scaffold_port

        config = tiny_config()
        config.pop("attention_bias")
        tensors = []
        for layer in range(2):
            for projection in ("q_proj", "k_proj", "v_proj"):
                tensors.append({
                    "key": f"model.layers.{layer}.self_attn.{projection}.bias",
                })

        biases = scaffold_port.dense_attention_biases({"tensors": tensors}, config)
        targets = scaffold_port.dense_target_tensors(config, biases)
        target_keys = {item["key"] for item in targets}

        self.assertEqual(
            biases,
            {"q_proj": True, "k_proj": True, "v_proj": True, "o_proj": False},
        )
        self.assertIn("model.layers.0.self_attn.q_proj.bias", target_keys)
        self.assertNotIn("model.layers.0.self_attn.o_proj.bias", target_keys)

    def test_partial_attention_bias_coverage_fails_closed(self) -> None:
        import scaffold_port

        config = tiny_config()
        config.pop("attention_bias")
        with self.assertRaisesRegex(scaffold_port.SkillError, "inconsistent q_proj bias coverage"):
            scaffold_port.dense_attention_biases(
                {"tensors": [{"key": "model.layers.0.self_attn.q_proj.bias"}]},
                config,
            )

    def test_declared_attention_bias_requires_every_projection(self) -> None:
        import scaffold_port

        config = tiny_config(attention_bias=True)
        tensors = []
        for layer in range(2):
            for projection in ("q_proj", "k_proj", "v_proj"):
                tensors.append({
                    "key": f"model.layers.{layer}.self_attn.{projection}.bias",
                })

        with self.assertRaisesRegex(
            scaffold_port.SkillError,
            "attention_bias=true conflicts with incomplete",
        ):
            scaffold_port.dense_attention_biases({"tensors": tensors}, config)

    def test_generated_package_is_clean_complete_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            model = root / "model"
            inspection = root / "inspection.json"
            first = root / "first"
            second = root / "second"
            write_model(model, tiny_config())
            inspect_model(model, inspection, no_site=True)

            first_result = scaffold(model, inspection, first, no_site=True)
            second_result = scaffold(model, inspection, second, no_site=True)

            self.assertEqual(first_result.returncode, 0, first_result.stdout + first_result.stderr)
            self.assertEqual(second_result.returncode, 0, second_result.stdout + second_result.stderr)
            expected = {
                "README.md",
                "capture.py",
                "config.json",
                "config.py",
                "generate.py",
                "model.py",
                "scaffold-manifest.json",
            }
            self.assertEqual({path.name for path in first.iterdir()}, expected)
            self.assertEqual({path.name for path in second.iterdir()}, expected)
            for name in sorted(expected):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)
            for name in ("capture.py", "config.py", "generate.py", "model.py"):
                source = (first / name).read_text(encoding="utf-8")
                ast.parse(source, filename=name)
                py_compile.compile(str(first / name), doraise=True)
                self.assertNotIn("TODO", source)
                self.assertRegex(
                    source.splitlines()[0],
                    r"\A# Generated by scaffold_port\.py 1\.0\.0 from inspection sha256:[0-9a-f]{64}\.\Z",
                )
            readme = (first / "README.md").read_text(encoding="utf-8")
            self.assertIn("model.layers.{i}.self_attn.q_proj.weight", readme)
            self.assertIn("starting implementation", readme)
            self.assertIn("parity validation", readme)
            manifest = json.loads((first / "scaffold-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["config_sha256"],
                hashlib.sha256((first / "config.json").read_bytes()).hexdigest(),
            )
            self.assertIn(
                {"key": "model.layers.0.self_attn.q_proj.weight", "shape": [8, 8]},
                manifest["tensors"],
            )


@unittest.skipUnless(HAS_MLX, "mlx is required for generated-model execution tests")
class ScaffoldPortMLXContractTests(unittest.TestCase):
    def test_tiny_gqa_forward_generate_and_cache_match_full_context(self) -> None:
        import mlx.core as mx

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            model_root = root / "source"
            inspection = root / "inspection.json"
            generated = root / "generated"
            write_model(model_root, tiny_config())
            inspect_model(model_root, inspection)
            result = scaffold(model_root, inspection, generated)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            previous_path = sys.path[:]
            prior_modules = {name: sys.modules.pop(name, None) for name in ("config", "model")}
            try:
                sys.path.insert(0, str(generated))
                from config import ModelConfig
                from model import build_model, greedy_generate

                config = ModelConfig.from_file(generated / "config.json")
                model = build_model(config)
                weights: dict[str, object] = {
                    "model.embed_tokens.weight": mx.random.normal((16, 8)),
                    "model.norm.weight": mx.ones((8,)),
                }
                for index in range(2):
                    prefix = f"model.layers.{index}"
                    weights.update({
                        f"{prefix}.input_layernorm.weight": mx.ones((8,)),
                        f"{prefix}.post_attention_layernorm.weight": mx.ones((8,)),
                        f"{prefix}.self_attn.q_proj.weight": mx.random.normal((8, 8)),
                        f"{prefix}.self_attn.k_proj.weight": mx.random.normal((4, 8)),
                        f"{prefix}.self_attn.v_proj.weight": mx.random.normal((4, 8)),
                        f"{prefix}.self_attn.o_proj.weight": mx.random.normal((8, 8)),
                        f"{prefix}.mlp.gate_proj.weight": mx.random.normal((16, 8)),
                        f"{prefix}.mlp.up_proj.weight": mx.random.normal((16, 8)),
                        f"{prefix}.mlp.down_proj.weight": mx.random.normal((8, 16)),
                    })
                model.load_weights(list(weights.items()), strict=True)
                mx.eval(model.parameters())

                input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
                full_logits, full_cache = model(input_ids)
                prefill_logits, cache = model(input_ids[:, :2])
                step_logits, stepped_cache = model(input_ids[:, 2:], cache=cache)
                generated_ids = greedy_generate(model, input_ids, max_new_tokens=3)
                mx.eval(full_logits, prefill_logits, step_logits, generated_ids)

                self.assertEqual(full_logits.shape, (1, 3, 16))
                self.assertEqual(prefill_logits.shape, (1, 2, 16))
                self.assertEqual(step_logits.shape, (1, 1, 16))
                self.assertEqual(generated_ids.shape, (1, 3))
                self.assertEqual(full_logits.dtype, mx.float32)
                self.assertIn(generated_ids.dtype, (mx.int32, mx.int64, mx.uint32, mx.uint64))
                self.assertEqual(len(full_cache), 2)
                self.assertEqual(len(stepped_cache), 2)
                self.assertEqual(stepped_cache[0][0].shape, (1, 1, 3, 4))
                self.assertTrue(
                    mx.allclose(
                        full_logits[:, -1, :],
                        step_logits[:, -1, :],
                        rtol=1e-5,
                        atol=1e-5,
                    ).item()
                )

                weights_path = generated / "weights.safetensors"
                mx.save_safetensors(str(weights_path), weights)
                generated_cli = subprocess.run(
                    [
                        sys.executable,
                        str(generated / "generate.py"),
                        "--weights",
                        str(weights_path),
                        "--token-ids",
                        "1",
                        "2",
                        "3",
                        "--max-new-tokens",
                        "3",
                    ],
                    cwd=generated,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    generated_cli.returncode,
                    0,
                    generated_cli.stdout + generated_cli.stderr,
                )
                self.assertEqual(
                    len(json.loads(generated_cli.stdout)["generated_token_ids"][0]),
                    3,
                )

                target_npz = generated / "target.npz"
                capture_cli = subprocess.run(
                    [
                        sys.executable,
                        str(generated / "capture.py"),
                        "--weights",
                        str(weights_path),
                        "--token-ids",
                        "1",
                        "2",
                        "3",
                        "--generate-steps",
                        "3",
                        "--output",
                        str(target_npz),
                    ],
                    cwd=generated,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    capture_cli.returncode,
                    0,
                    capture_cli.stdout + capture_cli.stderr,
                )
                manifest_validation = run_script(
                    SCRIPTS / "capture_oracle.py",
                    "--validate-manifest",
                    generated / "target.manifest.json",
                    no_site=True,
                )
                self.assertEqual(
                    manifest_validation.returncode,
                    0,
                    manifest_validation.stdout + manifest_validation.stderr,
                )
                import numpy as np

                with np.load(target_npz, allow_pickle=False) as archive:
                    self.assertEqual(
                        set(archive.files),
                        {
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
                        },
                    )
            finally:
                sys.path[:] = previous_path
                for name in ("config", "model"):
                    sys.modules.pop(name, None)
                    if prior_modules[name] is not None:
                        sys.modules[name] = prior_modules[name]


if __name__ == "__main__":
    unittest.main()
