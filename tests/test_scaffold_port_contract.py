"""Contracts for the trusted dense-decoder MLX scaffold generator."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import py_compile
import shutil
import struct
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

from mlx_keystone import require_mlx_keystone  # noqa: E402


def mlx_runtime_available() -> bool:
    if importlib.util.find_spec("mlx") is None:
        return False
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mlx.core as mx; value=mx.array([1], dtype=mx.int32); mx.eval(value)",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return probe.returncode == 0


HAS_MLX = mlx_runtime_available()


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


def tiny_ssm_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "architectures": ["MambaForCausalLM"],
        "conv_bias": True,
        "d_conv": 3,
        "d_model": 4,
        "d_state": 3,
        "expand": 2,
        "license": "apache-2.0",
        "model_type": "mamba",
        "n_layer": 2,
        "rms_norm_eps": 1e-5,
        "ssm_variant": "minimal_selective",
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "use_cache": True,
        "vocab_size": 11,
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


def add_qk_norm_weights(model_path: Path, *, layers: int, head_dim: int) -> None:
    raw = model_path.read_bytes()
    header_length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_length])
    payload = raw[8 + header_length :]
    offset = len(payload)
    for layer in range(layers):
        for name in ("q_norm", "k_norm"):
            key = f"model.layers.{layer}.self_attn.{name}.weight"
            byte_length = head_dim * 4
            header[key] = {
                "dtype": "F32",
                "shape": [head_dim],
                "data_offsets": [offset, offset + byte_length],
            }
            payload += b"\0" * byte_length
            offset += byte_length
    encoded_header = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded_header += b" " * (-len(encoded_header) % 8)
    model_path.write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + payload)


def write_ssm_model(root: Path, config: dict[str, object]) -> None:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    specs = {
        "backbone.layers.0.mixer.A_log": {
            "dtype": "F32",
            "shape": [8, 3],
            "data_offsets": [0, 96],
        },
        "backbone.layers.0.mixer.conv1d.weight": {
            "dtype": "F32",
            "shape": [8, 1, 3],
            "data_offsets": [96, 192],
        },
    }
    header = json.dumps(specs, separators=(",", ":"), sort_keys=True).encode("utf-8")
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + (b"\0" * 192)
    )


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
    def test_meaningfully_set_treats_numeric_zero_as_explicit(self) -> None:
        import scaffold_port

        self.assertTrue(scaffold_port._meaningfully_set(0))
        self.assertTrue(scaffold_port._meaningfully_set(0.0))
        for value in (None, False, "", [], {}):
            self.assertFalse(scaffold_port._meaningfully_set(value))

    def test_llama_rope_interleaving_is_value_gated(self) -> None:
        import scaffold_port

        standard_config = tiny_config(
            is_llama_config=True,
            rope_interleaved=False,
        )
        standard_config["transformers.js_config"] = {"dtype": "q4"}

        self.assertIs(scaffold_port.validate_dense_config(standard_config), standard_config)

        with self.assertRaisesRegex(
            scaffold_port.SkillError,
            "interleaved RoPE is not implemented",
        ):
            scaffold_port.validate_dense_config({**standard_config, "rope_interleaved": True})

    def test_use_mrope_is_value_gated(self) -> None:
        import scaffold_port

        # use_mrope=False is the standard 1D RoPE used by text-only Qwen2 models
        # (e.g. DeepSeek-R1-Distill-Qwen-7B); it must be accepted, not blocked.
        standard_config = tiny_config(use_mrope=False)
        self.assertIs(scaffold_port.validate_dense_config(standard_config), standard_config)

        with self.assertRaisesRegex(
            scaffold_port.SkillError,
            "M-RoPE.*not supported",
        ):
            scaffold_port.validate_dense_config({**standard_config, "use_mrope": True})

    def test_generator_registry_is_explicit_and_imports_no_ml_framework(self) -> None:
        import scaffold_port

        self.assertEqual(
            set(scaffold_port.FAMILY_GENERATORS),
            {
                "automatic-speech-recognition",
                "dense-decoder-transformer",
                "encoder-decoder-transformer",
                "encoder-transformer",
                "moe-decoder-transformer",
                "ssm-recurrent-hybrid",
            },
        )
        self.assertEqual(
            set(scaffold_port.FAMILY_FEATURE_ALLOWLISTS),
            set(scaffold_port.FAMILY_GENERATORS),
        )
        self.assertIs(scaffold_port.FEATURE_ALLOWLIST, scaffold_port.DENSE_CONFIG_FEATURE_ALLOWLIST)
        self.assertNotIn("mlx", scaffold_port.__dict__)
        self.assertNotIn("torch", scaffold_port.__dict__)

    def test_ssm_config_allowlist_and_hybrid_identity_fail_closed(self) -> None:
        import scaffold_port

        self.assertEqual(scaffold_port.validate_ssm_config(tiny_ssm_config())["d_state"], 3)
        with self.assertRaisesRegex(
            scaffold_port.SkillError,
            "unrecognized computation-relevant config key 'mystery_scan_mode'",
        ):
            scaffold_port.validate_ssm_config(tiny_ssm_config(mystery_scan_mode="fast"))
        with self.assertRaisesRegex(
            scaffold_port.SkillError,
            "hybrid/attention-mixed SSM config is not supported",
        ):
            scaffold_port.validate_ssm_config(tiny_ssm_config(
                model_type="jamba",
                architectures=["JambaForCausalLM"],
                layer_types=["mamba", "attention"],
            ))

    def test_ssm_identity_and_decoder_bypasses_fail_closed_individually(self) -> None:
        import scaffold_port

        cases = (
            (
                {"architectures": "JambaForCausalLM"},
                "architectures must be a list of strings for the pure-SSM generator",
            ),
            (
                {"architectures": ["UnknownForCausalLM"]},
                "architectures must be exactly ['MambaForCausalLM'] for model_type 'mamba'",
            ),
            (
                {"architectures": ["MambaAttentionForCausalLM"]},
                "architectures must be exactly ['MambaForCausalLM'] for model_type 'mamba'",
            ),
            (
                {"architectures": ["JambaForCausalLM"]},
                "hybrid/attention-mixed SSM config is not supported",
            ),
            (
                {"architectures": ["MambaForCausalLM", 7]},
                "architectures must be a list of strings for the pure-SSM generator",
            ),
            (
                {"is_decoder": False},
                "is_decoder must be true when set for the decoder-only SSM generator",
            ),
            (
                {"is_encoder_decoder": True},
                "is_encoder_decoder must be false when set for the decoder-only SSM generator",
            ),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(scaffold_port.SkillError) as raised:
                    scaffold_port.validate_ssm_config(tiny_ssm_config(**overrides))
                self.assertIn(message, str(raised.exception))

    def test_ssm_attention_and_moe_tensor_inventory_fails_closed(self) -> None:
        import scaffold_port

        for key in (
            "backbone.layers.0.self_attn.q_proj.weight",
            "backbone.layers.0.moe.experts.0.up_proj.weight",
            "backbone.layers.0.moe.router.weight",
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                    scaffold_port.SkillError,
                    "Pure-SSM generation rejects attention or MoE tensor namespaces",
                ):
                    scaffold_port.validate_ssm_inspection({"tensors": [{"key": key}]})

    def test_generated_ssm_package_is_routed_and_attests_recurrent_weights(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            model = root / "model"
            inspection = root / "inspection.json"
            generated = root / "generated"
            write_ssm_model(model, tiny_ssm_config())
            inspect_model(model, inspection, no_site=True)

            result = scaffold(model, inspection, generated, no_site=True)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)["family"], "ssm-recurrent-hybrid")
            model_source = (generated / "model.py").read_text(encoding="utf-8")
            ast.parse(model_source)
            self.assertIn(
                "input_coefficient = dt[:, :, None] * _exprel(mx, scaled_A)",
                model_source,
            )
            self.assertNotIn("class Attention", model_source)
            readme = (generated / "README.md").read_text(encoding="utf-8")
            self.assertIn("Real-checkpoint weight conversion", readme)
            manifest = json.loads(
                (generated / "scaffold-manifest.json").read_text(encoding="utf-8")
            )
            target_keys = {record["key"] for record in manifest["tensors"]}
            self.assertIn("model.layers.0.mixer.A_log", target_keys)
            self.assertIn("model.layers.1.mixer.conv1d.weight", target_keys)
            self.assertNotIn("model.layers.0.self_attn.q_proj.weight", target_keys)

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
            "- quantization_config is set\n"
            "- rope_scaling type 'yarn' is not supported; supported types: default, dynamic, linear\n"
            "- use_sliding_window=True is not supported\n"
            "- unrecognized computation-relevant config key 'mystery_attention_scale'\n",
        )
        self.assertFalse(output.exists())

    def test_qwen3_qk_norm_scaffolds_and_normalizes_before_rope(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            model = root / "model"
            inspection = root / "inspection.json"
            output = root / "generated"
            write_model(model, tiny_config(
                architectures=["Qwen3ForCausalLM"],
                head_dim=6,
                max_window_layers=2,
                model_type="qwen3",
                qk_norm=True,
                sliding_window=None,
                use_sliding_window=False,
            ))
            add_qk_norm_weights(model / "model.safetensors", layers=2, head_dim=6)
            inspect_model(model, inspection, no_site=True)

            result = scaffold(model, inspection, output, no_site=True)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("QK normalization", result.stderr)
            model_source = (output / "model.py").read_text(encoding="utf-8")
            self.assertIn(
                "self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)",
                model_source,
            )
            self.assertIn(
                "self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)",
                model_source,
            )
            self.assertIn(
                "attended.transpose(0, 2, 1, 3).reshape("
                "\n                batch, length, config.num_attention_heads * config.head_dim\n"
                "            )",
                model_source,
            )
            q_reshape = model_source.index("q = self.q_proj(x).reshape(")
            q_norm = model_source.index("q = self.q_norm(q)")
            k_norm = model_source.index("k = self.k_norm(k)")
            rope = model_source.index("q = self._apply_rope(q, offset, total_length)")
            self.assertLess(q_reshape, q_norm)
            self.assertLess(q_norm, k_norm)
            self.assertLess(k_norm, rope)
            manifest = json.loads(
                (output / "scaffold-manifest.json").read_text(encoding="utf-8")
            )
            target_shapes = {
                record["key"]: record["shape"] for record in manifest["tensors"]
            }
            target_keys = set(target_shapes)
            self.assertIn("model.layers.0.self_attn.q_norm.weight", target_keys)
            self.assertIn("model.layers.1.self_attn.k_norm.weight", target_keys)
            self.assertEqual(
                target_shapes["model.layers.0.self_attn.q_proj.weight"], [12, 8]
            )
            self.assertEqual(
                target_shapes["model.layers.0.self_attn.o_proj.weight"], [8, 12]
            )
            self.assertEqual(
                target_shapes["model.layers.0.self_attn.q_norm.weight"], [6]
            )
            self.assertNotIn(
                "hidden_size must equal num_attention_heads * head_dim", model_source
            )
            readme = (output / "README.md").read_text(encoding="utf-8")
            self.assertIn("per-head Q/K RMSNorm over `head_dim` before RoPE", readme)
            self.assertNotIn("QK normalization, MoE", readme)

    def test_explicit_null_head_dim_matches_missing_head_dim(self) -> None:
        import scaffold_port

        null_config = tiny_config(head_dim=None)
        self.assertIs(scaffold_port.validate_dense_config(null_config), null_config)
        with self.assertRaisesRegex(
            scaffold_port.SkillError,
            "divisible by num_attention_heads when head_dim is omitted",
        ):
            scaffold_port.validate_dense_config(tiny_config(head_dim=None, hidden_size=9))

    def test_generated_config_treats_explicit_null_head_dim_as_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            model = root / "model"
            inspection = root / "inspection.json"
            output = root / "generated"
            write_model(model, tiny_config(head_dim=None))
            inspect_model(model, inspection, no_site=True)

            result = scaffold(model, inspection, output, no_site=True)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            spec = importlib.util.spec_from_file_location(
                "generated_null_head_dim_config", output / "config.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
                parsed = module.ModelConfig.from_dict(tiny_config(head_dim=None))
                self.assertEqual(parsed.head_dim, 4)
                with self.assertRaisesRegex(
                    ValueError,
                    "divisible by num_attention_heads when head_dim is omitted",
                ):
                    module.ModelConfig.from_dict(tiny_config(head_dim=None, hidden_size=9))
            finally:
                sys.modules.pop(spec.name, None)

    def test_qk_norm_still_fails_closed_without_complete_weights(self) -> None:
        import scaffold_port

        with self.assertRaisesRegex(
            scaffold_port.SkillError,
            "complete q_norm/k_norm weights were not inspected",
        ):
            scaffold_port.dense_qk_norm_enabled(
                {"tensors": []},
                tiny_config(use_qk_norm=True),
            )

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
                    r"\A# Generated by scaffold_port\.py 1\.1\.0 from inspection sha256:[0-9a-f]{64}\.\Z",
                )
            self.assertNotIn("self.q_norm =", (first / "model.py").read_text(encoding="utf-8"))
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


def numpy_ssm_forward(
    input_ids: object,
    weights: dict[str, object],
    config: dict[str, object],
    np: object,
    *,
    euler_bug: bool = False,
) -> tuple[object, list[tuple[object, object]]]:
    """Independent FP32 oracle for the generated synthetic selective recurrence."""
    d_model = int(config["d_model"])
    d_state = int(config["d_state"])
    d_conv = int(config["d_conv"])
    d_inner = d_model * int(config["expand"])
    epsilon = float(config["rms_norm_eps"])
    hidden = weights["model.embed_tokens.weight"][input_ids].astype(np.float32)
    final_states = []
    for layer in range(int(config["n_layer"])):
        prefix = f"model.layers.{layer}"
        norm_weight = weights[f"{prefix}.norm.weight"]
        normalized = hidden * np.reciprocal(
            np.sqrt(np.mean(hidden * hidden, axis=-1, keepdims=True) + epsilon)
        ) * norm_weight
        projected = normalized @ weights[f"{prefix}.mixer.in_proj.weight"].T
        projected, gate = np.split(projected, 2, axis=-1)
        conv_state = np.zeros(
            (hidden.shape[0], d_conv - 1, d_inner),
            dtype=np.float32,
        )
        ssm_state = np.zeros((hidden.shape[0], d_inner, d_state), dtype=np.float32)
        branches = []
        A = -np.exp(weights[f"{prefix}.mixer.A_log"])
        for token in range(hidden.shape[1]):
            window = np.concatenate((conv_state, projected[:, token : token + 1, :]), axis=1)
            convolved = np.sum(
                window * weights[f"{prefix}.mixer.conv1d.weight"].T[None, :, :],
                axis=1,
            )
            if bool(config["conv_bias"]):
                convolved = convolved + weights[f"{prefix}.mixer.conv1d.bias"]
            activated = convolved / (1.0 + np.exp(-convolved))
            parameters = activated @ weights[f"{prefix}.mixer.x_proj.weight"].T
            dt_raw = parameters[:, :d_inner]
            B = parameters[:, d_inner : d_inner + d_state]
            C = parameters[:, d_inner + d_state :]
            dt_input = dt_raw + weights[f"{prefix}.mixer.dt_bias"]
            dt = np.logaddexp(dt_input, np.zeros_like(dt_input))
            scaled_A = dt[:, :, None] * A[None, :, :]
            decay = np.exp(scaled_A)
            coefficient = (
                dt[:, :, None]
                if euler_bug
                else dt[:, :, None] * numpy_exprel(scaled_A, np)
            )
            ssm_state = (
                decay * ssm_state
                + coefficient * B[:, None, :] * activated[:, :, None]
            )
            y = np.sum(ssm_state * C[:, None, :], axis=-1)
            y = y + weights[f"{prefix}.mixer.D"] * activated
            silu_gate = gate[:, token, :] / (1.0 + np.exp(-gate[:, token, :]))
            branch = (y * silu_gate) @ weights[f"{prefix}.mixer.out_proj.weight"].T
            branches.append(branch)
            conv_state = window[:, 1:, :]
        hidden = hidden + np.stack(branches, axis=1)
        final_states.append((conv_state, ssm_state))
    hidden = hidden * np.reciprocal(
        np.sqrt(np.mean(hidden * hidden, axis=-1, keepdims=True) + epsilon)
    ) * weights["model.norm.weight"]
    logits = hidden @ weights["model.embed_tokens.weight"].T
    return logits, final_states


def numpy_exprel(value: object, np: object) -> object:
    """Stable expm1(x) / x matching the generated MLX recurrence."""
    small = np.abs(value) < 1e-4
    square = value * value
    series = 1.0 + value * 0.5 + square / 6.0 + square * value / 24.0
    safe_value = np.where(small, np.ones_like(value), value)
    quotient = np.expm1(value) / safe_value
    return np.where(small, series, quotient)


@require_mlx_keystone(
    HAS_MLX,
    "a usable MLX runtime is required for generated-model execution tests",
)
class ScaffoldPortMLXContractTests(unittest.TestCase):
    @staticmethod
    def _weights(mx: object, *, dtype: object, scale: float = 1.0) -> dict[str, object]:
        weights: dict[str, object] = {
            "model.embed_tokens.weight": (mx.random.normal((16, 8)) * scale).astype(dtype),
            "model.norm.weight": mx.ones((8,), dtype=dtype),
        }
        for index in range(2):
            prefix = f"model.layers.{index}"
            weights.update({
                f"{prefix}.input_layernorm.weight": mx.ones((8,), dtype=dtype),
                f"{prefix}.post_attention_layernorm.weight": mx.ones((8,), dtype=dtype),
                f"{prefix}.self_attn.q_proj.weight": (mx.random.normal((8, 8)) * scale).astype(dtype),
                f"{prefix}.self_attn.k_proj.weight": (mx.random.normal((4, 8)) * scale).astype(dtype),
                f"{prefix}.self_attn.v_proj.weight": (mx.random.normal((4, 8)) * scale).astype(dtype),
                f"{prefix}.self_attn.o_proj.weight": (mx.random.normal((8, 8)) * scale).astype(dtype),
                f"{prefix}.mlp.gate_proj.weight": (mx.random.normal((16, 8)) * scale).astype(dtype),
                f"{prefix}.mlp.up_proj.weight": (mx.random.normal((16, 8)) * scale).astype(dtype),
                f"{prefix}.mlp.down_proj.weight": (mx.random.normal((8, 16)) * scale).astype(dtype),
            })
        return weights

    def test_ssm_exprel_zoh_is_finite_at_underflow_and_zero_limits(self) -> None:
        import mlx.core as mx
        import numpy as np

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            source = root / "source"
            inspection = root / "inspection.json"
            generated = root / "generated"
            write_ssm_model(source, tiny_ssm_config())
            inspect_model(source, inspection)
            result = scaffold(source, inspection, generated)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            previous_path = sys.path[:]
            prior_modules = {name: sys.modules.pop(name, None) for name in ("config", "model")}
            try:
                sys.path.insert(0, str(generated))
                from model import _exprel

                dt_np = np.asarray([[0.125, 1e-8, 0.5, 2.0]], dtype=np.float32)
                a_log_np = np.asarray([[-100.0, -80.0, -20.0, 0.0]], dtype=np.float32)
                dt = mx.array(dt_np)
                A = -mx.exp(mx.array(a_log_np))
                scaled_A = dt * A
                coefficient = dt * _exprel(mx, scaled_A)
                mx.eval(A, scaled_A, coefficient)

                coefficient_np = np.asarray(coefficient)
                expected = dt_np * numpy_exprel(dt_np * -np.exp(a_log_np), np)
                self.assertTrue(np.isfinite(np.asarray(A)).all())
                self.assertTrue(np.isfinite(coefficient_np).all())
                np.testing.assert_allclose(coefficient_np, expected, rtol=1e-6, atol=1e-8)
                np.testing.assert_allclose(
                    coefficient_np[:, :2],
                    dt_np[:, :2],
                    rtol=1e-6,
                    atol=0.0,
                )

                near_zero = mx.array([0.0, -0.0, 1e-8, -1e-8], dtype=mx.float32)
                limits = _exprel(mx, near_zero)
                mx.eval(limits)
                self.assertTrue(np.isfinite(np.asarray(limits)).all())
                np.testing.assert_allclose(
                    np.asarray(limits),
                    np.ones((4,), dtype=np.float32),
                    rtol=1e-6,
                    atol=1e-7,
                )
            finally:
                sys.path[:] = previous_path
                for name, module in prior_modules.items():
                    sys.modules.pop(name, None)
                    if module is not None:
                        sys.modules[name] = module

    def test_selective_ssm_matches_numpy_recurrence_and_carried_state(self) -> None:
        import mlx.core as mx
        import numpy as np

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            source = root / "source"
            inspection = root / "inspection.json"
            generated = root / "generated"
            config_data = tiny_ssm_config()
            write_ssm_model(source, config_data)
            inspect_model(source, inspection)
            result = scaffold(source, inspection, generated)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            rng = np.random.default_rng(7301)
            d_model = int(config_data["d_model"])
            d_state = int(config_data["d_state"])
            d_conv = int(config_data["d_conv"])
            d_inner = d_model * int(config_data["expand"])
            numpy_weights: dict[str, np.ndarray] = {
                "model.embed_tokens.weight": rng.normal(0.0, 0.25, (11, d_model)).astype(np.float32),
                "model.norm.weight": rng.uniform(0.8, 1.2, d_model).astype(np.float32),
            }
            for layer in range(int(config_data["n_layer"])):
                prefix = f"model.layers.{layer}"
                numpy_weights.update({
                    f"{prefix}.norm.weight": rng.uniform(0.8, 1.2, d_model).astype(np.float32),
                    f"{prefix}.mixer.in_proj.weight": rng.normal(
                        0.0, 0.35, (2 * d_inner, d_model)
                    ).astype(np.float32),
                    f"{prefix}.mixer.conv1d.weight": rng.normal(
                        0.0, 0.3, (d_inner, d_conv)
                    ).astype(np.float32),
                    f"{prefix}.mixer.conv1d.bias": rng.normal(0.0, 0.05, d_inner).astype(np.float32),
                    f"{prefix}.mixer.x_proj.weight": rng.normal(
                        0.0, 0.3, (d_inner + 2 * d_state, d_inner)
                    ).astype(np.float32),
                    f"{prefix}.mixer.dt_bias": rng.uniform(-0.7, 0.2, d_inner).astype(np.float32),
                    f"{prefix}.mixer.D": rng.uniform(0.5, 1.2, d_inner).astype(np.float32),
                    f"{prefix}.mixer.out_proj.weight": rng.normal(
                        0.0, 0.3, (d_model, d_inner)
                    ).astype(np.float32),
                })
                a_log = rng.uniform(-1.0, 0.2, (d_inner, d_state)).astype(np.float32)
                a_log[0, 0] = -100.0
                a_log[1, 1] = -80.0
                numpy_weights[f"{prefix}.mixer.A_log"] = a_log

            previous_path = sys.path[:]
            prior_modules = {name: sys.modules.pop(name, None) for name in ("config", "model")}
            try:
                sys.path.insert(0, str(generated))
                from config import ModelConfig
                from model import build_model, greedy_generate

                model = build_model(ModelConfig.from_file(generated / "config.json"))
                mlx_weights = {key: mx.array(value) for key, value in numpy_weights.items()}
                model.load_weights(list(mlx_weights.items()), strict=True)
                mx.eval(model.parameters())
                token_ids_np = np.asarray(
                    [[1, 4, 2, 7, 3], [6, 2, 9, 1, 5]],
                    dtype=np.int32,
                )
                token_ids = mx.array(token_ids_np)

                full_logits, full_state = model(token_ids)
                self.assertTrue(np.isfinite(np.asarray(full_logits)).all())
                for layer_state in full_state:
                    for value in layer_state:
                        self.assertTrue(np.isfinite(np.asarray(value)).all())
                for split_point in (1, 3):
                    prefix_logits, prefix_state = model(token_ids[:, :split_point])
                    suffix_logits, carried_state = model(
                        token_ids[:, split_point:],
                        state=prefix_state,
                    )
                    recurrent_logits = mx.concatenate((prefix_logits, suffix_logits), axis=1)
                    mx.eval(full_logits, recurrent_logits, full_state, carried_state)

                    # Checks recurrence/recompute equivalence on deterministic FP32 fixtures.
                    self.assertTrue(mx.allclose(
                        full_logits,
                        recurrent_logits,
                        rtol=1e-6,
                        atol=1e-6,
                    ).item())
                    for expected_layer, actual_layer in zip(
                        full_state,
                        carried_state,
                        strict=True,
                    ):
                        for expected, actual in zip(expected_layer, actual_layer, strict=True):
                            self.assertTrue(mx.allclose(
                                expected,
                                actual,
                                rtol=1e-6,
                                atol=1e-6,
                            ).item())

                cached_generation = greedy_generate(model, token_ids, max_new_tokens=4)
                sequence = token_ids
                recomputed_tokens = []
                for _ in range(4):
                    logits, _ = model(sequence)
                    next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
                    recomputed_tokens.append(next_token)
                    sequence = mx.concatenate((sequence, next_token[:, None]), axis=1)
                recomputed_generation = mx.stack(recomputed_tokens, axis=1)
                mx.eval(cached_generation, recomputed_generation)
                self.assertTrue(mx.array_equal(cached_generation, recomputed_generation).item())

                reference_logits, reference_state = numpy_ssm_forward(
                    token_ids_np,
                    numpy_weights,
                    config_data,
                    np,
                )
                np.testing.assert_allclose(
                    np.asarray(full_logits),
                    reference_logits,
                    rtol=2e-5,
                    atol=2e-5,
                )
                for mlx_layer, numpy_layer in zip(full_state, reference_state, strict=True):
                    for mlx_value, numpy_value in zip(mlx_layer, numpy_layer, strict=True):
                        np.testing.assert_allclose(
                            np.asarray(mlx_value),
                            numpy_value,
                            rtol=2e-5,
                            atol=2e-5,
                        )

                bugged_logits, _ = numpy_ssm_forward(
                    token_ids_np,
                    numpy_weights,
                    config_data,
                    np,
                    euler_bug=True,
                )
                bug_max_abs = float(np.max(np.abs(np.asarray(full_logits) - bugged_logits)))
                self.assertGreater(
                    bug_max_abs,
                    1e-4,
                    "seeded Euler-discretization bug must be detected by the FP32 parity gate",
                )

                weights_path = generated / "weights.safetensors"
                capture_path = generated / "ssm-capture.npz"
                mx.save_safetensors(str(weights_path), mlx_weights)
                capture_result = subprocess.run(
                    [
                        sys.executable,
                        str(generated / "capture.py"),
                        "--weights",
                        str(weights_path),
                        "--token-ids",
                        "1",
                        "4",
                        "2",
                        "7",
                        "3",
                        "--generate-steps",
                        "2",
                        "--output",
                        str(capture_path),
                    ],
                    cwd=generated,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    capture_result.returncode,
                    0,
                    capture_result.stdout + capture_result.stderr,
                )
                with np.load(capture_path, allow_pickle=False) as capture:
                    self.assertIn("layer.0.ssm", capture.files)
                    self.assertIn("layer.1.hidden", capture.files)
            finally:
                sys.path[:] = previous_path
                for name in ("config", "model"):
                    sys.modules.pop(name, None)
                    if prior_modules[name] is not None:
                        sys.modules[name] = prior_modules[name]

    def test_dynamic_ntk_greedy_cache_matches_full_recompute_across_threshold(self) -> None:
        import mlx.core as mx

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            model_root = root / "source"
            inspection = root / "inspection.json"
            generated = root / "generated"
            write_model(model_root, tiny_config(
                max_position_embeddings=8,
                rope_scaling={
                    "rope_type": "dynamic",
                    "factor": 2.0,
                    "original_max_position_embeddings": 4,
                },
            ))
            inspect_model(model_root, inspection)
            result = scaffold(model_root, inspection, generated)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            previous_path = sys.path[:]
            prior_modules = {name: sys.modules.pop(name, None) for name in ("config", "model")}
            try:
                sys.path.insert(0, str(generated))
                from config import ModelConfig
                from model import build_model, greedy_generate

                mx.random.seed(8002)
                model = build_model(ModelConfig.from_file(generated / "config.json"))
                model.load_weights(
                    list(self._weights(mx, dtype=mx.float32, scale=0.2).items()),
                    strict=True,
                )
                mx.eval(model.parameters())
                threshold_prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
                _, threshold_cache = model(threshold_prompt)
                with self.assertRaisesRegex(ValueError, "full-sequence recomputation"):
                    model(mx.array([[5]], dtype=mx.int32), cache=threshold_cache)

                class RecordingModel:
                    def __init__(self, wrapped: object):
                        self.wrapped = wrapped
                        self.config = wrapped.config
                        self.calls: list[tuple[int, bool]] = []

                    def __call__(self, input_ids: object, **kwargs: object) -> object:
                        self.calls.append((int(input_ids.shape[1]), kwargs.get("cache") is not None))
                        return self.wrapped(input_ids, **kwargs)

                prompt = mx.array([[1, 3, 5]], dtype=mx.int32)
                recording_model = RecordingModel(model)
                cached = greedy_generate(recording_model, prompt, max_new_tokens=5)
                sequence = prompt
                reference_tokens = []
                for _ in range(5):
                    logits, _ = model(sequence)
                    next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
                    reference_tokens.append(next_token)
                    sequence = mx.concatenate((sequence, next_token[:, None]), axis=1)
                reference = mx.stack(reference_tokens, axis=1)
                mx.eval(cached, reference)

                self.assertTrue(mx.array_equal(cached, reference).item())
                self.assertEqual(
                    recording_model.calls,
                    [(3, False), (1, True), (5, False), (6, False), (7, False)],
                )
                readme = (generated / "README.md").read_text(encoding="utf-8")
                self.assertIn("full-sequence recomputation", readme)
                self.assertIn("dynamic-NTK", readme)
            finally:
                sys.path[:] = previous_path
                for name in ("config", "model"):
                    sys.modules.pop(name, None)
                    if prior_modules[name] is not None:
                        sys.modules[name] = prior_modules[name]

    def test_float16_left_padding_is_finite_and_matches_unpadded_capture(self) -> None:
        import mlx.core as mx
        import numpy as np

        import capture_mlx

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            model_root = root / "source"
            inspection = root / "inspection.json"
            generated = root / "generated"
            write_model(model_root, tiny_config(torch_dtype="float16"))
            inspect_model(model_root, inspection)
            result = scaffold(model_root, inspection, generated)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            mx.random.seed(8003)
            previous_path = sys.path[:]
            prior_modules = {name: sys.modules.pop(name, None) for name in ("config", "model")}
            try:
                sys.path.insert(0, str(generated))
                from config import ModelConfig
                from model import build_model

                model = build_model(ModelConfig.from_file(generated / "config.json"))
                weights = self._weights(mx, dtype=mx.float16, scale=0.1)
                model.load_weights(list(weights.items()), strict=True)
                mx.eval(model.parameters())
                weights_path = generated / "weights.safetensors"
                mx.save_safetensors(str(weights_path), weights)
            finally:
                sys.path[:] = previous_path
                for name in ("config", "model"):
                    sys.modules.pop(name, None)
                    if prior_modules[name] is not None:
                        sys.modules[name] = prior_modules[name]

            batch_ids = np.asarray([[1, 2, 3], [0, 4, 5]], dtype=np.int32)
            batch_mask = np.asarray([[1, 1, 1], [0, 1, 1]], dtype=np.int32)
            batch = capture_mlx._capture(
                generated,
                weights_path,
                batch_ids,
                batch_mask,
                0,
                True,
                np,
                mx,
            )
            singles = [
                capture_mlx._capture(
                    generated,
                    weights_path,
                    ids,
                    np.ones_like(ids),
                    0,
                    True,
                    np,
                    mx,
                )
                for ids in (
                    np.asarray([[1, 2, 3]], dtype=np.int32),
                    np.asarray([[4, 5]], dtype=np.int32),
                )
            ]

            floating = [
                name for name, value in batch.items()
                if np.issubdtype(value.dtype, np.floating) and value.ndim >= 2
            ]
            self.assertIn("logits", floating)
            for name in floating:
                with self.subTest(tensor=name):
                    self.assertTrue(np.isfinite(batch[name]).all())
                    self.assertTrue(np.all(batch[name][1, 0] == 0))
                    np.testing.assert_allclose(
                        batch[name][0],
                        singles[0][name][0],
                        rtol=1e-2,
                        atol=1e-2,
                    )
                    np.testing.assert_allclose(
                        batch[name][1, 1:],
                        singles[1][name][0],
                        rtol=1e-2,
                        atol=1e-2,
                    )

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
