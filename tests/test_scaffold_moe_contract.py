"""Contracts for the additive sparse-MoE decoder scaffold generator."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import py_compile
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
SCAFFOLD = SCRIPTS / "scaffold_port.py"
CONVERT = SCRIPTS / "convert_checkpoint.py"
RUN_PARITY = SCRIPTS / "run_parity.py"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import run_parity  # noqa: E402
import scaffold_port  # noqa: E402
from mlx_keystone import require_mlx_keystone  # noqa: E402
from test_tooling import trusted_inspection_fixture  # noqa: E402


HAS_TORCH_STACK = all(
    importlib.util.find_spec(package) is not None
    for package in ("numpy", "safetensors", "torch", "transformers")
)


def mlx_runtime_available() -> bool:
    if importlib.util.find_spec("mlx") is None:
        return False
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mlx.core as mx; x=mx.array([1]); mx.eval(x); print(mx.default_device())",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


HAS_MLX = mlx_runtime_available()


def tiny_moe_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "architectures": ["Qwen2MoeForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "head_dim": 8,
        "hidden_act": "silu",
        "hidden_size": 32,
        "intermediate_size": 48,
        "license": "apache-2.0",
        "max_position_embeddings": 64,
        "mlp_bias": False,
        "model_type": "qwen2_moe",
        "moe_intermediate_size": 24,
        "norm_topk_prob": False,
        "num_attention_heads": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "rope_scaling": None,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 64,
    }
    config.update(overrides)
    return config


def run_script(script: Path, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *(str(value) for value in args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def write_generated_files(root: Path, files: dict[str, str]) -> None:
    root.mkdir()
    for name, source in files.items():
        (root / name).write_text(source, encoding="utf-8")


def trusted_moe_inspection(config: dict[str, object]) -> dict[str, object]:
    inspection = trusted_inspection_fixture(
        ["moe-decoder-transformer"],
        model_type=str(config["model_type"]),
        runbooks=["references/runbook-moe-transformer.md"],
    )
    tensors = [
        {
            "key": item["key"],
            "shape": item["shape"],
            "dtype": "F32",
            "parameters": 1,
            "estimated_bytes": 4,
            "file": "model.safetensors",
        }
        for item in scaffold_port.moe_source_tensors(config)
    ]
    inspection["tensors"] = tensors
    inspection["tensor_summary"] = {
        "count": len(tensors),
        "parameters": len(tensors),
        "estimated_bytes": 4 * len(tensors),
        "dtypes": {"F32": len(tensors)},
        "files": {"model.safetensors": len(tensors)},
        "metadata": {},
        "errors": [],
        "integrity_ok": True,
    }
    return inspection


class MoeScaffoldDependencyFreeContractTests(unittest.TestCase):
    def test_registry_aliases_and_feature_allowlist_are_explicit(self) -> None:
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
        self.assertIs(
            scaffold_port.FEATURE_ALLOWLIST,
            scaffold_port.DENSE_CONFIG_FEATURE_ALLOWLIST,
        )
        moe_features = scaffold_port.FAMILY_FEATURE_ALLOWLISTS["moe-decoder-transformer"]
        for key in (
            "num_experts",
            "num_local_experts",
            "num_experts_per_tok",
            "top_k",
            "norm_topk_prob",
            "moe_intermediate_size",
        ):
            self.assertIn(key, moe_features)

        registry = json.loads(
            (ROOT / "mlx-model-porting" / "assets" / "architectures.yaml").read_text(
                encoding="utf-8"
            )
        )
        record = next(
            item for item in registry["families"]
            if item["id"] == "moe-decoder-transformer"
        )
        self.assertTrue({
            "mixtral", "qwen2_moe", "qwen3_moe", "deepseek_v2", "deepseek_v3",
            "olmoe", "phimoe", "granitemoe",
        }.issubset(record["model_type_aliases"]))
        self.assertNotIn("mlx", scaffold_port.__dict__)
        self.assertNotIn("torch", scaffold_port.__dict__)

    def test_router_profiles_are_alias_specific_and_fail_closed(self) -> None:
        profiles = scaffold_port.MOE_ROUTER_PROFILES
        self.assertEqual(set(profiles), {
            "mixtral", "qwen2_moe", "qwen3_moe", "deepseek_v2", "deepseek_v3",
            "dbrx", "grok", "minimax", "ernie4_5_moe", "olmoe", "phimoe",
            "granitemoe",
        })
        self.assertEqual(profiles["mixtral"]["renormalize"], "always")
        self.assertEqual(profiles["granitemoe"]["renormalize"], "always")
        self.assertEqual(profiles["qwen2_moe"]["renormalize"], "config")
        self.assertEqual(profiles["phimoe"]["router"], "sparse-mixer")
        self.assertEqual(
            profiles["mixtral"]["architectures"],
            {"MixtralForCausalLM"},
        )
        self.assertEqual(
            profiles["granitemoe"]["architectures"],
            {"GraniteMoeForCausalLM"},
        )
        with self.assertRaisesRegex(
            scaffold_port.SkillError,
            "architectures do not match model_type 'mixtral'",
        ):
            scaffold_port.validate_moe_config(tiny_moe_config(model_type="mixtral"))
        for alias in ("phimoe", "qwen3_moe", "olmoe", "deepseek_v2", "deepseek_v3"):
            with self.subTest(alias=alias):
                with self.assertRaisesRegex(scaffold_port.SkillError, "is not supported"):
                    scaffold_port.validate_moe_config(tiny_moe_config(model_type=alias))

    def test_parameter_contract_uses_hf_router_and_per_expert_names(self) -> None:
        tensors = scaffold_port.moe_target_tensors(tiny_moe_config())
        by_name = {item["key"]: item["shape"] for item in tensors}
        self.assertEqual(by_name["model.layers.0.mlp.gate.weight"], [4, 32])
        self.assertEqual(
            by_name["model.layers.1.mlp.experts.3.gate_proj.weight"],
            [24, 32],
        )
        self.assertEqual(
            by_name["model.layers.1.mlp.experts.3.down_proj.weight"],
            [32, 24],
        )
        self.assertNotIn("model.layers.0.mlp.gate_proj.weight", by_name)
        self.assertFalse(any("shared_expert" in name for name in by_name))

    def test_unsupported_computation_features_fail_closed(self) -> None:
        cases = (
            (
                {"shared_expert_intermediate_size": 16},
                "shared experts are not supported",
            ),
            ({"expert_parallel_size": 2}, "single-device softmax router"),
            ({"output_router_logits": True}, "auxiliary router outputs"),
            ({"router_jitter_noise": 0.1}, "router_jitter_noise must be 0.0"),
            ({"decoder_sparse_step": 2}, "every decoder layer must be MoE"),
            ({"routed_scaling_factor": 0.0}, "routed_scaling_factor=0.0"),
            ({"pruned_heads": {"0": [1]}}, "pruned_heads must be empty"),
        )
        for override, message in cases:
            with self.subTest(override=override):
                with self.assertRaisesRegex(scaffold_port.SkillError, message):
                    scaffold_port.validate_moe_config(tiny_moe_config(**override))
        with self.assertRaisesRegex(scaffold_port.SkillError, "must not exceed"):
            scaffold_port.validate_moe_config(tiny_moe_config(num_experts_per_tok=5))

    def test_generation_is_deterministic_and_python_compiles(self) -> None:
        inspection = trusted_moe_inspection(tiny_moe_config())
        first = scaffold_port.generate_moe_decoder(inspection, tiny_moe_config())
        second = scaffold_port.generate_moe_decoder(inspection, tiny_moe_config())
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as raw_tmp:
            generated = Path(raw_tmp).resolve() / "generated"
            write_generated_files(generated, first)
            for name in ("capture.py", "config.py", "generate.py", "model.py"):
                py_compile.compile(str(generated / name), doraise=True)
        self.assertIn("class Expert", first["model.py"])
        self.assertIn("norm_topk_prob", first["model.py"])
        self.assertNotIn("TODO", first["model.py"])

    def test_inspected_tensor_topology_must_match_supported_profile_exactly(self) -> None:
        config = tiny_moe_config()
        cases = {
            "qk norm": "model.layers.0.self_attn.q_norm.weight",
            "MLA": "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "shared expert": "model.layers.0.mlp.shared_expert.gate_proj.weight",
            "router bias": "model.layers.0.mlp.gate.bias",
            "dense layer": "model.layers.0.mlp.gate_proj.weight",
        }
        for label, key in cases.items():
            with self.subTest(label=label):
                inspection = trusted_moe_inspection(config)
                inspection["tensors"].append({"key": key, "shape": [32], "dtype": "F32"})
                with self.assertRaisesRegex(scaffold_port.SkillError, "unsupported tensors"):
                    scaffold_port.generate_moe_decoder(inspection, config)

        inspection = trusted_moe_inspection(config)
        inspection["tensors"] = [
            item for item in inspection["tensors"]
            if item["key"] != "model.layers.1.mlp.experts.3.down_proj.weight"
        ]
        with self.assertRaisesRegex(scaffold_port.SkillError, "missing tensors"):
            scaffold_port.generate_moe_decoder(inspection, config)

    def test_mixtral_and_granite_force_selected_probability_renormalization(self) -> None:
        identities = {
            "mixtral": "MixtralForCausalLM",
            "granitemoe": "GraniteMoeForCausalLM",
        }
        for alias, architecture in identities.items():
            with self.subTest(alias=alias):
                config = tiny_moe_config(
                    model_type=alias,
                    architectures=[architecture],
                    norm_topk_prob=False,
                )
                generated = scaffold_port.generate_moe_decoder(
                    trusted_moe_inspection(config), config,
                )
                with tempfile.TemporaryDirectory() as raw_tmp:
                    package = Path(raw_tmp).resolve() / "package"
                    write_generated_files(package, generated)
                    previous_path = sys.path[:]
                    prior = sys.modules.pop("config", None)
                    try:
                        sys.path.insert(0, str(package))
                        from config import ModelConfig
                        self.assertTrue(ModelConfig.from_file(package / "config.json").norm_topk_prob)
                    finally:
                        sys.path[:] = previous_path
                        sys.modules.pop("config", None)
                        if prior is not None:
                            sys.modules["config"] = prior

    def test_packed_profiles_reject_unmapped_expert_biases(self) -> None:
        qwen = tiny_moe_config(mlp_bias=True)
        self.assertIs(scaffold_port.validate_moe_config(qwen), qwen)

        identities = {
            "mixtral": "MixtralForCausalLM",
            "granitemoe": "GraniteMoeForCausalLM",
        }
        for alias, architecture in identities.items():
            with self.subTest(alias=alias), self.assertRaisesRegex(
                scaffold_port.SkillError,
                "source expert-bias contract is not defined",
            ):
                scaffold_port.validate_moe_config(tiny_moe_config(
                    model_type=alias,
                    architectures=[architecture],
                    mlp_bias=True,
                ))


@unittest.skipUnless(
    HAS_TORCH_STACK,
    "torch, transformers, safetensors, and NumPy are required",
)
@require_mlx_keystone(HAS_MLX, "a usable MLX Metal runtime is required")
class MoeScaffoldSyntheticParityTests(unittest.TestCase):
    @staticmethod
    def _write_source(root: Path) -> None:
        import torch
        from transformers import MixtralConfig, MixtralForCausalLM

        torch.manual_seed(9201)
        config = MixtralConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
        )
        model = MixtralForCausalLM(config).eval()
        expert_generator = torch.Generator(device="cpu").manual_seed(9202)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".mlp.experts." in name and name.endswith(".weight"):
                    parameter.normal_(mean=0.0, std=0.2, generator=expert_generator)
        model.save_pretrained(root, safe_serialization=True)
        saved = json.loads((root / "config.json").read_text(encoding="utf-8"))
        saved["license"] = "apache-2.0"
        saved["norm_topk_prob"] = True
        (root / "config.json").write_text(
            json.dumps(saved, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _convert_with_real_schema_2_converter(
        source: Path,
        package: Path,
        mapping: Path,
        converted: Path,
    ) -> subprocess.CompletedProcess[str]:
        from safetensors.numpy import load_file

        source_weights = load_file(source / "model.safetensors")
        manifest = json.loads(
            (package / "scaffold-manifest.json").read_text(encoding="utf-8")
        )
        entries = []
        for item in manifest["tensors"]:
            target = item["key"]
            source_key = target
            source_key = source_key.replace(".mlp.gate.weight", ".block_sparse_moe.gate.weight")
            source_key = source_key.replace(".mlp.experts.", ".block_sparse_moe.experts.")
            source_key = source_key.replace(".gate_proj.weight", ".w1.weight")
            source_key = source_key.replace(".down_proj.weight", ".w2.weight")
            source_key = source_key.replace(".up_proj.weight", ".w3.weight")
            entries.append({
                "source": source_key,
                "source_shape": list(source_weights[source_key].shape),
                "target": target,
                "target_shape": item["shape"],
                "transforms": [{"op": "rename"}],
            })
        mapping.write_text(
            json.dumps({
                "schema_version": 2,
                "draft": False,
                "dtype_policy": "keep",
                "entries": entries,
                "ignore": [],
                "unresolved": [],
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return run_script(
            CONVERT,
            "--source", source,
            "--mapping", mapping,
            "--output", converted,
        )

    def test_tiny_moe_real_conversion_parity_and_seeded_negatives(self) -> None:
        import mlx.core as mx
        from safetensors.numpy import load_file, save_file

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp).resolve()
            source = root / "source"
            inspection = root / "inspection.json"
            package = root / "package"
            mapping = root / "WEIGHT_MAP.json"
            converted = root / "converted"
            passing_report = root / "passing.json"
            routing_failing_report = root / "routing-failing.json"
            failing_report = root / "failing.json"
            self._write_source(source)

            inspected = run_script(
                SCRIPTS / "inspect_model.py", source, "--output", inspection,
            )
            self.assertEqual(inspected.returncode, 0, inspected.stdout + inspected.stderr)
            scaffolded = run_script(
                SCAFFOLD,
                inspection,
                "--artifact-root", source,
                "--output", package,
            )
            self.assertEqual(scaffolded.returncode, 0, scaffolded.stdout + scaffolded.stderr)
            self.assertEqual(json.loads(scaffolded.stdout)["family"], "moe-decoder-transformer")
            conversion = self._convert_with_real_schema_2_converter(
                source, package, mapping, converted,
            )
            self.assertEqual(conversion.returncode, 0, conversion.stdout + conversion.stderr)
            conversion_report = json.loads(
                (converted / "conversion-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(json.loads(mapping.read_text(encoding="utf-8"))["schema_version"], 2)
            self.assertEqual(
                conversion_report["counts"]["mapped_source_tensors"],
                conversion_report["counts"]["target_tensors"],
            )

            previous_path = sys.path[:]
            prior_modules = {name: sys.modules.pop(name, None) for name in ("config", "model")}
            try:
                sys.path.insert(0, str(package))
                from config import ModelConfig
                from model import greedy_generate, load_model

                self.assertTrue(ModelConfig.from_file(package / "config.json").norm_topk_prob)
                model = load_model(package / "config.json", converted / "model.safetensors")
                input_ids = mx.array([[1, 5, 7, 9]], dtype=mx.int32)
                full_logits, _ = model(input_ids)
                _, cache = model(input_ids[:, :-1])
                step_logits, stepped_cache = model(input_ids[:, -1:], cache=cache)
                first_generated = greedy_generate(model, input_ids, 3)
                second_generated = greedy_generate(model, input_ids, 3)
                mx.eval(full_logits, step_logits, first_generated, second_generated)
                self.assertEqual(full_logits.shape, (1, 4, 64))
                self.assertEqual(first_generated.shape, (1, 3))
                self.assertEqual(stepped_cache[0][0].shape, (1, 2, 4, 8))
                self.assertTrue(mx.array_equal(first_generated, second_generated).item())
                self.assertTrue(
                    mx.allclose(
                        full_logits[:, -1, :],
                        step_logits[:, -1, :],
                        rtol=2e-3,
                        atol=2e-4,
                    ).item()
                )

                mlp = model.model.layers[0].mlp
                rows = mx.array([[1.0] * 32, [0.5] * 32, [-1.0] * 32, [-2.0] * 32])
                mlp.gate.weight = rows
                inactive = mlp.experts[3]
                inactive.gate_proj.weight = mx.full((24, 32), 1e20)
                inactive.up_proj.weight = mx.full((24, 32), 1e20)
                inactive.down_proj.weight = mx.full((32, 24), 1e20)
                nan_guard_output = mlp(mx.ones((1, 2, 32)))
                mx.eval(nan_guard_output)
                self.assertTrue(mx.all(mx.isfinite(nan_guard_output)).item())
            finally:
                sys.path[:] = previous_path
                for name in ("config", "model"):
                    sys.modules.pop(name, None)
                    if prior_modules[name] is not None:
                        sys.modules[name] = prior_modules[name]

            parity_args = (
                "--source-model", source,
                "--package", package,
                "--weights", converted,
                "--token-ids", "1", "5", "7", "9",
                "--generate-steps", "3",
                "--atol", "0.0002",
                "--rtol", "0.002",
                "--cosine-min", "0.9999",
            )
            passed = run_script(RUN_PARITY, *parity_args, "--output", passing_report)
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
            passing = json.loads(passing_report.read_text(encoding="utf-8"))
            run_parity.validate_parity_report(passing)
            self.assertTrue(passing["ok"], passing)
            self.assertEqual(
                [rung["name"] for rung in passing["rungs"]],
                [
                    "input_ids", "embed", "layer.0.hidden", "layer.1.hidden",
                    "final_norm", "logits", "generated_token_ids",
                ],
            )

            weights_path = converted / "model.safetensors"

            def attest_mutated_weights() -> None:
                report_path = converted / "conversion-report.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                raw = weights_path.read_bytes()
                report["outputs"]["weights"].update({
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                })
                report_path.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

            original_arrays = load_file(weights_path)
            arrays = {key: value.copy() for key, value in original_arrays.items()}
            for layer in range(2):
                gate = f"model.layers.{layer}.mlp.gate.weight"
                arrays[gate] = arrays[gate][[3, 1, 2, 0]].copy()
            temporary = converted / "model.routing-mutant.safetensors"
            save_file(arrays, temporary)
            os.replace(temporary, weights_path)
            attest_mutated_weights()

            routing_failed = run_script(
                RUN_PARITY, *parity_args, "--output", routing_failing_report,
            )
            self.assertEqual(
                routing_failed.returncode, 1,
                routing_failed.stdout + routing_failed.stderr,
            )
            routing_failure = json.loads(routing_failing_report.read_text(encoding="utf-8"))
            run_parity.validate_parity_report(routing_failure)
            self.assertEqual(routing_failure["summary"]["stopped_at"], "layer.0.hidden")

            arrays = {key: value.copy() for key, value in original_arrays.items()}
            for layer in range(2):
                for expert in range(4):
                    down = f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"
                    arrays[down] = (-3.0 * arrays[down]).copy()
            temporary = converted / "model.seeded-bug.safetensors"
            save_file(arrays, temporary)
            os.replace(temporary, weights_path)
            attest_mutated_weights()

            failed = run_script(RUN_PARITY, *parity_args, "--output", failing_report)
            self.assertEqual(failed.returncode, 1, failed.stdout + failed.stderr)
            failing = json.loads(failing_report.read_text(encoding="utf-8"))
            run_parity.validate_parity_report(failing)
            self.assertFalse(failing["ok"])
            self.assertEqual(failing["summary"]["stopped_at"], "layer.0.hidden")
            self.assertEqual(failing["summary"]["debug_target"], "layer 0")
            self.assertEqual(
                [rung["name"] for rung in failing["rungs"]],
                ["input_ids", "embed", "layer.0.hidden"],
            )


if __name__ == "__main__":
    unittest.main()
