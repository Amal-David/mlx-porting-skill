"""Contracts for the executable source-to-MLX parity ladder."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
CAPTURE_MLX = SCRIPTS / "capture_mlx.py"
RUN_PARITY = SCRIPTS / "run_parity.py"
DECODER_FIXTURE = ROOT / "tests" / "fixtures" / "models" / "decoder"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import capture_mlx  # noqa: E402
import run_parity  # noqa: E402
from mlx_keystone import require_mlx_keystone  # noqa: E402


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
            "import mlx.core as mx; value=mx.array([1], dtype=mx.int32); mx.eval(value)",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return probe.returncode == 0


HAS_MLX_RUNTIME = mlx_runtime_available()


def run_script(
    script: Path,
    *args: object,
    no_site: bool = False,
    disable_optional_tensor_libs: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable]
    if no_site:
        command.append("-S")
    command.extend([str(script), *(str(value) for value in args)])
    environment = os.environ.copy()
    if disable_optional_tensor_libs:
        environment["MLX_PORTING_DISABLE_OPTIONAL_TENSOR_LIBS"] = "1"
    return subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def tiny_config() -> dict[str, object]:
    return {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": 8,
        "hidden_act": "silu",
        "hidden_size": 32,
        "initializer_range": 0.02,
        "intermediate_size": 64,
        "license": "apache-2.0",
        "max_position_embeddings": 64,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 4,
        "pad_token_id": 0,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-5,
        "rope_scaling": None,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 128,
    }


def write_tiny_stock_llama(root: Path) -> None:
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    root.mkdir()
    config = tiny_config()
    torch.manual_seed(7004)
    model = LlamaForCausalLM(LlamaConfig(**config))
    attention_generator = torch.Generator(device="cpu").manual_seed(7005)
    with torch.no_grad():
        # Keep a transposed projection observable at the first parity rung.
        for name, parameter in model.named_parameters():
            if ".self_attn." in name and name.endswith(".weight"):
                parameter.normal_(mean=0.0, std=0.2, generator=attention_generator)
    model.train(False)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    (root / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_file(state, root / "model.safetensors")


def write_dependency_free_scaffold(root: Path) -> tuple[Path, Path, Path]:
    inspection = root / "inspection.json"
    package = root / "package"
    weights = root / "weights"
    inspected = run_script(
        SCRIPTS / "inspect_model.py",
        DECODER_FIXTURE,
        "--output",
        inspection,
        no_site=True,
    )
    if inspected.returncode != 0:
        raise AssertionError(inspected.stdout + inspected.stderr)
    scaffolded = run_script(
        SCRIPTS / "scaffold_port.py",
        inspection,
        "--artifact-root",
        DECODER_FIXTURE,
        "--output",
        package,
        no_site=True,
    )
    if scaffolded.returncode != 0:
        raise AssertionError(scaffolded.stdout + scaffolded.stderr)
    scaffold_manifest = json.loads(
        (package / "scaffold-manifest.json").read_text(encoding="utf-8")
    )
    weights.mkdir()
    weights_path = weights / "model.safetensors"
    weights_path.write_bytes(b"not-read-before-mlx-import")
    target_path = weights / "target-manifest.json"
    target_path.write_text(
        json.dumps({
            "schema_version": 1,
            "format": "safetensors",
            "file": "model.safetensors",
            "tensors": [
                {"key": item["key"], "shape": item["shape"], "dtype": "F32"}
                for item in scaffold_manifest["tensors"]
            ],
        }),
        encoding="utf-8",
    )
    def record(path: Path) -> dict[str, object]:
        raw = path.read_bytes()
        return {
            "name": path.name,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    tensor_count = len(scaffold_manifest["tensors"])
    (weights / "conversion-report.json").write_text(
        json.dumps({
            "schema_version": 1,
            "counts": {
                "ignored_source_tensors": 0,
                "mapped_source_tensors": tensor_count,
                "mapping_entries": tensor_count,
                "source_tensors": tensor_count,
                "target_tensors": tensor_count,
            },
            "transforms_applied": {"rename": tensor_count},
            "dtype_policy": {"global": "keep", "targets": {}},
            "memory_budget": {
                "limit_bytes": 1,
                "aggregate_materialized_bytes": 0,
                "estimated_peak_bytes": 0,
                "source_materialized_bytes": 0,
                "target_materialized_bytes": 0,
                "target_encoded_bytes": 0,
            },
            "inputs": {
                "mapping": {"name": "WEIGHT_MAP.json", "size_bytes": 0, "sha256": "0" * 64},
                "source_files": [
                    {"name": "source.safetensors", "size_bytes": 0, "sha256": "0" * 64}
                ],
            },
            "outputs": {
                "weights": {**record(weights_path), "writer": "test-fixture"},
                "target_manifest": record(target_path),
            },
        }),
        encoding="utf-8",
    )
    return inspection, package, weights


class ParityRunnerDependencyFreeContractTests(unittest.TestCase):
    def test_imports_and_help_do_not_load_execution_frameworks(self) -> None:
        self.assertNotIn("mlx", capture_mlx.__dict__)
        self.assertNotIn("numpy", capture_mlx.__dict__)
        self.assertNotIn("torch", run_parity.__dict__)
        for script in (CAPTURE_MLX, RUN_PARITY):
            with self.subTest(script=script.name):
                completed = run_script(script, "--help", no_site=True)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.assertEqual(completed.stderr, "")

    def test_argument_and_tolerance_validation_precede_optional_imports(self) -> None:
        cases = (
            (
                CAPTURE_MLX,
                ("--package", "p", "--weights", "w", "--output", "x.npz"),
                "error: provide --prompt, --prompts-file, or --token-ids\n",
            ),
            (
                RUN_PARITY,
                (
                    "--source-model", "s", "--package", "p", "--weights", "w",
                    "--token-ids", "1", "--atol", "nan",
                ),
                "error: --atol must be a finite non-negative number\n",
            ),
            (
                RUN_PARITY,
                (
                    "--source-model", "s", "--package", "p", "--weights", "w",
                    "--token-ids", "1", "--cosine-min", "2",
                ),
                "error: --cosine-min must be finite and between -1 and 1\n",
            ),
        )
        for script, args, expected in cases:
            with self.subTest(script=script.name, expected=expected):
                completed = run_script(script, *args, no_site=True)
                self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
                self.assertEqual(completed.stderr, expected)

    def test_capture_mlx_fails_closed_on_missing_mlx_and_package_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp).resolve()
            _, package, weights = write_dependency_free_scaffold(root)
            output = root / "target.npz"
            missing = run_script(
                CAPTURE_MLX,
                "--package", package, "--weights", weights,
                "--token-ids", "1", "2", "--output", output,
                no_site=True,
            )
            self.assertEqual(missing.returncode, 2, missing.stdout + missing.stderr)
            self.assertEqual(
                missing.stderr,
                "error: capture_mlx.py requires optional package 'mlx'; "
                "install it with python3 -m pip install mlx\n",
            )

            with (package / "model.py").open("a", encoding="utf-8") as handle:
                handle.write("\n# user modification\n")
            drifted = run_script(
                CAPTURE_MLX,
                "--package", package, "--weights", weights,
                "--token-ids", "1", "--output", output,
                no_site=True,
            )
            self.assertEqual(drifted.returncode, 2, drifted.stdout + drifted.stderr)
            self.assertIn("unvalidated modifications", drifted.stderr)

            self.assertIn("model.py", drifted.stderr)
            allowed = run_script(
                CAPTURE_MLX,
                "--package", package, "--weights", weights,
                "--token-ids", "1", "--output", output, "--allow-modified",
                no_site=True,
            )
            self.assertEqual(allowed.returncode, 2, allowed.stdout + allowed.stderr)
            self.assertIn("requires optional package 'mlx'", allowed.stderr)

            package = root / "package-config-drift"
            inspection = root / "inspection-config-drift.json"
            inspected = run_script(
                SCRIPTS / "inspect_model.py", DECODER_FIXTURE,
                "--output", inspection, no_site=True,
            )
            self.assertEqual(inspected.returncode, 0, inspected.stdout + inspected.stderr)
            scaffolded = run_script(
                SCRIPTS / "scaffold_port.py", inspection,
                "--artifact-root", DECODER_FIXTURE, "--output", package,
                no_site=True,
            )
            self.assertEqual(scaffolded.returncode, 0, scaffolded.stdout + scaffolded.stderr)
            with (package / "config.json").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            config_drift = run_script(
                CAPTURE_MLX,
                "--package", package, "--weights", weights,
                "--token-ids", "1", "--output", output,
                no_site=True,
            )
            self.assertEqual(config_drift.returncode, 2, config_drift.stdout + config_drift.stderr)
            self.assertIn("config.json", config_drift.stderr)

    def test_capture_mlx_rejects_stale_conversion_digests(self) -> None:
        for filename in ("model.safetensors", "target-manifest.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as raw_tmp:
                root = Path(raw_tmp).resolve()
                _, package, weights = write_dependency_free_scaffold(root)
                with (weights / filename).open("ab") as handle:
                    handle.write(b"\n " if filename == "target-manifest.json" else b"tampered")
                completed = run_script(
                    CAPTURE_MLX,
                    "--package", package,
                    "--weights", weights,
                    "--token-ids", "1", "2",
                    "--output", root / "target.npz",
                    no_site=True,
                )

            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            expected = (
                "target manifest"
                if filename == "target-manifest.json"
                else "converted weights"
            )
            self.assertIn(expected, completed.stderr)
            self.assertIn("does not match conversion-report.json attestation", completed.stderr)

    def test_token_id_mode_never_requires_a_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp).resolve()
            _, package, weights = write_dependency_free_scaffold(root)
            completed = run_script(
                CAPTURE_MLX,
                "--package", package, "--weights", weights,
                "--token-ids", "1,2,3", "--output", root / "target.npz",
                no_site=True,
            )
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertNotIn("tokenizer", completed.stderr.lower())
            self.assertIn("requires optional package 'mlx'", completed.stderr)

    def test_ladder_mapping_is_same_name_contiguous_and_ordered(self) -> None:
        keys = {
            "generated_token_ids", "logits", "layer.1.hidden", "attention_mask",
            "embed", "layer.0.hidden", "input_ids", "final_norm",
        }
        ladder = run_parity.build_parity_ladder(keys, keys)
        expected = [
            "input_ids", "embed", "layer.0.hidden", "layer.1.hidden",
            "final_norm", "logits", "generated_token_ids",
        ]
        self.assertEqual([rung["name"] for rung in ladder], expected)
        self.assertEqual(run_parity.tensor_key_mapping(ladder), {key: key for key in expected})
        with self.assertRaisesRegex(run_parity.SkillError, "contiguous"):
            run_parity.build_parity_ladder(keys - {"layer.0.hidden"}, keys)

    def test_report_schema_is_strict_and_dependency_free(self) -> None:
        report = {
            "schema_version": 1,
            "ok": True,
            "inputs": {
                "source_model": "/source", "package": "/package", "weights": "/weights",
                "mode": "dense-decoder", "input_mode": "token_ids",
                "prompts": None, "token_ids": [1, 2], "attention_mask": [[1, 1]],
                "generate_steps": 1, "seed": 0, "dtype_policy": "float32",
                "allow_modified": False, "fault_inject_target": None,
            },
            "tolerances": {"atol": 1e-5, "rtol": 1e-4, "cosine_min": -1.0},
            "rungs": [{
                "position": 0, "name": "input_ids", "source_key": "input_ids",
                "target_key": "input_ids", "exact": True, "pass": True,
                "max_abs": 0.0, "max_rel": 0.0, "cosine": 1.0,
            }],
            "summary": {
                "status": "pass", "evaluated_rungs": 1, "total_rungs": 1,
                "stopped_at": None, "debug_target": None,
                "message": "All parity rungs passed in runbook order.",
            },
        }
        self.assertIs(run_parity.validate_parity_report(report), report)
        report["unexpected"] = True
        with self.assertRaisesRegex(run_parity.SkillError, "invalid field set"):
            run_parity.validate_parity_report(report)

    def test_capture_manifest_validation_reuses_oracle_schema_without_site_packages(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "manifests" / "oracle.json"
        completed = run_script(CAPTURE_MLX, "--validate-manifest", fixture, no_site=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(completed.stdout, '{"ok": true, "schema_version": 1}\n')

    def test_run_parity_propagates_dependency_isolation_to_bounded_children(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp).resolve()
            _, package, weights = write_dependency_free_scaffold(root)
            completed = run_script(
                RUN_PARITY,
                "--source-model", DECODER_FIXTURE,
                "--package", package,
                "--weights", weights,
                "--token-ids", "1", "2",
                no_site=True,
            )
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertIn("capture_oracle.py failed with exit 2", completed.stderr)
            self.assertIn("requires optional package 'torch'", completed.stderr)


@unittest.skipUnless(HAS_TORCH_STACK, "torch, transformers, and safetensors are required")
class ParityRunnerTorchPreparationContractTests(unittest.TestCase):
    def test_stock_llama_constructs_offline_and_real_tools_prepare_converted_weights(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp).resolve()
            source = root / "source"
            inspection = root / "inspection.json"
            package = root / "package"
            draft = root / "WEIGHT_MAP.draft.json"
            mapping = root / "WEIGHT_MAP.json"
            converted = root / "converted"
            write_tiny_stock_llama(source)
            inspected = run_script(SCRIPTS / "inspect_model.py", source, "--output", inspection)
            self.assertEqual(inspected.returncode, 0, inspected.stdout + inspected.stderr)
            scaffolded = run_script(
                SCRIPTS / "scaffold_port.py", inspection,
                "--artifact-root", source, "--output", package,
            )
            self.assertEqual(scaffolded.returncode, 0, scaffolded.stdout + scaffolded.stderr)
            drafted = run_script(
                SCRIPTS / "convert_checkpoint.py",
                "--source", inspection,
                "--scaffold-manifest", package / "scaffold-manifest.json",
                "--emit-draft-map", draft,
            )
            self.assertEqual(drafted.returncode, 0, drafted.stdout + drafted.stderr)
            authored = json.loads(draft.read_text(encoding="utf-8"))
            self.assertEqual(authored["unresolved"], [], authored)
            authored["draft"] = False
            mapping.write_text(
                json.dumps(authored, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result = run_script(
                SCRIPTS / "convert_checkpoint.py",
                "--source", source, "--mapping", mapping, "--output", converted,
                disable_optional_tensor_libs=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((converted / "model.safetensors").is_file())


@unittest.skipUnless(
    HAS_TORCH_STACK,
    "torch, transformers, safetensors, and NumPy are required",
)
@require_mlx_keystone(HAS_MLX_RUNTIME, "a usable MLX runtime is required")
class ParityRunnerEndToEndContractTests(unittest.TestCase):
    def test_real_tool_chain_passes_then_seeded_target_fault_stops_at_layer_zero(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp).resolve()
            source = root / "source"
            inspection = root / "inspection.json"
            package = root / "package"
            draft = root / "WEIGHT_MAP.draft.json"
            mapping = root / "WEIGHT_MAP.json"
            converted = root / "converted"
            passing_report = root / "passing.json"
            failing_report = root / "failing.json"
            write_tiny_stock_llama(source)

            inspected = run_script(
                SCRIPTS / "inspect_model.py", source, "--output", inspection,
            )
            self.assertEqual(inspected.returncode, 0, inspected.stdout + inspected.stderr)
            scaffolded = run_script(
                SCRIPTS / "scaffold_port.py", inspection,
                "--artifact-root", source, "--output", package,
            )
            self.assertEqual(scaffolded.returncode, 0, scaffolded.stdout + scaffolded.stderr)
            drafted = run_script(
                SCRIPTS / "convert_checkpoint.py",
                "--source", inspection,
                "--scaffold-manifest", package / "scaffold-manifest.json",
                "--emit-draft-map", draft,
            )
            self.assertEqual(drafted.returncode, 0, drafted.stdout + drafted.stderr)
            authored = json.loads(draft.read_text(encoding="utf-8"))
            self.assertEqual(authored["unresolved"], [], authored)
            authored["draft"] = False
            mapping.write_text(
                json.dumps(authored, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            converted_result = run_script(
                SCRIPTS / "convert_checkpoint.py",
                "--source", source, "--mapping", mapping, "--output", converted,
                disable_optional_tensor_libs=True,
            )
            self.assertEqual(
                converted_result.returncode,
                0,
                converted_result.stdout + converted_result.stderr,
            )

            parity_args = (
                "--source-model", source, "--package", package, "--weights", converted,
                "--token-ids", "1", "5", "7", "9", "--generate-steps", "3",
                "--atol", "0.0002", "--rtol", "0.002", "--cosine-min", "0.9999",
            )
            passed = run_script(RUN_PARITY, *parity_args, "--output", passing_report)
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
            passing = json.loads(passing_report.read_text(encoding="utf-8"))
            run_parity.validate_parity_report(passing)
            self.assertTrue(passing["ok"], passing)
            self.assertEqual(passing["summary"]["status"], "pass")

            failed = run_script(
                RUN_PARITY,
                *parity_args,
                "--fault-inject-target", "layer.0.hidden",
                "--output", failing_report,
            )
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
            self.assertFalse(failing["rungs"][-1]["pass"])


if __name__ == "__main__":
    unittest.main()
