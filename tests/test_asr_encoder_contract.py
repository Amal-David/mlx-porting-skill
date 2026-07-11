"""Contracts for additive HuBERT/Wav2Vec2 acoustic-encoder scaffolding."""
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
REAL_HUBERT = Path.home() / ".cache" / "mlx-porting-work" / "hubert-base-ls960"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import capture_mlx  # noqa: E402
import scaffold_port  # noqa: E402
from _capture_common import validate_capture_manifest  # noqa: E402
from _scaffold_asr import asr_target_tensors, validate_asr_config  # noqa: E402
from mlx_keystone import require_mlx_keystone  # noqa: E402
from recommend_optimizations import validate_trusted_inspection  # noqa: E402


def tiny_asr_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "architectures": ["HubertModel"],
        "conv_dim": [4, 4],
        "conv_kernel": [3, 2],
        "conv_stride": [2, 2],
        "do_stable_layer_norm": False,
        "feat_extract_activation": "gelu",
        "feat_proj_layer_norm": True,
        "hidden_act": "gelu",
        "hidden_size": 8,
        "intermediate_size": 16,
        "layer_norm_eps": 1e-5,
        "license": "apache-2.0",
        "model_type": "hubert",
        "num_attention_heads": 2,
        "num_conv_pos_embedding_groups": 2,
        "num_conv_pos_embeddings": 4,
        "num_hidden_layers": 2,
    }
    config.update(overrides)
    return config


class ASREncoderContractTests(unittest.TestCase):
    def test_allowlist_and_target_contract_cover_hubert_encoder(self) -> None:
        config = tiny_asr_config()
        self.assertIs(validate_asr_config(config), config)
        targets = {item["key"]: item["shape"] for item in asr_target_tensors(config)}
        self.assertEqual(targets["feature_projection.projection.weight"], [8, 4])
        self.assertEqual(targets["encoder.pos_conv_embed.conv.weight_v"], [8, 4, 4])
        self.assertEqual(targets["encoder.layers.1.attention.q_proj.weight"], [8, 8])
        self.assertEqual(targets["encoder.layers.1.feed_forward.output_dense.weight"], [8, 16])
        self.assertNotIn("feature_extractor.conv_layers.0.conv.weight", targets)

    def test_whisper_and_decoding_graphs_fail_closed(self) -> None:
        with self.assertRaisesRegex(scaffold_port.SkillError, "Whisper/sequence-to-sequence"):
            validate_asr_config(tiny_asr_config(model_type="whisper", is_encoder_decoder=True))
        with self.assertRaisesRegex(scaffold_port.SkillError, "only model_type 'hubert' or 'wav2vec2'"):
            validate_asr_config(tiny_asr_config(model_type="conformer"))
        self.assertFalse(scaffold_port._is_whisper_config({
            "architectures": ["T5ForConditionalGeneration"],
            "is_encoder_decoder": True,
            "model_type": "t5",
        }))
        self.assertTrue(scaffold_port._is_whisper_config({
            "architectures": ["WhisperForConditionalGeneration"],
            "is_encoder_decoder": True,
            "model_type": "whisper",
        }))

    def test_activation_and_ctc_configs_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            scaffold_port.SkillError,
            "feat_extract_activation='gelu'",
        ):
            validate_asr_config(tiny_asr_config(feat_extract_activation="relu"))
        with self.assertRaisesRegex(scaffold_port.SkillError, r"rejects \*ForCTC"):
            validate_asr_config(tiny_asr_config(architectures=["HubertForCTC"]))
        for key in ("is_decoder", "is_encoder_decoder"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(scaffold_port.SkillError, f"{key}=false"):
                    validate_asr_config(tiny_asr_config(**{key: True}))

    def test_asr_capture_manifest_mode_is_explicit(self) -> None:
        fixture = json.loads(
            (ROOT / "tests" / "fixtures" / "manifests" / "oracle.json").read_text()
        )
        fixture["capture"] = {
            "mode": "asr",
            "waveform_samples": 4000,
            "seed": 7006,
            "dtype_policy": "float32",
        }
        self.assertIs(validate_capture_manifest(fixture), fixture)


def _mlx_stack_available() -> bool:
    packages = ("mlx", "numpy", "safetensors", "torch", "transformers")
    if not all(importlib.util.find_spec(package) is not None for package in packages):
        return False
    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core as mx; x=mx.array([1]); mx.eval(x)"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return probe.returncode == 0


def _real_hubert_available() -> bool:
    return (
        _mlx_stack_available()
        and (REAL_HUBERT / "config.json").is_file()
        and (REAL_HUBERT / "model.safetensors").is_file()
    )


def _build_asr_port(
    work: Path,
    source: Path,
    config: dict[str, object],
) -> tuple[Path, Path]:
    inspection_path = work / "inspection.json"
    inspected = subprocess.run(
        [sys.executable, str(SCRIPTS / "inspect_model.py"), str(source), "--output", str(inspection_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if inspected.returncode != 0:
        raise AssertionError(inspected.stdout + inspected.stderr)
    inspection = validate_trusted_inspection(json.loads(inspection_path.read_text()))
    if inspection["routing_decision"]["winner_family"] != "automatic-speech-recognition":
        raise AssertionError(inspection["routing_decision"])

    package = work / "package"
    scaffold_port._write_new_directory(
        package,
        scaffold_port.generate_asr_encoder(inspection, config),
    )
    scaffold_manifest = json.loads((package / "scaffold-manifest.json").read_text())
    source_shapes = {item["key"]: item["shape"] for item in inspection["tensors"]}
    targets = {item["key"]: item["shape"] for item in scaffold_manifest["tensors"]}
    aliases = {
        "encoder.pos_conv_embed.conv.weight_g": (
            "encoder.pos_conv_embed.conv.parametrizations.weight.original0"
        ),
        "encoder.pos_conv_embed.conv.weight_v": (
            "encoder.pos_conv_embed.conv.parametrizations.weight.original1"
        ),
    }
    entries = []
    mapped_sources = set()
    for key, shape in sorted(targets.items()):
        source_key = key if key in source_shapes else aliases.get(key)
        if source_key is None or source_key not in source_shapes:
            raise AssertionError(f"missing source tensor for {key}")
        entries.append({
            "source": source_key,
            "source_shape": source_shapes[source_key],
            "target": key,
            "target_shape": shape,
            "transforms": [{"op": "rename"}],
        })
        mapped_sources.add(source_key)
    mapping = {
        "schema_version": 2,
        "draft": False,
        "dtype_policy": "f32",
        "entries": entries,
        "ignore": [
            {
                "source": key,
                "reason": "raw-waveform frontend or training-only masking is out of scope",
            }
            for key in sorted(set(source_shapes) - mapped_sources)
        ],
        "unresolved": [],
    }
    mapping_path = work / "WEIGHT_MAP.json"
    mapping_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    converted = work / "converted"
    conversion = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "convert_checkpoint.py"),
            "--source",
            str(source),
            "--mapping",
            str(mapping_path),
            "--output",
            str(converted),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if conversion.returncode != 0:
        raise AssertionError(conversion.stdout + conversion.stderr)
    return package, converted


def _run_asr_parity(
    source: Path,
    package: Path,
    converted: Path,
    report: Path,
    *,
    seed: int,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_parity.py"),
            "--source-model",
            str(source),
            "--package",
            str(package),
            "--weights",
            str(converted),
            "--mode",
            "asr",
            "--waveform-samples",
            "4000",
            "--seed",
            str(seed),
            "--atol",
            "1e-4",
            "--rtol",
            "1e-4",
            "--cosine-min",
            "0.99999",
            "--output",
            str(report),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@require_mlx_keystone(
    _real_hubert_available(),
    "real local HuBERT checkpoint plus Torch, Transformers, safetensors, NumPy, and MLX are required",
)
class RealHubertEncoderParityTests(unittest.TestCase):
    def test_real_hubert_encoder_passes_every_rung(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw_tmp:
            work = Path(raw_tmp)
            config = json.loads((REAL_HUBERT / "config.json").read_text())
            package, converted = _build_asr_port(work, REAL_HUBERT, config)
            report_path = work / "parity-report.json"
            parity = _run_asr_parity(
                REAL_HUBERT, package, converted, report_path, seed=7006
            )
            self.assertEqual(parity.returncode, 0, parity.stdout + parity.stderr)
            report = json.loads(report_path.read_text())
            self.assertTrue(report["ok"])
            self.assertEqual(report["summary"]["total_rungs"], 15)
            self.assertTrue(all(rung["pass"] for rung in report["rungs"]))
            self.assertTrue(report["rungs"][0]["exact"])
            self.assertEqual(report["rungs"][0]["max_abs"], 0.0)

            import mlx.core as mx
            import numpy as np

            features = np.zeros((1, 7, int(config["conv_dim"][-1])), dtype=np.float32)
            perturbed = features.copy()
            perturbed[0, 3, 0] = 0.125
            baseline_capture = capture_mlx._capture_asr(
                package,
                converted / "model.safetensors",
                features,
                False,
                np,
                mx,
            )
            perturbed_capture = capture_mlx._capture_asr(
                package,
                converted / "model.safetensors",
                perturbed,
                False,
                np,
                mx,
            )
            self.assertFalse(np.array_equal(
                baseline_capture["final_hidden"],
                perturbed_capture["final_hidden"],
            ))

            second_report = work / "second-input.json"
            second = _run_asr_parity(
                REAL_HUBERT, package, converted, second_report, seed=7007
            )
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertTrue(json.loads(second_report.read_text())["ok"])

            from safetensors.numpy import load_file, save_file

            weights_path = converted / "model.safetensors"
            arrays = load_file(weights_path)
            bug_key = "feature_projection.projection.weight"
            arrays[bug_key] = np.zeros_like(arrays[bug_key])
            temporary = converted / "model.seeded-bug.safetensors"
            save_file(arrays, temporary)
            os.replace(temporary, weights_path)
            conversion_report_path = converted / "conversion-report.json"
            conversion_report = json.loads(conversion_report_path.read_text())
            raw = weights_path.read_bytes()
            conversion_report["outputs"]["weights"].update({
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
            conversion_report_path.write_text(
                json.dumps(conversion_report, indent=2, sort_keys=True) + "\n"
            )
            failing_report = work / "weight-fault.json"
            failed = _run_asr_parity(
                REAL_HUBERT, package, converted, failing_report, seed=7006
            )
            self.assertEqual(failed.returncode, 1, failed.stdout + failed.stderr)
            failure = json.loads(failing_report.read_text())
            self.assertFalse(failure["ok"])
            self.assertEqual(failure["summary"]["stopped_at"], "embed")


def _write_tiny_wav2vec2(source: Path, *, stable: bool) -> dict[str, object]:
    import torch
    from safetensors.torch import save_file
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    config = tiny_asr_config(
        architectures=["Wav2Vec2Model"],
        apply_spec_augment=False,
        do_stable_layer_norm=stable,
        model_type="wav2vec2",
        num_hidden_layers=2,
    )
    torch.manual_seed(7010 if stable else 7009)
    source_model = Wav2Vec2Model(Wav2Vec2Config(**config))
    source_model.train(False)
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in source_model.state_dict().items()
        },
        source / "model.safetensors",
    )
    return config


@require_mlx_keystone(
    _mlx_stack_available(),
    "Torch, Transformers, safetensors, NumPy, and a usable MLX runtime are required",
)
class SyntheticWav2Vec2ParityTests(unittest.TestCase):
    def test_unsupported_asr_interface_fails_with_skill_error(self) -> None:
        import torch

        class UnsupportedModel:
            class Config:
                architectures = ["T5Model"]
                num_hidden_layers = 1

            config = Config()

        import capture_oracle

        with self.assertRaisesRegex(
            scaffold_port.SkillError,
            "requires a built-in HuBERT/Wav2Vec2 base-model interface",
        ):
            capture_oracle.capture_asr_tensors(UnsupportedModel(), 400, 0, torch)

    def _assert_parity(self, *, stable: bool) -> dict[str, object]:
        with tempfile.TemporaryDirectory(dir=ROOT) as raw_tmp:
            work = Path(raw_tmp)
            source = work / "source"
            config = _write_tiny_wav2vec2(source, stable=stable)
            package, converted = _build_asr_port(work, source, config)
            report_path = work / "parity.json"
            completed = _run_asr_parity(
                source,
                package,
                converted,
                report_path,
                seed=7011 if stable else 7008,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads(report_path.read_text())
            self.assertTrue(report["ok"], report)
            self.assertEqual(
                [rung["name"] for rung in report["rungs"]],
                [
                    "input_features",
                    "embed",
                    "layer.0.hidden",
                    "layer.1.hidden",
                    "final_hidden",
                ],
            )
            self.assertTrue(report["rungs"][0]["exact"])
            self.assertEqual(report["rungs"][0]["max_abs"], 0.0)
            return report

    def test_wav2vec2_tuple_projection_passes_real_parity(self) -> None:
        self._assert_parity(stable=False)

    def test_wav2vec2_stable_layer_norm_passes_real_parity(self) -> None:
        self._assert_parity(stable=True)
