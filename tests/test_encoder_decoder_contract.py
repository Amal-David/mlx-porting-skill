"""Contracts for additive T5 encoder-decoder scaffolding and real MLX parity."""
from __future__ import annotations

import importlib.util
import hashlib
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
EXAMPLE = ROOT / "mlx-model-porting" / "examples" / "worked-port-t5-small"
T5_MODEL = Path(
    os.environ.get(
        "T5_SMALL_LOCAL_MODEL",
        str(Path.home() / ".cache" / "mlx-porting-work" / "t5-small"),
    )
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from mlx_keystone import require_mlx_keystone  # noqa: E402


def run_script(script: str, *args: object) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *(str(value) for value in args)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


class EncoderDecoderDependencyFreeContractTests(unittest.TestCase):
    def test_t5_config_allowlist_and_target_contract(self) -> None:
        import scaffold_port

        config = {
            "architectures": ["T5ForConditionalGeneration"],
            "d_ff": 16,
            "d_kv": 4,
            "d_model": 8,
            "decoder_start_token_id": 0,
            "is_encoder_decoder": True,
            "layer_norm_epsilon": 1e-6,
            "license": "apache-2.0",
            "model_type": "t5",
            "num_heads": 2,
            "num_layers": 1,
            "pad_token_id": 0,
            "relative_attention_num_buckets": 8,
            "relative_attention_max_distance": 16,
            "tie_word_embeddings": True,
            "vocab_size": 32,
        }
        self.assertIs(scaffold_port.validate_t5_config(config), config)
        targets = {item["key"] for item in scaffold_port.t5_target_tensors(config)}
        self.assertIn("shared.weight", targets)
        self.assertIn(
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            targets,
        )
        self.assertIn(
            "decoder.block.0.layer.1.EncDecAttention.k.weight",
            targets,
        )
        self.assertNotIn("lm_head.weight", targets)
        self.assertFalse(any("EncDecAttention.relative_attention_bias" in key for key in targets))

    def test_gated_and_unknown_t5_config_features_fail_closed(self) -> None:
        import scaffold_port

        config = {
            "d_ff": 16,
            "d_kv": 4,
            "d_model": 8,
            "decoder_start_token_id": 0,
            "feed_forward_proj": "gated-gelu",
            "is_encoder_decoder": True,
            "layer_norm_epsilon": 1e-6,
            "model_type": "t5",
            "num_heads": 2,
            "num_layers": 1,
            "pad_token_id": 0,
            "relative_attention_num_buckets": 8,
            "tie_word_embeddings": True,
            "unknown_attention_mode": "unsafe",
            "vocab_size": 32,
        }
        errors = scaffold_port.unsupported_t5_features(config)
        self.assertTrue(any("gated T5 variants fail closed" in error for error in errors))
        self.assertIn(
            "unrecognized computation-relevant config key 'unknown_attention_mode'",
            errors,
        )

    def test_encoder_decoder_parity_ladder_orders_cross_attention(self) -> None:
        import run_parity

        keys = {
            "input_ids",
            "attention_mask",
            "encoder.embed",
            "encoder.layer.0.hidden",
            "encoder.final_norm",
            "decoder_input_ids",
            "decoder.embed",
            "decoder.layer.0.cross_attention",
            "decoder.layer.0.hidden",
            "decoder.final_norm",
            "logits",
            "generated_token_ids",
        }
        ladder = run_parity.build_parity_ladder(keys, keys)
        self.assertEqual(
            [rung["name"] for rung in ladder],
            [
                "input_ids",
                "attention_mask",
                "encoder.embed",
                "encoder.layer.0.hidden",
                "encoder.final_norm",
                "decoder_input_ids",
                "decoder.embed",
                "decoder.layer.0.cross_attention",
                "decoder.layer.0.hidden",
                "decoder.final_norm",
                "logits",
                "generated_token_ids",
            ],
        )
        self.assertTrue(ladder[1]["exact"])

    def test_malformed_relative_bucket_geometry_fails_closed(self) -> None:
        import scaffold_port

        base = {
            "d_ff": 16,
            "d_kv": 4,
            "d_model": 8,
            "decoder_start_token_id": 0,
            "is_encoder_decoder": True,
            "layer_norm_epsilon": 1e-6,
            "model_type": "t5",
            "num_heads": 2,
            "num_layers": 1,
            "pad_token_id": 0,
            "relative_attention_num_buckets": 8,
            "relative_attention_max_distance": 16,
            "tie_word_embeddings": True,
            "vocab_size": 32,
        }
        for buckets in (1, 2, 3, 5):
            with self.subTest(buckets=buckets):
                config = {**base, "relative_attention_num_buckets": buckets}
                with self.assertRaisesRegex(
                    scaffold_port.SkillError, "even integer >= 4"
                ):
                    scaffold_port.validate_t5_config(config)
        with self.assertRaisesRegex(scaffold_port.SkillError, "must exceed"):
            scaffold_port.validate_t5_config(
                {**base, "relative_attention_max_distance": 2}
            )


def real_t5_available() -> bool:
    return (
        T5_MODEL.is_dir()
        and (T5_MODEL / "model.safetensors").is_file()
        and importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


@require_mlx_keystone(
    real_t5_available(),
    "cached prepared t5-small plus MLX, Torch, and Transformers are required",
)
class EncoderDecoderRealMLXContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name).resolve()
        cls.inspection = cls.root / "inspection.json"
        cls.package = cls.root / "package"
        cls.converted = cls.root / "converted"
        commands = (
            ("inspect_model.py", T5_MODEL, "--output", cls.inspection),
            (
                "scaffold_port.py",
                cls.inspection,
                "--artifact-root",
                T5_MODEL,
                "--output",
                cls.package,
            ),
            (
                "convert_checkpoint.py",
                "--source",
                T5_MODEL,
                "--mapping",
                EXAMPLE / "WEIGHT_MAP.json",
                "--output",
                cls.converted,
            ),
        )
        for script, *args in commands:
            result = run_script(script, *args)
            if result.returncode != 0:
                cls._temporary.cleanup()
                raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()
        super().tearDownClass()

    def run_real_parity(
        self,
        report: Path,
        *fixture_args: object,
        weights: Path | None = None,
        generate_steps: int = 8,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        result = run_script(
            "run_parity.py",
            "--mode",
            "encoder-decoder",
            "--source-model",
            T5_MODEL,
            "--package",
            self.package,
            "--weights",
            weights or self.converted,
            *fixture_args,
            "--generate-steps",
            generate_steps,
            "--atol",
            "1e-3",
            "--rtol",
            "1e-4",
            "--cosine-min",
            "0.999999",
            "--output",
            report,
        )
        if not report.is_file():
            raise AssertionError(result.stdout + result.stderr)
        payload = json.loads(report.read_text(encoding="utf-8"))
        return result, payload

    def corrupted_weights(self, name: str, keys: tuple[str, ...]) -> Path:
        import mlx.core as mx

        output = self.root / name
        shutil.copytree(self.converted, output)
        weights = mx.load(str(output / "model.safetensors"))
        mx.eval(*weights.values())
        for key in keys:
            weights[key] = mx.zeros_like(weights[key])
        mx.eval(*weights.values())
        mx.save_safetensors(str(output / "model.safetensors"), weights)
        weights_path = output / "model.safetensors"
        report_path = output / "conversion-report.json"
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
        return output

    def test_real_t5_full_ladder_and_incremental_cache(self) -> None:
        import mlx.core as mx

        report = self.root / "parity.json"
        fixture = (13959, 1566, 12, 2968, 10, 37, 629, 19, 1627, 5, 1)
        result, payload = self.run_real_parity(report, "--token-ids", *fixture)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["summary"]["total_rungs"], 27)
        self.assertEqual(payload["rungs"][1]["name"], "attention_mask")
        self.assertTrue(payload["rungs"][1]["exact"])
        self.assertEqual(payload["rungs"][-1]["name"], "generated_token_ids")

        previous_path = sys.path[:]
        prior = {name: sys.modules.pop(name, None) for name in ("config", "model")}
        try:
            sys.path.insert(0, str(self.package))
            from model import greedy_generate, load_model

            model = load_model(
                self.package / "config.json", self.converted / "model.safetensors"
            )
            encoder_ids = mx.array([fixture], dtype=mx.int32)
            encoder_mask = mx.ones(encoder_ids.shape, dtype=mx.int32)
            generated = greedy_generate(model, encoder_ids, 8, attention_mask=encoder_mask)
            mx.eval(generated)
            continuation = generated.tolist()[0]
            self.assertEqual(continuation, [644, 4598, 229, 19250, 5, 1, 1, 40])
            self.assertGreater(len(set(continuation)), 2)

            relative_attention = model.encoder.block[0].layer[0].SelfAttention
            relative_bias = relative_attention.relative_attention_bias
            exact_edge = relative_attention.position_bias(9, 9)
            distance_edge = relative_attention.position_bias(129, 129)
            buckets = relative_bias._bucket(mx.array([-7, -8, -127, -128, -129]))
            mx.eval(exact_edge, distance_edge, buckets)
            self.assertEqual(exact_edge.shape, (1, 8, 9, 9))
            self.assertEqual(distance_edge.shape, (1, 8, 129, 129))
            self.assertEqual(buckets.tolist()[0:2], [7, 8])
            self.assertEqual(buckets.tolist()[-2:], [15, 15])

            memory, _ = model.encode(encoder_ids, attention_mask=encoder_mask)
            start = mx.array([[0]], dtype=mx.int32)
            cross_projections = {
                id(projection)
                for block in model.decoder.block
                for projection in (
                    block.layer[1].EncDecAttention.k,
                    block.layer[1].EncDecAttention.v,
                )
            }
            linear_type = type(model.decoder.block[0].layer[1].EncDecAttention.k)
            original_call = linear_type.__call__
            projection_calls = 0

            def counted_call(instance, *args, **kwargs):
                nonlocal projection_calls
                if id(instance) in cross_projections:
                    projection_calls += 1
                return original_call(instance, *args, **kwargs)

            linear_type.__call__ = counted_call
            try:
                first_logits, cache, _ = model.decode(
                    start, memory, encoder_mask=encoder_mask
                )
                first_token = mx.argmax(first_logits[:, -1, :], axis=-1).astype(mx.int32)
                calls_after_prefill = projection_calls
                cached_logits, _, _ = model.decode(
                    first_token[:, None], memory, encoder_mask=encoder_mask, cache=cache
                )
            finally:
                linear_type.__call__ = original_call
            full_logits, _, _ = model.decode(
                mx.concatenate((start, first_token[:, None]), axis=1),
                memory,
                encoder_mask=encoder_mask,
            )
            mx.eval(cached_logits, full_logits)
            self.assertEqual(calls_after_prefill, 2 * model.config.num_layers)
            self.assertEqual(projection_calls, calls_after_prefill)
            self.assertTrue(
                mx.allclose(
                    cached_logits[:, -1, :],
                    full_logits[:, -1, :],
                    atol=1e-5,
                    rtol=1e-5,
                ).item()
            )
            self.assertEqual(cache[0][2].shape[2], encoder_ids.shape[1])
        finally:
            sys.path[:] = previous_path
            for name in ("config", "model"):
                sys.modules.pop(name, None)
                if prior[name] is not None:
                    sys.modules[name] = prior[name]

        relative_corrupt = self.corrupted_weights(
            "relative-corrupt",
            ("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",),
        )
        result, negative = self.run_real_parity(
            self.root / "relative-negative.json",
            "--token-ids",
            *fixture,
            weights=relative_corrupt,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(negative["summary"]["stopped_at"], "encoder.layer.0.hidden")

        cross_corrupt = self.corrupted_weights(
            "cross-kv-corrupt",
            (
                "decoder.block.0.layer.1.EncDecAttention.k.weight",
                "decoder.block.0.layer.1.EncDecAttention.v.weight",
            ),
        )
        result, negative = self.run_real_parity(
            self.root / "cross-kv-negative.json",
            "--token-ids",
            *fixture,
            weights=cross_corrupt,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(
            negative["summary"]["stopped_at"], "decoder.layer.0.cross_attention"
        )

    def test_real_t5_padded_two_prompt_batch(self) -> None:
        report = self.root / "padded-parity.json"
        result, payload = self.run_real_parity(
            report,
            "--prompt",
            "translate English to German: Hi.",
            "--prompt",
            "translate English to German: The house is wonderful and the garden is quiet.",
            generate_steps=4,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["summary"]["total_rungs"], 27)
        mask_rung = payload["rungs"][1]
        self.assertEqual(mask_rung["name"], "attention_mask")
        self.assertTrue(mask_rung["exact"])
        self.assertTrue(mask_rung["pass"])


if __name__ == "__main__":
    unittest.main()
