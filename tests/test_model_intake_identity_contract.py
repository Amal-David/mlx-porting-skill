"""Artifact identity and license gates for local static model intake."""
from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
INSPECTOR = ROOT / "mlx-model-porting" / "scripts" / "inspect_model.py"

MIT_LICENSE = """MIT License

Copyright (c) 2026 Fixture Author

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
"""


def safetensors_bytes() -> bytes:
    specs = {
        "model.embed_tokens.weight": {
            "dtype": "F16",
            "shape": [2, 2],
            "data_offsets": [0, 8],
        },
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F16",
            "shape": [2, 2],
            "data_offsets": [8, 16],
        },
        "model.layers.0.mlp.gate_proj.weight": {
            "dtype": "F16",
            "shape": [2, 2],
            "data_offsets": [16, 24],
        },
    }
    header = json.dumps(specs, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + (b"\0" * 24)


class ModelIntakeIdentityContractTests(unittest.TestCase):
    def write_model(
        self,
        root: Path,
        *,
        license_value: str | None = "apache-2.0",
        license_file: bool = False,
    ) -> None:
        root.mkdir()
        config: dict[str, object] = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 2,
            "intermediate_size": 2,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 2,
        }
        if license_value is not None:
            config["license"] = license_value
        (root / "config.json").write_text(
            json.dumps(config, indent=2) + "\n",
            encoding="utf-8",
        )
        (root / "model.safetensors").write_bytes(safetensors_bytes())
        if license_file:
            (root / "LICENSE").write_text(MIT_LICENSE, encoding="utf-8")

    def inspect(
        self,
        model: Path,
        output: Path,
        *args: str,
    ) -> dict[str, object]:
        completed = subprocess.run(
            [
                sys.executable,
                str(INSPECTOR),
                str(model),
                "--output",
                str(output),
                *args,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return json.loads(output.read_text(encoding="utf-8"))

    def test_local_identity_is_deterministic_portable_and_changes_on_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp) / "model"
            self.write_model(root)
            output = root / "inspection.json"
            markdown = root / "inspection.md"

            first = self.inspect(root, output, "--markdown", str(markdown))
            second = self.inspect(root, output, "--markdown", str(markdown))

            first_identity = first["artifact_identity"]
            second_identity = second["artifact_identity"]
            self.assertEqual(first_identity, second_identity)
            self.assertEqual(first_identity["status"], "verified")
            self.assertTrue(first_identity["immutable"])
            self.assertRegex(first_identity["fingerprint"], r"\Asha256:[0-9a-f]{64}\Z")
            self.assertEqual(
                [record["path"] for record in first_identity["manifest"]],
                ["config.json", "model.safetensors"],
            )
            self.assertEqual(first["source"]["input"], "model")
            self.assertEqual(first["local_path"], "model")
            for record in first_identity["manifest"]:
                path = PurePosixPath(record["path"])
                self.assertFalse(path.is_absolute())
                self.assertNotIn("..", path.parts)
                self.assertNotIn(raw_tmp, record["path"])
                content = (root / record["path"]).read_bytes()
                self.assertEqual(record["size_bytes"], len(content))
                self.assertEqual(record["sha256"], hashlib.sha256(content).hexdigest())

            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["hidden_size"] = 4
            (root / "config.json").write_text(
                json.dumps(config, indent=2) + "\n",
                encoding="utf-8",
            )
            changed = self.inspect(root, output, "--markdown", str(markdown))
            self.assertNotEqual(
                changed["artifact_identity"]["fingerprint"],
                first_identity["fingerprint"],
            )

    def test_missing_and_placeholder_license_evidence_block_recommendation(self) -> None:
        for name, value in (("missing", None), ("unknown", "unknown"), ("restricted", "proprietary")):
            with self.subTest(case=name), tempfile.TemporaryDirectory() as raw_tmp:
                root = Path(raw_tmp) / "model"
                self.write_model(root, license_value=value)
                report = self.inspect(root, Path(raw_tmp) / "inspection.json")

                self.assertEqual(report["artifact_identity"]["status"], "verified")
                self.assertEqual(report["license"]["status"], "review-required")
                self.assertTrue(report["license"]["requires_review"])
                self.assertEqual(report["license"]["accepted_evidence"], [])
                self.assertTrue(
                    any("license evidence is missing or unacceptable" in blocker for blocker in report["recommendation_blockers"]),
                    report["recommendation_blockers"],
                )
                self.assertIsNone(report["recommended_family"])
                self.assertEqual(report["recommended_families"], [])
                self.assertEqual(report["recommended_runbooks"], [])

    def test_readable_artifact_bound_license_text_is_acceptable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp) / "model"
            self.write_model(root, license_value=None, license_file=True)
            report = self.inspect(root, Path(raw_tmp) / "inspection.json")

            self.assertEqual(report["license"]["status"], "acceptable-evidence")
            self.assertFalse(report["license"]["requires_review"])
            self.assertEqual(report["license"]["license_files"], ["LICENSE"])
            self.assertEqual(
                report["license"]["accepted_evidence"][0]["kind"],
                "license-file",
            )
            self.assertEqual(report["recommended_family"], "dense-decoder-transformer")
            self.assertFalse(
                any("license evidence" in blocker for blocker in report["recommendation_blockers"]),
                report["recommendation_blockers"],
            )

    def test_truncated_inventory_has_no_immutable_identity_or_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp) / "model"
            self.write_model(root)
            report = self.inspect(
                root,
                Path(raw_tmp) / "inspection.json",
                "--max-files",
                "1",
            )

            self.assertEqual(report["artifact_identity"]["status"], "incomplete")
            self.assertFalse(report["artifact_identity"]["immutable"])
            self.assertIsNone(report["artifact_identity"]["fingerprint"])
            self.assertTrue(report["artifact_identity"]["errors"])
            self.assertTrue(
                any("immutable artifact identity is missing or incomplete" in blocker for blocker in report["recommendation_blockers"]),
                report["recommendation_blockers"],
            )
            self.assertIsNone(report["recommended_family"])

    def test_single_file_intake_binds_sibling_config_used_for_routing(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp) / "model"
            self.write_model(root)
            report = self.inspect(
                root / "model.safetensors",
                Path(raw_tmp) / "inspection.json",
            )

            paths = [record["path"] for record in report["artifact_identity"]["manifest"]]
            self.assertEqual(paths, ["config.json", "model.safetensors"])
            self.assertEqual(report["artifact_identity"]["status"], "verified")
            self.assertEqual(report["license"]["status"], "acceptable-evidence")
            self.assertEqual(report["recommended_family"], "dense-decoder-transformer")

    def test_output_cannot_overwrite_source_config(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp) / "model"
            self.write_model(root)
            config = root / "config.json"
            original = config.read_bytes()

            completed = subprocess.run(
                [sys.executable, str(INSPECTOR), str(root), "--output", str(config)],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertIn("must not overwrite model artifact config.json", completed.stderr)
            self.assertEqual(config.read_bytes(), original)

    def test_markdown_cannot_overwrite_license_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp) / "model"
            self.write_model(root, license_file=True)
            license_path = root / "LICENSE"
            original = license_path.read_bytes()
            output = Path(raw_tmp) / "inspection.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(INSPECTOR),
                    str(root),
                    "--output",
                    str(output),
                    "--markdown",
                    str(license_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertIn("must not overwrite model artifact LICENSE", completed.stderr)
            self.assertEqual(license_path.read_bytes(), original)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
