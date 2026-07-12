from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import nightly_knowledge_curator as nightly  # noqa: E402


class NightlyKnowledgeCuratorHardeningTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "POSIX process-group behavior")
    def test_timeout_terminates_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "descendant-survived"
            child = f"import pathlib,time; time.sleep(0.6); pathlib.Path({str(marker)!r}).write_text('leaked')"
            parent = (
                "import subprocess,sys,time; "
                f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
                "time.sleep(10)"
            )
            started = time.monotonic()

            with self.assertRaisesRegex(nightly.SkillError, "timed out"):
                nightly.run_command(
                    [sys.executable, "-c", parent],
                    ROOT,
                    timeout=0.1,
                )

            self.assertLess(time.monotonic() - started, 2.0)
            time.sleep(0.8)
            self.assertFalse(marker.exists(), "nightly command left a descendant running")

    def test_noisy_output_is_byte_bounded_and_retains_tail(self) -> None:
        code = "import sys; sys.stdout.write('HEAD' + ('x' * 2000000) + 'TAIL')"

        record = nightly.run_command(
            [sys.executable, "-c", code],
            ROOT,
            timeout=5,
        )

        self.assertFalse(record["timed_out"])
        self.assertEqual(record["returncode"], 0)
        self.assertLessEqual(len(record["stdout"].encode("utf-8")), nightly.RECEIPT_OUTPUT_BYTES)
        self.assertTrue(record["stdout"].endswith("TAIL"))

    def test_command_receipt_and_output_redact_common_secret_forms(self) -> None:
        secrets = (
            "aws-secret-value",
            "private-key-value",
            "database-password",
            "dXNlcjpwYXNzd29yZA==",
            "bearer-value",
            "header-api-key-value",
            "url-password",
        )
        lines = (
            "AWS_SECRET_ACCESS_KEY=aws-secret-value",
            "PRIVATE_KEY=private-key-value",
            "DATABASE_URL=postgresql://db-user:database-password@db.example/app",
            "Authorization: Basic dXNlcjpwYXNzd29yZA==",
            "Authorization: Bearer bearer-value",
            "X-API-Key: header-api-key-value",
            "postgresql://url-user:url-password@db.example/app",
        )
        code = f"print({chr(10).join(lines)!r})"

        record = nightly.run_command(
            [sys.executable, "-c", code],
            ROOT,
            timeout=5,
        )

        serialized = json.dumps(record)
        for secret in secrets:
            self.assertNotIn(secret, serialized)
        self.assertIn("[REDACTED]", serialized)

    def test_failed_command_redacts_secret_from_exception(self) -> None:
        code = (
            "import sys; "
            "print('Authorization: Bearer failure-secret-value', file=sys.stderr); "
            "sys.exit(3)"
        )

        with self.assertRaises(nightly.SkillError) as raised:
            nightly.run_command(
                [sys.executable, "-c", code],
                ROOT,
                timeout=5,
            )

        message = str(raised.exception)
        self.assertNotIn("failure-secret-value", message)
        self.assertIn("[REDACTED]", message)


if __name__ == "__main__":
    unittest.main()
