from __future__ import annotations

import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import _common as common  # noqa: E402


class FakeEntry:
    def __init__(self, root: Path, index: int):
        self.name = f"file-{index:05d}.txt"
        self.path = str(root / self.name)

    def is_symlink(self) -> bool:
        return False

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        del follow_symlinks
        return False

    def is_file(self, *, follow_symlinks: bool = True) -> bool:
        del follow_symlinks
        return True


class CountingScandir:
    def __init__(self, root: Path, total: int):
        self.root = root
        self.total = total
        self.yielded = 0

    def __enter__(self) -> "CountingScandir":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self):
        for index in range(self.total):
            self.yielded += 1
            yield FakeEntry(self.root, index)


class CommonHardeningContractTests(unittest.TestCase):
    def test_redaction_covers_assignments_headers_and_credential_urls(self) -> None:
        payload = {
            "messages": [
                "AWS_SECRET_ACCESS_KEY=aws-secret-value",
                "PRIVATE_KEY='private-key-value'",
                "DATABASE_URL=postgresql://db-user:db-password@db.example/app",
                "Authorization: Basic dXNlcjpwYXNzd29yZA==",
                "Proxy-Authorization=Bearer proxy-bearer-value",
                "X-API-Key: header-api-key-value",
                "X-Auth-Token=header-auth-token-value",
                "postgresql://url-user:url-password@db.example/app",
                "https://api.example/items?access_token=query-token-value&safe=yes",
                "tokenizer=safe-tokenizer-name",
            ],
            "AWS_SECRET_ACCESS_KEY": "dict-aws-secret-value",
            "PRIVATE_KEY": "dict-private-key-value",
            "DATABASE_URL": "postgresql://dict-user:dict-password@db.example/app",
        }

        serialized = json.dumps(common.redact_secrets(payload))

        for secret in (
            "aws-secret-value",
            "private-key-value",
            "db-password",
            "dXNlcjpwYXNzd29yZA==",
            "proxy-bearer-value",
            "header-api-key-value",
            "header-auth-token-value",
            "url-password",
            "query-token-value",
            "dict-aws-secret-value",
            "dict-private-key-value",
            "dict-password",
        ):
            self.assertNotIn(secret, serialized)
        self.assertIn("[REDACTED]", serialized)
        self.assertIn("safe-tokenizer-name", serialized)

    def test_directory_enumeration_stops_at_the_global_entry_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            listing = CountingScandir(root, 10_000)
            with mock.patch.object(common.os, "scandir", return_value=listing):
                files, truncated = common.bounded_files(root, 1)
            self.assertTrue(truncated)
            self.assertEqual(len(files), 1)
            self.assertEqual(listing.yielded, 1_025)

    def test_structured_inputs_are_size_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oversized.json"
            with path.open("wb") as handle:
                handle.truncate(common.MAX_STRUCTURED_BYTES + 1)
            with self.assertRaisesRegex(common.SkillError, "Structured input exceeds"):
                common.load_structured(path)

    def test_atomic_write_preserves_existing_regular_file_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            path.write_text("old\n", encoding="utf-8")
            path.chmod(0o644)
            common.atomic_write_text(path, "new\n")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
            self.assertEqual(path.read_text(encoding="utf-8"), "new\n")

    def test_post_popen_setup_failure_still_terminates_process(self) -> None:
        process = mock.Mock()
        process.stdout = io.BytesIO()
        process.stderr = io.BytesIO()
        process.pid = 999_999
        process.returncode = None
        process.wait.return_value = -9
        process.poll.return_value = None

        with (
            mock.patch.object(common.subprocess, "Popen", return_value=process),
            mock.patch.object(common, "terminate_process_tree") as terminate,
            mock.patch.object(common.threading.Thread, "start", side_effect=RuntimeError("thread failure")),
            self.assertRaisesRegex(RuntimeError, "thread failure"),
        ):
            common.run_process_capture([sys.executable, "-c", "pass"])

        terminate.assert_called_once_with(process)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)


if __name__ == "__main__":
    unittest.main()
