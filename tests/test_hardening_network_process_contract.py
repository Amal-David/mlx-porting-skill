from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mlx-model-porting" / "scripts"
WORKFLOW = ROOT / ".github" / "workflows" / "daily-research.yml"
UPDATE_FIXTURE = ROOT / "tests" / "fixtures" / "updates" / "offline.json"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import _common as common  # noqa: E402
import collect_contributors  # noqa: E402
import update_sources  # noqa: E402
import validate_sources  # noqa: E402


class BlockingStream:
    def __init__(self) -> None:
        self._closed = threading.Event()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def read(self, _size: int) -> bytes:
        self._closed.wait(timeout=5)
        return b""

    def close(self) -> None:
        self._closed.set()


class InterruptingProcess:
    def __init__(self) -> None:
        self.stdout = BlockingStream()
        self.stderr = BlockingStream()
        self.pid = 999_999
        self.returncode: int | None = None
        self.wait_calls = 0
        self.terminated = False

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise KeyboardInterrupt
        self.returncode = -9
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.terminated = True
        self.returncode = -9


class ReversedScandir:
    def __init__(self, entries: list[object]):
        self.entries = entries

    def __enter__(self) -> "ReversedScandir":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self):
        return iter(self.entries)


class NetworkProcessHardeningContractTests(unittest.TestCase):
    def test_base_exception_terminates_process_and_closes_capture_readers(self) -> None:
        process = InterruptingProcess()

        def terminate(fake: InterruptingProcess) -> None:
            fake.terminated = True
            fake.returncode = -9

        try:
            with (
                mock.patch.object(common.subprocess, "Popen", return_value=process),
                mock.patch.object(common, "terminate_process_tree", side_effect=terminate),
                self.assertRaises(KeyboardInterrupt),
            ):
                common.run_process_capture(["fixture"])
            self.assertTrue(process.terminated)
            self.assertGreaterEqual(process.wait_calls, 2)
            self.assertTrue(process.stdout.closed)
            self.assertTrue(process.stderr.closed)
        finally:
            process.stdout.close()
            process.stderr.close()

    @unittest.skipIf(os.name == "nt", "POSIX process-group behavior")
    def test_leader_exit_cannot_bypass_timeout_or_leak_descendants(self) -> None:
        inherited_pipe_parent = (
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3)'])"
        )
        started = time.monotonic()
        _, timed_out = common.run_process_capture(
            [sys.executable, "-c", inherited_pipe_parent],
            timeout=0.2,
        )
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertFalse(timed_out)

        detached_stdio_parent = (
            "import subprocess,sys; "
            "p=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
            "print(p.pid, flush=True)"
        )
        completed, _ = common.run_process_capture(
            [sys.executable, "-c", detached_stdio_parent],
            timeout=0.2,
        )
        child_pid = int(completed.stdout.strip())
        deadline = time.monotonic() + 1.0
        while True:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            if time.monotonic() >= deadline:
                self.fail("descendant survived after its process-group leader exited")
            time.sleep(0.02)

    def test_bounded_files_sorts_before_truncating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "z.txt").write_text("z", encoding="utf-8")
            real_scandir = common.os.scandir

            def reversed_scandir(path: object) -> ReversedScandir:
                entries = list(real_scandir(path))
                entries.sort(key=lambda entry: entry.name, reverse=True)
                return ReversedScandir(entries)

            with mock.patch.object(common.os, "scandir", side_effect=reversed_scandir):
                files, truncated = common.bounded_files(root, 1)
            self.assertTrue(truncated)
            self.assertEqual([path.name for path in files], ["a.txt"])

    def test_bounded_files_reports_symlink_loops_as_skill_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "loop").symlink_to("loop")
            with self.assertRaisesRegex(common.SkillError, "symlink escapes"):
                common.bounded_files(root, 10)

    def test_token_rejects_plaintext_origin(self) -> None:
        with self.assertRaisesRegex(common.SkillError, "HTTPS"):
            collect_contributors.GitHubClient(
                timeout=1,
                token="secret-token",
                token_origin="http://api.github.test",
            )

    def test_custom_https_origin_never_receives_default_github_token(self) -> None:
        client = collect_contributors.GitHubClient(
            timeout=1,
            token="top-secret-token",
            token_origin="https://attacker.example",
        )

        headers = client.request_headers(
            "https://attacker.example/repos/owner/repo/contributors"
        )

        self.assertNotIn("Authorization", headers)

    def test_live_contributor_base_must_be_public_https(self) -> None:
        args = mock.Mock(
            repo="owner/repo",
            requested_count=1,
            per_page=1,
            api_base="https://127.0.0.1",
            offline_fixture=None,
        )

        with self.assertRaisesRegex(common.SkillError, "public HTTPS"):
            collect_contributors.validate_args(args)

    def test_contributor_redirect_is_rejected_before_private_destination_is_opened(self) -> None:
        handler = collect_contributors.SameOriginTokenRedirectHandler(
            collect_contributors.url_origin("https://api.example.test")
        )
        private_resolution = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443))
        ]
        with (
            mock.patch.object(
                validate_sources.socket,
                "getaddrinfo",
                return_value=private_resolution,
            ),
            self.assertRaisesRegex(collect_contributors.urllib.error.URLError, "non-public"),
        ):
            handler.redirect_request(
                collect_contributors.urllib.request.Request(
                    "https://api.example.test/repos/owner/repo/contributors"
                ),
                None,
                302,
                "Found",
                {},
                "https://metadata.example/latest",
            )

    def test_contributor_http_error_redacts_reflected_token_and_headers(self) -> None:
        token = "top-secret-token"
        url = "https://api.github.com/repos/owner/repo/contributors"
        client = collect_contributors.GitHubClient(
            timeout=1,
            token=token,
            token_origin="https://api.github.com",
        )
        error = collect_contributors.urllib.error.HTTPError(
            url,
            401,
            "Unauthorized",
            {},
            io.BytesIO(
                f"Authorization: Bearer {token}\nreflected={token}".encode("utf-8")
            ),
        )
        self.addCleanup(error.close)
        client.opener.open = mock.Mock(side_effect=error)

        with mock.patch.object(
            validate_sources.socket,
            "getaddrinfo",
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
            ],
        ), self.assertRaises(common.SkillError) as raised:
            client.fetch_json(url)

        message = str(raised.exception)
        self.assertNotIn(token, message)
        self.assertIn("[REDACTED]", message)

    def test_invalid_api_origin_fails_without_urlparse_traceback(self) -> None:
        for url in (
            "https://api.github.test:not-a-port",
            "https://user:password@api.github.test",
            "file:///tmp/contributors.json",
        ):
            with self.subTest(url=url), self.assertRaisesRegex(
                common.SkillError,
                "Invalid contributor API URL",
            ):
                collect_contributors.url_origin(url)

    def test_update_token_is_github_https_only_and_redirect_strips_auth(self) -> None:
        with self.assertRaisesRegex(common.SkillError, "outside https://api.github.com"):
            update_sources.request_json(
                "http://api.github.com/repos/ml-explore/mlx",
                timeout=1,
                token="secret-token",
            )
        request = update_sources.urllib.request.Request(
            "https://api.github.com/repos/ml-explore/mlx",
            headers={"Authorization": "Bearer secret-token"},
        )
        handler = collect_contributors.SameOriginTokenRedirectHandler(
            collect_contributors.url_origin("https://api.github.com")
        )
        public_resolution = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
        with mock.patch.object(
            validate_sources.socket,
            "getaddrinfo",
            return_value=public_resolution,
        ):
            redirected = handler.redirect_request(
                request,
                None,
                302,
                "Found",
                {},
                "https://example.invalid/collect",
            )
        self.assertIsNotNone(redirected)
        self.assertNotIn("Authorization", dict(redirected.header_items()))

    def test_github_fallback_is_recorded_as_incomplete_collection(self) -> None:
        errors: list[str] = []
        watchlist = {"repositories": [{"repo": "ml-explore/mlx", "topics": ["core"]}]}
        fallback = {
            "kind": "repository",
            "repo": "ml-explore/mlx",
            "metadata_fallback": "git ls-remote HEAD",
        }
        with (
            mock.patch.object(update_sources, "request_json", side_effect=TimeoutError("offline")),
            mock.patch.object(update_sources, "git_head_candidate", return_value=fallback),
        ):
            records = update_sources.github_candidates(watchlist, timeout=1, errors=errors)
        self.assertEqual(records, [fallback])
        self.assertEqual(len(errors), 1)
        self.assertIn("metadata incomplete", errors[0])

    def test_network_reads_and_contributor_pagination_are_bounded(self) -> None:
        response = mock.Mock()
        response.read.return_value = b"x" * (update_sources.MAX_NETWORK_RESPONSE_BYTES + 1)
        response.geturl.return_value = "https://api.github.com/oversized"
        with self.assertRaisesRegex(common.SkillError, "response exceeds"):
            update_sources.read_bounded_response(response)

        class EndlessPages:
            def __init__(self) -> None:
                self.page = 0

            def fetch_json(self, url: str):
                self.page += 1
                next_url = f"https://api.github.test/repos/owner/repo/contributors?page={self.page + 1}"
                return (
                    [{"login": f"user-{self.page}"}],
                    {"Link": f'<{next_url}>; rel="next"'},
                    200,
                    url,
                )

        start = "https://api.github.test/repos/owner/repo/contributors?page=1"
        with self.assertRaisesRegex(common.SkillError, "pagination exceeds"):
            collect_contributors.collect_pages(
                EndlessPages(),  # type: ignore[arg-type]
                start,
                collect_contributors.MAX_CONTRIBUTOR_PAGES + 1,
                keep_logins=True,
            )

    def test_link_pagination_cannot_change_origin(self) -> None:
        api_base = "https://api.github.test"
        start = collect_contributors.contributors_url(api_base, "owner/repo", 1)
        hostile = "https://example.invalid/repos/owner/repo/contributors?page=2&per_page=1"
        fixture = {
            "linked_pages": [
                {
                    "url": start,
                    "headers": {"Link": f'<{hostile}>; rel="next"'},
                    "body": [{"login": "safe"}],
                },
                {"url": hostile, "body": [{"login": "exfiltrated"}]},
            ],
            "anonymous_pages": [],
        }
        client = collect_contributors.FixtureClient(fixture, api_base, "owner/repo", 1)
        with self.assertRaisesRegex(common.SkillError, "same-origin"):
            collect_contributors.collect_pages(client, start, 2, keep_logins=True)

    def test_base_snapshot_wins_when_candidates_return_to_default_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.json"
            previous = root / "automation.json"
            output = root / "output.json"
            command = [
                sys.executable,
                str(SCRIPTS / "update_sources.py"),
                "--offline-fixture",
                str(UPDATE_FIXTURE),
                "--output",
                str(base),
            ]
            first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            stale = json.loads(base.read_text(encoding="utf-8"))
            stale["generated_at"] = "2026-07-09T00:00:00+00:00"
            stale["papers"] = []
            previous.write_text(json.dumps(stale, indent=2) + "\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    *command[:-1],
                    str(output),
                    "--previous",
                    str(previous),
                    "--base",
                    str(base),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(output.read_bytes(), base.read_bytes())

    def test_workflow_uses_automation_baseline_and_closes_stale_state(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn('previous_candidate_path="$RUNNER_TEMP/automation-update-candidates.json"', workflow)
        self.assertIn('ref=automation%2Fdaily-mlx-research', workflow)
        self.assertIn('--previous "$previous_candidate_path"', workflow)
        self.assertIn('GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}', workflow)
        self.assertIn('--base mlx-model-porting/assets/update-candidates.json', workflow)
        self.assertIn('cmp -s "$candidate_path" mlx-model-porting/assets/update-candidates.json', workflow)
        self.assertIn('automation_snapshot_path="$RUNNER_TEMP/current-automation-update-candidates.json"', workflow)
        self.assertIn('cmp -s "$candidate_path" "$automation_snapshot_path"', workflow)
        self.assertIn("Automation candidate snapshot is unchanged; branch was not rewritten", workflow)
        self.assertIn('gh pr close "$pr_number" --delete-branch', workflow)
        self.assertIn('git push origin --delete "$branch"', workflow)
        self.assertIn('git checkout -B "$branch" "origin/${default_branch}"', workflow)


if __name__ == "__main__":
    unittest.main()
