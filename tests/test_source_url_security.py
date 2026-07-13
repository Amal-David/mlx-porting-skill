"""SSRF and immutable-reference contracts for live source validation."""
from __future__ import annotations

import json
import shutil
import socket
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "mlx-model-porting"
SCRIPTS = SKILL / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import validate_sources  # noqa: E402


def resolution(address: str) -> list[tuple[object, ...]]:
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    return [(family, socket.SOCK_STREAM, 6, "", (address, 443))]


class SourceURLSecurityTests(unittest.TestCase):
    def test_https_structure_rejects_userinfo_and_non_https_schemes(self) -> None:
        credential_url = "https://" + "user:password" + "@example.com/source"
        self.assertIn("userinfo", validate_sources.https_url_structure_error(credential_url) or "")
        self.assertIn(
            "scheme must be https",
            validate_sources.https_url_structure_error("http://example.com/source") or "",
        )
        self.assertIn(
            "globally routable",
            validate_sources.https_url_structure_error("https://127.0.0.1/source") or "",
        )
        self.assertIn(
            "localhost",
            validate_sources.https_url_structure_error("https://localhost/source") or "",
        )
        self.assertIsNone(validate_sources.https_url_structure_error("https://example.com/source"))

    def test_public_url_gate_rejects_loopback_private_link_local_and_mixed_dns(self) -> None:
        for addresses in (
            ["127.0.0.1"],
            ["10.0.0.8"],
            ["169.254.169.254"],
            ["::1"],
            ["93.184.216.34", "127.0.0.1"],
        ):
            answers = [item for address in addresses for item in resolution(address)]
            with self.subTest(addresses=addresses), mock.patch.object(
                validate_sources.socket,
                "getaddrinfo",
                return_value=answers,
            ), self.assertRaisesRegex(urllib.error.URLError, "non-public"):
                validate_sources.require_public_https_url("https://example.com/source")

    def test_redirect_handler_revalidates_every_destination_before_following(self) -> None:
        handler = validate_sources.PublicHTTPSRedirectHandler()
        with mock.patch.object(
            validate_sources.socket,
            "getaddrinfo",
            return_value=resolution("169.254.169.254"),
        ), self.assertRaisesRegex(urllib.error.URLError, "non-public"):
            handler.redirect_request(
                urllib.request.Request("https://example.com/start"),
                None,
                302,
                "Found",
                {},
                "https://metadata.example/latest",
            )

    def test_check_url_never_opens_an_unsafe_initial_destination(self) -> None:
        opener = mock.Mock()
        with mock.patch.object(
            validate_sources.urllib.request,
            "build_opener",
            return_value=opener,
        ), mock.patch.object(
            validate_sources.socket,
            "getaddrinfo",
            return_value=resolution("127.0.0.1"),
        ):
            result = validate_sources.check_url(
                {"id": "unsafe", "url": "https://metadata.example/source"},
                1.0,
            )
        self.assertFalse(result["ok"])
        self.assertIn("non-public", result["error"])
        opener.open.assert_not_called()

    def test_request_fails_closed_when_connected_peer_was_not_vetted(self) -> None:
        raw_socket = mock.Mock()
        raw_socket.getpeername.return_value = ("127.0.0.1", 443)

        with mock.patch.object(
            validate_sources.socket,
            "getaddrinfo",
            return_value=resolution("93.184.216.34"),
        ) as resolver, mock.patch.object(
            validate_sources.socket,
            "socket",
            return_value=raw_socket,
        ):
            result = validate_sources.check_url(
                {"id": "rebind", "url": "https://original.example/source"},
                1.0,
            )

        self.assertFalse(result["ok"])
        self.assertIn("outside vetted address set", result["error"])
        resolver.assert_called_once_with(
            "original.example",
            443,
            type=socket.SOCK_STREAM,
        )
        raw_socket.connect.assert_called_once_with(("93.184.216.34", 443))

    def test_pinned_connection_keeps_tls_hostname_on_original_host(self) -> None:
        raw_socket = mock.Mock()
        raw_socket.getpeername.return_value = ("93.184.216.34", 443)
        tls_socket = mock.Mock()
        tls_context = mock.Mock()
        tls_context.wrap_socket.return_value = tls_socket
        with mock.patch.object(
            validate_sources.socket,
            "getaddrinfo",
            return_value=resolution("93.184.216.34"),
        ), mock.patch.object(
            validate_sources.socket,
            "socket",
            return_value=raw_socket,
        ):
            destination = validate_sources.require_public_https_url(
                "https://original.example/source"
            )
            connection = validate_sources.PinnedHTTPSConnection(
                "original.example",
                destination=destination,
                context=tls_context,
                timeout=1.0,
            )
            connection.connect()

        self.assertEqual(connection.server_hostname, "original.example")
        raw_socket.connect.assert_called_once_with(("93.184.216.34", 443))
        tls_context.wrap_socket.assert_called_once_with(
            raw_socket,
            server_hostname="original.example",
        )
        self.assertIs(connection.sock, tls_socket)

    def test_url_check_deadline_bounds_dns_and_connections_and_caps_address_attempts(self) -> None:
        many_answers = [
            item
            for suffix in range(1, 13)
            for item in resolution(f"93.184.216.{suffix}")
        ]

        with self.subTest("synchronous resolver is inside the overall deadline"):
            opener = mock.Mock()

            def blocking_resolver(*args: object, **kwargs: object) -> list[tuple[object, ...]]:
                time.sleep(0.2)
                return many_answers

            started = time.monotonic()
            with mock.patch.object(
                validate_sources.urllib.request,
                "build_opener",
                return_value=opener,
            ), mock.patch.object(
                validate_sources.socket,
                "getaddrinfo",
                side_effect=blocking_resolver,
            ):
                result = validate_sources.check_url(
                    {"id": "slow-dns", "url": "https://example.com/source"},
                    0.04,
                )
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 1.0)
            self.assertIn("overall network deadline exceeded", result["error"])
            opener.open.assert_not_called()

        with self.subTest("connection attempts are capped after a large DNS answer"):
            attempts: list[tuple[object, ...]] = []

            class RefusingSocket:
                def settimeout(self, timeout: float) -> None:
                    self.timeout = timeout

                def connect(self, sockaddr: tuple[object, ...]) -> None:
                    attempts.append(sockaddr)
                    raise socket.timeout("black hole")

                def close(self) -> None:
                    return None

            with mock.patch.object(
                validate_sources.socket,
                "getaddrinfo",
                return_value=many_answers,
            ), mock.patch.object(
                validate_sources.socket,
                "socket",
                side_effect=lambda *args: RefusingSocket(),
            ):
                result = validate_sources.check_url(
                    {"id": "many-addresses", "url": "https://example.com/source"},
                    1.0,
                )
            self.assertFalse(result["ok"])
            self.assertEqual(len(attempts), validate_sources.MAX_HTTPS_ADDRESS_ATTEMPTS)

        with self.subTest("one blocking connection cannot overrun the overall deadline"):
            attempts = []

            class BlockingSocket(RefusingSocket):
                def connect(self, sockaddr: tuple[object, ...]) -> None:
                    attempts.append(sockaddr)
                    time.sleep(0.2)
                    raise socket.timeout("black hole")

            started = time.monotonic()
            with mock.patch.object(
                validate_sources.socket,
                "getaddrinfo",
                return_value=many_answers,
            ), mock.patch.object(
                validate_sources.socket,
                "socket",
                side_effect=lambda *args: BlockingSocket(),
            ):
                result = validate_sources.check_url(
                    {"id": "slow-connect", "url": "https://example.com/source"},
                    0.04,
                )
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 1.0)
            self.assertIn("overall network deadline exceeded", result["error"])
            self.assertLessEqual(len(attempts), validate_sources.MAX_HTTPS_ADDRESS_ATTEMPTS)

    def test_synthesized_github_sources_require_full_matching_commit_refs(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            skill = Path(raw_tmp) / "skill"
            shutil.copytree(SKILL / "assets", skill / "assets")
            source_path = skill / "assets" / "sources.yaml"
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            source = next(item for item in payload["sources"] if item["id"] == "mlx-lm-cache")
            source["url"] = source["url"].replace(source["snapshot"], source["snapshot"][:12])
            source["snapshot"] = source["snapshot"][:12]
            source_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            _, errors = validate_sources.validate(skill, False, 1.0, 1)
            self.assertTrue(
                any("must use a full 40-hex commit URL" in error for error in errors),
                errors,
            )


if __name__ == "__main__":
    unittest.main()
