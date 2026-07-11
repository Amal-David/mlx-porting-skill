from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_ROOTS = {
    ".git", ".superplan", ".agents", ".amazonq", ".claude", ".codex",
    ".cursor", ".gemini", ".opencode", ".pagecast", ".pytest_cache",
    ".ruff_cache", ".wrangler", "__pycache__", "dist-pages", "node_modules",
}


def shipped_utf8_text() -> tuple[tuple[Path, str], ...]:
    records: list[tuple[Path, str]] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if set(relative.parts).intersection(IGNORED_ROOTS):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "\x00" in text:
            continue
        records.append((relative, text))
    return tuple(records)


class DistributionPortabilityTests(unittest.TestCase):
    def test_byte_exact_attestations_are_opaque_to_git_diff_and_merge(self) -> None:
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
        self.assertIn(
            "mlx-model-porting/assets/benchmarks/attestations/** -diff -merge",
            attributes,
        )

    def test_contributor_instructions_are_checkout_and_home_agnostic(self) -> None:
        for name in ("AGENTS.md", "CLAUDE.md"):
            with self.subTest(file=name):
                text = (ROOT / name).read_text(encoding="utf-8")
                self.assertNotIn("/" + "Users/", text)
                self.assertNotIn(str(ROOT), text)
                self.assertIn(
                    "If Superplan is unavailable, continue with the repository-native workflow",
                    text,
                )

    def test_shipped_text_has_no_private_home_or_ephemeral_benchmark_paths(self) -> None:
        patterns = {
            "macOS home": re.compile("/" + r"Users/[A-Za-z0-9._-]+/"),
            "Linux home": re.compile("/" + r"home/[A-Za-z0-9._-]+/"),
            "root home": re.compile("/" + r"root/[.A-Za-z0-9_-]"),
            "macOS temporary home": re.compile(
                "/private/var/" + r"folders/[A-Za-z0-9]{2}/[A-Za-z0-9_-]{8,}/",
            ),
            "macOS temporary alias": re.compile(
                "/var/" + r"folders/[A-Za-z0-9]{2}/[A-Za-z0-9_-]{8,}/",
            ),
            "Windows home": re.compile(
                r"[A-Za-z]:(?:\\|/)" + r"Users(?:\\|/)[A-Za-z0-9._-]+(?:\\|/)",
            ),
            "legacy benchmark temp": re.compile("/tmp/" + "t206"),
            "nightly scratch": re.compile("mlx-porting-skill-" + "nightly-"),
        }
        offenders: list[str] = []
        for relative, text in shipped_utf8_text():
            for label, pattern in patterns.items():
                if pattern.search(text):
                    offenders.append(f"{relative}: {label}")
        self.assertEqual(offenders, [], "non-portable shipped paths:\n" + "\n".join(offenders))

    def test_shipped_text_has_no_embedded_secret_material(self) -> None:
        patterns = {
            "GitHub classic token": re.compile("ghp_" + r"[A-Za-z0-9]{30,}"),
            "GitHub fine-grained token": re.compile("github_pat_" + r"[A-Za-z0-9_]{30,}"),
            "OpenAI-style key": re.compile("sk-" + r"[A-Za-z0-9]{20,}"),
            "AWS access key": re.compile("AKIA" + r"[0-9A-Z]{16}"),
            "Hugging Face token": re.compile("hf_" + r"[A-Za-z0-9]{30,}"),
            "Slack token": re.compile("xox" + r"[abprs]-[A-Za-z0-9-]{20,}"),
            "Stripe live key": re.compile("sk_" + r"live_[A-Za-z0-9]{20,}"),
            "Google API key": re.compile("AI" + r"za[0-9A-Za-z_-]{32,}"),
            "private key": re.compile(
                "-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            ),
        }
        offenders: list[str] = []
        scanned_suffixes: set[str] = set()
        for relative, text in shipped_utf8_text():
            scanned_suffixes.add(relative.suffix.lower())
            for label, pattern in patterns.items():
                if pattern.search(text):
                    offenders.append(f"{relative}: {label}")
        self.assertTrue({".html", ".js", ".css", ".svg"}.issubset(scanned_suffixes))
        self.assertEqual(offenders, [], "embedded secret material:\n" + "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
