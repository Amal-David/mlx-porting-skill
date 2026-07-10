"""Offline contracts for local Markdown links, anchors, and documented CLIs."""
from __future__ import annotations

import argparse
import ast
import html
import re
import shlex
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
INLINE_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
REFERENCE_LINK_RE = re.compile(r"(?m)^\[[^\]]+\]:\s*(\S+)\s*$")
FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
SCRIPT_PREFIX = "mlx-model-porting/scripts/"
IGNORED_TOP_LEVEL_DIRS = {
    ".agents", ".claude", ".codex", ".git", ".superplan", "node_modules",
}
ARCHIVAL_RESEARCH_ROOT = ROOT / "mlx-model-porting" / "research-runs"
ARCHIVAL_EXCLUSION_REASON = (
    "research-runs are immutable evidence receipts that preserve historical researcher prose; "
    "their commands are not current user documentation"
)


def documentation_markdown() -> tuple[Path, ...]:
    paths: list[Path] = []
    for path in ROOT.rglob("*.md"):
        relative = path.relative_to(ROOT)
        if relative.parts and relative.parts[0] in IGNORED_TOP_LEVEL_DIRS:
            continue
        if path.is_relative_to(ARCHIVAL_RESEARCH_ROOT):
            continue
        paths.append(path)
    return tuple(sorted(paths))


DOCUMENTATION_MARKDOWN = documentation_markdown()


def link_destination(raw: str) -> str:
    value = raw.strip()
    if value.startswith("<") and ">" in value:
        return value[1:value.index(">")]
    return value.split(maxsplit=1)[0]


def github_heading_anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match is None:
            continue
        heading = html.unescape(match.group(1))
        heading = re.sub(r"<[^>]+>", "", heading)
        heading = re.sub(r"!?\[([^\]]+)\]\([^)]+\)", r"\1", heading)
        heading = re.sub(r"[`*_~]", "", heading).strip().lower()
        slug = "".join(character for character in heading if character.isalnum() or character in " _-")
        slug = re.sub(r"\s+", "-", slug)
        suffix = counts.get(slug, 0)
        counts[slug] = suffix + 1
        anchors.add(slug if suffix == 0 else f"{slug}-{suffix}")
    anchors.update(
        re.findall(
            r"<(?:a\s+[^>]*name|[a-zA-Z][^>]*\sid)=[\"']([^\"']+)[\"']",
            path.read_text(encoding="utf-8"),
        ),
    )
    return anchors


def html_ids(path: Path) -> set[str]:
    return set(re.findall(r"\bid=[\"']([^\"']+)[\"']", path.read_text(encoding="utf-8")))


class ContractArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def ast_value(node: ast.AST, constants: dict[str, object]) -> object:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        pass
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.Name) and node.id in {"int", "float", "str", "Path"}:
        return {"int": int, "float": float, "str": str, "Path": Path}[node.id]
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "argparse" and node.attr == "REMAINDER":
            return argparse.REMAINDER
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {"list", "tuple", "set"} and len(node.args) == 1:
            value = ast_value(node.args[0], constants)
            return {"list": list, "tuple": tuple, "set": set}[node.func.id](value)
    raise ValueError(f"unsupported static argparse value: {ast.dump(node, include_attributes=False)}")


def script_cli_parser(script: Path) -> ContractArgumentParser:
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    constants: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        try:
            constants[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue

    group_required: dict[str, bool] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        call = node.value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr != "add_mutually_exclusive_group":
            continue
        required = next((keyword.value for keyword in call.keywords if keyword.arg == "required"), None)
        group_required[node.targets[0].id] = bool(ast_value(required, constants)) if required else False

    parser = ContractArgumentParser(prog=script.name, add_help=False)
    groups = {
        name: parser.add_mutually_exclusive_group(required=required)
        for name, required in group_required.items()
    }
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
    ]
    for call in sorted(calls, key=lambda node: (node.lineno, node.col_offset)):
        names = [ast_value(argument, constants) for argument in call.args]
        if not names or not all(isinstance(name, str) for name in names):
            raise AssertionError(f"non-static argparse argument name in {script}:{call.lineno}")
        receiver = call.func.value.id if isinstance(call.func.value, ast.Name) else "parser"
        destination = groups.get(receiver, parser)
        kwargs: dict[str, object] = {}
        for keyword in call.keywords:
            if keyword.arg not in {"action", "choices", "nargs", "required", "type"}:
                continue
            kwargs[keyword.arg] = ast_value(keyword.value, constants)
        destination.add_argument(*names, **kwargs)
    return parser


def documented_script_commands(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8")
    snippets = [*FENCE_RE.findall(text), *INLINE_CODE_RE.findall(text)]
    commands: list[list[str]] = []
    for snippet in snippets:
        logical = re.sub(r"\\\s*\n\s*", " ", snippet)
        for line in logical.splitlines():
            candidate = line.strip().removeprefix("$ ")
            if not candidate.startswith(f"python3 {SCRIPT_PREFIX}"):
                continue
            commands.append(shlex.split(candidate))
    return commands


class DocumentationContractTests(unittest.TestCase):
    def test_current_shipped_markdown_local_links_and_fragments_resolve(self) -> None:
        self.assertTrue(DOCUMENTATION_MARKDOWN, "no shipped documentation found")
        relative_docs = {path.relative_to(ROOT) for path in DOCUMENTATION_MARKDOWN}
        self.assertIn(Path("mlx-model-porting/SKILL.md"), relative_docs)
        self.assertTrue(any(path.parts[:1] == ("adapters",) for path in relative_docs))
        self.assertTrue(any(path.parts[:2] == ("mlx-model-porting", "references") for path in relative_docs))
        self.assertTrue(ARCHIVAL_EXCLUSION_REASON)
        self.assertFalse(any(path.is_relative_to(ARCHIVAL_RESEARCH_ROOT) for path in DOCUMENTATION_MARKDOWN))
        anchor_cache: dict[Path, set[str]] = {}
        for document in DOCUMENTATION_MARKDOWN:
            text = document.read_text(encoding="utf-8")
            destinations = [*INLINE_LINK_RE.findall(text), *REFERENCE_LINK_RE.findall(text)]
            for raw in destinations:
                destination = link_destination(raw)
                parsed = urlsplit(destination)
                if parsed.scheme in {"http", "https", "mailto"} or parsed.netloc:
                    continue
                with self.subTest(document=document.name, destination=destination):
                    self.assertFalse(parsed.scheme, f"unsupported URI scheme in documentation: {destination}")
                    path_text = unquote(parsed.path)
                    if not path_text:
                        target = document
                    elif path_text.startswith("/"):
                        target = ROOT / path_text.lstrip("/")
                    else:
                        target = document.parent / path_text
                    target = target.resolve(strict=False)
                    self.assertTrue(
                        target.is_relative_to(ROOT.resolve()),
                        f"local documentation link escapes the repository: {destination}",
                    )
                    self.assertTrue(target.exists(), f"missing local documentation target: {destination}")
                    if not parsed.fragment:
                        continue
                    self.assertTrue(target.is_file(), f"fragment target is not a file: {destination}")
                    fragment = unquote(parsed.fragment)
                    if target.suffix.lower() == ".md":
                        anchors = anchor_cache.setdefault(target, github_heading_anchors(target))
                    elif target.suffix.lower() in {".html", ".htm"}:
                        anchors = anchor_cache.setdefault(target, html_ids(target))
                    else:
                        self.fail(f"cannot resolve a fragment on this file type: {destination}")
                    self.assertIn(fragment, anchors, f"missing local anchor: {destination}")

    def test_documented_python_commands_satisfy_real_cli_semantics(self) -> None:
        commands: list[tuple[Path, list[str]]] = []
        for document in DOCUMENTATION_MARKDOWN:
            commands.extend((document, command) for command in documented_script_commands(document))
        self.assertGreaterEqual(len(commands), 30, "too few executable script examples are documented")

        parser_cache: dict[Path, ContractArgumentParser] = {}
        for document, command in commands:
            self.assertGreaterEqual(len(command), 2, f"malformed documented command in {document}")
            script_token = command[1]
            script = ROOT / script_token
            with self.subTest(document=document.name, command=shlex.join(command)):
                self.assertTrue(script_token.startswith(SCRIPT_PREFIX))
                self.assertTrue(script.is_file(), f"documented script does not exist: {script_token}")
                parser = parser_cache.setdefault(script, script_cli_parser(script))
                try:
                    parser.parse_args(command[2:])
                except (ValueError, argparse.ArgumentError) as exc:
                    self.fail(f"documented command violates {script.name} positional/choice/required semantics: {exc}")

    def test_cli_contract_rejects_bad_actions_and_missing_required_destinations(self) -> None:
        manifest_parser = script_cli_parser(ROOT / SCRIPT_PREFIX / "manifest.py")
        with self.assertRaisesRegex(ValueError, "invalid choice"):
            manifest_parser.parse_args(["destroy"])
        benchmark_parser = script_cli_parser(ROOT / SCRIPT_PREFIX / "validate_benchmarks.py")
        with self.assertRaises(ValueError):
            benchmark_parser.parse_args([])
        installer_parser = script_cli_parser(ROOT / SCRIPT_PREFIX / "install_skill.py")
        with self.assertRaisesRegex(ValueError, "one of the arguments"):
            installer_parser.parse_args(["--dry-run"])

    def test_validation_document_lists_the_complete_deterministic_release_gate(self) -> None:
        validation = (ROOT / "VALIDATION.md").read_text(encoding="utf-8")
        offline_match = re.search(
            r"(?ms)^## Offline (?:release gates|gate commands)\s*$\n(.*?)(?=^## |\Z)",
            validation,
        )
        self.assertIsNotNone(offline_match, "VALIDATION.md needs an offline release-gate section")
        offline = offline_match.group(1)
        for command in (
            "audit_skill.py --strict",
            "validate_sources.py mlx-model-porting",
            "validate_benchmarks.py check",
            "generate_claim_catalog.py --check",
            "generate_evidence_index.py --check",
            "generate_site_data.py --check",
            "manifest.py check",
            "node --check site/app.js",
            "python3 -m unittest discover -s tests -v",
            "git diff --check",
        ):
            self.assertIn(command, offline, f"offline release gate is not documented: {command}")
        self.assertNotIn("--check-urls", offline, "live URL checks must not be in the offline gate")

        network_match = re.search(
            r"(?ms)^## (?:What requires|Requires) network access[^\n]*\n(.*?)(?=^## |\Z)",
            validation,
        )
        self.assertIsNotNone(network_match, "VALIDATION.md needs a separate network validation section")
        self.assertIn("--check-urls", network_match.group(1))


if __name__ == "__main__":
    unittest.main()
