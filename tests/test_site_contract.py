"""Static public-site contracts using only the Python standard library."""
from __future__ import annotations

import ast
import json
import re
import shlex
import stat
import unittest
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PAGES = (SITE / "index.html", SITE / "docs" / "index.html")
SITE_FILES = (*PAGES, SITE / "styles.css", SITE / "app.js", SITE / "data.js")


class ParsedPage(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.ids: set[str] = set()
        self.duplicate_ids: set[str] = set()
        self.labels: dict[str, str] = {}
        self.code_blocks: list[str] = []
        self.audited_source_hrefs: list[str] = []
        self.text_parts: list[str] = []
        self._label_for: str | None = None
        self._label_parts: list[str] = []
        self._code_depth = 0
        self._code_parts: list[str] = []
        self._inside_audited_sources = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        self.tags.append((tag, values))
        identifier = values.get("id")
        if identifier:
            if identifier in self.ids:
                self.duplicate_ids.add(identifier)
            self.ids.add(identifier)
        if tag == "label":
            self._label_for = values.get("for")
            self._label_parts = []
        if tag == "table" and "data-audited-sources" in values:
            self._inside_audited_sources = True
        if tag == "a" and self._inside_audited_sources and values.get("href"):
            self.audited_source_hrefs.append(values["href"] or "")
        if tag == "code":
            if self._code_depth == 0:
                self._code_parts = []
            self._code_depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "label" and self._label_for:
            self.labels[self._label_for] = " ".join("".join(self._label_parts).split())
            self._label_for = None
            self._label_parts = []
        if tag == "table" and self._inside_audited_sources:
            self._inside_audited_sources = False
        if tag == "code" and self._code_depth:
            self._code_depth -= 1
            if self._code_depth == 0:
                self.code_blocks.append("".join(self._code_parts))
                self._code_parts = []

    def handle_data(self, data: str) -> None:
        self.text_parts.append(data)
        if self._label_for is not None:
            self._label_parts.append(data)
        if self._code_depth:
            self._code_parts.append(data)

    @property
    def text(self) -> str:
        return " ".join("".join(self.text_parts).split())


def parse_page(path: Path) -> ParsedPage:
    parser = ParsedPage()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser


def class_names(attrs: dict[str, str | None]) -> set[str]:
    return set((attrs.get("class") or "").split())


def is_remote(value: str) -> bool:
    parsed = urlsplit(value)
    return bool(parsed.scheme or parsed.netloc)


def resolve_local_reference(page: Path, value: str) -> tuple[Path, str]:
    parsed = urlsplit(value)
    path_text = unquote(parsed.path)
    if not path_text:
        target = page
    elif path_text.startswith("/"):
        target = SITE / path_text.lstrip("/")
    else:
        target = page.parent / path_text
    target = target.resolve(strict=False)
    if target.is_dir() or path_text.endswith("/"):
        target /= "index.html"
    return target, unquote(parsed.fragment)


def argparse_options(script: Path) -> set[str]:
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    options: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if argument.value.startswith("--"):
                    options.add(argument.value)
    return options


def documented_commands(page: ParsedPage) -> list[list[str]]:
    commands: list[list[str]] = []
    for block in page.code_blocks:
        logical = re.sub(r"\\\s*\n\s*", " ", block)
        for line in logical.splitlines():
            stripped = line.strip()
            if not stripped.startswith("python3 mlx-model-porting/scripts/"):
                continue
            commands.append(shlex.split(stripped))
    return commands


class SiteContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pages = {path: parse_page(path) for path in PAGES}

    def test_all_local_href_and_src_references_resolve(self) -> None:
        parser_cache = dict(self.pages)
        for page, parsed_page in self.pages.items():
            self.assertFalse(parsed_page.duplicate_ids, f"duplicate ids in {page}: {parsed_page.duplicate_ids}")
            for tag, attrs in parsed_page.tags:
                for attribute in ("href", "src"):
                    value = attrs.get(attribute)
                    if not value or is_remote(value):
                        continue
                    if attribute == "href" and urlsplit(value).path.endswith("/"):
                        self.fail(f"directory-style local link is not file:// safe in {page}: {value}")
                    target, fragment = resolve_local_reference(page, value)
                    with self.subTest(page=page.name, tag=tag, attribute=attribute, value=value):
                        self.assertTrue(
                            target.is_relative_to(SITE.resolve()),
                            f"local reference escapes site root: {value} -> {target}",
                        )
                        self.assertTrue(target.is_file(), f"missing local asset: {value} -> {target}")
                        if fragment and target.suffix.lower() in {".html", ".htm"}:
                            target_page = parser_cache.setdefault(target, parse_page(target))
                            self.assertIn(fragment, target_page.ids, f"missing fragment #{fragment} in {target}")

    def test_executable_style_font_and_image_dependencies_are_local(self) -> None:
        dependency_values: list[tuple[Path, str, str]] = []
        for page, parsed_page in self.pages.items():
            for tag, attrs in parsed_page.tags:
                rel = set((attrs.get("rel") or "").lower().split())
                if tag == "script" and attrs.get("src"):
                    dependency_values.append((page, "script", attrs["src"] or ""))
                if tag == "link" and rel.intersection({"stylesheet", "icon", "preload", "modulepreload"}):
                    dependency_values.append((page, "link", attrs.get("href") or ""))
                if tag in {"img", "image", "source"}:
                    if attrs.get("src"):
                        dependency_values.append((page, tag, attrs["src"] or ""))
                    if attrs.get("href"):
                        dependency_values.append((page, tag, attrs["href"] or ""))
                    for candidate in (attrs.get("srcset") or "").split(","):
                        if candidate.strip():
                            dependency_values.append((page, f"{tag}-srcset", candidate.split()[0]))
        for page, kind, value in dependency_values:
            with self.subTest(page=page.name, kind=kind, value=value):
                self.assertFalse(
                    value.lower().startswith(("http://", "https://", "//")),
                    f"remote {kind} dependency in {page}: {value}",
                )

        css = (SITE / "styles.css").read_text(encoding="utf-8")
        self.assertIsNone(
            re.search(r"(?:url\(|@import\s+)[^\n)]*(?:https?:)?//", css, re.IGNORECASE),
            "remote dependency appears in site/styles.css",
        )

    def test_accessibility_landmarks_search_and_navigation_controls(self) -> None:
        for page, parsed_page in self.pages.items():
            main_ids = {
                attrs.get("id")
                for tag, attrs in parsed_page.tags
                if tag == "main" and attrs.get("id")
            }
            self.assertTrue(main_ids, f"missing labeled main landmark in {page}")
            skip_links = [
                attrs for tag, attrs in parsed_page.tags
                if tag == "a" and "skip-link" in class_names(attrs)
            ]
            self.assertTrue(skip_links, f"missing skip link in {page}")
            for skip in skip_links:
                self.assertIn((skip.get("href") or "").removeprefix("#"), main_ids)

        landing = self.pages[SITE / "index.html"]
        docs = self.pages[SITE / "docs" / "index.html"]
        for parsed_page, toggle_attribute in (
            (landing, "data-nav-toggle"),
            (docs, "data-docs-nav-toggle"),
        ):
            toggles = [attrs for tag, attrs in parsed_page.tags if tag == "button" and toggle_attribute in attrs]
            self.assertEqual(len(toggles), 1, f"expected one {toggle_attribute} button")
            toggle = toggles[0]
            self.assertIn(toggle.get("aria-expanded"), {"false", "true"})
            self.assertIn(toggle.get("aria-controls"), parsed_page.ids)

        search_inputs = [
            attrs for tag, attrs in docs.tags
            if tag == "input" and attrs.get("id") == "docs-search"
        ]
        self.assertEqual(len(search_inputs), 1, "documentation search must be a real input")
        search = search_inputs[0]
        self.assertEqual(search.get("type"), "search")
        self.assertTrue(docs.labels.get("docs-search"), "documentation search input needs a visible label")

        close_controls = [
            attrs for tag, attrs in docs.tags
            if tag == "button" and "data-docs-nav-close" in attrs
        ]
        self.assertEqual(len(close_controls), 1, "mobile docs drawer needs an accessible close control")
        self.assertTrue(close_controls[0].get("aria-label"))

        css = (SITE / "styles.css").read_text(encoding="utf-8")
        reduced_motion = re.search(
            r"@media\s*\(\s*prefers-reduced-motion\s*:\s*reduce\s*\)",
            css,
            re.IGNORECASE,
        )
        self.assertIsNotNone(reduced_motion, "missing reduced-motion media query")
        reduced_tail = css[reduced_motion.start():] if reduced_motion else ""
        self.assertIn("animation-duration", reduced_tail)
        self.assertIn("transition-duration", reduced_tail)

    def test_progressive_enhancement_and_interaction_fallbacks_are_present(self) -> None:
        css = (SITE / "styles.css").read_text(encoding="utf-8")
        app = (SITE / "app.js").read_text(encoding="utf-8")
        self.assertIn('document.documentElement.classList.add("js")', app)
        self.assertIn("html:not(.js)", css)
        self.assertIn(".js .site-header .nav-links", css)
        self.assertIn(".js .docs-sidebar", css)
        self.assertIn('toggleAttribute("inert"', app)
        self.assertIn('event.key === "Escape"', app)
        self.assertIn("section.dataset.search", app)
        self.assertIn('status.setAttribute("aria-live", "polite")', app)
        self.assertIn("Press Command+C or Control+C", app)
        self.assertIn('heading.focus({ preventScroll: true })', app)
        self.assertIn("Use your browser’s Find command", self.pages[SITE / "docs" / "index.html"].text)
        self.assertRegex(css, r"html:not\(\.js\)[^{]*\.search-wrap")
        self.assertRegex(css, r"html:not\(\.js\)[^{]*\.docs-search-fallback\s*\{")

    def test_documentation_color_tokens_meet_normal_text_contrast(self) -> None:
        css = (SITE / "styles.css").read_text(encoding="utf-8")

        def color(token: str) -> str:
            match = re.search(rf"{re.escape(token)}:\s*(#[0-9a-fA-F]{{6}})", css)
            self.assertIsNotNone(match, f"missing color token {token}")
            return match.group(1)

        def luminance(hex_color: str) -> float:
            channels = [int(hex_color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
            linear = [
                channel / 12.92 if channel <= 0.04045
                else ((channel + 0.055) / 1.055) ** 2.4
                for channel in channels
            ]
            return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

        def contrast(foreground: str, background: str) -> float:
            lighter, darker = sorted((luminance(foreground), luminance(background)), reverse=True)
            return (lighter + 0.05) / (darker + 0.05)

        self.assertGreaterEqual(contrast(color("--paper-muted"), color("--paper")), 4.5)
        self.assertGreaterEqual(contrast(color("--aluminum-600"), color("--graphite-950")), 4.5)
        self.assertIn("color-scheme: light", css)
        self.assertRegex(css, r"\.docs-page\s+:focus-visible\s*\{[^}]*#116f65")
        self.assertRegex(css, r"\.docs-page\s+\.label\s*\{[^}]*#116f65")

    def test_legacy_network_visualization_smoke_and_worker_remnants_are_absent(self) -> None:
        combined = "\n".join(path.read_text(encoding="utf-8") for path in SITE_FILES).lower()
        for remnant in (
            "vis-network",
            "vis.network",
            "cdnjs",
            "cdn.jsdelivr",
            "unpkg.com",
            "model-advisor-worker",
            "offline smoke",
            "offline-smoke",
            "smoke-local",
        ):
            with self.subTest(remnant=remnant):
                self.assertNotIn(remnant, combined)
        self.assertIsNone(re.search(r"\bworkers?\b", combined), "worker-era copy or code remains in site")

    def test_generated_data_loads_before_site_application(self) -> None:
        for page, parsed_page in self.pages.items():
            scripts = [
                PurePosixPath(urlsplit(attrs.get("src") or "").path).name
                for tag, attrs in parsed_page.tags
                if tag == "script" and attrs.get("src")
            ]
            with self.subTest(page=page):
                self.assertIn("data.js", scripts)
                self.assertIn("app.js", scripts)
                self.assertLess(scripts.index("data.js"), scripts.index("app.js"))

    def test_audited_source_links_are_immutable(self) -> None:
        docs = self.pages[SITE / "docs" / "index.html"]
        source_registry = json.loads(
            (ROOT / "mlx-model-porting" / "assets" / "sources.yaml").read_text(encoding="utf-8")
        )
        registered_urls = {source["url"] for source in source_registry["sources"]}
        self.assertGreaterEqual(len(docs.audited_source_hrefs), 5)
        immutable_path = re.compile(
            r"/(?:blob|tree)/[0-9a-f]{40}(?:/|$)",
            re.IGNORECASE,
        )
        for href in docs.audited_source_hrefs:
            parsed = urlsplit(href)
            with self.subTest(href=href):
                self.assertEqual(parsed.scheme, "https")
                self.assertEqual(parsed.netloc, "github.com")
                self.assertRegex(parsed.path, immutable_path)
                self.assertIn(href, registered_urls)

    def test_workflow_evidence_and_limitations_sections_are_present(self) -> None:
        landing = self.pages[SITE / "index.html"]
        docs = self.pages[SITE / "docs" / "index.html"]
        landing_sections = {
            attrs.get("id") for tag, attrs in landing.tags if tag == "section" and attrs.get("id")
        }
        docs_sections = {
            attrs.get("id") for tag, attrs in docs.tags if tag == "section" and attrs.get("id")
        }
        self.assertTrue({"workflow", "evidence", "limits"}.issubset(landing_sections))
        self.assertTrue({"workflow", "evidence", "limitations"}.issubset(docs_sections))

        docs_text = docs.text.lower()
        for workflow_term in (
            "inspect",
            "route",
            "port",
            "validat",
            "optimiz",
            "benchmark",
            "publish",
            "review depth",
            "support scope",
            "promotion-ready",
            "limitations",
        ):
            with self.subTest(term=workflow_term):
                self.assertIn(workflow_term, docs_text)

    def test_controlled_evidence_and_advisor_states_are_complete(self) -> None:
        docs_text = self.pages[SITE / "docs" / "index.html"].text
        taxonomy = json.loads(
            (ROOT / "mlx-model-porting" / "assets" / "recommendation-taxonomy.yaml").read_text(
                encoding="utf-8",
            ),
        )
        sources = json.loads(
            (ROOT / "mlx-model-porting" / "assets" / "sources.yaml").read_text(encoding="utf-8"),
        )
        for bucket in taxonomy["advisor_buckets"]:
            with self.subTest(advisor_bucket=bucket["id"]):
                self.assertIn(bucket["label"], docs_text)
        for support_scope in sources["support_scope_definitions"]:
            with self.subTest(support_scope=support_scope):
                self.assertIn(support_scope, docs_text)

    def test_documented_scripts_exist_and_documented_flags_match_argparse(self) -> None:
        html = "\n".join(page.read_text(encoding="utf-8") for page in PAGES)
        script_references = sorted(set(re.findall(
            r"mlx-model-porting/scripts/[A-Za-z0-9_.-]+\.py",
            html,
        )))
        self.assertGreaterEqual(len(script_references), 12)
        for reference in script_references:
            path = ROOT / reference
            with self.subTest(reference=reference):
                self.assertTrue(path.is_file(), f"documented script does not exist: {reference}")
                self.assertTrue(stat.S_ISREG(path.lstat().st_mode))

        commands = [
            command
            for parsed_page in self.pages.values()
            for command in documented_commands(parsed_page)
        ]
        self.assertGreaterEqual(len(commands), 15)
        for command in commands:
            reference = command[1]
            script = ROOT / reference
            separator = command.index("--") if "--" in command else len(command)
            documented_flags = {
                token.split("=", 1)[0]
                for token in command[2:separator]
                if token.startswith("--")
            }
            available_flags = argparse_options(script)
            with self.subTest(script=reference, flags=sorted(documented_flags)):
                self.assertTrue(
                    documented_flags.issubset(available_flags),
                    f"unknown documented flags for {reference}: "
                    f"{sorted(documented_flags - available_flags)}",
                )


if __name__ == "__main__":
    unittest.main()
