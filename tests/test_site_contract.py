"""Static public-site contracts using only the Python standard library."""
from __future__ import annotations

import ast
import json
import re
import shlex
import stat
import unittest
from html import unescape
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PAGES = (SITE / "index.html", SITE / "docs" / "index.html")
SITE_FILES = (
    *PAGES,
    SITE / "styles.css",
    SITE / "app.js",
    SITE / "atlas.js",
    SITE / "optimization.js",
    SITE / "data.js",
)


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
        landing_main = [
            attrs for tag, attrs in landing.tags
            if tag == "main" and attrs.get("id") == "main"
        ]
        self.assertEqual(len(landing_main), 1)
        self.assertEqual(landing_main[0].get("tabindex"), "-1")
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
        docs_html = (SITE / "docs" / "index.html").read_text(encoding="utf-8")
        bootstrap = '<script>document.documentElement.classList.add("js")</script>'
        self.assertIn(bootstrap, docs_html)
        self.assertLess(
            docs_html.index(bootstrap),
            docs_html.index('<link rel="stylesheet" href="../styles.css">'),
            "mobile drawer layout must be selected before the stylesheet paints",
        )
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

    def test_homepage_leads_with_porting_and_preserves_optional_learnmlx_module(self) -> None:
        path = SITE / "index.html"
        html = path.read_text(encoding="utf-8")
        landing = self.pages[path]
        section_ids = [
            attrs.get("id")
            for tag, attrs in landing.tags
            if tag == "section" and attrs.get("id")
        ]
        self.assertIn("learn", section_ids)
        self.assertIn("atlas-preview", section_ids)
        self.assertLess(section_ids.index("atlas-preview"), section_ids.index("workflow"))
        self.assertLess(section_ids.index("workflow"), section_ids.index("learn"))
        self.assertLess(section_ids.index("routes"), section_ids.index("learn"))
        self.assertLess(section_ids.index("learn"), section_ids.index("evidence"))

        self.assertIn("Port the model. Prove the result.", landing.text)
        self.assertIn("Start a model port", landing.text)
        self.assertIn("LearnMLX · optional depth module", landing.text)
        self.assertIn("Use it as a reference, not a prerequisite", landing.text)
        self.assertIn("See the whole port before you touch code.", landing.text)
        self.assertIn("Qwen2.5-0.5B-Instruct", landing.text)
        self.assertIn("Whisper-style ASR", landing.text)
        self.assertIn("FLUX-style diffusion/flow", landing.text)
        self.assertIn("Proven guided lab", landing.text)
        self.assertGreaterEqual(landing.text.count("Runbook simulation"), 2)
        self.assertIn('./docs/index.html#quick-start', html)
        self.assertIn('./docs/index.html#mlx-fundamentals', html)
        self.assertGreaterEqual(html.count('./docs/index.html#porting-atlas'), 4)
        self.assertIn('./docs/index.html#qwen-worked-port', html)

        radios = [
            attrs
            for tag, attrs in landing.tags
            if tag == "input" and attrs.get("type") == "radio" and attrs.get("name") == "preview-model"
        ]
        self.assertEqual(len(radios), 3)
        for radio in radios:
            self.assertTrue(radio.get("id"))
            self.assertTrue(landing.labels.get(radio["id"] or ""))
        self.assertEqual(html.count("data-preview-path="), 3)
        self.assertEqual(html.count('class="atlas-preview-steps"'), 3)
        for preview in re.findall(
            r'<ol class="atlas-preview-steps"[^>]*>(.*?)</ol>',
            html,
            re.DOTALL,
        ):
            self.assertLess(preview.index("Prove parity"), preview.index("Optimize"))
            self.assertIn("Gated by parity", preview)

        official_links = {
            attrs.get("href")
            for tag, attrs in landing.tags
            if tag == "a" and (attrs.get("href") or "").startswith("https://ml-explore.github.io/mlx/")
        }
        self.assertTrue({
            "https://ml-explore.github.io/mlx/build/html/usage/quick_start.html",
            "https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html",
            "https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html",
            "https://ml-explore.github.io/mlx/build/html/usage/compile.html",
        }.issubset(official_links))

        brand_links = [
            attrs
            for tag, attrs in landing.tags
            if tag == "a" and "brand" in class_names(attrs)
        ]
        self.assertEqual(len(brand_links), 1)
        self.assertIsNone(brand_links[0].get("aria-label"))

        proof_loops = [
            attrs for tag, attrs in landing.tags
            if tag == "figure" and "data-port-loop" in attrs
        ]
        self.assertEqual(len(proof_loops), 1)
        self.assertIn("Porting is a proof loop", landing.text)
        for expression in (
            "f_torch(x; W_src)",
            "W_MLX = P(W_src)",
            "Δ = ||y_src − y_MLX||",
            "arg max phase_time",
            "unlock only when Δ ≤ ε",
        ):
            with self.subTest(hero_expression=expression):
                self.assertIn(expression, landing.text)
        loop_steps = [
            attrs.get("data-loop-step")
            for _tag, attrs in landing.tags
            if attrs.get("data-loop-step")
        ]
        self.assertEqual(loop_steps, ["classify", "translate", "parity", "profile"])
        pause_buttons = [
            attrs for tag, attrs in landing.tags
            if tag == "button" and "data-port-loop-toggle" in attrs
        ]
        self.assertEqual(len(pause_buttons), 1)
        self.assertEqual(pause_buttons[0].get("aria-pressed"), "false")
        loop_svgs = [
            attrs for tag, attrs in landing.tags
            if tag == "svg" and "port-loop-wires" in class_names(attrs)
        ]
        self.assertEqual(len(loop_svgs), 1)
        self.assertEqual(loop_svgs[0].get("aria-hidden"), "true")
        self.assertNotIn("<canvas", html.lower())
        self.assertNotIn("<video", html.lower())

        css = (SITE / "styles.css").read_text(encoding="utf-8")
        self.assertIn(".atlas-preview-board", css)
        self.assertIn(".atlas-preview-steps", css)
        self.assertRegex(css, r"\.contribute-layout\s*>\s*\*\s*\{[^}]*min-width:\s*0")
        self.assertRegex(css, r"@media\s*\(max-width:\s*640px\)[\s\S]*\.atlas-preview-steps")
        self.assertIn("@keyframes port-loop-current", css)
        self.assertIn("@keyframes port-loop-focus", css)
        self.assertRegex(
            css,
            r"\.port-loop-current\s*\{[^}]*animation:\s*port-loop-current\s+8s\s+linear\s+infinite",
        )
        self.assertRegex(
            css,
            r"@media\s*\(prefers-reduced-motion:\s*reduce\)[\s\S]*"
            r"\.port-loop-current[^}]*animation:\s*none",
        )
        self.assertRegex(
            css,
            r"@media\s*\(max-width:\s*640px\)[\s\S]*"
            r"\.port-loop-canvas\s*\{[^}]*grid-template-columns:\s*repeat\(2,",
        )
        self.assertRegex(
            css,
            r"@media\s*\(max-width:\s*640px\)[\s\S]*"
            r"\.js\s+\[data-port-loop-toggle\]\s*\{[^}]*display:\s*none",
        )
        self.assertRegex(
            css,
            r"@media\s*\(max-width:\s*640px\)\s+and\s+"
            r"\(prefers-reduced-motion:\s*reduce\)[\s\S]*"
            r"\.port-loop-gate\s*\{[^}]*transform:\s*none",
        )
        app = (SITE / "app.js").read_text(encoding="utf-8")
        self.assertIn('querySelector("[data-port-loop-toggle]")', app)
        self.assertIn('classList.toggle("is-paused")', app)

    def test_documentation_is_learning_first_before_the_executable_field_manual(self) -> None:
        path = SITE / "docs" / "index.html"
        html = path.read_text(encoding="utf-8")
        docs = self.pages[path]
        section_ids = [
            attrs.get("id")
            for tag, attrs in docs.tags
            if tag == "section" and attrs.get("id")
        ]
        learning_order = [
            "mlx-fundamentals",
            "what-is-porting",
            "read-the-model",
            "pytorch-to-mlx",
            "correctness-rail",
            "porting-atlas",
            "qwen-worked-port",
            "profile-bottleneck",
            "optimization-atlas",
            "benchmark-honestly",
            "publish-proof",
            "glossary",
        ]
        self.assertEqual(section_ids[:len(learning_order)], learning_order)
        self.assertLess(section_ids.index("glossary"), section_ids.index("quick-start"))
        self.assertGreater(section_ids.index("advanced-workshop"), section_ids.index("limitations"))
        for legacy in (
            "quick-start",
            "install",
            "workflow",
            "routes",
            "inspect",
            "oracle",
            "weights",
            "parity",
            "optimize",
            "benchmarks",
            "evidence",
            "validate",
            "sources",
            "contribute",
            "limitations",
        ):
            self.assertIn(legacy, section_ids)

        self.assertIn("Learn MLX by reading a port from the inside out.", docs.text)
        self.assertIn("A port is not a file conversion", docs.text)
        self.assertLess(html.index('id="mlx-fundamentals"'), html.index('id="quick-start"'))
        self.assertLess(html.index('id="what-is-porting"'), html.index("python3 mlx-model-porting/scripts/"))
        teaching_labels = [
            "Definition",
            "Why it matters",
            "PyTorch / CUDA",
            "MLX translation",
            "Example",
            "Common failure",
            "Proof check",
            "Next step",
        ]
        learning_frames = re.findall(
            r'<div class="learning-frame"[^>]*>(.*?)</div>',
            html,
            re.DOTALL,
        )
        self.assertGreaterEqual(len(learning_frames), 10)
        for index, frame in enumerate(learning_frames, start=1):
            with self.subTest(learning_frame=index):
                self.assertEqual(re.findall(r"<strong>([^<]+)</strong>", frame), teaching_labels)

        mlx_section_match = re.search(
            r'<section class="[^"]*\bdoc-section\b[^"]*" id="mlx-fundamentals".*?</section>',
            html,
            re.DOTALL,
        )
        self.assertIsNotNone(mlx_section_match)
        mlx_section = mlx_section_match.group(0) if mlx_section_match else ""
        for term in ("streams", "mlx.nn.Module", "function transforms"):
            with self.subTest(mlx_fundamental=term):
                self.assertIn(term, mlx_section)

        translation_text = re.search(
            r'<section class="[^"]*\bdoc-section\b[^"]*" id="pytorch-to-mlx".*?</section>',
            html,
            re.DOTALL,
        )
        self.assertIsNotNone(translation_text)
        translation = translation_text.group(0) if translation_text else ""
        for term in (
            "torch.Tensor",
            '.to(&quot;cuda&quot;)',
            "mx.array",
            "unified memory",
            "lazy",
            "mx.eval",
            "state_dict",
            "weight map",
            "NCHW",
            "autocast",
            "explicit state",
            "torch.compile",
            "mx.compile",
            "CUDA extensions",
            "custom Metal",
            "synchronization",
            "DLPack",
        ):
            with self.subTest(translation_term=term):
                self.assertIn(term, translation)

        learning = json.loads(
            (ROOT / "mlx-model-porting" / "assets" / "learning_paths.json").read_text(
                encoding="utf-8",
            ),
        )
        sources = json.loads(
            (ROOT / "mlx-model-porting" / "assets" / "sources.yaml").read_text(encoding="utf-8"),
        )
        sources_by_id = {source["id"]: source for source in sources["sources"]}
        expected_learning_links = {
            sources_by_id[source_id]["url"]
            for source_id in learning["official_learning_source_ids"]
        }
        rendered_links = {
            attrs.get("href")
            for tag, attrs in docs.tags
            if tag == "a" and attrs.get("href")
        }
        self.assertTrue(
            expected_learning_links.issubset(rendered_links),
            f"missing canonical learning links: {sorted(expected_learning_links - rendered_links)}",
        )

        glossary_match = re.search(
            r'<section class="[^"]*\bdoc-section\b[^"]*" id="glossary".*?</section>',
            html,
            re.DOTALL,
        )
        self.assertIsNotNone(glossary_match)
        glossary = (glossary_match.group(0) if glossary_match else "").casefold()
        for term in (
            "mlx",
            "eager graph",
            "source oracle",
            "parity",
            "route",
            "runbook",
            "checkpoint",
            "weight map",
            "kv cache",
            "schema-2",
            "golden scenario",
            "receipt",
            "promotion-ready",
            "execution attestation",
            "targetprofile",
            "source-key coverage",
            "proof boundary",
        ):
            with self.subTest(glossary_term=term):
                self.assertIn(term, glossary)

        css = (SITE / "styles.css").read_text(encoding="utf-8")
        self.assertRegex(
            css,
            r"\.learning-frame\s*\{[^}]*grid-template-columns:\s*repeat\(4,",
            "the eight-field teaching frame should render as a balanced four-column grid",
        )
        self.assertRegex(
            css,
            r"@media\s*\(max-width:\s*640px\)[\s\S]*"
            r"\.learning-frame\s+article:nth-last-child\(2\)\s*\{[^}]*"
            r"border-bottom:\s*1px\s+solid",
            "the mobile teaching frame must retain a divider before its final card",
        )
        self.assertRegex(
            css,
            r"\.doc-section\s+\.inline-code\s*\{[^}]*overflow-wrap:\s*anywhere",
            "long technical identifiers must wrap instead of widening mobile docs",
        )

    def test_documentation_porting_atlas_progressively_enhances_four_journeys(self) -> None:
        path = SITE / "docs" / "index.html"
        html = path.read_text(encoding="utf-8")
        docs = self.pages[path]

        self.assertIn('id="porting-atlas"', html)
        self.assertLess(html.index('id="correctness-rail"'), html.index('id="porting-atlas"'))
        self.assertLess(html.index('id="porting-atlas"'), html.index('id="qwen-worked-port"'))

        fallbacks = [
            attrs for tag, attrs in docs.tags
            if tag == "div" and "data-atlas-fallback" in attrs
        ]
        self.assertEqual(len(fallbacks), 1)
        self.assertNotIn("hidden", fallbacks[0], "the complete linear atlas must work without JavaScript")

        roots = [
            attrs for tag, attrs in docs.tags
            if tag == "div" and "data-atlas-root" in attrs
        ]
        self.assertEqual(len(roots), 1)
        self.assertIn("hidden", roots[0], "enhancement must stay hidden until validated data mounts")

        journey_ids = [
            "qwen25-dense-decoder",
            "whisper-style-asr",
            "flux-style-diffusion",
            "llava-style-vlm",
        ]
        radios = [
            attrs for tag, attrs in docs.tags
            if tag == "input" and attrs.get("type") == "radio" and attrs.get("name") == "atlas-journey"
        ]
        self.assertEqual([radio.get("value") for radio in radios], journey_ids)
        for radio in radios:
            self.assertTrue(radio.get("id"))
            self.assertTrue(docs.labels.get(radio.get("id") or ""))

        checkpoint_ids = ["inspect", "oracle", "implement", "map", "parity", "profile", "optimize", "publish"]
        node_buttons = [
            attrs for tag, attrs in docs.tags
            if tag == "button" and attrs.get("data-atlas-node")
        ]
        self.assertEqual([button.get("data-atlas-node") for button in node_buttons], checkpoint_ids)
        self.assertEqual(node_buttons[0].get("tabindex"), "0")
        self.assertTrue(all(button.get("type") == "button" for button in node_buttons))
        self.assertEqual(html.count('class="atlas-connector" aria-hidden="true"'), 7)

        canonical_nodes = {
            node["id"]: node for node in json.loads(
                (ROOT / "mlx-model-porting" / "assets" / "learning_paths.json").read_text(
                    encoding="utf-8",
                ),
            )["checkpoint_nodes"]
        }
        for button in node_buttons:
            node = canonical_nodes[button["data-atlas-node"]]
            self.assertEqual(button.get("data-node-state"), node["evidence_state"])

        fallback_blocks = re.findall(
            r'<details class="atlas-fallback-journey"[^>]*data-journey="([^"]+)"[^>]*>(.*?)</details>',
            html,
            re.DOTALL,
        )
        self.assertEqual([journey_id for journey_id, _ in fallback_blocks], journey_ids)
        learning = json.loads(
            (ROOT / "mlx-model-porting" / "assets" / "learning_paths.json").read_text(
                encoding="utf-8",
            ),
        )
        journey_by_id = {journey["id"]: journey for journey in learning["journeys"]}
        for journey_id, block in fallback_blocks:
            with self.subTest(fallback_journey=journey_id):
                self.assertEqual(
                    re.findall(r'data-atlas-fallback-step="([^"]+)"', block),
                    checkpoint_ids,
                )
                rendered_notes = {
                    checkpoint_id: " ".join(unescape(note).split())
                    for checkpoint_id, note in re.findall(
                        r'data-atlas-fallback-step="([^"]+)"[^>]*>'
                        r'<strong>[^<]+</strong><span>(.*?)</span>',
                        block,
                        re.DOTALL,
                    )
                }
                self.assertEqual(
                    rendered_notes,
                    journey_by_id[journey_id]["checkpoint_notes"],
                    "the no-JavaScript guide must remain canonical rather than paraphrasing the data",
                )
                rendered_text = " ".join(unescape(re.sub(r"<[^>]+>", " ", block)).split())
                self.assertIn(journey_by_id[journey_id]["title"], rendered_text)
                canonical_path = " → ".join(
                    component["title"] for component in journey_by_id[journey_id]["component_path"]
                )
                self.assertIn(canonical_path, rendered_text)
                self.assertIn(journey_by_id[journey_id]["proof_boundary"], rendered_text)
        for text in (
            "Proven — pinned checkpoint proof",
            "Simulation — not a completed checkpoint port",
            "Qwen2.5 dense decoder",
            "Whisper-style ASR",
            "FLUX-style diffusion/flow",
            "LLaVA-style VLM",
        ):
            self.assertIn(text, docs.text)

        self.assertEqual(len([
            attrs for tag, attrs in docs.tags
            if tag == "input" and attrs.get("type") == "checkbox" and "data-atlas-path-only" in attrs
        ]), 1)
        for control in ("data-atlas-previous", "data-atlas-next", "data-atlas-copy"):
            self.assertEqual(len([
                attrs for tag, attrs in docs.tags
                if tag == "button" and control in attrs
            ]), 1)
        export_outputs = [
            attrs for tag, attrs in docs.tags
            if tag == "textarea" and "data-atlas-export-output" in attrs
        ]
        self.assertEqual(len(export_outputs), 1)
        self.assertIn("readonly", export_outputs[0])
        live_regions = [
            attrs for _tag, attrs in docs.tags
            if "data-atlas-live" in attrs
        ]
        self.assertEqual(len(live_regions), 1)
        self.assertEqual(live_regions[0].get("aria-live"), "polite")
        proof_boundaries = [
            attrs for _tag, attrs in docs.tags if "data-atlas-proof-boundary" in attrs
        ]
        self.assertEqual(len(proof_boundaries), 1)

        scripts = [
            PurePosixPath(urlsplit(attrs.get("src") or "").path).name
            for tag, attrs in docs.tags
            if tag == "script" and attrs.get("src")
        ]
        self.assertIn("atlas.js", scripts)
        self.assertLess(scripts.index("data.js"), scripts.index("atlas.js"))

        css = (SITE / "styles.css").read_text(encoding="utf-8")
        for selector in (".interactive-atlas", ".atlas-node-list", ".atlas-detail-panel", ".atlas-fallback-journey"):
            self.assertIn(selector, css)
        self.assertRegex(
            css,
            r"@media\s*\(max-width:\s*640px\)[\s\S]*\.atlas-node-list\s*\{[^}]*grid-template-columns:\s*1fr",
        )
        self.assertRegex(
            css,
            r"@media\s*\(prefers-reduced-motion:\s*reduce\)[\s\S]*\.atlas-connector",
        )

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
