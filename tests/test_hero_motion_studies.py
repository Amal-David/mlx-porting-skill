"""Contracts for the review-only mathematical hero motion studies."""
from __future__ import annotations

import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "prototypes" / "hero-motion"
PAGES = (
    STUDIES / "index.html",
    STUDIES / "matrix-port.html",
    STUDIES / "parity-trace.html",
)


class StudyParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.ids: set[str] = set()
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        self.tags.append((tag, values))
        if values.get("id"):
            self.ids.add(values["id"] or "")

    def handle_data(self, data: str) -> None:
        self.text_parts.append(data)

    @property
    def text(self) -> str:
        return " ".join("".join(self.text_parts).split())


def parse(path: Path) -> StudyParser:
    parser = StudyParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser


class HeroMotionStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pages = {path: parse(path) for path in PAGES}
        cls.css = (STUDIES / "motion.css").read_text(encoding="utf-8")
        cls.js = (STUDIES / "motion.js").read_text(encoding="utf-8")

    def test_studies_are_self_contained_and_local(self) -> None:
        for page, parsed in self.pages.items():
            for tag, attrs in parsed.tags:
                value = attrs.get("href") if tag == "a" or tag == "link" else attrs.get("src")
                if not value or value.startswith("#"):
                    continue
                with self.subTest(page=page.name, reference=value):
                    parts = urlsplit(value)
                    self.assertFalse(parts.scheme or parts.netloc, "review studies must not load remote assets")
                    target = (page.parent / parts.path).resolve()
                    self.assertTrue(target.exists(), f"missing local asset: {target}")

    def test_gallery_exposes_two_distinct_review_choices(self) -> None:
        gallery = self.pages[STUDIES / "index.html"]
        links = {attrs.get("href") for tag, attrs in gallery.tags if tag == "a"}
        self.assertIn("./matrix-port.html", links)
        self.assertIn("./parity-trace.html", links)
        self.assertIn("Neither is connected to the production landing page", gallery.text)

    def test_each_loop_has_timing_controls_and_an_accessible_static_explanation(self) -> None:
        expected = {"matrix-port.html": "24", "parity-trace.html": "30"}
        for page in PAGES[1:]:
            parsed = self.pages[page]
            bodies = [attrs for tag, attrs in parsed.tags if tag == "body"]
            figures = [attrs for tag, attrs in parsed.tags if tag == "figure"]
            buttons = [attrs for tag, attrs in parsed.tags if tag == "button"]
            with self.subTest(page=page.name):
                self.assertEqual(bodies[0].get("data-duration"), expected[page.name])
                self.assertTrue(any("data-pause" in attrs for attrs in buttons))
                self.assertTrue(any("data-restart" in attrs for attrs in buttons))
                self.assertEqual(len(figures), 1)
                self.assertIn(figures[0].get("aria-labelledby"), parsed.ids)
                self.assertIn(figures[0].get("aria-describedby"), parsed.ids)

    def test_tensor_study_teaches_an_index_preserving_layout_map(self) -> None:
        text = self.pages[STUDIES / "matrix-port.html"].text
        for concept in ("O×I×H×W", "O×H×W×I", "(o, i, h, w)", "(o, h, w, i)", "Δ ≤ ε"):
            with self.subTest(concept=concept):
                self.assertIn(concept, text)

    def test_trace_study_localizes_and_repairs_the_first_divergence(self) -> None:
        text = self.pages[STUDIES / "parity-trace.html"].text
        for concept in ("FIRST DIVERGENCE", "[ B, T, T ]", "[ B, 1, T, T ]", "expand_dims", "∀ checkpoints"):
            with self.subTest(concept=concept):
                self.assertIn(concept, text)

    def test_motion_has_explicit_reduced_motion_and_playback_states(self) -> None:
        self.assertIn("@media (prefers-reduced-motion: reduce)", self.css)
        self.assertIn("animation-play-state: paused", self.css)
        self.assertIn("animation-fill-mode", self.css)
        self.assertIn("visibilitychange", self.js)
        self.assertIn("aria-pressed", self.js)

    def test_production_landing_page_does_not_reference_the_unapproved_studies(self) -> None:
        production = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("prototypes/hero-motion", production)
        self.assertNotIn("matrix-port.html", production)
        self.assertNotIn("parity-trace.html", production)


if __name__ == "__main__":
    unittest.main()
