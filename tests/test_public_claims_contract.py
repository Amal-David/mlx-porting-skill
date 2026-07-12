"""Keep public multiplier claims subordinate to the canonical claim catalogue."""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "mlx-model-porting" / "assets" / "effective_claims.json"
MULTIPLIER_RE = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"(?P<value>\d+(?:\.\d+)?x(?:\s*[-–—]\s*\d+(?:\.\d+)?x)?)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
CATALOG_BOUNDARY_RE = re.compile(
    r"\b(?:"
    r"catalogued\s+(?:range|observation)|ceiling|floor|hypoth(?:esis|eses|etical)|"
    r"not\s+(?:a\s+)?(?:portable\s+)?guarantee|not\s+a\s+claim|"
    r"performance[_ -]observation|observation|observed|rejected|"
    r"planning\s+band|profile[- ]eligible|profile[- ]required|receipt|regress(?:ed|ion)?|"
    r"requires?\s+(?:local\s+)?(?:measurement|profiling|validation)|source[- ]reported|"
    r"unmeasured|withheld"
    r")\b",
    re.IGNORECASE,
)
NON_CLAIM_BOUNDARY_RE = re.compile(
    r"\b(?:"
    r"ceiling|floor|hypoth(?:esis|eses|etical)|"
    r"not\s+(?:a\s+)?(?:portable\s+)?guarantee|not\s+a\s+claim|"
    r"planning\s+band|requires?\s+(?:local\s+)?(?:measurement|profiling|validation)|"
    r"unmeasured|withheld"
    r")\b",
    re.IGNORECASE,
)
GENERATED_OBSERVATION_RE = re.compile(r"`(?:performance_observation|rejected)`", re.IGNORECASE)
TARGET_BOUNDARY_RE = re.compile(
    r"(?:"
    r"\d+(?:\.\d+)?x(?:\s*(?:/|,|\bor\b)\s*\d+(?:\.\d+)?x)*\s+targets?\b|"
    r"\b(?:research|optimization|performance|speedup|stack)\s+targets?\b|"
    r"\btargets?\s+(?:of\s+)?\d+(?:\.\d+)?x\b|"
    r"\d+(?:\.\d+)?x\s+stack\s+discipline\b"
    r")",
    re.IGNORECASE,
)

# Generated evidence tables and raw assets are provenance/data surfaces, not
# promotional prose. Archived research runs intentionally preserve historical
# researcher language verbatim. The public-copy lint instead covers every current
# product, adapter, skill, runbook, example, and site narrative surface.
PUBLIC_ROOT_FILES = (
    "README.md",
    "VALIDATION.md",
    "RESEARCH_REPORT.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
)
PUBLIC_TREES = (
    ROOT / "adapters",
    ROOT / "mlx-model-porting" / "references",
    ROOT / "mlx-model-porting" / "examples",
    ROOT / "site",
)
PUBLIC_SINGLE_FILES = (
    ROOT / "mlx-model-porting" / "SKILL.md",
    ROOT / "mlx-model-porting" / "assets" / "BENCHMARK_REPORT.md",
)


def normalize_multiplier(value: str) -> str:
    return re.sub(r"\s*[-–—]\s*", "-", value.strip().lower())


def canonical_multipliers() -> set[str]:
    payload = json.loads(CLAIMS.read_text(encoding="utf-8"))
    values: set[str] = set()
    for claim in payload.get("claims", []):
        if not isinstance(claim, dict):
            continue
        for field in ("observed_range", "profile_eligible_range", "effective_range"):
            raw = claim.get(field)
            if not isinstance(raw, str):
                continue
            normalized = normalize_multiplier(raw)
            values.add(normalized)
            values.update(normalize_multiplier(match.group("value")) for match in MULTIPLIER_RE.finditer(raw))
            values.update(re.findall(r"\d+(?:\.\d+)?x", normalized))
    return values


def public_copy_files() -> tuple[Path, ...]:
    paths = [ROOT / relative for relative in PUBLIC_ROOT_FILES]
    paths.extend(PUBLIC_SINGLE_FILES)
    for tree in PUBLIC_TREES:
        paths.extend(
            path for path in tree.rglob("*")
            if path.is_file() and path.suffix.lower() in {".md", ".html", ".js"}
        )
    return tuple(sorted(set(paths)))


def lint_text(
    text: str,
    allowed: set[str],
    *,
    generated_observation_report: bool = False,
) -> list[tuple[int, str, str]]:
    lines = text.splitlines()
    offenders: list[tuple[int, str, str]] = []
    for index, line in enumerate(lines):
        for match in MULTIPLIER_RE.finditer(line):
            value = normalize_multiplier(match.group("value"))
            canonical = value in allowed
            # Boundary language must accompany the number itself. Nearby mentions
            # of a target device, baseline, candidate, or held state cannot launder
            # an unsupported sentence into public copy.
            catalog_bounded = (
                CATALOG_BOUNDARY_RE.search(line) is not None
                or TARGET_BOUNDARY_RE.search(line) is not None
            )
            non_claiming = (
                NON_CLAIM_BOUNDARY_RE.search(line) is not None
                or TARGET_BOUNDARY_RE.search(line) is not None
                or (generated_observation_report and GENERATED_OBSERVATION_RE.search(line) is not None)
            )
            if not canonical and not non_claiming:
                offenders.append((index + 1, value, "not in effective_claims.json and lacks boundary language"))
            elif canonical and not catalog_bounded:
                offenders.append((index + 1, value, "catalogued number is presented without provenance/boundary language"))
    return offenders


class PublicClaimsContractTests(unittest.TestCase):
    def test_public_multiplier_copy_is_catalogued_or_explicitly_non_claiming(self) -> None:
        allowed = canonical_multipliers()
        self.assertTrue(allowed, "effective claim catalogue exposes no auditable multiplier values")
        offenders: list[str] = []
        for path in public_copy_files():
            for line, value, reason in lint_text(
                path.read_text(encoding="utf-8"),
                allowed,
                generated_observation_report=path.name == "BENCHMARK_REPORT.md",
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{line}: {value}: {reason}")
        self.assertEqual(offenders, [], "unsafe public numeric claims:\n" + "\n".join(offenders))

    def test_uncatalogued_or_context_free_static_speedups_are_rejected(self) -> None:
        allowed = canonical_multipliers()
        self.assertTrue(lint_text("This port is 99x faster.", allowed))
        self.assertTrue(lint_text("This port is 4.3x faster.", allowed))
        self.assertTrue(lint_text("We observed a 99x speedup.", allowed))
        self.assertTrue(lint_text("A source-reported 99x speedup is available.", allowed))
        self.assertFalse(lint_text("A 99x research target is not a measured claim.", allowed))
        self.assertTrue(
            lint_text(
                "Target hardware is Apple Silicon and the baseline is held fixed.\n"
                "This port is 99x faster.\n"
                "The candidate will be profiled later.",
                allowed,
            ),
            "nearby generic target/baseline/candidate language must not mask a static claim",
        )


if __name__ == "__main__":
    unittest.main()
