#!/usr/bin/env python3
"""Generate a runbook-grounded MLX port plan from an inspection report."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from _common import SkillError, load_structured

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PORT_PLAN.md from inspect_model.py JSON")
    parser.add_argument("inspection")
    parser.add_argument("--techniques", default=str(SKILL_ROOT / "assets" / "techniques.yaml"))
    parser.add_argument("--optimization-guidance", default=str(SKILL_ROOT / "assets" / "optimization_guidance.yaml"))
    parser.add_argument("--output", default="PORT_PLAN.md")
    parser.add_argument("--family", help="Override detected architecture family")
    return parser.parse_args()


def relevant(tech: dict[str, Any], family: str) -> bool:
    values = [str(x).lower() for x in tech.get("applies_to", [])]
    f = family.lower()
    tokens = set(f.replace("-", " ").split())
    for value in values:
        if value == "all" or value == f or value in f or f in value:
            return True
        if "-" not in value and value in tokens:
            return True
    return False


def main() -> int:
    args = parse_args()
    try:
        inspection = load_structured(args.inspection)
        techniques = load_structured(args.techniques).get("techniques", [])
        optimization_guidance_path = Path(args.optimization_guidance)
        optimization_methods = (
            load_structured(optimization_guidance_path).get("methods", [])
            if optimization_guidance_path.exists()
            else []
        )
        candidates = inspection.get("architecture_candidates", [])
        family = args.family or inspection.get("recommended_family")
        if not family:
            raise SkillError("No detected family; pass --family after manual architecture review")
        selected = next((c for c in candidates if c.get("family") == family), None)
        runbook = (selected or {}).get("runbook") or inspection.get("recommended_runbook") or "manual selection required"
        proven = [t for t in techniques if relevant(t, family) and t.get("status") != "research-candidate"]
        research = [t for t in techniques if relevant(t, family) and t.get("status") == "research-candidate"]
        optimization_shortlist = [m for m in optimization_methods if relevant(m, family)]
        optimization_shortlist.sort(
            key=lambda m: (
                {"native-mlx": 0, "official-mlx-project": 1, "proven-mlx-port": 2, "research-candidate": 3}.get(str(m.get("status")), 9),
                str(m.get("category", "")),
                str(m.get("id", "")),
            )
        )
        risks = inspection.get("risks", [])
        license_info = inspection.get("license", {})
        source = inspection.get("source", {})
        tensors = inspection.get("tensor_summary", {})

        lines = [
            "# MLX port plan",
            "",
            "## Source and target",
            "",
            f"- Source: `{source.get('input', inspection.get('local_path', 'unknown'))}`",
            f"- Requested/pinned revision: `{source.get('revision') or 'REQUIRED BEFORE IMPLEMENTATION'}`",
            f"- Local inspection path: `{inspection.get('local_path', '')}`",
            f"- Static tensor count: {tensors.get('count', 0)}",
            f"- Static parameter count: {int(tensors.get('parameters', 0)):,}",
            f"- License evidence: {license_info.get('declared') or license_info.get('license_files') or 'MISSING — STOP FOR REVIEW'}",
            "- Target package: decide among MLX-LM, MLX-VLM, MLX-Audio, or standalone MLX after source comparison.",
            "- Target Mac/workload: **fill in chip, memory, OS, MLX version, input lengths, batch/concurrency, latency/memory/quality objective**.",
            "",
            "## Architecture route",
            "",
            f"- Family: `{family}`",
            f"- Runbook: `{runbook}`",
            f"- Detection evidence: {', '.join((selected or {}).get('evidence', [])) or 'manual override'}",
            f"- State/cache contract: {(selected or {}).get('state', 'must be documented')}",
            "",
            "## Static risk gates",
            "",
        ]
        if risks:
            lines.extend(f"- **{r.get('severity')} / {r.get('type')}**: {r.get('detail')}" for r in risks)
        else:
            lines.append("- No high-signal static risk was found; still review code, license, and provenance.")

        lines += [
            "",
            "## Source oracle",
            "",
            "- [ ] Pin source repository, model, tokenizer/processor, codec/vocoder, and generation revisions.",
            "- [ ] Save post-preprocessing minimal, ordinary, and boundary fixtures.",
            "- [ ] Capture semantic intermediate checkpoints described by the selected runbook.",
            "- [ ] Capture deterministic end-to-end output and task-quality baseline.",
            "- [ ] Capture full versus incremental/chunked state behavior.",
            "",
            "## Weight conversion",
            "",
            "- [ ] Export source tensor manifest without importing untrusted model code.",
            "- [ ] Create explicit `WEIGHT_MAP.json` entries for every source/target tensor.",
            "- [ ] Record fused/split QKV, gate/up, convolution layout, codebook, and tied/shared transforms.",
            "- [ ] Explain every ignored source and generated target tensor.",
            "- [ ] Validate shape coverage with `validate_weight_map.py`.",
            "",
            "## Implementation phases",
            "",
            "1. Implement config and preprocessing contract.",
            "2. Implement a readable primitive/block oracle in eager floating point.",
            "3. Achieve block and staged intermediate parity.",
            "4. Assemble end-to-end model and deterministic output.",
            "5. Add exact cache/recurrent/streaming state and compare against full recomputation.",
            "6. Save/reload and package a smoke-test fixture.",
            "7. Establish a baseline profile and benchmark report.",
            "8. Run one optimization experiment at a time.",
            "9. Quantize only after unquantized quality and performance are recorded.",
            "10. Publish only after license/provenance and clean-environment load tests.",
            "",
            "## Optimization shortlist",
            "",
            "These are starting hypotheses, not accepted changes. Keep or reject each one through the validation gate and benchmark metadata.",
            "",
        ]
        for method in optimization_shortlist[:10]:
            gates = method.get("validation_gates", [])
            gate = gates[0] if gates else "validation gate required"
            lines.append(
                f"- **{method['id']}** (`{method['status']}`): {method['recommendation']} "
                f"Expected effect: {method['expected_effect']} First gate: {gate}"
            )
        if not optimization_shortlist:
            lines.append("- None auto-selected; profile and consult `assets/optimization_guidance.yaml` manually.")
        lines += [
            "",
            "## Proven/native optimization candidates",
            "",
        ]
        for tech in proven[:16]:
            lines.append(f"- **{tech['title']}** (`{tech['status']}`): use when {tech['use_when']} Gate: {tech['validation_gate']}")
        if not proven:
            lines.append("- None auto-selected; profile and consult the registry manually.")
        lines += ["", "## Research candidates — experiments, not assumed support", ""]
        for tech in research[:12]:
            lines.append(f"- **{tech['title']}**: {tech['use_when']} Avoid when {tech['avoid_when']} Gate: {tech['validation_gate']}")
        if not research:
            lines.append("- None selected by family matching.")
        lines += [
            "",
            "## Validation matrix",
            "",
            "| Stage | Fixture | Metric/tolerance | Status |",
            "|---|---|---|---|",
            "| Preprocessing | minimal/ordinary/boundary | exact or declared | pending |",
            "| Weight coverage | all tensors | 100% explained | pending |",
            "| Primitive/block | deterministic tensors | per-op tolerance | pending |",
            "| End-to-end | deterministic input | logits/latent/output | pending |",
            "| Cache/state | full vs incremental/chunks | declared tolerance | pending |",
            "| Task quality | representative set | baseline-relative | pending |",
            "| Performance | target workload | raw baseline/candidate | pending |",
            "",
            "## Stop conditions",
            "",
            "- Missing or incompatible license/provenance.",
            "- Required behavior exists only in unreviewed remote code.",
            "- First divergence cannot be localized through staged checkpoints.",
            "- Candidate optimization fails correctness/quality or lacks end-to-end benefit.",
            "- Custom kernel lacks a reference fallback and shape/dtype coverage.",
        ]
        Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(args.output)
        return 0
    except SkillError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
