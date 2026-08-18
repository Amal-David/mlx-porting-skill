#!/usr/bin/env python3
"""Validate an evidence pack before it is merged into the graph.

    python3 mlx-model-porting/graph/tools/validate_pack.py packs/<pack-dir>
    python3 mlx-model-porting/graph/tools/validate_pack.py --all

An evidence pack is one directory holding:

    manifest.json     who measured it, on what, when, with which runtimes
    graph-delta.json  a schema-valid v2 graph document holding only new nodes
                      and edges; edges may reference ids already in the graph
    receipts/         optional raw measurement receipts, referenced by sha256

Standard library only. The gate is deliberately stricter than the graph
validator in three ways:

1. Scrub. No credentials, no absolute local paths, and no bare home
   directories anywhere in the pack, including the manifest and receipts.
2. Provenance ceiling. A pack may not assert `official_verified`,
   `local_measured`, `replicated`, or `code_verified` on anything that carries
   a measurement. An incoming measurement is a `contributor_claim` until
   somebody else reproduces it. Non-measurement nodes may still be
   `code_verified` when they only record what published code says.
3. No redefinition. A pack may add nodes and edges. It may not silently
   redefine a node that already exists in the compiled graph; changing an
   existing node is a graph edit, reviewed as one.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _graphlib as gl  # noqa: E402

MANIFEST_REQUIRED = (
    "pack_version",
    "contributor",
    "submitted_at",
    "hardware",
    "runtime",
    "summary",
)
MANIFEST_OPTIONAL = ("receipts", "notes", "replicates")

HARDWARE_REQUIRED = ("node_id", "chip")
RUNTIME_REQUIRED = ("os", "packages")

# A measurement claim can never enter the graph above this ceiling.
CONTRIBUTOR_PROVENANCE = frozenset(
    {"contributor_claim", "author_claim", "research_inference", "transfer_inference"}
)

HANDLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_manifest(manifest, where):
    problems = []
    if not isinstance(manifest, dict):
        return ["%s: manifest must be a JSON object" % where]
    unknown = set(manifest) - set(MANIFEST_REQUIRED) - set(MANIFEST_OPTIONAL)
    if unknown:
        problems.append("%s: manifest has unknown fields %s" % (where, sorted(unknown)))
    for field in MANIFEST_REQUIRED:
        if field not in manifest:
            problems.append("%s: manifest is missing %r" % (where, field))
    if manifest.get("pack_version") != 1:
        problems.append("%s: manifest pack_version must be 1" % where)
    handle = manifest.get("contributor")
    if not isinstance(handle, str) or not HANDLE_RE.match(handle):
        problems.append(
            "%s: contributor must be a plain handle (letters, digits, dot, dash, "
            "underscore); do not put an email address or a real name here" % where
        )
    submitted = manifest.get("submitted_at")
    if not isinstance(submitted, str) or not gl.ISO_DATE_RE.match(submitted):
        problems.append("%s: submitted_at must be an ISO date" % where)

    hardware = manifest.get("hardware")
    if not isinstance(hardware, dict):
        problems.append("%s: hardware must be an object" % where)
    else:
        for field in HARDWARE_REQUIRED:
            if field not in hardware:
                problems.append("%s: hardware is missing %r" % (where, field))
        node_id = hardware.get("node_id")
        if isinstance(node_id, str) and not node_id.startswith("hardware:"):
            problems.append(
                "%s: hardware.node_id must name a hardware node (hardware:...)" % where
            )
        for key, value in hardware.items():
            if not isinstance(value, (str, int, bool)) or isinstance(value, float):
                problems.append("%s: hardware[%r] must be a string, integer, or boolean" % (where, key))

    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        problems.append("%s: runtime must be an object" % where)
    else:
        for field in RUNTIME_REQUIRED:
            if field not in runtime:
                problems.append("%s: runtime is missing %r" % (where, field))
        packages = runtime.get("packages")
        if not isinstance(packages, dict) or not packages:
            problems.append(
                "%s: runtime.packages must be a non-empty object of pinned versions "
                "(for example {\"mlx\": \"0.30.4\"})" % where
            )
        else:
            for name, version in sorted(packages.items()):
                if not isinstance(version, str) or not version:
                    problems.append("%s: runtime.packages[%r] must be a version string" % (where, name))

    receipts = manifest.get("receipts", [])
    if not isinstance(receipts, list):
        problems.append("%s: receipts must be an array" % where)
        receipts = []
    for index, item in enumerate(receipts):
        if not isinstance(item, dict):
            problems.append("%s: receipts[%d] must be an object" % (where, index))
            continue
        unknown = set(item) - {"path", "sha256", "role"}
        if unknown:
            problems.append("%s: receipts[%d] has unknown fields %s" % (where, index, sorted(unknown)))
        for field in ("path", "sha256"):
            if field not in item:
                problems.append("%s: receipts[%d] is missing %r" % (where, index, field))
        if isinstance(item.get("sha256"), str) and not SHA256_RE.match(item["sha256"]):
            problems.append("%s: receipts[%d].sha256 must be 64 lowercase hex characters" % (where, index))
        path = item.get("path")
        if isinstance(path, str) and (path.startswith("/") or ".." in Path(path).parts):
            problems.append(
                "%s: receipts[%d].path must be relative to the pack directory and must not escape it"
                % (where, index)
            )
    return problems


def _validate_delta(delta, compiled, where):
    existing_ids = {
        item["id"] for item in compiled.get("nodes", []) if isinstance(item, dict)
    }
    problems = list(gl.validate_document(delta, where))
    problems.extend(
        gl.validate_corpus([(where, delta)], context=[("compiled graph", compiled)])
    )

    declared = set()
    for item in delta.get("nodes", []):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        declared.add(item["id"])
        if item["id"] in existing_ids:
            problems.append(
                "%s: node %s already exists in the compiled graph; a pack adds nodes, it "
                "does not redefine them" % (where, item["id"])
            )
        provenance = item.get("provenance")
        if "effect" in item and provenance not in CONTRIBUTOR_PROVENANCE:
            problems.append(
                "%s: node %s carries a measurement with provenance %r; an incoming pack may "
                "claim at most %s. Replication upgrades provenance later; a pack never does."
                % (where, item["id"], provenance, "contributor_claim")
            )
        if provenance in ("official_verified", "local_measured", "replicated"):
            problems.append(
                "%s: node %s asserts provenance %r, which only the graph maintainers may set"
                % (where, item["id"], provenance)
            )

    known = declared | existing_ids
    for index, item in enumerate(delta.get("edges", [])):
        if not isinstance(item, dict):
            continue
        for field in ("from", "to"):
            value = item.get(field)
            if isinstance(value, str) and value not in known:
                problems.append(
                    "%s edges[%d]: %s references %r, which is neither in this pack nor in the "
                    "compiled graph" % (where, index, field, value)
                )
    return problems


def validate_pack(pack_dir, compiled):
    pack_dir = Path(pack_dir)
    label = pack_dir.name
    problems = []

    manifest_path = pack_dir / "manifest.json"
    delta_path = pack_dir / "graph-delta.json"
    for path in (manifest_path, delta_path):
        if not path.exists():
            problems.append("%s: missing %s" % (label, path.name))
    if problems:
        return problems

    try:
        manifest = gl.load_strict(manifest_path)
    except gl.GraphError as exc:
        return ["%s: %s" % (label, exc)]
    try:
        delta = gl.load_strict(delta_path)
    except gl.GraphError as exc:
        return ["%s: %s" % (label, exc)]

    problems.extend(_validate_manifest(manifest, label + "/manifest.json"))
    problems.extend(_validate_delta(delta, compiled, label + "/graph-delta.json"))

    # Scrub the whole pack directory, including receipts and any stray file.
    for path in sorted(pack_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(pack_dir).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            problems.append("%s/%s: pack files must be UTF-8 text" % (label, relative))
            continue
        for name, pattern in gl.SECRET_PATTERNS:
            if pattern.search(text):
                problems.append("%s/%s: possible %s" % (label, relative, name))
        if gl.PRIVATE_PATH_RE.search(text):
            problems.append(
                "%s/%s: absolute private path; use a URL or a descriptive locator"
                % (label, relative)
            )

    # Receipt digests must match the bytes actually shipped.
    for item in manifest.get("receipts", []) or []:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        digest = item.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            continue
        target = pack_dir / path
        if not target.exists():
            problems.append("%s: receipt %s is declared but missing" % (label, path))
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != digest:
            problems.append(
                "%s: receipt %s digest mismatch (declared %s, actual %s)"
                % (label, path, digest, actual)
            )
    return problems


def _compiled(compiled_path):
    path = Path(compiled_path)
    if not path.exists():
        return {"nodes": [], "edges": []}
    return gl.load_strict(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packs", nargs="*", help="pack directories to validate")
    parser.add_argument("--all", action="store_true", help="validate every pack under packs/")
    parser.add_argument("--compiled", default=str(gl.COMPILED_PATH))
    args = parser.parse_args(argv)

    targets = [Path(item) for item in args.packs]
    if args.all or not targets:
        targets = sorted(
            path for path in gl.PACKS_DIR.iterdir()
            if path.is_dir() and (path / "manifest.json").exists()
        )
    if not targets:
        print("no packs to validate")
        return 0

    compiled = _compiled(args.compiled)
    problems = []
    for pack in targets:
        found = validate_pack(pack, compiled)
        print("  %-46s %s" % (pack.name, "OK" if not found else "FAIL (%d)" % len(found)))
        problems.extend(found)

    if problems:
        print("FAIL %d problem(s):" % len(problems))
        for problem in problems:
            print("  - %s" % problem)
        return 1
    print("OK %d pack(s) valid" % len(targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
