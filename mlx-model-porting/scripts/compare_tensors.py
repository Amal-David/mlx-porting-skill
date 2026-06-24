#!/usr/bin/env python3
"""Compare portable NPZ source-oracle tensors with shape and numerical metrics."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from _common import SkillError, dump_json, load_structured


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tensors in two .npz archives")
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--mapping", help="JSON/YAML mapping of source key to target key")
    parser.add_argument("--include", help="Regex selecting source keys")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--cosine-min", type=float, default=-1.0, help="Fail if cosine is below this; -1 disables")
    parser.add_argument("--output")
    parser.add_argument("--no-fail", action="store_true")
    parser.add_argument("--allow-empty", action="store_true", help="Permit zero selected tensors")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        try:
            import numpy as np
        except ImportError as exc:
            raise SkillError("numpy is required for compare_tensors.py") from exc
        src = np.load(args.source, allow_pickle=False)
        dst = np.load(args.target, allow_pickle=False)
        mapping: dict[str, str] = {}
        if args.mapping:
            raw = load_structured(args.mapping)
            if isinstance(raw, dict) and "mapping" in raw:
                raw = raw["mapping"]
            if not isinstance(raw, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in raw.items()):
                raise SkillError("mapping must be an object of source-key -> target-key")
            mapping = raw
        pattern = re.compile(args.include) if args.include else None
        source_keys = [k for k in src.files if pattern is None or pattern.search(k)]
        rows: list[dict[str, Any]] = []
        failures: list[str] = []
        if not source_keys and not args.allow_empty:
            failures.append("no source tensors selected; use --allow-empty only for intentional report-only checks")
        for key in source_keys:
            target_key = mapping.get(key, key)
            if target_key not in dst.files:
                failures.append(f"missing target key {target_key} for source {key}")
                rows.append({"source": key, "target": target_key, "ok": False, "error": "missing target"})
                continue
            a = np.asarray(src[key])
            b = np.asarray(dst[target_key])
            if a.shape != b.shape:
                failures.append(f"shape mismatch {key} {a.shape} vs {target_key} {b.shape}")
                rows.append({"source": key, "target": target_key, "source_shape": list(a.shape), "target_shape": list(b.shape), "ok": False, "error": "shape mismatch"})
                continue
            aa = a.astype(np.complex128 if np.iscomplexobj(a) or np.iscomplexobj(b) else np.float64, copy=False)
            bb = b.astype(np.complex128 if np.iscomplexobj(a) or np.iscomplexobj(b) else np.float64, copy=False)
            finite = bool(np.all(np.isfinite(aa)) and np.all(np.isfinite(bb)))
            diff = np.abs(aa - bb)
            denom = np.maximum(np.abs(aa), 1e-30)
            rel = diff / denom
            flat_a = aa.reshape(-1)
            flat_b = bb.reshape(-1)
            # Avoid BLAS-backed norm/vdot here: tiny oracle archives should not
            # spawn large thread pools or inherit vendor BLAS deadlocks in CI.
            norm_a = float(np.sqrt(np.sum(np.abs(flat_a) ** 2, dtype=np.float64)))
            norm_b = float(np.sqrt(np.sum(np.abs(flat_b) ** 2, dtype=np.float64)))
            norm = norm_a * norm_b
            dot = np.sum(np.conjugate(flat_a) * flat_b)
            cosine = float(np.real(dot) / norm) if norm else (1.0 if np.array_equal(flat_a, flat_b) else 0.0)
            allclose = bool(np.allclose(aa, bb, atol=args.atol, rtol=args.rtol, equal_nan=False))
            cosine_ok = args.cosine_min < 0 or cosine >= args.cosine_min
            ok = finite and allclose and cosine_ok
            row = {
                "source": key,
                "target": target_key,
                "shape": list(a.shape),
                "source_dtype": str(a.dtype),
                "target_dtype": str(b.dtype),
                "finite": finite,
                "max_abs": float(diff.max(initial=0.0)),
                "mean_abs": float(diff.mean()) if diff.size else 0.0,
                "max_rel": float(rel.max(initial=0.0)),
                "cosine": cosine,
                "allclose": allclose,
                "ok": ok,
            }
            rows.append(row)
            if not ok:
                failures.append(f"numerical mismatch {key}: max_abs={row['max_abs']:.6g}, max_rel={row['max_rel']:.6g}, cosine={cosine:.8f}")
        unmapped_target = sorted(set(dst.files) - {mapping.get(k, k) for k in source_keys})
        report = {
            "schema_version": 1,
            "ok": not failures,
            "atol": args.atol,
            "rtol": args.rtol,
            "cosine_min": args.cosine_min,
            "compared": len(rows),
            "rows": rows,
            "uncompared_target_keys": unmapped_target,
            "failures": failures,
            "report_only": bool(args.no_fail),
        }
        text = dump_json(report, args.output)
        if args.output is None:
            sys.stdout.write(text)
        return 0 if (not failures or args.no_fail) else 1
    except (SkillError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
