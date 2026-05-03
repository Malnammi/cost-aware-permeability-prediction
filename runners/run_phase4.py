#!/usr/bin/env python
"""
Phase 4 Generalization Validation - Nested LOWO Runner.

Runs nested leave-one-well-out evaluation for shortlisted Phase 3 subsets.
Supports partial execution by outer fold and subset IDs for cluster partitioning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nested_cv import load_phase4_config, run_nested_lowo


def _parse_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    raw = value.strip().lower()
    if raw == "all":
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_str_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    raw = value.strip().lower()
    if raw == "all":
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 nested LOWO run for unbiased generalization estimates.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to phase4 config (default: configs/phase4_config_bundled.json).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: results/phase4_generalization/run).",
    )
    parser.add_argument(
        "--subset_ids",
        type=str,
        default=None,
        help="Comma-separated subset IDs to run, or 'all' (default: all).",
    )
    parser.add_argument(
        "--outer_folds",
        type=str,
        default=None,
        help="Comma-separated held-out wells (e.g., A,B), or 'all' (default: all).",
    )
    parser.add_argument(
        "--hp_budget",
        type=int,
        default=None,
        help="Override HP budget per subset per outer fold.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume mode and recompute all selected fold/subset pairs.",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        help="Print resolved config and exit.",
    )
    parser.add_argument(
        "--list_subsets",
        action="store_true",
        help="List candidate subsets and exit.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress verbose output.",
    )
    args = parser.parse_args()

    config = load_phase4_config(args.config)

    if args.show_config:
        print("Phase 4 config:")
        print(json.dumps(config, indent=2))
        return

    if args.list_subsets:
        print("Candidate subsets:")
        for subset in config["candidate_subsets"]:
            subset_id = subset["subset_id"]
            label = subset.get("label", f"subset_{subset_id}")
            features = ",".join(subset.get("features", []))
            print(f"  - {subset_id} [{label}] -> {features}")
        return

    project_root = Path(__file__).parent.parent
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root / "results" / "phase4_generalization" / "run"
    )

    subset_ids = _parse_int_list(args.subset_ids)
    outer_folds = _parse_str_list(args.outer_folds)

    run_nested_lowo(
        config=config,
        output_dir=output_dir,
        subset_ids=subset_ids,
        outer_folds=outer_folds,
        hp_budget=args.hp_budget,
        resume=not args.no_resume,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
