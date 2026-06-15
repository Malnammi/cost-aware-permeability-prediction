#!/usr/bin/env python
"""
Generate a schema-matched synthetic surrogate dataset.

This script creates a CSV with the same column layout as the without-outlier-adaptive scheme.

It is intended for reproducibility and pipeline execution tests.
It is not intended for scientific benchmarking.

Usage:
    python datasets/generate_synthetic_surrogate.py
    python datasets/generate_synthetic_surrogate.py --n-rows 3000 --seed 42
    python datasets/generate_synthetic_surrogate.py --output datasets/synthetic_custom.csv
    python datasets/generate_synthetic_surrogate.py --depth-min 0 --depth-max 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Physical reference ranges from Supplementary Table S2.
PHYSICAL_RANGES: dict[str, tuple[float, float]] = {
    "RT": (0.2, 2000.0),
    "CT": (0.0005, 5.0),
    "SWT": (0.0, 1.0),
    "GR": (0.0, 150.0),
    "NPHI": (-0.05, 0.45),
    "RHOB": (1.95, 2.95),
    "DRHO": (-0.25, 0.25),
    "DT": (40.0, 140.0),
    "PEF": (1.0, 6.0),
    "CALI": (4.0, 22.0),
    "MSFL": (0.2, 2000.0),
    "PHIT": (0.0, 0.40),
}

SOURCE_VALUES = np.array(["A", "B", "C", "D", "E", "F", "G"], dtype=object)
ZONE_VALUES = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)

OUTPUT_COLUMNS = [
    "DEPTH",
    "CKHL_SM",
    "CPOR_SM",
    "CALI",
    "CT",
    "DRHO",
    "GR",
    "MSFL",
    "NPHI",
    "PHIT",
    "RHOB",
    "RT",
    "SWT",
    "CT_missing",
    "RT_missing",
    "SWT_missing",
    "Source",
    "Zone",
    "DT",
    "PEF",
    "MSFL_missing_source",
    "DT_missing_source",
    "PEF_missing_source",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_output = (
        Path(__file__).resolve().parent / "synthetic_without_outlier_adaptive.csv"
    )
    parser = argparse.ArgumentParser(
        description="Generate schema-matched synthetic surrogate CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output CSV path (default: {default_output})",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=2284,
        help="Number of rows to generate (default: 2284).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--depth-min",
        type=float,
        default=0.0,
        help="Minimum synthetic relative depth (default: 0.0).",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=2000.0,
        help="Maximum synthetic relative depth (default: 2000.0).",
    )
    return parser.parse_args()


def _sample_log_uniform(
    rng: np.random.Generator,
    low: float,
    high: float,
    size: int,
) -> np.ndarray:
    """Sample from log-uniform distribution between low and high."""
    return np.power(10.0, rng.uniform(np.log10(low), np.log10(high), size=size))


def _build_sources(rng: np.random.Generator, n_rows: int) -> np.ndarray:
    """Return balanced Source labels over A..G."""
    repeats = int(np.ceil(n_rows / len(SOURCE_VALUES)))
    sources = np.tile(SOURCE_VALUES, repeats)[:n_rows].copy()
    rng.shuffle(sources)
    return sources


def _build_continuous(
    rng: np.random.Generator, n_rows: int
) -> dict[str, np.ndarray]:
    """
    Build synthetic continuous features.

    Features are sampled independently from bounded ranges to preserve schema
    and valid value domains. This is for code-path testing only.
    """
    cpor_sm = rng.uniform(0.03, 0.35, size=n_rows)

    cali = rng.uniform(PHYSICAL_RANGES["CALI"][0], PHYSICAL_RANGES["CALI"][1], size=n_rows)
    drho = rng.uniform(PHYSICAL_RANGES["DRHO"][0], PHYSICAL_RANGES["DRHO"][1], size=n_rows)
    gr = rng.uniform(PHYSICAL_RANGES["GR"][0], PHYSICAL_RANGES["GR"][1], size=n_rows)
    nphi = rng.uniform(PHYSICAL_RANGES["NPHI"][0], PHYSICAL_RANGES["NPHI"][1], size=n_rows)
    phit = rng.uniform(PHYSICAL_RANGES["PHIT"][0], PHYSICAL_RANGES["PHIT"][1], size=n_rows)
    rhob = rng.uniform(PHYSICAL_RANGES["RHOB"][0], PHYSICAL_RANGES["RHOB"][1], size=n_rows)
    swt = rng.uniform(PHYSICAL_RANGES["SWT"][0], PHYSICAL_RANGES["SWT"][1], size=n_rows)
    dt = rng.uniform(PHYSICAL_RANGES["DT"][0], PHYSICAL_RANGES["DT"][1], size=n_rows)
    pef = rng.uniform(PHYSICAL_RANGES["PEF"][0], PHYSICAL_RANGES["PEF"][1], size=n_rows)

    rt = _sample_log_uniform(
        rng, PHYSICAL_RANGES["RT"][0], PHYSICAL_RANGES["RT"][1], n_rows
    )
    ct = _sample_log_uniform(
        rng, PHYSICAL_RANGES["CT"][0], PHYSICAL_RANGES["CT"][1], n_rows
    )
    msfl = _sample_log_uniform(
        rng, PHYSICAL_RANGES["MSFL"][0], PHYSICAL_RANGES["MSFL"][1], n_rows
    )

    return {
        "CPOR_SM": cpor_sm,
        "CALI": cali,
        "CT": ct,
        "DRHO": drho,
        "GR": gr,
        "MSFL": msfl,
        "NPHI": nphi,
        "PHIT": phit,
        "RHOB": rhob,
        "RT": rt,
        "SWT": swt,
        "DT": dt,
        "PEF": pef,
    }


def _build_relative_depth(
    rng: np.random.Generator,
    source: np.ndarray,
    depth_min: float,
    depth_max: float,
) -> np.ndarray:
    """
    Build synthetic relative depth per source.

    Each source receives a monotonic synthetic depth track rescaled to
    [depth_min, depth_max]. This is a relative coordinate, not measured depth.
    """
    depth = np.empty(source.shape[0], dtype=float)

    for src in SOURCE_VALUES:
        idx = np.flatnonzero(source == src)
        if idx.size == 0:
            continue
        if idx.size == 1:
            depth[idx] = depth_min
            continue

        increments = rng.lognormal(mean=0.0, sigma=0.35, size=idx.size)
        cumulative = np.cumsum(increments)
        scaled = depth_min + (cumulative - cumulative.min()) * (
            (depth_max - depth_min) / (cumulative.max() - cumulative.min())
        )
        depth[idx] = scaled

    return depth


def _build_zone_from_depth(
    depth: np.ndarray,
    depth_min: float,
    depth_max: float,
) -> np.ndarray:
    """
    Assign Zone from DEPTH using equal-width bins.

    The full depth interval [depth_min, depth_max] is split into 11 uniform
    bins mapped to integer zones 0..10.
    """
    edges = np.linspace(depth_min, depth_max, num=len(ZONE_VALUES) + 1)
    zone = np.digitize(depth, edges[1:-1], right=False).astype(int)
    return np.clip(zone, ZONE_VALUES.min(), ZONE_VALUES.max())


def _build_target(
    rng: np.random.Generator,
    depth: np.ndarray,
    continuous: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Build positive synthetic CKHL_SM values.

    Target is generated from a simple random linear model in normalized feature
    space, then back-transformed from log10-space.
    """
    feature_names = [
        "DEPTH",
        "CPOR_SM",
        "CALI",
        "CT",
        "DRHO",
        "GR",
        "MSFL",
        "NPHI",
        "PHIT",
        "RHOB",
        "RT",
        "SWT",
        "DT",
        "PEF",
    ]
    feature_arrays = [
        depth,
        continuous["CPOR_SM"],
        continuous["CALI"],
        continuous["CT"],
        continuous["DRHO"],
        continuous["GR"],
        continuous["MSFL"],
        continuous["NPHI"],
        continuous["PHIT"],
        continuous["RHOB"],
        continuous["RT"],
        continuous["SWT"],
        continuous["DT"],
        continuous["PEF"],
    ]
    x = np.column_stack(feature_arrays).astype(float)

    # Min-max normalize each feature to avoid scale domination.
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    x_range = np.where((x_max - x_min) > 0, x_max - x_min, 1.0)
    x_norm = (x - x_min) / x_range

    # Random linear weights (user-requested), resampled each generator run.
    weights = rng.uniform(-1.0, 1.0, size=len(feature_names))
    linear_signal = (x_norm * weights).sum(axis=1) / len(feature_names)
    log_perm = 1.0 + linear_signal + rng.normal(loc=0.0, scale=0.20, size=x.shape[0])
    ckhl_sm = np.power(10.0, log_perm)
    ckhl_sm = np.clip(ckhl_sm, 0.01, 5000.0)
    return ckhl_sm


def generate_synthetic_dataframe(
    n_rows: int,
    seed: int,
    depth_min: float = 0.0,
    depth_max: float = 2000.0,
) -> pd.DataFrame:
    """Generate schema-matched synthetic dataframe."""
    if n_rows <= 0:
        raise ValueError("--n-rows must be a positive integer.")
    if depth_max <= depth_min:
        raise ValueError("--depth-max must be greater than --depth-min.")

    rng = np.random.default_rng(seed)
    continuous = _build_continuous(rng, n_rows)

    source = _build_sources(rng, n_rows)
    depth = _build_relative_depth(rng, source, depth_min=depth_min, depth_max=depth_max)
    zone = _build_zone_from_depth(depth, depth_min=depth_min, depth_max=depth_max)

    # Per-sample missing flags for cleaned sentinel channels.
    ct_missing = rng.binomial(1, 0.02, size=n_rows).astype(np.int8)
    rt_missing = rng.binomial(1, 0.02, size=n_rows).astype(np.int8)
    swt_missing = rng.binomial(1, 0.02, size=n_rows).astype(np.int8)

    # Source-level missingness for selected curves.
    msfl_missing_source = np.isin(source, np.array(["G"], dtype=object)).astype(np.int8)
    dt_missing_source = np.isin(source, np.array(["F", "G"], dtype=object)).astype(np.int8)
    pef_missing_source = np.isin(source, np.array(["E", "F", "G"], dtype=object)).astype(np.int8)

    ct = continuous["CT"].copy()
    rt = continuous["RT"].copy()
    swt = continuous["SWT"].copy()
    msfl = continuous["MSFL"].copy()
    dt = continuous["DT"].copy()
    pef = continuous["PEF"].copy()

    ct[ct_missing == 1] = np.nan
    rt[rt_missing == 1] = np.nan
    swt[swt_missing == 1] = np.nan

    msfl[msfl_missing_source == 1] = np.nan
    dt[dt_missing_source == 1] = np.nan
    pef[pef_missing_source == 1] = np.nan

    ckhl_sm = _build_target(rng, depth=depth, continuous=continuous)

    df = pd.DataFrame(
        {
            "DEPTH": depth,
            "CKHL_SM": ckhl_sm,
            "CPOR_SM": continuous["CPOR_SM"],
            "CALI": continuous["CALI"],
            "CT": ct,
            "DRHO": continuous["DRHO"],
            "GR": continuous["GR"],
            "MSFL": msfl,
            "NPHI": continuous["NPHI"],
            "PHIT": continuous["PHIT"],
            "RHOB": continuous["RHOB"],
            "RT": rt,
            "SWT": swt,
            "CT_missing": ct_missing,
            "RT_missing": rt_missing,
            "SWT_missing": swt_missing,
            "Source": source,
            "Zone": zone,
            "DT": dt,
            "PEF": pef,
            "MSFL_missing_source": msfl_missing_source,
            "DT_missing_source": dt_missing_source,
            "PEF_missing_source": pef_missing_source,
        }
    )
    return df[OUTPUT_COLUMNS]


def main() -> None:
    """Run generator and write synthetic CSV."""
    args = parse_args()
    output_path: Path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_dataframe(
        n_rows=args.n_rows,
        seed=args.seed,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )
    df.to_csv(output_path, index=False)

    print(f"Wrote synthetic surrogate dataset: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Schema columns: {', '.join(df.columns)}")


if __name__ == "__main__":
    main()
