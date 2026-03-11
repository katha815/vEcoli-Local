#!/usr/bin/env python3
"""Record final observed generation for each variant/seed lineage.

This scans the success output partitioned as:
  out/<project>/success/experiment_id=<project>/variant=<v>/lineage_seed=<s>/generation=<g>/...

and writes one row per (variant, lineage_seed) with the maximum generation found.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd

VARIANT_RE = re.compile(r"^variant=(\d+)$")
SEED_RE = re.compile(r"^lineage_seed=(\d+)$")
GEN_RE = re.compile(r"^generation=(\d+)$")


def parse_int(name: str, pattern: re.Pattern[str], label: str) -> int:
    match = pattern.match(name)
    if not match:
        raise ValueError(f"Expected {label} directory, got: {name}")
    return int(match.group(1))


def infer_variant_key(project: str) -> str:
    project_lower = project.lower()
    if "media" in project_lower or "condition" in project_lower:
        return "condition"
    return "gene_knockout"


def build_variant_label(variant_id: int, variant_metadata: dict[str, object]) -> str:
    variant_info = variant_metadata.get(str(variant_id), f"variant_{variant_id}")

    if isinstance(variant_info, str):
        return variant_info

    if isinstance(variant_info, dict):
        genes = variant_info.get("genes_to_knockout", [])
        if genes:
            return f"KO: {', '.join(genes)}"
        return str(variant_info.get("condition", variant_info))

    return f"variant_{variant_id}"


def load_variant_metadata(
    project: str, out_root: Path, variant_key: str | None
) -> dict[str, object]:
    metadata_path = out_root / project / "variant_sim_data" / "metadata.json"
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    key = variant_key or infer_variant_key(project)
    variant_metadata = metadata.get(key, {})
    return variant_metadata if isinstance(variant_metadata, dict) else {}


def collect_last_generations(
    project: str, out_root: Path, variant_key: str | None = None
) -> pd.DataFrame:
    base = out_root / project / "success" / f"experiment_id={project}"
    rows: list[dict[str, object]] = []
    variant_metadata = load_variant_metadata(project, out_root, variant_key)

    if not base.exists():
        raise FileNotFoundError(f"Success path not found: {base}")

    for variant_dir in sorted(base.glob("variant=*")):
        if not variant_dir.is_dir():
            continue
        variant = parse_int(variant_dir.name, VARIANT_RE, "variant")
        label = build_variant_label(variant, variant_metadata)

        for seed_dir in sorted(variant_dir.glob("lineage_seed=*")):
            if not seed_dir.is_dir():
                continue
            seed = parse_int(seed_dir.name, SEED_RE, "lineage_seed")

            generations = []
            for gen_dir in seed_dir.glob("generation=*"):
                if not gen_dir.is_dir():
                    continue
                generations.append(parse_int(gen_dir.name, GEN_RE, "generation"))

            if not generations:
                # Keep a row for visibility if lineage folder exists but has no generation folders.
                rows.append(
                    {
                        "project": project,
                        "variant": variant,
                        "label": label,
                        "lineage_seed": seed,
                        "last_generation": pd.NA,
                    }
                )
                continue

            last_generation = max(generations)
            rows.append(
                {
                    "project": project,
                    "variant": variant,
                    "label": label,
                    "lineage_seed": seed,
                    "last_generation": last_generation,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["variant", "lineage_seed"]).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default="gene_ko_441imported_2seeds",
        help="Project folder name under out/ (default: gene_ko_441imported_2seeds)",
    )
    parser.add_argument(
        "--out-root",
        default="/user/home/il22158/work/vEcoli/out",
        help="Root output directory containing project folders",
    )
    parser.add_argument(
        "--output-dir",
        default="/user/home/il22158/work/vEcoli/reading/results/success",
        help="Directory to store results CSV",
    )
    parser.add_argument(
        "--variant-key",
        default=None,
        help="Optional metadata key for labels (e.g., condition, gene_knockout)",
    )
    args = parser.parse_args()

    df = collect_last_generations(
        project=args.project,
        out_root=Path(args.out_root),
        variant_key=args.variant_key,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.project}_stop_generation.csv"
    df.to_csv(out_csv, index=False)

    print(f"Wrote {len(df)} rows to {out_csv}")
    if not df.empty:
        print(
            df["last_generation"]
            .value_counts(dropna=False)
            .sort_index()
            .rename("count")
        )


if __name__ == "__main__":
    main()
