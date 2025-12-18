#!/usr/bin/env python3
"""
Gene Activity Screen for vEcoli Simulations

Screen gene activity (mRNA and protein expression) across multiple variants.
Tests if genes are transcribed and translated during simulation.
"""

import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Add reading directory to path
sys.path.insert(0, "/user/home/il22158/work/vEcoli/reading")
from gene_expression_trace import track_knockout_dynamics


def screen_gene_activity_multi_variant(
    gene_list, project, variants, generations, output_dir, downsample_sec=20
):
    """
    Screen gene activity across multiple variants. Saves separate CSV per variant.

    Args:
        gene_list: List of gene IDs to test
        project: Project folder name
        variants: List of variant IDs (e.g., [0, 1, 2, 3, 4])
        generations: List of generation numbers
        output_dir: Output directory for results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for variant in variants:
        print(f"\n{'=' * 80}\nVariant {variant}\n{'=' * 80}")
        variant_results = []

        for i, gene_id in enumerate(gene_list, 1):
            print(f"[{i}/{len(gene_list)}] {gene_id}...", end=" ")

            try:
                result = track_knockout_dynamics(
                    gene_id=gene_id,
                    project_folder=project,
                    variants=[variant],
                    generations=generations,
                    figsize=(16, 10),
                    plot=True,
                    save=False,
                    downsample_sec=downsample_sec,
                )
                from gene_expression_trace import save_results

                if result is None:
                    print("not found")
                    variant_results.append(
                        {
                            "gene_id": gene_id,
                            "variant": variant,
                            "status": "not_found",
                            "all_transcription_success": False,
                            "all_translation_success": False,
                            "max_mrna": 0,
                            "max_protein": 0,
                            "mean_mrna": 0,
                            "mean_protein": 0,
                        }
                    )
                    continue
                # Use save_results to get summary statistics for total initiations
                summary_df = save_results(
                    result,
                    gene_id,
                    project,
                    [variant],
                    save_dir=output_dir,
                    save_pic=True,
                    save_csv=True,
                )
                total_tc = (
                    summary_df["total_tc_initiations"].sum()
                    if "total_tc_initiations" in summary_df
                    else 0
                )
                total_tl = (
                    summary_df["total_tl_initiations"].sum()
                    if "total_tl_initiations" in summary_df
                    else 0
                )

                # Restore mean/max calculation from concatenated time series
                all_mrna, all_protein = [], []
                for _, data in result.items():
                    all_mrna.extend(data["mRNA"])
                    all_protein.extend(data["protein"])
                max_mrna = float(np.max(all_mrna)) if all_mrna else 0
                max_protein = float(np.max(all_protein)) if all_protein else 0
                mean_mrna = float(np.mean(all_mrna)) if all_mrna else 0
                mean_protein = float(np.mean(all_protein)) if all_protein else 0

                variant_results.append(
                    {
                        "gene_id": gene_id,
                        "variant": variant,
                        "status": "analyzed",
                        "all_transcription_success": total_tc,
                        "all_translation_success": total_tl,
                        "max_mrna": max_mrna,
                        "max_protein": max_protein,
                        "mean_mrna": mean_mrna,
                        "mean_protein": mean_protein,
                    }
                )
                print(
                    f"Trancripted units over {generations} generation: {total_tc} \
                      Translation units over {generations} generation: {total_tl}"
                )
            except Exception as e:
                print(f"error: {str(e)[:50]}")
                variant_results.append(
                    {
                        "gene_id": gene_id,
                        "variant": variant,
                        "status": "error",
                        "all_transcription_success": False,
                        "all_translation_success": False,
                        "max_mrna": 0,
                        "max_protein": 0,
                        "mean_mrna": 0,
                        "mean_protein": 0,
                    }
                )

        # Save this variant's results to separate file
        variant_df = pd.DataFrame(variant_results)
        output_file = (
            f"{output_dir}/{project}_variant{variant}_{len(gene_list)}genes.csv"
        )
        variant_df.to_csv(output_file, index=False)

        n_mrna = (variant_df["all_transcription_success"] > 0).sum()
        n_protein = (variant_df["all_translation_success"] > 0).sum()
        print(f"\nVariant {variant} Summary:")
        print(
            f"  Genes that successully transcript: {n_mrna}({100 * n_mrna / len(variant_df):.1f}%)"
        )
        print(
            f"  Genes that successully translate: {n_protein}({100 * n_protein / len(variant_df):.1f}%)"
        )
        print(f"  ✓ Saved: {output_file}")
        import sys

        sys.stdout.flush()

    print(f"\n{'=' * 80}\nAll variants complete!\n{'=' * 80}")
    import sys

    sys.stdout.flush()


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Screen gene activity across variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Test 2 genes (default)
    python gene_screen.py
    
    # Test 50 genes
    python gene_screen.py --subset 50
    
    # Test ALL 4747 genes
    python gene_screen.py --subset 0
        """,
    )

    parser.add_argument(
        "--generations",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Generations to simulate (default: 1-8)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="all_media_conditions1",
        help="Project folder name (default: all_media_conditions1)",
    )
    parser.add_argument(
        "--variants",
        "-v",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Variant IDs to test (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--gene-list",
        type=str,
        default="/user/home/il22158/work/vEcoli/reading/results/knockout_experiment/knocked_out_genes.txt",
        help="Path to a file with gene IDs to test, currently set to .../knocked_out_genes.txt",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=-1,
        help="Number of genes to test (default: -1 for EG10003 and EG10030). Use 0 for all genes \
                        and positive integers for N of gene list.",
    )
    parser.add_argument(
        "--downsample_sec",
        "-d",
        type=int,
        default=10,
        help="Downsampling factor in seconds for gene_expression_trace (default: 10)",
    )
    args = parser.parse_args()

    # Load genes
    genes_df = pd.read_csv(
        "/user/home/il22158/work/vEcoli/reconstruction/ecoli/flat/genes.tsv",
        sep="\t",
        comment="#",
    )
    if args.gene_list:
        # Read gene IDs from file (supporting both Python list and one-per-line)
        with open(args.gene_list) as f:
            content = f.read()
            if content.strip().startswith("["):  # Python list
                import ast

                test_genes = ast.literal_eval(content)
                # Flatten if nested
                test_genes = [
                    g
                    for sub in test_genes
                    for g in (sub if isinstance(sub, list) else [sub])
                ]
            else:  # One per line
                test_genes = [
                    line.strip() for line in content.splitlines() if line.strip()
                ]
        print(f"Testing custom gene list: {test_genes}")
    elif args.subset == 0:
        test_genes = genes_df["id"].tolist()
        print(f"Testing ALL {len(test_genes)} genes across 5 variants")
    elif args.subset == -1:
        test_genes = ["EG10003", "EG10030"]  # Example specific genes
        print(f"Testing specific genes: {test_genes}")
    else:
        test_genes = genes_df["id"].head(args.subset).tolist()
        print(f"Testing {args.subset} genes across {len(args.variants)} variants")

    print(f"Total tests: {len(test_genes) * len(args.variants)}\n")

    # Run the screen
    screen_gene_activity_multi_variant(
        gene_list=test_genes,
        project=args.project,
        variants=args.variants,
        generations=args.generations,
        output_dir="/user/home/il22158/work/vEcoli/reading/results/gene_activity_screen",
        downsample_sec=args.downsample_sec,
    )


if __name__ == "__main__":
    main()
