"""
Classify genes by operon structure (single-gene vs multi-gene operons)
Output format matches metabolic_genes classifier
"""

import pickle
import pandas as pd

# Load sim_data
kb_path = "/user/home/il22158/work/vEcoli/out/all_media_conditions_test/parca/kb/simData.cPickle"
print("Loading sim_data...")
with open(kb_path, "rb") as f:
    sim_data = pickle.load(f)

# Load FULL genes.tsv
genes_df_full = pd.read_csv(
    "/user/home/il22158/work/vEcoli/reconstruction/ecoli/flat/genes.tsv",
    sep="\t",
    comment="#",
)

# # shortened for demo
# genes_df_full = genes_df_full[10:20]

print(f"Loaded {len(genes_df_full)} genes from genes.tsv")
print("Analyzing operon structure...\n")

# Build gene → operon size mapping
gene_to_operon_size = {}
gene_to_tu_id = {}
gene_to_other_genes = {}

for cistron in sim_data.process.transcription.cistron_data.struct_array:
    gene_id = cistron["gene_id"]
    cistron_id = cistron["id"]

    # Get TU containing this cistron
    rna_idxs = sim_data.process.transcription.cistron_id_to_rna_indexes(cistron_id)

    # Convert to Python int
    if hasattr(rna_idxs, "__iter__"):
        rna_idx = int(list(rna_idxs)[0])
    else:
        rna_idx = int(rna_idxs)

    # Get TU info
    rna = sim_data.process.transcription.rna_data.struct_array[rna_idx]
    rna_id = rna["id"]

    # Find all genes in this TU
    cistrons_in_tu = sim_data.process.transcription.rna_id_to_cistron_indexes(rna_id)
    operon_size = len(cistrons_in_tu)

    # Get other genes in operon
    other_genes = []
    for cidx in cistrons_in_tu:
        other_cistron = sim_data.process.transcription.cistron_data.struct_array[
            int(cidx)
        ]
        other_gene_id = other_cistron["gene_id"]
        if other_gene_id != gene_id:
            other_genes.append(other_gene_id)

    gene_to_operon_size[gene_id] = operon_size
    gene_to_tu_id[gene_id] = rna_id
    gene_to_other_genes[gene_id] = other_genes

# Mark genes as single-gene operon or not
genes_df_full["is_single_gene_operon"] = genes_df_full["id"].map(
    lambda gid: gene_to_operon_size.get(gid, None) == 1
)

# Add additional info columns
genes_df_full["operon_size"] = genes_df_full["id"].map(
    lambda gid: gene_to_operon_size.get(gid, None)
)
genes_df_full["tu_id"] = genes_df_full["id"].map(
    lambda gid: gene_to_tu_id.get(gid, None)
)
genes_df_full["other_genes_in_operon"] = genes_df_full["id"].map(
    lambda gid: ",".join(gene_to_other_genes.get(gid, []))
)

# Save
path = "/user/home/il22158/work/vEcoli/reading/results/"
output_file = f"{path}operon_classification.tsv"
genes_df_full.to_csv(output_file, sep="\t", index=False)

print(f"✓ Saved to {output_file}")
print(f"\nTotal genes: {len(genes_df_full)}")
print(
    f"Single-gene operons: {genes_df_full['is_single_gene_operon'].sum()} ({100 * genes_df_full['is_single_gene_operon'].sum() / len(genes_df_full):.1f}%)"
)
print(
    f"Multi-gene operons: {(~genes_df_full['is_single_gene_operon']).sum()} ({100 * (~genes_df_full['is_single_gene_operon']).sum() / len(genes_df_full):.1f}%)"
)

# Operon size distribution
print("\nOperon size distribution:")
size_counts = genes_df_full["operon_size"].value_counts().sort_index()
for size, count in size_counts.head(10).items():
    if pd.notna(size):
        print(
            f"  {int(size)} gene(s): {count} genes ({100 * count / len(genes_df_full):.1f}%)"
        )

print(f"\nLargest operon: {genes_df_full['operon_size'].max():.0f} genes")

# Test: reload and verify
test_df = pd.read_csv(output_file, sep="\t")
print(
    f"\n✓ Verified: Reloaded {len(test_df)} genes (same as original {len(genes_df_full)})"
)
print(f"  Single-gene operons: {test_df['is_single_gene_operon'].sum()}")
print(f"  Multi-gene operons: {(~test_df['is_single_gene_operon']).sum()}")

print("\n=== Sample single-gene operons ===")
print(
    test_df[test_df["is_single_gene_operon"]][
        ["id", "symbol", "operon_size", "tu_id"]
    ].head(5)
)

print("\n=== Sample multi-gene operons ===")
multi_gene = test_df[~test_df["is_single_gene_operon"]].sort_values(
    "operon_size", ascending=False
)
print(multi_gene[["id", "symbol", "operon_size", "other_genes_in_operon"]].head(5))

# Check our test genes
print("\n=== Your test genes ===")
for gene_id in ["EG10527", "EG10528"]:
    gene_row = test_df[test_df["id"] == gene_id]
    if not gene_row.empty:
        print(f"\n{gene_id} ({gene_row['symbol'].values[0]}):")
        print(f"  Single-gene operon: {gene_row['is_single_gene_operon'].values[0]}")
        print(f"  Operon size: {gene_row['operon_size'].values[0]:.0f}")
        print(f"  TU: {gene_row['tu_id'].values[0]}")
        if not gene_row["is_single_gene_operon"].values[0]:
            print(f"  Other genes: {gene_row['other_genes_in_operon'].values[0]}")
