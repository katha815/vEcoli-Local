# """Method 1 with save"""
# import pandas as pd
# import requests
# from time import sleep

# genes_df = pd.read_csv('/user/home/il22158/work/vEcoli/reconstruction/ecoli/flat/genes.tsv',
#                        sep='\t', comment='#')

# def check_metabolic_biocyc(gene_id):
#     """Query BioCyc API to check if gene is metabolic"""
#     url = f"https://websvc.biocyc.org/getxml?ECOLI:{gene_id}"
#     try:
#         response = requests.get(url, timeout=10)
#         if response.status_code == 200:
#             text = response.text.lower()
#             metabolic_keywords = ['metabol']
#             is_metabolic = any(kw in text for kw in metabolic_keywords)
#             return is_metabolic
#         return None
#     except Exception as e:
#         return None

# # # Check genes (shortened for demo)
# # genes_df1 = genes_df.head(10).copy()
# genes_df1 = genes_df.copy()
# print(f"Checking {len(genes_df1)} genes from BioCyc (demo)...")
# results = []

# for idx, gene in genes_df1.iterrows():
#     is_metabolic = check_metabolic_biocyc(gene['id'])
#     results.append(is_metabolic if is_metabolic is not None else False)

#     if (idx + 1) % 100 == 0:
#         print(f"Processed {idx + 1}/{len(genes_df1)} genes...")
#     sleep(0.1)

# genes_df1['is_metabolic_biocyc'] = results

# # Merge results back to full dataframe
# genes_df['is_metabolic_biocyc'] = False  # Initialize all as False
# genes_df.loc[genes_df['id'].isin(genes_df1['id']), 'is_metabolic_biocyc'] = genes_df1['is_metabolic_biocyc'].values

# # Save
# path = '/user/home/il22158/work/vEcoli/reading/results/'
# genes_df.to_csv(f'{path}metabolic_genes_method1.tsv', sep='\t', index=False)

# print(f"\n✓ Saved to {path}metabolic_genes_method1.tsv")
# print(f"Total: {len(genes_df)} genes | Metabolic: {genes_df['is_metabolic_biocyc'].sum()}")

# # Verify reload
# test_df = pd.read_csv(f'{path}metabolic_genes_method1.tsv', sep='\t')
# print(f"✓ Verified reload: {len(test_df)} genes, {test_df['is_metabolic_biocyc'].sum()} metabolic")
# print("\nSample metabolic genes:")
# print(test_df[test_df['is_metabolic_biocyc']][['id', 'symbol']].head(5))

"""Method 2 save - all genes marked"""

import pickle
import pandas as pd
import ast

# Load data
kb_path = "/user/home/il22158/work/vEcoli/out/all_media_conditions_test/parca/kb/simData.cPickle"
with open(kb_path, "rb") as f:
    sim_data = pickle.load(f)

# Load FULL genes.tsv (not shortened)
genes_df_full = pd.read_csv(
    "/user/home/il22158/work/vEcoli/reconstruction/ecoli/flat/genes.tsv",
    sep="\t",
    comment="#",
)
# # shortened for demo
# genes_df_full = genes_df_full[10:20]
rnas_df = pd.read_csv(
    "/user/home/il22158/work/vEcoli/reconstruction/ecoli/flat/rnas.tsv",
    sep="\t",
    comment="#",
)

# Build monomer to gene mapping
rnas_df["monomer_ids_parsed"] = rnas_df["monomer_ids"].apply(ast.literal_eval)
monomer_to_gene = {}
for _, row in rnas_df.iterrows():
    for monomer in row["monomer_ids_parsed"]:
        monomer_to_gene[monomer] = row["gene_id"]

# Get all metabolic genes
all_catalyst_monomers = set(
    [cat.split("[")[0] for cat in sim_data.process.metabolism.catalyst_ids]
)
all_metabolic_gene_ids = set(
    [monomer_to_gene[mono] for mono in all_catalyst_monomers if mono in monomer_to_gene]
)

# Mark ALL genes as metabolic or not
genes_df_full["is_metabolic"] = genes_df_full["id"].isin(all_metabolic_gene_ids)

# Save
path = "/user/home/il22158/work/vEcoli/reading/results/"
genes_df_full.to_csv(f"{path}metabolic_genes_method2.tsv", sep="\t", index=False)
print(f"Saved Method 2 results to {path}metabolic_genes_method2.tsv")
print(f"Total genes: {len(genes_df_full)}")
print(
    f"Metabolic genes: {genes_df_full['is_metabolic'].sum()} ({100 * genes_df_full['is_metabolic'].sum() / len(genes_df_full):.1f}%)"
)
print(f"Non-metabolic genes: {(~genes_df_full['is_metabolic']).sum()}")
print(
    f"Catalysts not mapped: {len([m for m in all_catalyst_monomers if m not in monomer_to_gene])}"
)

# Test: reload and verify
test_df = pd.read_csv(f"{path}metabolic_genes_method2.tsv", sep="\t")
print(
    f"\n✓ Verified: Reloaded {len(test_df)} genes (same as original {len(genes_df_full)})"
)
print(f"  Metabolic: {test_df['is_metabolic'].sum()}")
print(f"  Non-metabolic: {(~test_df['is_metabolic']).sum()}")
print("\nSample metabolic genes:")
print(test_df[test_df["is_metabolic"]][["id", "symbol"]].head(5))
