# %%
import pandas as pd

path = "/user/home/il22158/work/vEcoli/reading/results/knockout_experiment/metabolic_genes_method2.tsv"
with open(path) as f:
    df = pd.read_csv(f, sep="\t")
print(df.head())

# %%
is_metabolic = df["is_metabolic"]
sum = is_metabolic.eq(True).sum()
print(f"Metabolic genes: ({sum})")
# %%
