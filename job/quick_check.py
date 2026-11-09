# %%
# Quick check the last launched run: does the gene list of all configurations match the input gene_list?
# Also there should be 462 genes overall.
import json
from pathlib import Path

last_run_config_path = "/user/home/il22158/work/vEcoli/job/generated_runs/launcher/20261106T154839Z/configs"
gene_list_path = (
    "/user/home/il22158/work/vEcoli/reading/imported/Single_KO_RNA_names.txt"
)

config_dir = Path(last_run_config_path)
gene_list = Path(gene_list_path)

# Gather all genes referenced in the generated config files.
config_genes = set()
config_files = sorted(config_dir.glob("*.json"))
if not config_files:
    raise FileNotFoundError(f"No JSON config files found in {config_dir}")

for config_path in config_files:
    with config_path.open() as f:
        config = json.load(f)

    value = (
        config.get("variants", {})
        .get("gene_knockout", {})
        .get("genes_to_knockout", {})
        .get("value", [])
    )
    if not value:
        raise ValueError(f"No genes_to_knockout found in {config_path}")

    for item in value:
        if isinstance(item, list):
            config_genes.update(item)
        elif isinstance(item, str):
            config_genes.add(item)
        else:
            raise TypeError(
                f"Unexpected gene format in {config_path}: {type(item).__name__}"
            )

with gene_list.open() as f:
    input_genes = {line.strip() for line in f if line.strip()}

missing_from_configs = sorted(input_genes - config_genes)
missing_from_input = sorted(config_genes - input_genes)

print(f"Checked {len(config_files)} config files")
print(f"Unique genes in configs: {len(config_genes)}")
print(f"Unique genes in input list: {len(input_genes)}")

if len(config_genes) != 462 or len(input_genes) != 462:
    raise AssertionError(
        f"Expected 462 genes overall, got {len(config_genes)} config genes and {len(input_genes)} input genes"
    )

if missing_from_configs or missing_from_input:
    raise AssertionError(
        "Gene mismatch detected: "
        f"missing_from_configs={missing_from_configs[:10]}, "
        f"missing_from_input={missing_from_input[:10]}"
    )

print(
    "PASS: all generated config genes match the input gene list and the total is 462."
)


# %%
