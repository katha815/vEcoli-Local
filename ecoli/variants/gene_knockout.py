"""
Knockout expression of a gene

Modifies:
        sim_data.process.transcription.rna_synth_prob
        sim_data.process.transcription.rna_expression
        sim_data.process.transcription.exp_free
        sim_data.process.transcription.exp_ppgpp
        sim_data.process.transcription.attenuation_basal_prob_adjustments
        sim_data.process.transcription_regulation.basal_prob
        sim_data.process.transcription_regulation.delta_prob

Expected variant indices (depends on length of sim_data.process.transcription.rna_data):
        0: control
        1-4692: gene index to knockout
"""

CONTROL_OUTPUT = dict(shortName="control", desc="Control simulation")


def gene_knockout(sim_data, index):
    rna_data = sim_data.process.transcription.rna_data

    nGenes = len(rna_data)
    nConditions = nGenes + 1

    if index % nConditions == 0:
        return CONTROL_OUTPUT, sim_data

    geneIndex = (index - 1) % nConditions
    factor = 0  # Knockout expression
    sim_data.adjust_final_expression([geneIndex], [factor])
    geneID = rna_data["id"][geneIndex]

    return dict(
        shortName="{}_KO".format(geneID), desc="Complete knockout of {}.".format(geneID)
    ), sim_data


def apply_variant(sim_data, params):
    """Variant entrypoint used by vEcoli `create_variants` runner.

    Expects `params` to contain a key `genes_to_knockout` which is a list
    of gene identifiers. Each gene identifier may be a string or a one-item
    list (configs often provide nested lists). For each gene ID, find the
    corresponding cistron(s) and the TU (RNA) index(es) that contain it,
    then call `sim_data.adjust_final_expression` to set expression to 0.
    This preserves the original semantics of the old implementation
    (immediate renormalization via `adjust_final_expression`).
    """
    # Normalize params shape: support nested lists like [["EG10109"]]
    genes_param = params.get("genes_to_knockout", [])
    # Flatten one level if necessary
    flat_genes = []
    for item in genes_param:
        if isinstance(item, (list, tuple)) and len(item) == 1:
            flat_genes.append(item[0])
        else:
            flat_genes.append(item)

    # Collect TU indices to knock out. `adjust_final_expression` expects
    # integer indices into `sim_data.process.transcription.rna_data` (TU index).
    tu_indices = []
    transcription = sim_data.process.transcription
    cistron_array = transcription.cistron_data.struct_array

    for gene_id in flat_genes:
        if not isinstance(gene_id, str):
            print(f"Warning: expected gene ID string, got {gene_id!r}; skipping")
            continue

        # Find cistron(s) for this gene_id
        found = False
        for cistron in cistron_array:
            if cistron["gene_id"] == gene_id:
                cistron_id = cistron["id"]
                rna_idxs = transcription.cistron_id_to_rna_indexes(cistron_id)
                if hasattr(rna_idxs, "__iter__"):
                    for i in rna_idxs:
                        tu_indices.append(int(i))
                else:
                    tu_indices.append(int(rna_idxs))
                found = True
                break

        if not found:
            print(f"Warning: gene ID not found in cistron data: {gene_id}")

    # Remove duplicates while preserving order
    seen = set()
    unique_tu_indices = []
    for i in tu_indices:
        if i not in seen:
            seen.add(i)
            unique_tu_indices.append(i)

    if len(unique_tu_indices) > 0:
        # Set all matched TU indices to factor 0 (complete knockout)
        factors = [0.0] * len(unique_tu_indices)
        sim_data.adjust_final_expression(unique_tu_indices, factors)

    return sim_data
