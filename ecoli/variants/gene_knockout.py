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

Logic:
    gene knockout =
        identify cistron with gene_id
        get its TUs
        disable those TUs only
"""

CONTROL_OUTPUT = dict(shortName="control", desc="Control simulation")


def _as_single_item_list(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _collect_tu_indexes_for_gene(transcription, gene_id):
    tu_indexes = []
    cistron_array = transcription.cistron_data.struct_array

    for cistron in cistron_array:
        if cistron["gene_id"] == gene_id:
            cistron_id = cistron["id"]
            rna_idxs = transcription.cistron_id_to_rna_indexes(cistron_id)
            if hasattr(rna_idxs, "__iter__"):
                for i in rna_idxs:
                    tu_indexes.append(int(i))
            else:
                tu_indexes.append(int(rna_idxs))
            return tu_indexes

    return []


def _collect_tu_indexes_for_rna_id(transcription, rna_id):
    for index, rna in enumerate(transcription.rna_data.struct_array):
        if rna["id"] == rna_id:
            return [int(index)]
    return []


def gene_knockout(sim_data, index):  # Imported function from wcEcoli
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


def apply_variant(sim_data, params):  # Transplanted function
    """Variant entrypoint used by vEcoli `create_variants` runner.

    Expects `params` to contain a key `genes_to_knockout`, which may contain
    either gene IDs or TU/RNA IDs. Each entry may be a string or a one-item
    list (configs often provide nested lists). Gene IDs are resolved through
    cistron data; TU/RNA IDs are resolved directly from `rna_data`.

    The resolved TU indexes are passed to `sim_data.adjust_final_expression`
    to preserve the original knockout semantics.
    """
    # Normalize params shape: support nested lists like [["EG10109"]]
    genes_param = params.get("genes_to_knockout", [])
    # Flatten one level if necessary
    flat_genes = []
    for item in genes_param:
        flat_genes.append(_as_single_item_list(item))

    # Collect TU indices to knock out. `adjust_final_expression` expects
    # integer indices into `sim_data.process.transcription.rna_data` (TU index).
    tu_indices = []
    transcription = sim_data.process.transcription

    for knockout_id in flat_genes:
        if not isinstance(knockout_id, str):
            print(
                f"Warning: expected gene or TU ID string, got {knockout_id!r}; skipping"
            )
            continue

        gene_tu_indexes = _collect_tu_indexes_for_gene(transcription, knockout_id)
        if gene_tu_indexes:
            tu_indices.extend(gene_tu_indexes)
            continue

        rna_tu_indexes = _collect_tu_indexes_for_rna_id(transcription, knockout_id)
        if rna_tu_indexes:
            tu_indices.extend(rna_tu_indexes)
            continue

        print(f"Warning: gene or TU ID not found in transcription data: {knockout_id}")

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
