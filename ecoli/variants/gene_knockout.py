from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


def apply_variant(
    sim_data: "SimulationDataEcoli", params: dict[str, Any]
) -> "SimulationDataEcoli":
    """
    Knockout genes by setting their transcription unit's synthesis probability to 0.
    Note: This knocks out the entire operon containing the gene.
    """
    if not hasattr(sim_data, "genetic_perturbations"):
        sim_data.genetic_perturbations = {}

    for gene_id in params["genes_to_knockout"]:
        # Find cistron for this gene
        for cistron in sim_data.process.transcription.cistron_data.struct_array:
            if cistron["gene_id"] == gene_id:
                cistron_id = cistron["id"]

                # Get TU (RNA) containing this cistron
                rna_idxs = sim_data.process.transcription.cistron_id_to_rna_indexes(
                    cistron_id
                )

                # Handle single value or array
                if hasattr(rna_idxs, "__iter__"):
                    rna_idx_list = [int(i) for i in rna_idxs]
                else:
                    rna_idx_list = [int(rna_idxs)]

                # Set synthesis probability to 0 for all TUs containing this gene
                for ridx in rna_idx_list:
                    rna = sim_data.process.transcription.rna_data.struct_array[ridx]
                    rna_id = rna["id"]  # This is like "TU00036[c]"
                    sim_data.genetic_perturbations[rna_id] = 0.0
                    print(f"Knocking out {gene_id}: {cistron_id} → TU {rna_id}")

                break  # Found the gene, move to next

    return sim_data
