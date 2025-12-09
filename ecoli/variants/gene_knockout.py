from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    # This block ONLY runs during type checking (mypy, pylance, etc.)
    # It does NOT run during actual execution
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


def apply_variant(
    sim_data: "SimulationDataEcoli", params: dict[str, Any]
) -> "SimulationDataEcoli":
    """
    Knockout genes by setting their RNA synthesis probability to 0.

    Args:
        sim_data: Simulation data to modify
        params: Parameter dictionary with format:
            {
                "genes_to_knockout": list of gene IDs (e.g., ["EG10367_RNA", "EG10544_RNA"])
            }
    """
    # Initialize genetic_perturbations dict if it doesn't exist
    if not hasattr(sim_data, "genetic_perturbations"):
        sim_data.genetic_perturbations = {}

    # Set synthesis probability to 0 for knocked out genes
    for gene_id in params["genes_to_knockout"]:
        sim_data.genetic_perturbations[gene_id] = 0.0

    return sim_data
