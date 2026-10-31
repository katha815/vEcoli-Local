#!/usr/bin/env python3
"""Generate and submit one Slurm job per gene/media knockout run,
and also performs growth-rate extraction for each resulting project folder,
with designed generation windows.

Each submitted Slurm job:
1. writes a temporary config derived from the base template,
2. runs the vEcoli workflow for that specific gene/media pair, and
3. runs growth-rate extraction for the resulting project folder.

This keeps the launcher small, reproducible, and easy to parallelize without
manually maintaining hundreds of JSON files.
"""

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


ROOT_DIR = Path("/user/home/il22158/work/vEcoli")
DEFAULT_TEMPLATE_CONFIG = ROOT_DIR / "configs/N_gene_knockout_KO_sample.json"
DEFAULT_GENE_LIST = ROOT_DIR / "reading/imported/Single_KO_RNA_names.txt"
DEFAULT_MEDIA_WINDOWS = ROOT_DIR / "job/tdi_summary_40trial_5media_composite_tdi.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "job/generated_runs"
DEFAULT_SLURM_LOG_DIR = ROOT_DIR.parent / "slurm_logs"
DEFAULT_LINEAGE_SEEDS = [100, 101]
DEFAULT_MEDIA = ["with_aa", "acetate", "no_ox", "succinate"]
TOTAL_MEDIA = ["with_aa", "acetate", "no_ox", "succinate", "basal"]


class JobSpec:
    def __init__(
        self,
        gene: Optional[str],
        media: str,
        generation_start: int,
        generation_end: int,
    ):
        self.gene = gene
        self.media = media
        self.generation_start = generation_start
        self.generation_end = generation_end

    @property
    def project(self) -> str:
        if self.gene is None:
            return f"wildtype_{self.media}"
        return f"gene_knockout_{self.gene}_{self.media}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Create one simulation job per gene/media pair (knockout) or per medium "
            "(wildtype) and submit it to Slurm."
        ),
        epilog=(
            "The launcher reads genes from the gene list file and uses the media\n"
            "window CSV to map each selected medium to a generation range for\n"
            "growth-rate extraction. Use --simulation-type wildtype to create\n"
            "one wildtype job per selected medium, without gene knockouts. Pass\n"
            "multiple media values after --media to include several conditions.\n\n"
            "Test run that only produces configuration and submission code:\n"
            "  python job/write_for_new_job.py --dry-run\n\n"
            "Test run with job submission:\n"
            "  printf 'EG10001\\n' > /tmp/one_gene_ko.txt && python job/write_for_new_job.py --genes-file /tmp/one_gene_ko.txt --media with_aa\n\n"
            "Wildtype dry run:\n"
            "  python job/write_for_new_job.py --simulation-type wildtype --media no_ox acetate succinate with_aa --dry-run\n\n"
            "Default full run:\n"
            "  python job/write_for_new_job.py\n\n"
            "Run growth-rate extraction after each simulation:\n"
            "  python job/write_for_new_job.py --extract-growth-rate\n\n"
            "Where to find results:\n"
            "  Slurm stdout/stderr: /user/home/il22158/work/slurm_logs/<job>.%j.out\n"
            "  Launcher artifacts: /user/home/il22158/work/vEcoli/job/generated_runs/launcher/<run-id>/\n"
            "  Workflow output: /user/home/il22158/work/vEcoli/out/<project>/nextflow/\n\n"
        ),
    )
    parser.add_argument(
        "--template-config",
        type=Path,
        default=DEFAULT_TEMPLATE_CONFIG,
        help=(
            "Base JSON config used as the template for each job. The launcher"
            " copies this file and sets experiment_id, condition (medium),"
            " and generations. In knockout mode it also sets one knockout"
            " gene in variants."
        ),
    )
    parser.add_argument(
        "--genes-file",
        type=Path,
        default=DEFAULT_GENE_LIST,
        help=(
            "Text file containing one gene ID per line. The default is the"
            " full Single_KO_RNA_names.txt list for knockout simulations; use"
            " a custom file to test one gene or a smaller subset. Not used"
            " for wildtype simulations."
        ),
    )
    parser.add_argument(
        "--simulation-type",
        choices=["knockout", "wildtype"],
        default="knockout",
        help=(
            "Simulation type. 'knockout' (default) creates one gene-knockout"
            " job per gene and medium; 'wildtype' creates one unmodified job"
            " per medium."
        ),
    )
    parser.add_argument(
        "--media-windows",
        type=Path,
        default=DEFAULT_MEDIA_WINDOWS,
        help=(
            "CSV file with Media and Proposed generation window columns. The"
            " Media value must match --media, and the generation window is used"
            " for growth_rate_extract.py."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where generated configs, manifests, and Slurm scripts"
            " are written. This is a launcher workspace, not the simulation"
            " output directory."
        ),
    )
    parser.add_argument(
        "--slurm-log-dir",
        type=Path,
        default=DEFAULT_SLURM_LOG_DIR,
        help=(
            "Directory for Slurm stdout/stderr files. The default points to"
            " /user/home/il22158/work/slurm_logs."
        ),
    )
    parser.add_argument(
        "--lineage-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_LINEAGE_SEEDS,
        help=(
            "Lineage seeds for optional growth_rate_extract.py. Used only"
            " with --extract-growth-rate. Default: 100 101."
        ),
    )
    parser.add_argument(
        "--media",
        nargs="+",
        choices=TOTAL_MEDIA,
        default=DEFAULT_MEDIA,
        help=(
            "One or more media conditions to process (for example:"
            " --media no_ox acetate succinate with_aa). Each selected medium"
            " must appear in the media-windows CSV. Knockout mode creates a"
            " job per gene and medium; wildtype mode creates one job per"
            " medium. Each job sets its medium using the top-level 'condition'"
            " config option. The default is with_aa, acetate, no_ox, and succinate."
            " Choices: " + ", ".join(TOTAL_MEDIA)
        ),
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help=(
            "Generation count written into each generated config. This is the"
            " simulation length, not the extraction window. Default is 20."
        ),
    )
    parser.add_argument(
        "--account",
        default="emat024603",
        help=" Slurm account to use in the SBATCH header. Default is emat024603.",
    )
    parser.add_argument(
        "--partition",
        default="compute",
        help="Slurm partition to use in the SBATCH header. Default is compute.",
    )
    parser.add_argument(
        "--time",
        default="14-00:00:00",
        help="Slurm time limit to request for each job. Default is 14-00:00:00.",
    )
    parser.add_argument(
        "--mem",
        default="16G",
        help="Memory request for each job. Default is 16G.",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=4,
        help="CPU request for each job. Default is 4.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write configs and Slurm scripts without submitting them.",
    )
    parser.add_argument(
        "--extract-growth-rate",
        action="store_true",
        help=(
            "Run growth_rate_extract.py after the simulation in each Slurm job."
            " By default, extraction is skipped so it can be run later."
        ),
    )
    return parser.parse_args()


def load_genes(genes_file: Path) -> List[str]:
    genes = []
    with genes_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            gene = line.strip()
            if gene and not gene.startswith("#"):
                genes.append(gene)
    return genes


def parse_generation_window(window_text: str) -> Tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", window_text)
    if not match:
        raise ValueError(f"Invalid generation window: {window_text!r}")
    generation_start = int(match.group(1))
    generation_end = int(match.group(2))
    if generation_end < generation_start:
        raise ValueError(f"Generation window is reversed: {window_text!r}")
    return generation_start, generation_end


def load_media_windows(
    media_windows_file: Path, selected_media: Set[str]
) -> Dict[str, Tuple[int, int]]:
    media_windows = {}
    with media_windows_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            media = row.get("Media", "").strip()
            if media not in selected_media:
                continue
            window_text = row.get("Proposed generation window", "").strip()
            if not media or not window_text:
                continue
            media_windows[media] = parse_generation_window(window_text)

    missing_media = sorted(selected_media.difference(media_windows))
    if missing_media:
        raise RuntimeError(
            "Missing generation windows for media: " + ", ".join(missing_media)
        )
    return media_windows


def make_job_spec(
    gene: str, media: str, media_windows: Dict[str, Tuple[int, int]]
) -> JobSpec:
    generation_start, generation_end = media_windows[media]
    return JobSpec(
        gene=gene,
        media=media,
        generation_start=generation_start,
        generation_end=generation_end,
    )


def build_config(
    template_config: Dict[str, object], job: JobSpec, generations: int
) -> Dict[str, object]:
    config = json.loads(json.dumps(template_config))
    config["experiment_id"] = job.project
    config["condition"] = job.media
    config["generations"] = generations
    if job.gene is None:
        config["variants"] = {}
        config["skip_baseline"] = False
    else:
        config["variants"] = {
            "gene_knockout": {
                "genes_to_knockout": {"value": [[job.gene]]},
            }
        }
    return config


def build_slurm_script(
    job: JobSpec,
    config_path: Path,
    args: argparse.Namespace,
    slurm_log_dir: Path,
) -> str:
    generation_args = " ".join(
        str(generation)
        for generation in range(job.generation_start, job.generation_end + 1)
    )
    growth_rate_extraction = ""
    if args.extract_growth_rate:
        growth_rate_extraction = f"""
python /user/home/il22158/work/vEcoli/reading/growth_rate_extract.py \\
    --all \\
    --projects {job.project} \\
    --save-timeseries \\
    --suffix {job.project} \\
    --lineage-seeds {" ".join(str(seed) for seed in args.lineage_seeds)} \\
    --generations {generation_args}
"""
    return f"""#!/bin/bash
#SBATCH --job-name={job.project}
#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --chdir=/user/home/il22158
#SBATCH --account={args.account}
#SBATCH --output={slurm_log_dir}/{job.project}.%j.out
#SBATCH --mem={args.mem}
#SBATCH --cpus-per-task={args.cpus_per_task}

set -euo pipefail

WORK_DIR="/user/home/il22158/work/vEcoli"
cd "$WORK_DIR" || exit 1

source "$WORK_DIR/.venv/bin/activate"

module load languages/java-sdk/22.0.2 openssh/9.7p1-uyheegq git
module list

nextflow -version

python runscripts/workflow.py --config {config_path}
{growth_rate_extraction}
"""


def submit_job(script_path: Path) -> None:
    subprocess.run(["sbatch", str(script_path)], check=True)


def main() -> None:
    args = parse_args()
    launcher_command = " ".join(shlex.quote(arg) for arg in sys.argv)
    launch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if not args.template_config.exists():
        raise FileNotFoundError(f"Template config not found: {args.template_config}")
    if args.simulation_type == "knockout" and not args.genes_file.exists():
        raise FileNotFoundError(f"Gene list not found: {args.genes_file}")
    if not args.media_windows.exists():
        raise FileNotFoundError(f"Media window CSV not found: {args.media_windows}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.slurm_log_dir.mkdir(parents=True, exist_ok=True)

    with args.template_config.open("r", encoding="utf-8") as handle:
        template_config = json.load(handle)

    genes = (
        load_genes(args.genes_file) if args.simulation_type == "knockout" else [None]
    )
    selected_media = set(args.media)
    media_windows = load_media_windows(args.media_windows, selected_media)

    run_dir = args.output_dir / "launcher" / launch_id
    configs_dir = run_dir / "configs"
    scripts_dir = run_dir / "slurm"
    run_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "launcher_command.txt").write_text(
        launcher_command + "\n", encoding="utf-8"
    )
    scripts_dir = run_dir / "slurm"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    submission_records = []
    script_paths = []
    for gene in genes:
        for media in args.media:
            job = make_job_spec(gene, media, media_windows)
            config = build_config(template_config, job, args.generations)

            config_path = configs_dir / f"{job.project}.json"
            config_path.write_text(
                json.dumps(config, indent=2) + "\n", encoding="utf-8"
            )

            script_path = scripts_dir / f"{job.project}.sh"
            slurm_script = build_slurm_script(
                job, config_path, args, args.slurm_log_dir
            )
            script_path.write_text(slurm_script, encoding="utf-8")

            submission_records.append(
                (
                    job.project,
                    f"sbatch {shlex.quote(str(script_path))}",
                    slurm_script,
                )
            )
            script_paths.append(script_path)

    manifest_path = run_dir / "submission_manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for project, submit_command, slurm_script in submission_records:
            handle.write(f"Job: {project}\n")
            handle.write(f"Submission command: {submit_command}\n")
            handle.write("Submitted Slurm script:\n")
            handle.write(slurm_script)
            handle.write("\n" + "=" * 80 + "\n\n")

    print(f"Wrote {len(script_paths)} job definitions to {manifest_path}")
    if args.dry_run:
        print(f"Dry run only; no jobs submitted. Scripts are under {scripts_dir}")
    else:
        for script_path in script_paths:
            submit_job(script_path)
        print(f"Submitted {len(script_paths)} Slurm jobs")


if __name__ == "__main__":
    main()
