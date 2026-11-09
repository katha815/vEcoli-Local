This launcher creates and submits gene-knockout or wildtype jobs for selected media conditions.
It supports dry runs, custom gene lists, multiple media, generation windows for growth-rate extraction,
and single-job batching for multiple knockouts in one Slurm submission.

Usage:
    python job/write_for_new_job.py

For each launch, a run ID is generated and saved with the launcher command, submission manifest, and per-job config files under:
    job/generated_runs/launcher/<run-id>/

Previous jobs `launcher_command.txt` and `submission_manifest.txt` can be used as reference examples for new submissions.

Useful options:
    python job/write_for_new_job.py --dry-run
    python job/write_for_new_job.py --single_job
    python job/write_for_new_job.py --simulation-type wildtype --media with_aa acetate

Default gene list: 
    `reading/imported/Single_KO_RNA_names.txt`.