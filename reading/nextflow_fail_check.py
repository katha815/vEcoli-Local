# This code checks the simulation output's simdata tp ensure it doesn't fail at a
# nextflow level
from pathlib import Path

c = 0
project_suffix = ["s1", "errored", "errored_no_overlap", "errored_overlap_s2"]
search_dir = Path("/user/home/il22158/work/vEcoli/out")

for project in sorted(search_dir.iterdir()):
    if not project.is_dir() or not project.name.endswith(tuple(project_suffix)):
        continue

    simdata = project / "variant_sim_data"
    valid_files = [
        path.name
        for path in simdata.glob("*.cPickle")
        if path.is_file() and path.stat().st_size > 0
    ]
    if valid_files:
        continue
    else:
        c += 1
        print(f"{project.name}: missing non-empty .cPickle file")

print(f"Total missing non-empty .cPickle files: {c}")
