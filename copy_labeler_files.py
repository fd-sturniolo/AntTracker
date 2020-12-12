import shutil
from pathlib import Path

antlabeler_files = set([p.name for p in Path("dist/AntLabeler/").glob("*")])
anttracker_files = set([p.name for p in Path("dist/AntTracker/").glob("*")])

for file in antlabeler_files - anttracker_files:
    from_file = Path(f"dist/AntLabeler/{file}")
    to_file = Path(f"dist/AntTracker/{file}")
    if from_file.is_file():
        shutil.copy(from_file, to_file)
    elif from_file.is_dir():
        shutil.copytree(from_file, to_file)
