import os
from pathlib import Path

os.environ['PATH'] = str((Path(__file__).parent.parent.parent / "lib").resolve()) + os.pathsep + os.environ['PATH']
