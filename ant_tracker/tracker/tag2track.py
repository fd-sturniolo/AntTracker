import argparse
import json
import numpy as np
from packaging.version import Version
from pathlib import Path

from .blob import Blob
from .info import TracksInfo
from .parameters import SegmenterParameters, TrackerParameters
from .track import Track

parser = argparse.ArgumentParser(description="Convert .tag file to .trk file format")
parser.add_argument('file')
parser.add_argument('--output', '-o', type=str, default=None,
                    help="Archivo de salida")
args = parser.parse_args()
file: str = args.file
output = args.output

if not file.endswith('.tag'):
    raise ValueError('Must be a .tag file!')
print(f"Converting {file}")
filename = file[:-4]
if output is None:
    output = Path(f'{filename}-gt.trk' if output is None else output)
print(f"Output will be {output}")
video_filename = f"{filename}.mp4"

with open(f'{filename}.tag', 'r') as f:
    manual_track = json.load(f)

ants = manual_track["ants"]
shape = manual_track["videoShape"]
closed_tracks = []
for ant in ants:
    i = ant["id"]
    blobs = dict()
    areas = ant["areasByFrame"]
    for area_by_frame in areas:
        frame, area = area_by_frame["frame"], area_by_frame["area"]
        mask = np.zeros(shape, dtype='uint8')
        mask[tuple(area)] = 1
        blob = Blob(mask=mask)
        blobs[frame] = blob
    closed_tracks.append(Track(i - 1, dict(sorted(blobs.items())), force_load_to=ant["loaded"]))

info = TracksInfo(
    video_path=video_filename,
    tracks=sorted(closed_tracks, key=lambda t: t.id),
    segmenter_version=Version("0"),
    segmenter_parameters=SegmenterParameters.mock(),
    tracker_version=Version("0"),
    tracker_parameters=TrackerParameters.mock(),
)

print("Done")
info.save(output)
print(f"Saved to {output}")
