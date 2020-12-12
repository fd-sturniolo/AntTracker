import argparse
import pathlib
import pims

from .parameters import DohSegmenterParameters, LogWSegmenterParameters, TrackerParameters
from .segmenter import DohSegmenter, LogWSegmenter, Segmenter
from .tracking import Tracker

def main(file: pathlib.Path, resolution, segmentver, play, outdir="./data"):
    seg_params = {
        1: LogWSegmenterParameters, 2: DohSegmenterParameters
    }[segmentver](dict(
        approx_tolerance=0.25 if resolution == "low" else 1,
        gaussian_sigma={"low": 8, "med": 14, "high": 16}[resolution],
        minimum_ant_radius={"low": 4, "med": 8, "high": 10}[resolution],
    ))
    video = pims.PyAVReaderIndexed(file)

    segmenter: Segmenter = {
        1: LogWSegmenter(video, params=seg_params),
        2: DohSegmenter(video, params=seg_params)
    }[segmentver]

    # segmenter.segment()
    # with open('dia.prl', 'w') as f:
    #     f.write(segmenter.serialize())

    track_params = TrackerParameters(
        use_defaults=True,
        max_distance_between_assignments=seg_params.minimum_ant_radius * 8,
    )
    tracker = Tracker(file, segmenter, params=track_params)
    if play:
        tracker.track_viz(video, step_by_step=False, fps=60)
    else:
        tracker.track_viz()

    outfile = pathlib.Path(outdir, f"{file.stem}-{tracker.version}.trk")

    tracker.info().save(outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generar .trk a partir de un video")
    parser.add_argument('file')
    parser.add_argument('--resolution', '-r', type=str, choices=['low', 'med', 'high'], help="Resolución del video",
                        required=True)
    parser.add_argument('--segmentver', '-s', type=int, choices=[1, 2], help="Versión del segmentador",
                        required=True)
    parser.add_argument('--play', '-p', type=bool, default=False)
    parser.add_argument('--outputDir', '-o', type=str, default='./data',
                        help="Directorio de salida")
    args = parser.parse_args()

    main(pathlib.Path(args.file), args.resolution, args.segmentver, args.play, pathlib.Path(args.outputDir))
