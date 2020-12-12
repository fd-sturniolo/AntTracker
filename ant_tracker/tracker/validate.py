from dataclasses import dataclass, field

import argparse
import numpy as np
import pims
import skimage.draw as skdraw
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from pathlib import Path
from pims.process import crop
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff
from typing import List, Optional, overload, Dict, Tuple

from .ant_labeler_info import LabelingInfo
from .common import Video, to_json, Side, blend, crop_from_rect
from .info import TracksInfo, TracksCompleteInfo, reposition_into_crop, Direction
from .plotcommon import Animate, Log1pInvNormalize
from .track import Track

frame_grace_margin = 10

def get_distances(tracks_truth: List[Track], tracks_tracked: List[Track], distances_filename=None) -> np.ndarray:
    def hausdorff(u: np.ndarray, v: np.ndarray):
        return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])

    import os.path
    if distances_filename and os.path.isfile(distances_filename):
        return np.load(distances_filename)

    distances = np.ndarray((len(tracks_truth), len(tracks_tracked)), dtype=float)

    for i, t1 in enumerate(tracks_truth):
        p1 = t1.path()
        for j, t2 in enumerate(tracks_tracked):
            p2 = t2.path()
            distances[i, j] = hausdorff(p1, p2)
    if distances_filename:
        np.save(distances_filename, distances)
    return distances

def draw_results(video: Video, tracks_truth: List[Track], tracks_tracked: List[Track], distances: np.ndarray,
                 idx_truth: np.ndarray,
                 idx_tracked: np.ndarray):
    fig_comp: Figure = plt.figure(constrained_layout=False)
    fig_comp.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    gs = fig_comp.add_gridspec(2, 2, height_ratios=[1, 0.5], wspace=0.0, hspace=0.1)
    ax_truth: Axes = fig_comp.add_subplot(gs[0, 0])
    ax_track: Axes = fig_comp.add_subplot(gs[0, 1])
    # ax_distances: Axes = fig.add_subplot(gs[1, :])
    plt.figure()
    ax_distances: Axes = plt.gca()
    ax_distances.set_ylabel("ID's etiquetadas")
    ax_distances.set_xlabel("ID's trackeadas")

    ax_truth.set_title('Ground Truth')
    ax_track.set_title('Tracking')

    normalize = Log1pInvNormalize(vmin=distances.min(), vmax=distances.max())
    colormap = plt.get_cmap('viridis')

    ax_distances.imshow(distances, norm=Log1pInvNormalize())
    for i, j in zip(idx_truth, idx_tracked):
        # if distances[i, j] > percentile_50:
        #     continue
        ax_distances.annotate('!', (j, i), color=(0.8, 0, 0))

    ax_frame_slider = fig_comp.add_axes([0.1, 0.05, 0.8, 0.04])
    frame_slider = Slider(ax_frame_slider, 'Frame', 0, len(video), 0, valfmt="%d", valstep=1)

    def __frame_update_fn(val):
        nonlocal frame_n
        frame_n = int(val)
        draw_figure()

    frame_slider.on_changed(__frame_update_fn)

    def on_key_press(event):
        nonlocal play, frame_slider
        if event.key == 'a':
            frame_slider.set_val((frame_slider.val - 1) % len(video))
        elif event.key == 'd':
            frame_slider.set_val((frame_slider.val + 1) % len(video))
        elif event.key == 'p':
            play = not play
        elif event.key == 'escape' or event.key == 'q':
            import sys
            sys.exit()

    fig_comp.canvas.mpl_connect('key_press_event', on_key_press)

    def draw_figure():
        nonlocal frame_slider, artists, annotations
        frame_slider.val = frame_n
        frame = video[frame_n]
        for a in artists:
            ax_track.artists.remove(a)
        for ann in annotations:
            ann.remove()
        annotations = []
        artists = []
        fig_comp.suptitle(f"{frame_n=}")
        frame_truth = Track.draw_tracks(tracks_truth, frame, frame_n).copy()
        frame_tracked = Track.draw_tracks(tracks_tracked, frame, frame_n).copy()

        for track in tracks_tracked:
            if (blob := track.at(frame_n)) is not None:
                bbox = (blob
                        .bbox
                        .scale(frame.shape, extra_pixels=8)
                        .square(frame.shape)
                        )
                rr, cc = skdraw.rectangle((bbox.y0, bbox.x0), (bbox.y1, bbox.x1))
                color = (20, 230, 20) if blob.is_fully_visible(0.05) else (230, 20, 20)
                frame_tracked[rr, cc] = blend(frame_tracked[rr, cc], color, 0.3)

        for track in tracks_truth:
            if (blob := track.at(frame_n)) is not None:
                bbox = (blob
                        .bbox
                        .scale(frame.shape, extra_pixels=8)
                        .square(frame.shape)
                        )
                rr, cc = skdraw.rectangle((bbox.y0, bbox.x0), (bbox.y1, bbox.x1))

                color = (20, 230, 20) if blob.is_fully_visible(0.05) else (230, 20, 20)
                frame_truth[rr, cc] = blend(frame_truth[rr, cc], color, 0.3)

        Animate.draw(ax_truth, frame_truth)
        Animate.draw(ax_track, frame_tracked)

        for i, j in zip(idx_truth, idx_tracked):
            # if distances[i, j] > percentile_50:
            #     # print(f"Truth {i} -> Track {j}: distance too high: {distances[i, j]}")
            #     continue
            from matplotlib.patches import ConnectionPatch

            blob_truth = tracks_truth[i].at(frame_n)
            blob_tracked = tracks_tracked[j].at(frame_n)
            if blob_truth is None or blob_tracked is None:
                continue

            norm_distance = normalize(distances[i, j])

            annotations.extend([
                ax_truth.annotate(f"{distances[i, j]:.2f}", blob_truth.center_xy),
            ])
            con = ConnectionPatch(xyA=blob_truth.center_xy, xyB=blob_tracked.center_xy,
                                  coordsA="data", coordsB="data",
                                  axesA=ax_truth, axesB=ax_track,
                                  color=colormap(norm_distance),
                                  lw=(norm_distance * 3).astype(int) + 1)
            a = ax_track.add_artist(con)
            artists.append(a)
        fig_comp.canvas.draw_idle()

    artists = []
    annotations = []
    frame_n = 0
    exit_flag = False
    play = False
    last_drawn_frame_n = -1
    while not exit_flag and frame_n < len(video):
        if last_drawn_frame_n != frame_n or play:
            draw_figure()
            last_drawn_frame_n = frame_n
            if play:
                frame_n += 1
        plt.pause(0.001)
@dataclass
class ExportLikeMeasures:
    total_EN: int = field(default=0)
    total_SN: int = field(default=0)
    speed_mean: float = field(default=0)
    area_median: float = field(default=0)
    length_median: float = field(default=0)
    width_median: float = field(default=0)
@dataclass
class Measures:
    label: ExportLikeMeasures = field(default=ExportLikeMeasures())
    track: ExportLikeMeasures = field(default=ExportLikeMeasures())

def export_like_measures(label: LabelingInfo, tracked: TracksCompleteInfo, trackfilter: Dict = None):
    measures = Measures()
    tracks_label = label.filter_tracks(**trackfilter) if trackfilter else label.filter_tracks()
    tracks_tracked = (tracked.filter_tracks(last_frame=label.last_tracked_frame(), **trackfilter) if trackfilter else
                      tracked.filter_tracks(last_frame=label.last_tracked_frame()))
    for m, info, tracks in zip([measures.label, measures.track], [label, tracked], [tracks_label, tracks_tracked]):
        total_ants = 0
        for track in tracks:
            direction = info.track_direction(track, tracked.nest_side)
            if direction in (Direction.EN, Direction.SN):
                m.total_EN += 1 if direction == Direction.EN else 0
                m.total_SN += 1 if direction == Direction.SN else 0
                m.speed_mean += track.speed_mean
                m.area_median += track.area_median
                m.length_median += track.length_median
                m.width_median += track.width_median
                total_ants += 1
        m.speed_mean /= total_ants
        m.area_median /= total_ants
        m.length_median /= total_ants
        m.width_median /= total_ants
    return measures

@overload
def measure_all_data(truth: LabelingInfo, tracked: TracksInfo, *, trackfilter: Dict = ...) -> Dict: ...
@overload
def measure_all_data(truth: LabelingInfo, tracked: TracksInfo, *, trackfilter: Dict = ...,
                     cachedir: Path, file_truth: Path, file_tracked: Path) -> Dict: ...
@overload
def measure_all_data(truth: LabelingInfo, tracked: TracksInfo, *, trackfilter: Dict = ...,
                     cachedir: Path, file_truth: Path, file_tracked: Path, return_for_plots=True) -> Tuple: ...
def measure_all_data(truth: LabelingInfo, tracked: TracksInfo, *, trackfilter: Dict = None,
                     cachedir: Path = None, file_truth: Path = None, file_tracked: Path = None,
                     return_for_plots=False):
    """Obtiene un conjunto de datos validando ``tracked`` contra ``truth``. Estos datos están divididos en 3 subgrupos,
    y cada uno continene las siguientes medidas de error:

    - "mean_direction": error en la dirección media (en grados desde la horizontal)
    - "mean_speed": error en la velocidad media
    - "max_speed": error en la velocidad máxima
    - "median_area": error en el área mediana ocupada
    - "max_area": error en el área máxima ocupada

    Los tres subgrupos antes mencionados son:

    - `measures_dev` contiene medidas de error track-por-track promediadas. También cantidades:

        - "discarded_by_video_cutoff": Cantidad de tracks descartados de la comparación porque comenzaron en el centro (los primeros en cuadro al comenzar la grabación)
        - "discarded_by_direction": Cantidad de tracks descartados de ``tracked`` porque no seguían la misma dirección que su correspondiente en ``truth``
        - "discarded_by_ontological_inertia": Cantidad de tracks descartados de ``tracked`` porque desaparecieron en el centro
        - "total_real": Cantidad total de tracks en ``truth``
        - "total_tracked": Cantidad total de tracks en ``tracked``
        - "total_assigned": Cantidad de tracks en ``tracked`` que se asignaron a tracks en ``truth``
        - "total_tracked_non_discarded": Cantidad en ``tracked`` que no fueron descartados por las razones anteriores
        - "total_real_interesting": Cantidad en ``truth`` que no comenzaron en el centro

    - `measures_test` contiene medidas de error calculadas globalmente, y los números "total_real_interesting" y "total_tracked_non_discarded" que son iguales a sus correspondientes en `measures_dev`

    - `per_ant_comparisons` tiene medidas de error track-por-track (que promediadas resultan en las que se encuen en `measures_dev`. Además, contiene:

        - "assignment": tupla (i,j) que indica que el track ``i`` en ``truth`` se corresponde con el track ``j`` en ``tracked``
        - "distance": distancia de Hausdorff entre los dos tracks
        - "first_blob_distance": distancia en píxeles entre los primeros blobs de cada track
        - "direction_of_travel": tupla ((SideIn_i,SideOut_i)),(SideIn_j,SideOut_j)) que indica desde dónde entró y salió cada track
    """
    if trackfilter:
        tracks_truth = truth.filter_tracks(**trackfilter)
        tracks_tracked = tracked.filter_tracks(last_frame=truth.last_tracked_frame(), **trackfilter)
    else:
        tracks_truth = truth.filter_tracks()
        tracks_tracked = tracked.filter_tracks(last_frame=truth.last_tracked_frame())
    if cachedir is not None:
        distances_filename = Path(f'{cachedir}/distances__{file_truth.stem}__{file_tracked.stem}.npy')
    else:
        distances_filename = ""
    distances = get_distances(tracks_truth, tracks_tracked, distances_filename)
    idx_truth, idx_tracked = linear_sum_assignment(distances)
    assigned_distances = distances[idx_truth, idx_tracked]
    percentile_25 = np.percentile(assigned_distances, 25)
    percentile_50 = np.percentile(assigned_distances, 50)
    percentile_75 = np.percentile(assigned_distances, 75)
    interquartile_range = percentile_75 - percentile_25

    per_ant_measures = [
        "mean_direction",
        "mean_speed",
        "max_speed",
        "median_area",
        "max_area"
    ]
    comparisons = []
    i: int
    j: int
    for i, j in zip(idx_truth, idx_tracked):
        # if distances[i, j] > percentile_50:
        #     # only ants that we're confident in
        #     continue

        track_truth = tracks_truth[i]
        track_tracked = tracks_tracked[j]

        comparisons.append({
            "assignment":          (i, j),
            "distance":            distances[i, j],
            "first_blob_distance": np.linalg.norm(track_truth.first_blob().center - track_tracked.first_blob().center),
            "direction_of_travel": (
                track_truth.direction_of_travel(truth.video_shape),
                track_tracked.direction_of_travel(tracked.video_shape)
            ),
            "errors":              {
                "mean_direction": track_truth.direction_mean - track_tracked.direction_mean,
                "mean_speed":     track_truth.speed_mean - track_tracked.speed_mean,
                "max_speed":      track_truth.speed_max - track_tracked.speed_max,
                "median_area":    track_truth.area_median - track_tracked.area_median,
                "max_area":       track_truth.areas.max() - track_tracked.areas.max(),
            }
        })

    measures_dev = {
        "errors":                           {measure: 0 for measure in per_ant_measures},
        # "discarded_by_hausdorff": 0,
        "discarded_by_video_cutoff":        0,
        "discarded_by_direction":           0,
        "discarded_by_ontological_inertia": 0,
        "total_real":                       len(tracks_truth),
        "total_tracked":                    len(tracks_tracked),
        "total_assigned":                   len(idx_tracked),
        "total_tracked_non_discarded":      0,
        "total_real_interesting":           0,
    }
    n = 0
    for comp in comparisons:
        # if comp["distance"] > percentile_50:
        #     global_measures["discarded_by_hausdorff"] += 1
        #     # in dev, don't consider bad assignments
        #     continue
        if Side.Center in comp["direction_of_travel"][0]:
            measures_dev["discarded_by_video_cutoff"] += 1
            # we only check ants that started on an edge, so as to get rid of bad tracks by ontological inertia
            continue
        if comp["direction_of_travel"][0] != comp["direction_of_travel"][1]:
            measures_dev["discarded_by_direction"] += 1
            # if traveling in the wrong direction
            continue
        if Side.Center in comp["direction_of_travel"][1]:
            measures_dev["discarded_by_ontological_inertia"] += 1
            # if ant appears/disappears in center
            continue
        n += 1
        for measure in per_ant_measures:
            measures_dev["errors"][measure] += comp["errors"][measure] ** 2
    if n != 0:
        for measure in per_ant_measures:
            measures_dev["errors"][measure] /= n
    measures_dev["total_tracked_non_discarded"] = n
    measures_dev["total_real_interesting"] = measures_dev["total_real"] - measures_dev[
        "discarded_by_video_cutoff"]

    interesting_truth_tracks = [track for track in tracks_truth if
                                Side.Center not in track.direction_of_travel(truth.video_shape)]
    non_discarded_tracks = [track for track in tracks_tracked if
                            Side.Center not in track.direction_of_travel(tracked.video_shape)]

    measures_test = {
        "errors":                      {measure: 0 for measure in per_ant_measures},
        "total_real_interesting":      len(interesting_truth_tracks),
        "total_tracked_non_discarded": len(non_discarded_tracks),
    }

    global_truth_measures = {measure: 0 for measure in per_ant_measures}
    for track in interesting_truth_tracks:
        global_truth_measures["mean_direction"] += track.direction_mean
        global_truth_measures["mean_speed"] += track.speed_mean
        global_truth_measures["max_speed"] += track.speed_max
        global_truth_measures["median_area"] += track.area_median
        global_truth_measures["max_area"] += track.areas.max()
    if len(interesting_truth_tracks) != 0:
        global_truth_measures = {measure: value / len(interesting_truth_tracks) for measure, value in
                                 global_truth_measures.items()}
    global_track_measures = {measure: 0 for measure in per_ant_measures}
    for track in non_discarded_tracks:
        global_track_measures["mean_direction"] += track.direction_mean
        global_track_measures["mean_speed"] += track.speed_mean
        global_track_measures["max_speed"] += track.speed_max
        global_track_measures["median_area"] += track.area_median
        global_track_measures["max_area"] += track.areas.max()
    if len(non_discarded_tracks) != 0:
        global_track_measures = {measure: value / len(non_discarded_tracks) for measure, value in
                                 global_track_measures.items()}

    measures_test["errors"] = {
        measure: abs(global_truth_measures[measure] - global_track_measures[measure]) /
                 global_truth_measures[measure] for measure in per_ant_measures
    }

    measures = {'measures_dev': measures_dev, 'measures_test': measures_test, 'per_ant_comparisons': comparisons}
    if return_for_plots:
        return measures, distances, idx_truth, idx_tracked
    return measures

def main():
    parser = argparse.ArgumentParser(description="Comparar dos archivos .trk")

    parser.add_argument('truth')
    parser.add_argument('tracked')
    parser.add_argument('--video', '-v', type=str, default=None,
                        help="Video del que fueron generados, para graficar ambos")
    parser.add_argument('--output', '-o', type=str, default="data", help="Directorio de salida")
    parser.add_argument('--cache', type=bool, default=False, help="Cachear distancias entre tracks de .trk y .tag")

    import pathlib
    args = parser.parse_args()
    file_tracked = pathlib.Path(args.tracked)
    file_truth = pathlib.Path(args.truth)

    complete = False
    try:
        tracked = TracksCompleteInfo.load(file_tracked)
        complete = True
    except:
        tracked = TracksInfo.load(file_tracked)
    truth = LabelingInfo.load(file_truth)
    if complete:
        truth = reposition_into_crop(truth, tracked.crop_rect)

    video: Optional[Video] = None
    if args.video is not None:
        video = pims.PyAVReaderIndexed(args.video)[:]
        if complete:
            video = crop(video, crop_from_rect(video.frame_shape[0:2], tracked.crop_rect))

    if args.cache:
        measures, distances, idx_truth, idx_tracked = measure_all_data(truth, tracked,
                                                                       cachedir=Path(".cachedata"),
                                                                       file_truth=file_truth, file_tracked=file_tracked,
                                                                       return_for_plots=True)
    else:
        measures, distances, idx_truth, idx_tracked = measure_all_data(truth, tracked, return_for_plots=True)

    results_filename = pathlib.Path(args.output) / 'validation__{file_truth.stem}__{file_tracked.stem}.json'
    with open(results_filename, 'w') as f:
        f.write(to_json(measures))

    if video is not None:
        draw_results(video=video,
                     tracks_truth=truth.filter_tracks(),
                     tracks_tracked=tracked.filter_tracks(last_frame=truth.last_tracked_frame()),
                     distances=distances,
                     idx_truth=idx_truth,
                     idx_tracked=idx_tracked)

if __name__ == '__main__':
    main()
