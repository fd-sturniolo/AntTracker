import numpy as np
from packaging.version import Version
from pathlib import Path
from scipy.spatial.distance import cdist
from typing import List, NewType, Optional, Tuple, Union, TypedDict, Dict

from .blob import Blob
from .common import FrameNumber, Video, ensure_path, to_json, encode_np_randomstate, decode_np_randomstate
from .info import TracksInfo
from .ongoing_track import OngoingTrack
from .parameters import TrackerParameters, SegmenterParameters
from .segmenter import Blobs, Segmenter, LogWSegmenter
from .track import Track, TrackId
from ..version import __version__

BlobIndex = NewType('BlobIndex', int)
Assignment = Tuple[TrackId, BlobIndex]

class Tracker:
    # region Init
    def __init__(self, video_path: Union[Path, str], segmenter: Segmenter = None, *,
                 params: TrackerParameters = TrackerParameters(use_defaults=True)):
        self.video_path = video_path
        self.version = __version__
        self.__tracks: Optional[List[Track]] = None
        self.params = params
        if segmenter is None:  # from deserialization
            return
        self.segmenter = segmenter
        self.video_length = segmenter.video_length
        self.video_shape = segmenter.video_shape
        self.last_id: TrackId = TrackId(0)
        self.__inprogress_tracks: List[Track] = []
        self.__last_tracked_frame: FrameNumber = -1
        self.__random_state = np.random.RandomState()

    # endregion

    # region Tracking

    @property
    def last_tracked_frame(self):
        return self.__last_tracked_frame

    def next_id(self) -> TrackId:
        i = self.last_id
        self.last_id += 1
        return i

    @staticmethod
    def kalman_get_measurement(blobs: List[Blob]) -> List[Tuple[float, float]]:
        return [(blob.center[0], blob.center[1]) for blob in blobs]

    @property
    def is_finished(self):
        return self.__tracks is not None

    @property
    def tracks(self) -> List[Track]:
        if not self.is_finished:
            list(self.track_progressive())
        return self.__tracks

    def track_progressive_continue(self):
        last_frame = self.__last_tracked_frame
        if last_frame == -1:  # first time, from track_progressive()
            prev_blobs = []
        else:
            prev_blobs = [track.at(last_frame) for track in self.__inprogress_tracks if
                          track.at(last_frame) is not None]
        frames_with_blobs = self.segmenter.segment_rolling_from(last_frame + 1, prev_blobs)
        tracks = self.__inprogress_tracks
        for frame, blobs in frames_with_blobs:
            if frame == 0:
                for blob in blobs:
                    track_id = self.next_id()
                    track = OngoingTrack(
                        id=track_id, blobs={0: blob},
                        frames_until_close=self.params.frames_until_close,
                        imshape=self.video_shape,
                        a_sigma=self.params.a_sigma,
                        random_state=self.__random_state,
                    )
                    tracks.append(track)
                continue

            ongoing_tracks = Tracker.ongoing(tracks)

            # region Get detections and estimates
            detections = Tracker.kalman_get_measurement(blobs)
            number_of_detections = len(detections)
            estimates_and_ids = [(track.predict(), track.id) for track in ongoing_tracks]
            estimates = [estimate for estimate, track_id in estimates_and_ids]
            number_of_estimates = len(estimates)
            # endregion

            id_assignments = []
            new_blob_idxs = []
            lost_ids = []
            if number_of_estimates == 0:
                if number_of_detections != 0:
                    new_blob_idxs = list(range(len(blobs)))
            elif number_of_detections == 0:
                if number_of_estimates != 0:
                    lost_ids = [track_id for estimate, track_id in estimates_and_ids]
            else:
                # region Calculate the distance matrix and apply Munkres algorithm
                estimates_nd = np.array(estimates)[:, 0:2]  # get only position, not velocity
                detections_nd = np.array(detections)

                estimate_areas = np.array(
                    [[Track.get(ongoing_tracks, track_id).at(frame - 1).area] for _, track_id in
                     estimates_and_ids])
                estimates_y_x_area = np.hstack((estimates_nd, estimate_areas))

                detection_areas = np.array([[b.area] for b in blobs])
                detections_y_x_area = np.hstack((detections_nd, detection_areas))

                dist: np.ndarray = cdist(estimates_y_x_area, detections_y_x_area).T
                from scipy.optimize import linear_sum_assignment
                det, est = linear_sum_assignment(dist)
                indexes = [(estim, detect) for estim, detect in zip(est, det)]

                # endregion

                # region Find lost assignments and new tracks
                id_assignments: List[Assignment] = []
                lost_ids: List[TrackId] = []
                for estim, detect in indexes:
                    track_id = estimates_and_ids[estim][1]
                    detect_index_in_frame: BlobIndex = detect
                    track_yx = Track.get(ongoing_tracks, track_id).at(frame - 1).center
                    detect_yx = blobs[detect_index_in_frame].center
                    distance = np.linalg.norm(track_yx - detect_yx)
                    if distance < self.params.max_distance_between_assignments:
                        id_assignments.append((track_id, detect_index_in_frame))
                    else:
                        # Far away assignments are tracks that were lost
                        lost_ids.append(track_id)
                        pass
                prev_ids = [track_id for estimate, track_id in estimates_and_ids]
                # Tracks that weren't assigned are lost
                lost_ids.extend([track_id
                                 for track_id in prev_ids
                                 if
                                 track_id not in [prev_id for prev_id, next_id in id_assignments]
                                 and track_id not in lost_ids
                                 ])
                # Detections that weren't assigned are new tracks
                new_blob_idxs = [blob_index
                                 for blob_index in range(len(blobs))
                                 if blob_index not in [blob_idx for track_id, blob_idx in id_assignments]
                                 ]

                # endregion

            # region Update filters for each track
            for prev_track, blob_index in id_assignments:
                track = OngoingTrack.get(tracks, prev_track)
                next_blob = blobs[blob_index]
                track.update(frame, next_blob)
            for track_id in lost_ids:
                track = OngoingTrack.get(tracks, track_id)
                track.update(frame)
            # endregion

            # region Close tracks that were lost for too many frames
            tracks = [(track.as_closed() if track.closed else track)
                      if isinstance(track, OngoingTrack) else track
                      for track in tracks]
            # endregion

            # region Create new tracks
            for blob_index in new_blob_idxs:
                blob = blobs[blob_index]
                new_track = OngoingTrack(
                    id=self.next_id(),
                    blobs={frame: blob},
                    frames_until_close=self.params.frames_until_close,
                    imshape=self.video_shape,
                    a_sigma=self.params.a_sigma,
                    random_state=self.__random_state,
                )
                tracks.append(new_track)
            # endregion

            self.__inprogress_tracks = tracks
            self.__last_tracked_frame = frame

            yield frame

        self.__tracks = [track.as_closed() for track in tracks]

    def track_progressive(self):
        yield from self.track_progressive_continue()

    def track_viz(self, video: Optional[Video] = None, *, step_by_step=False, fps=10):
        # region Drawing
        if video is not None:
            from matplotlib import pyplot as plt
            from .plotcommon import Animate
            fig, ax = plt.subplots(1, 2)
            axli = ax.flatten()
            fig_track, ax_track = plt.subplots(1, 1)
        if step_by_step:
            process_next_frame = False
            if video:
                def on_key_press(event):
                    nonlocal process_next_frame
                    if event.key == ' ':
                        process_next_frame = True
                    elif event.key == 'escape':
                        import sys
                        sys.exit()

                plt.gcf().canvas.mpl_connect('key_press_event', on_key_press)
        # endregion

        tracks: List[Track] = []
        frame: FrameNumber
        for frame, blobs in self.segmenter.frames_with_blobs:
            if frame == 0:
                for blob in blobs:
                    track_id = self.next_id()
                    track = OngoingTrack(
                        id=track_id, blobs={0: blob},
                        frames_until_close=self.params.frames_until_close,
                        imshape=self.video_shape,
                        a_sigma=self.params.a_sigma,
                        random_state=self.__random_state,
                    )
                    tracks.append(track)
                continue

            print(f"Tracking at {frame=}")
            ongoing_tracks = Tracker.ongoing(tracks)

            # region Drawing
            if video is not None:
                axli[0].clear()
                axli[1].clear()
                axli[0].set_title(f'Frame {frame - 1}')
                axli[1].set_title(f'Frame {frame}')
                prev_image = OngoingTrack.draw_tracks(ongoing_tracks, video[frame - 1], frame - 1)
                Animate.draw(axli[0], prev_image, override_hash=True)
                image = Blob.draw_blobs(blobs, video[frame])
                Animate.draw(axli[1], image, override_hash=True)
                track_img = OngoingTrack.draw_tracks(
                    sorted(ongoing_tracks, key=lambda t: -1 if t.is_currently_lost else 0), video[frame - 1],
                    frame - 1)
                for track in tracks:
                    track_img = track.draw_track_line(frame, track_img)
                Animate.draw(ax_track, track_img)

            # endregion

            # region Get detections and estimates
            detections = Tracker.kalman_get_measurement(blobs)
            number_of_detections = len(detections)
            # number_of_detections = len(blobs)
            estimates_and_ids = [(track.predict(), track.id) for track in ongoing_tracks]
            estimates = [estimate for estimate, track_id in estimates_and_ids]
            number_of_estimates = len(estimates)
            # endregion

            id_assignments = []
            new_blob_idxs = []
            lost_ids = []
            if number_of_estimates == 0:
                if number_of_detections != 0:
                    new_blob_idxs = list(range(len(blobs)))
            elif number_of_detections == 0:
                if number_of_estimates != 0:
                    lost_ids = [track_id for estimate, track_id in estimates_and_ids]
            else:
                # region Calculate the distance matrix and apply Munkres algorithm
                estimates_nd = np.array(estimates)[:, 0:2]  # get only position, not velocity
                detections_nd = np.array(detections)

                estimate_areas = np.array(
                    [[Track.get(ongoing_tracks, track_id).at(frame - 1).area] for _, track_id in
                     estimates_and_ids])
                estimates_y_x_area = np.hstack((estimates_nd, estimate_areas))

                detection_areas = np.array([[b.area] for b in blobs])
                detections_y_x_area = np.hstack((detections_nd, detection_areas))

                dist: np.ndarray = cdist(estimates_y_x_area, detections_y_x_area).T
                # distance_btwn_estimate_n_detections = dist.copy()
                from scipy.optimize import linear_sum_assignment
                det, est = linear_sum_assignment(dist)
                indexes = [(estim, detect) for estim, detect in zip(est, det)]

                # endregion

                # region Find lost assignments and new tracks
                id_assignments: List[Assignment] = []
                lost_ids: List[TrackId] = []
                for estim, detect in indexes:
                    track_id = estimates_and_ids[estim][1]
                    detect_index_in_frame: BlobIndex = detect
                    track_yx = Track.get(ongoing_tracks, track_id).at(frame - 1).center
                    detect_yx = blobs[detect_index_in_frame].center
                    distance = np.linalg.norm(track_yx - detect_yx)
                    # print(f'(Track:{track_id}->Detect:{detect_index_in_frame}): {distance=}')
                    if distance < self.params.max_distance_between_assignments:
                        id_assignments.append((track_id, detect_index_in_frame))
                    else:
                        # Far away assignments are tracks that were lost
                        lost_ids.append(track_id)
                        pass
                prev_ids = [track_id for estimate, track_id in estimates_and_ids]
                # Tracks that weren't assigned are lost
                lost_ids.extend([track_id
                                 for track_id in prev_ids
                                 if
                                 track_id not in [prev_id for prev_id, next_id in id_assignments]
                                 and track_id not in lost_ids
                                 ])
                # next_ids: Sequence[BlobIndex] = range(len(blobs))
                # Detections that weren't assigned are new tracks
                new_blob_idxs = [blob_index
                                 for blob_index in range(len(blobs))
                                 if blob_index not in [blob_idx for track_id, blob_idx in id_assignments]
                                 ]

                # endregion

            # region Update filters for each track
            for prev_track, blob_index in id_assignments:
                track = OngoingTrack.get(tracks, prev_track)
                next_blob = blobs[blob_index]
                track.update(frame, next_blob)
            for track_id in lost_ids:
                track = OngoingTrack.get(tracks, track_id)
                track.update(frame)
            # endregion

            # region Close tracks that were lost for too many frames
            tracks = [(track.as_closed() if track.closed else track)
                      if isinstance(track, OngoingTrack) else track
                      for track in tracks]
            # endregion

            # region Create new tracks
            for blob_index in new_blob_idxs:
                blob = blobs[blob_index]
                new_track = OngoingTrack(
                    id=self.next_id(),
                    blobs={frame: blob},
                    frames_until_close=self.params.frames_until_close,
                    imshape=self.video_shape,
                    a_sigma=self.params.a_sigma,
                    random_state=self.__random_state,
                )
                tracks.append(new_track)
            # endregion

            self.__inprogress_tracks = tracks
            self.__last_tracked_frame = frame

            # region Drawing
            blob_correlation = Tracker.correlate_blobs(frame, ongoing_tracks, blobs, id_assignments)

            if video is not None:
                for prev_blob, next_blob in blob_correlation:
                    from matplotlib.patches import ConnectionPatch
                    con = ConnectionPatch(xyA=prev_blob.center_xy, xyB=next_blob.center_xy,
                                          coordsA="data", coordsB="data",
                                          axesA=axli[0], axesB=axli[1], color=(1, 0, 0, 0.2), lw=1)
                    axli[1].add_artist(con)
                for lost_id in lost_ids:
                    lost_track = OngoingTrack.get(ongoing_tracks, lost_id)
                    if (lost_blob := lost_track.at(frame)) is not None:
                        # because it could've been closed in this frame, thus not having a blob
                        prev_image = lost_blob.draw_label(prev_image, 'LOST', size=10)
                        Animate.draw(axli[0], prev_image, override_hash=True)
                    # axli[0].text(lost_blob.center_xy[0], lost_blob.center_xy[1] - 10, 'LOST')
                for blob_index in new_blob_idxs:
                    new_blob = blobs[blob_index]
                    image = new_blob.draw_label(image, 'NEW', size=10)
                    Animate.draw(axli[1], image, override_hash=True)
                    # axli[1].text(new_blob.center_xy[0], new_blob.center_xy[1] - 10, 'NEW')

                plt.draw()
            if step_by_step:
                while not process_next_frame:
                    plt.pause(0.05)
                process_next_frame = False

            if video is not None:
                plt.pause(1 / fps)

            # endregion

        self.__tracks = [track.as_closed() for track in tracks]

    @staticmethod
    def correlate_blobs(
            frame: FrameNumber,
            tracks: List['OngoingTrack'],
            blobs: Blobs,
            id_assignments: List[Assignment]) -> List[Tuple[Blob, Blob]]:
        correlation = []
        for prev_id, next_id in id_assignments:
            track = OngoingTrack.get(tracks, prev_id)
            prev_blob = track.at(frame)
            next_blob = blobs[next_id]
            correlation.append((prev_blob, next_blob))
        return correlation

    @staticmethod
    def ongoing(tracks: List[Track]) -> List[OngoingTrack]:
        return [track for track in tracks if isinstance(track, OngoingTrack)]

    @staticmethod
    def closed(tracks: List[Track]) -> List[Track]:
        return [track for track in tracks if isinstance(track, Track) and not isinstance(track, OngoingTrack)]

    # endregion

    # region Serialization
    def save_unfinished(self, file: Union[Path, str]):
        if self.is_finished: raise ValueError("El tracking ya fue finalizado, use info()")
        file = ensure_path(file)
        with file.open('w') as f:
            f.write(to_json(self.encode_unfinished()))

    @classmethod
    def load_unfinished(cls, file: Union[Path, str], video: Video, video_path: Union[Path, str]):
        file = ensure_path(file)
        with file.open('r') as f:
            import json
            self = cls.decode_unfinished(json.load(f), video, video_path)
        return self

    class UnfinishedSerial(TypedDict):
        __closed_tracks: List[Track.Serial]
        __ongoing_tracks: List[OngoingTrack.Serial]
        __last_tracked_frame: int
        __np_randomstate: Tuple
        segmenter_parameters: Dict
        tracker_parameters: Dict
        video_length: int
        video_shape: Tuple[int, int]

    def encode_unfinished(self) -> 'Tracker.UnfinishedSerial':
        return {
            '__closed_tracks':      [Track.encode(track) for track in Tracker.closed(self.__inprogress_tracks)],
            '__ongoing_tracks':     [OngoingTrack.encode(track) for track in Tracker.ongoing(self.__inprogress_tracks)],
            '__last_tracked_frame': self.__last_tracked_frame,
            '__np_randomstate':     encode_np_randomstate(self.__random_state.get_state()),
            'segmenter_parameters': self.segmenter.params.encode(),
            'tracker_parameters':   self.params.encode(),
            'video_length':         self.video_length,
            'video_shape':          self.video_shape,
        }

    @classmethod
    def decode_unfinished(cls, serial: 'Tracker.UnfinishedSerial', video: Video, video_path: Path):
        video_shape = serial['video_shape']
        self = cls(video_path, params=TrackerParameters.decode(serial['tracker_parameters']))
        self.segmenter = LogWSegmenter(video, SegmenterParameters.decode(serial['segmenter_parameters']))
        self.video_path = video_path
        self.video_length = serial['video_length']
        self.video_shape = video_shape
        self.__last_tracked_frame = serial['__last_tracked_frame']

        self.__random_state = np.random.RandomState()
        self.__random_state.set_state(decode_np_randomstate(serial['__np_randomstate']))

        closed_tracks = [Track.decode(track, video_shape) for track in serial['__closed_tracks']]
        ongoing_tracks = [OngoingTrack.decode(
            track,
            self.params.a_sigma,
            video_shape,
            self.params.frames_until_close,
            self.__random_state
        ) for track in serial['__ongoing_tracks']]

        self.__inprogress_tracks = closed_tracks + ongoing_tracks
        self.last_id = TrackId(max([track.id for track in self.__inprogress_tracks]) + 1)

        return self

    def info(self):
        if not self.is_finished: raise ValueError('track() first!')
        return TracksInfo(
            video_path=self.video_path,
            version=self.version,
            segmenter_parameters=self.segmenter.params,
            tracker_parameters=self.params,
            tracks=self.tracks,
        )
    # endregion
