from dataclasses import dataclass
from enum import Enum
from typing import Dict

from ..tracker.common import Side, Rect
from ..tracker.parameters import SegmenterParameters, TrackerParameters

class SelectionStep(Enum):
    SizeMarker, TrackingArea, AntFrame1, AntFrame2, NestSide, Done = range(6)
    First = SizeMarker

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return not (self <= other)

    def __ge__(self, other):
        return not (self < other)

    def next(self):
        if self != SelectionStep.Done: return SelectionStep(self.value + 1)

    def back(self):
        if self != SelectionStep.First: return SelectionStep(self.value - 1)

@dataclass
class ExtractedParameters:
    segmenter_parameters: SegmenterParameters
    tracker_parameters: TrackerParameters
    rect_data: Dict[SelectionStep, Rect]
    nest_side: Side

    def encode(self):
        return {
            'segmenter_parameters': dict(self.segmenter_parameters.items()),
            'tracker_parameters':   dict(self.tracker_parameters.items()),
            'rect_data':            {step.name: rect for step, rect in self.rect_data.items()},
            'nest_side':            self.nest_side.name,
        }

    @classmethod
    def decode(cls, d):
        return cls(
            SegmenterParameters(d['segmenter_parameters']),
            TrackerParameters(d['tracker_parameters']),
            {SelectionStep[step]: Rect(*rect) for step, rect in d['rect_data'].items()},
            Side[d['nest_side']],
        )
