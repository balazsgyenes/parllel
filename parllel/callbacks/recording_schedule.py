from __future__ import annotations

from typing import Literal, Sequence

from parllel.cages import Cage
from parllel.transforms import RecordVectorizedVideo
from parllel.types import BatchSpec

from .callback import Callback


class RecordingSchedule(Callback):
    def __init__(
        self,
        video_recorder_transform: RecordVectorizedVideo,
        record_interval_steps: int,
        video_length: int,
        batch_spec: BatchSpec,
        cages: Sequence[Cage] | None = None,
        begin_or_end_recording_at_interval: Literal["begin", "end"] = "end",
    ) -> None:
        self.video_recorder_transform = video_recorder_transform
        self.cages = list(cages) if cages is not None else []
        self.record_interval = int(record_interval_steps)
        self.video_length = int(video_length)
        self.batch_size = batch_spec.size
        self.batch_B = batch_spec.B
        assert begin_or_end_recording_at_interval in ("begin", "end")
        if begin_or_end_recording_at_interval == "end":
            # adjust delay such that recordings are always finished at the requested intervals
            # start recording one batch earlier because runner might trigger logger dump up to
            # one batch before nominal logging interval
            # TODO: this won't work for recurrent case where stop is not deterministic
            self.steps_since_last = self.video_length * self.batch_B + batch_spec.size
        elif begin_or_end_recording_at_interval == "start":
            # force first batch to be recorded
            self.steps_since_last = self.record_interval

        self.recording = False

    def __call__(self, elapsed_steps: int) -> None:
        if self.recording:
            # transform will automatically stop recording when desired video
            # length has been reached, but the cages need to be told to stop
            # rendering
            if not self.video_recorder_transform.recording:
                for env in self.cages:
                    env.render = False
                self.recording = False

        # we may start recording again immediately after stopping
        if not self.recording:
            # check if the requested starting point is in the coming batch
            if self.steps_since_last + self.batch_size > self.record_interval:
                # calculate exact time point where recording should start
                delay_steps = max(0, self.record_interval - self.steps_since_last)
                delay_t = delay_steps // self.batch_B

                self.video_recorder_transform.start_recording(
                    video_name_suffix=str(elapsed_steps + delay_steps),
                    delay_t=delay_t,
                )

                for env in self.cages:
                    env.render = True

                self.steps_since_last -= self.record_interval
                self.recording = True

        # update counters after processing batch
        self.steps_since_last += self.batch_size
