from __future__ import annotations

from typing import Literal, Sequence

import parllel.logger as logger
from parllel.cages import Cage
from parllel.transforms import RecordVectorizedVideo
from parllel.types import BatchSpec

from .callback import Callback


class RecordingSchedule(Callback):
    def __init__(
        self,
        video_recorder_transform: RecordVectorizedVideo,
        trigger: Literal["on_sample", "on_eval"],
        cages: Sequence[Cage] | None = None,
        record_interval_steps: int | None = None,
        batch_spec: BatchSpec | None = None,
        offset_steps: int | None = None,
    ) -> None:
        self.video_recorder_transform = video_recorder_transform
        self.cages = list(cages) if cages is not None else []

        if trigger == "on_eval":
            if record_interval_steps is not None or batch_spec is not None:
                raise ValueError(
                    "If recording every evaluation, `record_interval_steps` and `batch_spec` should not be passed as arguments."
                )
            self.pre_evaluation = self._pre_evaluation
            self.post_evaluation = self._post_evaluation

        elif trigger == "on_sample":
            if record_interval_steps is None or batch_spec is None:
                raise ValueError(
                    "If recording during sampling, you must provide both `record_interval_steps` and `batch_spec`."
                )
            self.pre_sampling = self._pre_sampling
            self.post_sampling = self._post_sampling

            self.record_interval = int(record_interval_steps)
            self.batch_size = batch_spec.size
            self.batch_B = batch_spec.B
            if offset_steps is None:
                # record start of training by default
                self.steps_since_last = self.record_interval
            else:
                # otherwise use user-requested offset
                self.steps_since_last = -offset_steps
        else:
            raise ValueError(f"Unknown trigger {trigger}")

        self.recording = False

    def _pre_sampling(self, elapsed_steps: int) -> None:
        # check if the requested starting point is in the coming batch
        if (
            not self.recording
            and self.steps_since_last + self.batch_size > self.record_interval
        ):
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

    def _post_sampling(self, elapsed_steps: int) -> None:
        # update counters after processing batch
        self.steps_since_last += self.batch_size

        if self.recording and not self.video_recorder_transform.recording:
            # transform will automatically stop recording when desired video
            # length has been reached, but the cages need to be told to stop
            # rendering
            for env in self.cages:
                env.render = False
            self.recording = False

    def _pre_evaluation(self, elapsed_steps: int) -> None:
        self.video_recorder_transform.start_recording(
            video_name_suffix=str(elapsed_steps),
        )

        for env in self.cages:
            env.render = True
        self.recording = True

    def _post_evaluation(self, elapsed_steps: int) -> None:
        if self.video_recorder_transform.recording:
            logger.warn(
                f"Video recorder was not able to collect the requested number of video frames during evaluation."
            )
            self.video_recorder_transform.stop_recording()

        for env in self.cages:
            env.render = False
        self.recording = False
