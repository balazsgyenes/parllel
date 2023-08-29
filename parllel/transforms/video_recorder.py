from __future__ import annotations

import itertools
from os import PathLike
from pathlib import Path
from typing import Iterator, Mapping, TypeVar

import gymnasium as gym
import numba
import numpy as np

try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
except ImportError as e:
    raise gym.error.DependencyNotInstalled(
        "moviepy is not installed, run `pip install moviepy`"
    ) from e

import parllel.logger as logger
from parllel import Array, ArrayDict

from .transform import BatchTransform


class RecordVectorizedVideo(BatchTransform):
    def __init__(
        self,
        output_dir: str | PathLike,
        sample_tree: ArrayDict[Array],
        buffer_key_to_record: str,  # e.g. "observation" or "env_info.rendering"
        record_every_n_steps: int,
        video_length: int,
        env_fps: int = 30,
        tiled_height: int | None = None,
        tiled_width: int | None = None,
        torch_order: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.record_every = int(record_every_n_steps)
        self.length = int(video_length)
        self.keys = buffer_key_to_record.split(".")
        self.env_fps = env_fps
        self.torch_order = torch_order

        self.output_dir.mkdir(parents=True)

        images = dict_get_nested(sample_tree, self.keys)
        batch_T, batch_B, n_channels, height, width = images.shape
        self.batch_B = n_images = batch_B
        self.batch_size = batch_T * batch_B

        if tiled_height is not None and tiled_width is not None:
            tiled_height, tiled_width = int(tiled_height), int(tiled_width)
            if tiled_height * tiled_width < n_images:
                raise ValueError(
                    f"Tiled height and width {tiled_height}x{tiled_width} are "
                    f"too small to allocate space for {n_images} images."
                )
            self.tiled_height, self.tiled_width = tiled_height, tiled_width
        else:
            self.tiled_height, self.tiled_width = self.find_tiling(n_images)
            logger.debug(
                f"Automatically chose a tiling of {self.tiled_height}x"
                f"{self.tiled_width} for vectorized video recording."
            )

        self.total_t = 0  # used for naming video files
        # force first batch to be recorded
        self.steps_since_last_recording = self.record_every
        self.recording = False

    @staticmethod
    def find_tiling(n_images: int) -> tuple[int, int]:
        """Find a tiling to represent `n_images` images in one big PxQ tiling.
        P and Q are chosen to be factors of n_images, as long as this is possible
        with an aspect ratio between 1:1 and 2:1. Otherwise P and Q are chosen to
        be as close to each other as possible.
        """
        # first, try to factorize n_images with an aspect ratio between 1:1
        # and 2:1
        max_tiled_height = int(np.floor(np.sqrt(n_images)))
        min_tiled_height = int(np.ceil(np.sqrt(n_images / 2)))
        try:
            tiled_height = next(
                i
                for i in range(max_tiled_height, min_tiled_height - 1, -1)
                if n_images % i == 0
            )
            tiled_width = n_images // tiled_height
            return (tiled_height, tiled_width)

        except StopIteration:
            pass

        # if such factors do not exist, construct a grid that is roughly
        # square. Additional tiles will be filled in with black
        tiled_height = int(np.ceil(np.sqrt(n_images)))
        tiled_width = int(np.ceil(float(n_images) / tiled_height))
        return (tiled_height, tiled_width)

    def __call__(self, sample_tree: ArrayDict[Array]) -> ArrayDict[Array]:
        # check if we should start recording
        if not self.recording:
            if self.steps_since_last_recording + self.batch_size >= self.record_every:
                self._start_recording()

        if self.recording:
            self._record_batch(sample_tree)

        # update counters after processing batch
        self.total_t += self.batch_size
        self.steps_since_last_recording += self.batch_size

        return sample_tree

    def _start_recording(self) -> None:
        # calculate exact time point where recording should start
        offset_steps = max(0, self.record_every - self.steps_since_last_recording)
        self.offset_t = offset_steps // self.batch_B
        start_total_t = self.total_t + offset_steps

        self.path = self.output_dir / f"policy_step_{start_total_t}.mp4"
        logger.log(f"Started recording video of policy to {self.path}")

        self.recorded_frames = []
        self.steps_since_last_recording -= self.record_every
        self.recording = True

    def _record_batch(self, sample_tree: ArrayDict[Array]) -> None:
        images_batch = dict_get_nested(sample_tree, self.keys)
        valid_batch = sample_tree.get("valid", None)

        # convert to numpy arrays
        images_batch = np.asarray(images_batch)
        if valid_batch is not None:
            valid_batch = np.asarray(valid_batch)

        # if this is the start of recording, delay start to arrive at exact
        # desired start point
        if len(self.recorded_frames) == 0:
            images_batch = images_batch[self.offset_t :]
            if valid_batch is not None:
                valid_batch = valid_batch[self.offset_t :]

        # loop through time, saving images from each step to file
        for images, valid in zip_with_valid(images_batch, valid_batch):
            tiled_image = tile_images(
                images=images,
                valid=valid,
                tiled_height=self.tiled_height,
                tiled_width=self.tiled_width,
            )
            self.recorded_frames.append(tiled_image)

            if len(self.recorded_frames) >= self.length:
                self._stop_recording()
                break

            # if all environments are done, continue recording from next batch
            if valid is not None and not np.any(valid):
                break

    def _stop_recording(self) -> None:
        # TODO: use weakref to ensure this gets closed even if training ends
        # during video recording (user forgets to call close method)
        clip = ImageSequenceClip(self.recorded_frames, fps=self.env_fps)
        clip.write_videofile(str(self.path), logger=None)
        self.recording = False
        logger.debug(f"Finished recording video of policy to {self.path}")

    def close(self) -> None:
        if self.recording:
            self._stop_recording()


ValueType = TypeVar("ValueType")


def dict_get_nested(buffer: Mapping[str, ValueType], keys: list[str]) -> ValueType:
    result = buffer
    for key in keys:
        result = result[key]
    return result


def zip_with_valid(
    array: np.ndarray,
    valid: np.ndarray | None,
) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
    if valid is None:
        valid = itertools.repeat(None)
    yield from zip(array, valid)


@numba.njit
def tile_images(
    images: np.ndarray,
    valid: np.ndarray | None,
    tiled_height: int,
    tiled_width: int,
    torch_order: bool = False,
) -> np.ndarray:
    """Write images from individual environments into a preallocated tiled
    frame. Transposes image channels from torch order to image order and
    freezes images if the image data is no longer valid.
    """
    if torch_order:
        # move channel dimension to the end
        images_nhwc = images.transpose(0, 2, 3, 1)
    else:
        images_nhwc = images

    n_images, height, width, n_channels = images_nhwc.shape

    # [Height * height, Width * width, n_channels]
    tiled_frame = np.zeros(
        shape=(tiled_height * height, tiled_width * width, n_channels),
        dtype=np.uint8,
    )

    # [Height, height, Width, width, n_channels]
    tiled_frame_writable = tiled_frame.reshape(
        (tiled_height, height, tiled_width, width, n_channels)
    )

    # [Height, Width, height, width, n_channels]
    tiled_frame_writable = tiled_frame_writable.transpose(0, 2, 1, 3, 4)

    b = 0
    for i in range(tiled_height):
        for j in range(tiled_width):
            if valid is None or valid[b]:
                # if time step is not valid, just leave black
                tiled_frame_writable[i, j] = images_nhwc[b]
            b += 1
            if b == n_images:
                return tiled_frame

    return tiled_frame
