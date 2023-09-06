from __future__ import annotations

import functools
import itertools
from operator import getitem
from os import PathLike
from pathlib import Path
from typing import Iterator, Mapping, TypeVar

import gymnasium as gym
import numba
import numpy as np

import parllel.logger as logger
from parllel import Array, ArrayDict

from .transform import Transform


class RecordVectorizedVideo(Transform):
    def __init__(
        self,
        output_dir: str | PathLike,
        sample_tree: ArrayDict[Array],
        buffer_key_to_record: str,  # e.g. "observation" or "env_info.rendering"
        video_length: int,
        env_fps: int = 30,
        n_envs: int | None = None,
        tiled_height: int | None = None,
        tiled_width: int | None = None,
        torch_order: bool = False,  # TODO: replace with channel spec
        use_wandb: bool = False,  # TODO: replace with wandb logwriter
    ) -> None:
        self.output_dir = Path(output_dir)
        self.length = int(video_length)
        self.keys = buffer_key_to_record.split(".")
        self.env_fps = env_fps
        self.torch_order = torch_order
        self.use_wandb = use_wandb

        try:
            import moviepy
        except ImportError as e:
            raise gym.error.DependencyNotInstalled(
                "moviepy is not installed, run `pip install moviepy`"
            ) from e

        if self.use_wandb:
            try:
                import wandb
            except ImportError as e:
                raise gym.error.DependencyNotInstalled(
                    "wandb is not installed, run `pip install wandb`"
                ) from e

        self.output_dir.mkdir(parents=True)

        images = dict_get_nested(sample_tree, self.keys)
        if len(images.shape) != 5:
            # TODO: maybe specify error message based on user-given data order
            raise ValueError(
                f"Expected images to be 5-dimensional (batch_T, batch_B, and 3 image dimensions), not {len(images.shape)}-dimensional."
            )

        batch_B = batch_B = images.shape[1]
        if n_envs is not None:
            if n_envs > batch_B:
                raise ValueError(
                    f"Number of requested environment ({n_envs}) greater than number of environments available ({batch_B})."
                )

            self.n_images = n_envs
        else:
            self.n_images = images.shape[1]

        if tiled_height is not None and tiled_width is not None:
            tiled_height, tiled_width = int(tiled_height), int(tiled_width)
            if tiled_height * tiled_width < self.n_images:
                raise ValueError(
                    f"Tiled height and width {tiled_height}x{tiled_width} are "
                    f"too small to allocate space for {self.n_images} images."
                )
            self.tiled_height, self.tiled_width = tiled_height, tiled_width
        else:
            self.tiled_height, self.tiled_width = self.find_tiling(self.n_images)
            logger.debug(
                f"Automatically chose a tiling of {self.tiled_height}x"
                f"{self.tiled_width} for vectorized video recording."
            )

        self.recording = False
        self.recorded_frames = []

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

    def start_recording(self, video_name_suffix: str, delay_t: int = 0) -> None:
        self.recording = True
        self.delay_t = delay_t
        self.video_name_suffix = video_name_suffix
        logger.debug(f"Started recording video of policy.")

    def __call__(self, sample_tree: ArrayDict[Array]) -> ArrayDict[Array]:
        if self.recording:
            images_batch = dict_get_nested(sample_tree, self.keys)
            assert isinstance(images_batch, Array)
            assert (n_batch_dims := len(images_batch.batch_shape)) in (1, 2)
            valid_batch = sample_tree.get("valid", None)

            # convert to numpy arrays
            images_batch = np.asarray(images_batch)
            if valid_batch is not None:
                valid_batch = np.asarray(valid_batch)

            # ensure 2 batch dimensions and index time and batch accordingly
            b_loc = slice(self.n_images)
            if n_batch_dims == 2:
                # if this is the start of recording, delay start to arrive at exact desired start
                # point
                t_loc = (
                    slice(self.delay_t, None)
                    if len(self.recorded_frames) == 0
                    else slice(None)
                )
            elif n_batch_dims == 1:
                assert self.delay_t == 0
                # add leading singleton time dimension
                t_loc = np.newaxis
            else:
                raise ValueError(
                    f"Received images with {n_batch_dims} batch dimensions."
                )
            images_batch = images_batch[t_loc, b_loc]
            if valid_batch is not None:
                valid_batch = valid_batch[t_loc, b_loc]

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
                    self.stop_recording()
                    break

                # if all environments are done, continue recording from next batch
                if valid is not None and not np.any(valid):
                    break

        return sample_tree

    def stop_recording(self) -> None:
        # TODO: replace this with call to logger, which handles outputs
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        clip = ImageSequenceClip(self.recorded_frames, fps=self.env_fps)
        path = self.output_dir / f"policy_step_{self.video_name_suffix}.mp4"
        clip.write_videofile(str(path), logger=None)
        logger.info(f"Saved video of policy to {path}.")

        if self.use_wandb:
            import wandb

            if wandb.run is not None:
                # NOTE: this will add an extra step in the log
                wandb.log(
                    {"video": wandb.Video(str(path), fps=self.env_fps, format="mp4")}
                )

        self.recording = False
        del self.delay_t
        del self.video_name_suffix
        self.recorded_frames.clear()

    def close(self) -> None:
        # TODO: should this be called by a finalizer?
        if self.recording:
            self.stop_recording()


ValueType = TypeVar("ValueType")


def dict_get_nested(mapping: Mapping[str, ValueType], keys: list[str]) -> ValueType:
    return functools.reduce(getitem, keys, mapping)


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
