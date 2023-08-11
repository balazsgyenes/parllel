from __future__ import annotations

import distutils.spawn
import distutils.version
import itertools
import os
import subprocess
from pathlib import Path
from typing import Iterator, Mapping, Optional, TypeVar

import numba
import numpy as np

import parllel.logger as logger
from parllel import Array, ArrayDict

try:
    import wandb

    # catch if wandb is a local import, e.g. if a wandb folder exists in the
    # current directory
    has_wandb = wandb.__file__ is not None
except ImportError:
    has_wandb = False

from .transform import BatchTransform


class RecordVectorizedVideo(BatchTransform):
    def __init__(
        self,
        output_dir: Path,
        batch_buffer: ArrayDict[Array],
        buffer_key_to_record: str,  # e.g. "observation" or "env_info.rendering"
        record_every_n_steps: int,
        video_length: int,
        env_fps: int | None = None,
        output_fps: int = 30,
        tiled_height: int | None = None,
        tiled_width: int | None = None,
        use_wandb: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.record_every = int(record_every_n_steps)
        self.length = int(video_length)
        self.keys = buffer_key_to_record.split(".")
        self.env_fps = env_fps if env_fps is not None else output_fps
        self.output_fps = output_fps
        self.use_wandb = (
            has_wandb and use_wandb
        )  # TODO: warning or error if use_wandb and !has_wandb

        self.output_dir.mkdir(parents=True)

        images = dict_get_nested(batch_buffer, self.keys)
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
            self.tiled_height, self.tiled_width = find_tiling(n_images)
            logger.debug(
                f"Automatically chose a tiling of {self.tiled_height}x"
                f"{self.tiled_width} for vectorized video recording."
            )

        self.tiled_frame = np.zeros(
            (self.tiled_height * height, self.tiled_width * width, n_channels),
            dtype=np.uint8,
        )

        # for writing to the tiled frame on each step, create a view where the
        # individual tiles can be indexed
        self.tiled_frame_writable = self.tiled_frame.reshape(
            self.tiled_height, height, self.tiled_width, width, n_channels
        )

        self.total_t = 0  # used for naming video files
        # force first batch to be recorded
        self.steps_since_last_recording = self.record_every
        self.recording = False

    def __call__(self, batch_samples: ArrayDict[Array]) -> ArrayDict[Array]:
        # check if we should start recording
        if not self.recording:
            if self.steps_since_last_recording + self.batch_size >= self.record_every:
                self._start_recording()

        if self.recording:
            self._record_batch(batch_samples)

        # update counters after processing batch
        self.total_t += self.batch_size
        self.steps_since_last_recording += self.batch_size

        return batch_samples

    def _start_recording(self) -> None:
        # calculate exact time point where recording should start
        offset_steps = max(0, self.record_every - self.steps_since_last_recording)
        self.offset_t = offset_steps // self.batch_B
        start_total_t = self.total_t + offset_steps

        self.path = self.output_dir / f"policy_video_step_{start_total_t}.mp4"
        logger.log(f"Started recording video of policy to {self.path}")
        self.recorder = ImageEncoder(
            output_path=str(self.path),
            frame_shape=self.tiled_frame.shape,
            frames_per_sec=self.env_fps,
            output_frames_per_sec=self.output_fps,
        )
        self.steps_since_last_recording -= self.record_every
        self.recorded_frames = 0
        self.recording = True

    def _record_batch(self, batch_samples: ArrayDict[Array]) -> None:
        images_batch = dict_get_nested(batch_samples, self.keys)
        valid_batch = batch_samples.get("valid", None)

        # convert to numpy arrays
        images_batch = np.asarray(images_batch)
        if valid_batch is not None:
            valid_batch = np.asarray(valid_batch)

        # if this is the start of recording, delay start to arrive at exact
        # desired start point
        if self.recorded_frames == 0:
            images_batch = images_batch[self.offset_t :]
            if valid_batch is not None:
                valid_batch = valid_batch[self.offset_t :]

        # loop through time, saving images from each step to file
        for images, valid in zip_with_valid(images_batch, valid_batch):
            write_tiles_to_frame(
                images=images,
                valid=valid,
                tiled_height=self.tiled_height,
                tiled_width=self.tiled_width,
                tiled_frame=self.tiled_frame_writable,
            )
            self.recorder.capture_frame(self.tiled_frame)
            self.recorded_frames += 1

            if self.recorded_frames >= self.length:
                self._stop_recording()
                break

            # if all environments are done, continue recording from next batch
            if valid is not None and not np.any(valid):
                break

    def _stop_recording(self) -> None:
        # TODO: use weakref to ensure this gets closed even if training ends
        # during video recording (user forgets to call close method)
        self.recorder.close()
        self.recording = False
        logger.debug(f"Finished recording video of policy to {self.path}")
        if self.use_wandb:
            wandb.log({"video": wandb.Video(str(self.path), format="mp4")})

    def close(self) -> None:
        if self.recording:
            self._stop_recording()


ValueType = TypeVar("ValueType")


def dict_get_nested(buffer: Mapping[str, ValueType], keys: list[str]) -> ValueType:
    result = buffer
    for key in keys:
        result = result[key]
    return result


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


def zip_with_valid(
    array: np.ndarray,
    valid: np.ndarray | None,
) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
    if valid is None:
        valid = itertools.repeat(None)
    yield from zip(array, valid)


@numba.njit
def write_tiles_to_frame(
    images: np.ndarray,
    valid: np.ndarray | None,
    tiled_height: int,
    tiled_width: int,
    tiled_frame: np.ndarray,
) -> None:
    """Write images from individual environments into a preallocated tiled
    frame. Transposes image channels from torch order to image order and
    freezes images if the image data is no longer valid.
    """
    n_images = images.shape[0]
    images_hwc = images.transpose(0, 2, 3, 1)  # move channel dimension to the end

    b = 0
    for i in range(tiled_height):
        for j in range(tiled_width):
            if valid is None or valid[b]:
                # if time step is not valid, just leave the existing tile
                # this appears to "freeze" an environments after it is done
                tiled_frame[i, :, j] = images_hwc[b]
            b += 1
            if b == n_images:
                return


class ImageEncoder(object):
    """Writes a sequence of frames to an mp4 file on disk.

    Copied from gym v0.21
    """

    def __init__(self, output_path, frame_shape, frames_per_sec, output_frames_per_sec):
        self.proc: Optional[subprocess.Popen] = None
        self.output_path = output_path
        # Frame shape should be lines-first, so w and h are swapped
        h, w, pixfmt = frame_shape
        if pixfmt != 3 and pixfmt != 4:
            raise ValueError(
                "Your frame has shape {}, but we require (w,h,3) or (w,h,4), i.e., RGB values for a w-by-h image, with an optional alpha channel.".format(
                    frame_shape
                )
            )
        self.wh = (w, h)
        self.includes_alpha = pixfmt == 4
        self.frame_shape = frame_shape
        self.frames_per_sec = frames_per_sec
        self.output_frames_per_sec = output_frames_per_sec

        if distutils.spawn.find_executable("avconv") is not None:
            self.backend = "avconv"
        elif distutils.spawn.find_executable("ffmpeg") is not None:
            self.backend = "ffmpeg"
        else:
            raise RuntimeError(
                """Found neither the ffmpeg nor avconv executables. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`."""
            )

        self.start()

    @property
    def version_info(self):
        return {
            "backend": self.backend,
            "version": str(
                subprocess.check_output(
                    [self.backend, "-version"], stderr=subprocess.STDOUT
                )
            ),
            "cmdline": self.cmdline,
        }

    def start(self):
        if self.backend == "ffmpeg":
            self.cmdline = (
                self.backend,
                "-nostats",
                "-loglevel",
                "error",  # suppress warnings
                "-y",
                # input
                "-f",
                "rawvideo",
                "-s:v",
                "{}x{}".format(*self.wh),
                "-pix_fmt",
                ("bgr32" if self.includes_alpha else "bgr24"),
                "-r",
                "%d" % self.frames_per_sec,
                "-i",
                "-",  # this used to be /dev/stdin, which is not Windows-friendly
                # output
                "-an",
                "-r",
                "%d" % self.frames_per_sec,
                "-vcodec",
                "libx264", # NOTE: browsers apparently don't support mpeg4 codecs anymore, also much smaller files
                "-pix_fmt",
                "yuv420p",
                "-r",
                "%d" % self.output_frames_per_sec,
                self.output_path,
            )
        else:
            self.cmdline = (
                self.backend,
                "-nostats",
                "-loglevel",
                "error",  # suppress warnings
                "-y",
                # input
                "-f",
                "rawvideo",
                "-s:v",
                "{}x{}".format(*self.wh),
                "-pix_fmt",
                ("rgb32" if self.includes_alpha else "rgb24"),
                "-framerate",
                "%d" % self.frames_per_sec,
                "-i",
                "-",  # this used to be /dev/stdin, which is not Windows-friendly
                # output
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "%d" % self.output_frames_per_sec,
                self.output_path,
            )

        logger.debug('Starting %s with "%s"', self.backend, " ".join(self.cmdline))
        if hasattr(os, "setsid"):  # setsid not present on Windows
            self.proc = subprocess.Popen(
                self.cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid
            )
        else:
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

    def capture_frame(self, frame):
        if not isinstance(frame, (np.ndarray, np.generic)):
            raise TypeError(
                "Wrong type {} for {} (must be np.ndarray or np.generic)".format(
                    type(frame), frame
                )
            )
        if frame.shape != self.frame_shape:
            raise ValueError(
                "Your frame has shape {}, but the VideoRecorder is configured for shape {}.".format(
                    frame.shape, self.frame_shape
                )
            )
        if frame.dtype != np.uint8:
            raise TypeError(
                "Your frame has data type {}, but we require uint8 (i.e. RGB values from 0-255).".format(
                    frame.dtype
                )
            )

        try:
            if distutils.version.LooseVersion(
                np.__version__
            ) >= distutils.version.LooseVersion("1.9.0"):
                self.proc.stdin.write(frame.tobytes())
            else:
                self.proc.stdin.write(frame.tostring())
        except Exception as e:
            stdout, stderr = self.proc.communicate()
            logger.error("VideoRecorder encoder failed: %s", stderr)

    def close(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            logger.error("VideoRecorder encoder exited with status {}".format(ret))
