from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from gym.wrappers.monitoring import video_recorder
import numba
import numpy as np
try:
    import wandb
except ImportError:
    wandb = None

from parllel.buffers import Buffer, Samples
import parllel.logger as logger

from .transform import BatchTransform


class RecordVectorizedVideo(BatchTransform):
    def __init__(self,
        output_dir: Path,
        batch_buffer: Samples,
        buffer_key_to_record: str, # e.g. "observation" or "env_info.rendering"
        record_every_n_steps: int,
        video_length: int,
        env_fps: int = 4,
        output_fps: Optional[int] = None,
        tiled_height: Optional[int] = None,
        tiled_width: Optional[int] = None,
    ) -> None:
        # TODO: add max_envs to restrict the number of envs to include in video
        # TODO: add image resizing if renderings are too large
        self.output_dir = Path(output_dir)
        self.record_every = int(record_every_n_steps)
        self.length = int(video_length)
        self.keys = buffer_key_to_record.split(".")
        self.env_fps = env_fps
        self.output_fps = output_fps if output_fps is not None else env_fps

        self.output_dir.mkdir(parents=True)

        images = buffer_get_nested(batch_buffer.env, self.keys)
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

        self.total_t = 0 # used for naming video files
        # force first batch to be recorded
        self.steps_since_last_recording = self.record_every
        self.recording = False

    def __call__(self, batch_samples: Samples) -> Samples:
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
        self.recorder = video_recorder.ImageEncoder(
            output_path=str(self.path),
            frame_shape=self.tiled_frame.shape,
            frames_per_sec=self.env_fps,
            output_frames_per_sec=self.output_fps,
        )
        self.steps_since_last_recording -= self.record_every
        self.recorded_frames = 0
        self.recording = True

    def _record_batch(self, batch_samples: Samples) -> None:
        images_batch = buffer_get_nested(batch_samples.env, self.keys)
        valid_batch = getattr(batch_samples.env, "valid", None)
        
        # convert to numpy arrays
        images_batch = np.asarray(images_batch)
        if valid_batch is not None:
            valid_batch = np.asarray(valid_batch)

        # if this is the start of recording, delay start to arrive at exact
        # desired start point
        if self.recorded_frames == 0:
            images_batch = images_batch[self.offset_t:]
            if valid_batch is not None:
                valid_batch = valid_batch[self.offset_t:]

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
        if logger.use_wandb:
            logger.debug("Uploading recorded video to wandb...")
            wandb.log({"policy_videos": wandb.Video(str(self.path))}, commit=False)

    def close(self) -> None:
        if self.recording:
            self._stop_recording()


def buffer_get_nested(buffer: Buffer, keys: List[str]) -> Buffer:
    result = buffer
    for key in keys:
        result = getattr(result, key)
    return result


def find_tiling(n_images: int) -> Tuple[int, int]:
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
            i for i in range(max_tiled_height, min_tiled_height - 1, -1) if n_images % i == 0
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
    valid: Optional[np.ndarray],
) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
    if valid is None:
        for elem in array:
            yield elem, valid
    else:
        yield from zip(array, valid)


@numba.njit
def write_tiles_to_frame(
    images: np.ndarray,
    valid: Optional[np.ndarray],
    tiled_height: int,
    tiled_width: int,
    tiled_frame: np.ndarray,
) -> np.ndarray:
    """Write images from individual environments into a preallocated tiled
    frame. Transposes image channels from torch order to image order and
    freezes images if the image data is no longer valid.
    """
    n_images = images.shape[0]
    images_hwc = images.transpose(0, 2, 3, 1) # move channel dimension to the end

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
