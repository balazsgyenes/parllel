from pathlib import Path
from typing import List, Optional, Tuple

from gym.wrappers.monitoring import video_recorder
from numba import njit
import numpy as np

from parllel.buffers import Buffer, Samples
import parllel.logger as logger

from .transform import StepTransform


class RecordVectorizedVideo(StepTransform):
    def __init__(self,
        output_dir: Path,
        batch_buffer: Samples,
        buffer_key_to_record: str, # e.g. "observation" or "env_info.rendering"
        record_every_n_steps: int,
        video_length: int,
        env_fps: int, # TODO: maybe grab this from example env metadata
        output_fps: int = 30,
        tiled_height: Optional[int] = None,
        tiled_width: Optional[int] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.record_every = record_every_n_steps
        self.length = video_length
        self.keys = buffer_key_to_record.split(".")
        self.env_fps = env_fps
        self.output_fps = output_fps

        self.output_dir.mkdir(parents=True)

        images = buffer_get_nested(batch_buffer.env, self.keys)[0]
        n_images, n_channels, height, width = images.shape
        
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
                f"Automatically chose a tiling of {tiled_height}x{tiled_width} "
                "for vectorized video recording."
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

        self.total_t = 0
        self.recording = False

    def __call__(self, batch_samples: Samples, t: int) -> Samples:
        # TODO: save batch size as variable
        batch_B = batch_samples.env.done[t].shape[0]
        self.total_t += batch_B

        # check if we should start recording
        # TODO: does this need to be corrected for recurrent batches that
        # break early? maybe a batch transform would be better
        # TODO: maybe force recording to start at the beginning of a batch so
        # that all environments start out valid?
        if not self.recording:
            # TODO: count steps since last recording to ensure trigger is not
            # missed if batch_size does not divide record_every
            self.recording = (self.total_t % self.record_every == 0)
            if self.recording:
                self._start_recording()

        if self.recording:
            images = buffer_get_nested(batch_samples.env, self.keys)[t]
            images = np.asarray(images)
            valid = getattr(batch_samples.env, "valid", None)
            if valid is not None:
                valid = np.asarray(valid[t])
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

    def _start_recording(self) -> None:
        path = self.output_dir / f"policy_video_step_{self.total_t}.mp4"
        logger.log(f"Started recording video of policy to {path}")
        self.recorder = video_recorder.ImageEncoder(
            output_path=str(path),
            frame_shape=self.tiled_frame.shape,
            frames_per_sec=self.env_fps,
            output_frames_per_sec=self.output_fps,
        )
        self.recorded_frames = 0

    def _stop_recording(self) -> None:
        # TODO: use weakref to ensure this gets closed even if training ends
        # during video recording (transform has no close method)
        path = self.recorder.version_info["cmdline"][-1]
        self.recorder.close()
        self.recording = False
        logger.debug(f"Finished recording video of policy to {path}")

    def close(self) -> None:
        # TODO: call this in the cleanup of build method
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
            i for i in range(max_tiled_height, min_tiled_height - 1) if i % n_images == 0
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


@njit
def write_tiles_to_frame(
    images: np.ndarray,
    valid: Optional[np.ndarray],
    tiled_height: int,
    tiled_width: int,
    tiled_frame: np.ndarray,
) -> np.ndarray:
    """Write images from individual environments into a preallocated tiled
    frame.
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
