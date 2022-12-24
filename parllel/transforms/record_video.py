from pathlib import Path
from typing import Optional, Sequence, Tuple

from gym.wrappers.monitoring import video_recorder
import numpy as np

from parllel.buffers import Samples, NamedTuple
import parllel.logger as logger

from .transform import StepTransform


def tile_images(img_nhwc: Sequence[np.ndarray]) -> np.ndarray:  # pragma: no cover
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.

    :param img_nhwc: list or array of images, ndim=4 once turned into array. img nhwc
        n = batch index, h = height, w = width, c = channel
    :return: img_HWc, ndim=3

    # TODO: calculate new_height and new_width once and preallocate 
    """
    img_nhwc = np.asarray(img_nhwc)
    img_nhwc = np.moveaxis(img_nhwc, -3, -1) # move channel dimension to the end
    n_images, height, width, n_channels = img_nhwc.shape # TODO: verify
    tiled_height = int(np.ceil(np.sqrt(n_images)))
    tiled_width = int(np.ceil(float(n_images) / tiled_height))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(n_images, tiled_height * tiled_width)])
    # img_HWhwc
    out_image = img_nhwc.reshape((tiled_height, tiled_width, height, width, n_channels))
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape((tiled_height * height, tiled_width * width, n_channels))
    return out_image


class RecordVectorizedVideo(StepTransform):
    def __init__(self,
        output_dir: Path,
        buffer_key_to_record: str, # e.g. "observation" or "env_info.rendering"
        record_every_n_steps: int,
        video_length: int,
        tiled_shape: Tuple[int], # TODO: calculate this on init
        env_fps: int, # TODO: maybe grab this from example env metadata
        output_fps: int = 30,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.record_every = record_every_n_steps
        self.length = video_length
        self.key = buffer_key_to_record
        self.output_fps = output_fps

        self.output_dir.mkdir(parents=True)

        self.total_t = 0
        self.recording = False
        self.tiled_shape = tiled_shape
        self.env_fps = env_fps

    def __call__(self, batch_samples: Samples, t: int) -> Samples:
        # TODO: save batch size as variable
        batch_B = batch_samples.env.done[t].shape[0]
        self.total_t += batch_B

        # check if we should start recording
        # TODO: does this need to be corrected for recurrent batches that
        # break early? maybe a batch transform would be better
        if not self.recording:
            # TODO: count steps since last recording to ensure trigger is not
            # missed if batch_size does not divide record_every
            self.recording = (self.total_t % self.record_every == 0)
            if self.recording:
                self._start_recording()

        if self.recording:
            images = getattr(batch_samples.env, self.key)[t]
            # TODO: freeze frames of environments that are done
            images_tiled = tile_images(np.asarray(images))
            self.recorder.capture_frame(images_tiled)
            self.recorded_frames += 1
            if self.recorded_frames >= self.length:
                self._stop_recording()

    def _start_recording(self) -> None:
        path = self.output_dir / f"policy_video_step_{self.total_t}.mp4"
        logger.log(f"Started recording video of policy to {path}")
        self.recorder = video_recorder.ImageEncoder(
            output_path=str(path),
            frame_shape=self.tiled_shape,
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
