# because this module is full of types from 3rd party packages, treat type
# annotations as strings and do not evaluate them
from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
import sys
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from gym.wrappers.monitoring import video_recorder
import numpy as np
# from matplotlib import pyplot as plt

try:
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.tensorboard.summary import hparams
except ImportError:
    torch = None
    SummaryWriter = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import wandb
except ImportError:
    wandb = None


class Video:
    """
    Stores data that should be saved to a video and its associated metadata.

    :param data_or_path: numpy array to create the video from or the filepath
        where the video can be found
    :param fps: frames per second that the video was "recorded" at
    :param outout_fps: frames per second that the video should be written at,
        if different from the recorded fps.
    :param file_format: if the video is a file, what file format (available
        formats are "gif", "mp4", "webm" or "ogg")
    :param channel_format: if the video is a numpy array, the order of the
        axes (available formats are "CHW" and "HWC")
    """

    def __init__(self,
        data_or_path: Union[np.ndarray, PathLike],
        fps: Union[float, int],
        output_fps: Optional[int] = None,
        file_format: Optional[str] = None,
        channel_format: Optional[str] = None,
    ) -> None:
        self.data_or_path = data_or_path
        self.fps = fps
        self.output_fps = output_fps or fps
        self.file_format = file_format
        self.channel_format = channel_format

        self._is_file = not isinstance(data_or_path, np.ndarray)

    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        if self._is_file:
            return None
        return self.data_or_path.shape[1:]

    @property
    def is_file(self) -> bool:
        return self._is_file

    def as_video_file(self, path: PathLike, file_format: str) -> Video:
        """Return a new Video object representing the video file that this
        video has been converted to.
        """
        return Video(
            data_or_path=path,
            fps=self.output_fps,
            output_fps=None,
            file_format=file_format,
            channel_format=None,
        )


class Figure:
    """
    Figure data class storing a matplotlib figure and whether to close the figure after logging it

    :param figure: figure to log
    :param close: if true, close the figure after logging it
    """

    def __init__(self, figure: plt.figure, close: bool) -> None:
        self.figure = figure
        self.close = close


class Image:
    """
    Image data class storing an image and data format

    :param image: image to log
    :param dataformats: Image data format specification of the form NCHW, NHWC, CHW, HWC, HW, WH, etc.
        More info in add_image method doc at https://pytorch.org/docs/stable/tensorboard.html
        Gym envs normally use 'HWC' (channel last)
    """

    def __init__(self, image: Union[np.ndarray, str], dataformats: str) -> None:
        self.image = image
        self.dataformats = dataformats


class HParam:
    """
    Hyperparameter data class storing hyperparameters and metrics in dictionaries

    :param hparam_dict: key-value pairs of hyperparameters to log
    :param metric_dict: key-value pairs of metrics to log
        A non-empty metrics dict is required to display hyperparameters in the corresponding Tensorboard section.
    """

    def __init__(self,
        hparam_dict: Dict[str, Union[bool, str, float, int, None]],
        metric_dict: Dict[str, Union[float, int]],
    ) -> None:
        self.hparam_dict = hparam_dict
        if not metric_dict:
            raise Exception("`metric_dict` must not be empty to display hyperparameters to the HPARAMS tensorboard tab.")
        self.metric_dict = metric_dict


class FormatUnsupportedError(NotImplementedError):
    """
    Custom error to display informative message when
    a value is not supported by some formats.

    :param unsupported_formats: A sequence of unsupported formats,
        for instance ``["stdout"]``.
    :param value_description: Description of the value that cannot be logged by this format.
    """

    def __init__(self, unsupported_formats: Sequence[str], value_description: str):
        if len(unsupported_formats) > 1:
            format_str = f"formats {', '.join(unsupported_formats)} are"
        else:
            format_str = f"format {unsupported_formats[0]} is"
        super().__init__(
            f"The {format_str} not supported for the {value_description} value logged.\n"
            f"You can exclude formats via the `exclude` parameter of the logger's `record` function."
        )


class LogWriter:

    _writer_classes = {}

    def __init_subclass__(cls, /, name: Optional[str] = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # register subclass if defined with a name
        if name is not None:
            cls._writer_classes[name] = cls

    def __new__(cls, *args, name: Optional[str] = None, **kwargs):
        # if instantiating a subclass directly, just create that class
        if cls != LogWriter:
            return super().__new__(cls)
        # otherwise look up name in dictionary of registered subclasses
        if name is None:
            raise ValueError("Missing required keyword-only argument 'name'")
        if name not in cls._writer_classes:
            raise RuntimeError(f"No writer registered under name {name}")
        return super().__new__(cls._writer_classes[name])

    def __init__(self, filename: PathLike, name: Optional[str] = None) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """
        Close owned resources
        """
        pass


class KeyValueWriter:
    """
    Key Value writer
    """
    def write(self, key_values: Dict[str, Any], step: int = 0) -> None:
        """
        Write a dictionary to file

        :param key_values:
        :param step:
        """
        raise NotImplementedError


class VideoWriter:
    """A logwriter type that can accept video as input. Returns a new Video
    object if the video was encoded to file as part of logging, otherwise
    returns None.
    """
    def write_video(self, key: str, video: Video, step: int) -> Optional[Video]:
        raise NotImplementedError


class MessageWriter:
    """
    A writer capable of writing messages submitted using e.g. `logger.log` or
    `logger.warn`.
    """
    def write_message(self, sequence: List) -> None:
        """
        Write a message to the log file.

        :param sequence:
        # TODO: update these parameters and type hints
        """
        raise NotImplementedError


class TxtFileWriter(KeyValueWriter, MessageWriter, LogWriter, name="txt"):
    """A human-readable output format producing ASCII tables of key-value pairs.

    Set attribute ``max_length`` to change the maximum length of keys and values
    to write to output (or specify it when calling ``__init__``).

    :param filename_or_file: the file to write the log to
    :param max_length: the maximum length of keys and values to write to output.
        Outputs longer than this will be truncated. An error will be raised
        if multiple keys are truncated to the same value. The maximum output
        width will be ``2*max_length + 7``. The default of 36 produces output
        no longer than 79 characters wide.
    """

    def __init__(self,
        filename: PathLike,
        max_length: int = 36,
        name: Optional[str] = None,
    ) -> None:
        self.max_length = max_length
        self.file = open(filename, "wt")

    def write(self, key_values: Dict[str, Any], step: int = 0) -> None:
        # Create strings for printing
        key2str = {}
        for key, value in key_values.items():

            if isinstance(value, (Video, Figure, Image, HParam)):
                raise FormatUnsupportedError(["stdout", "log"], type(value).__name__)

            elif isinstance(value, float):
                # Align left
                value_str = f"{value:<8.3g}"
            else:
                value_str = str(value)

            tag = None
            if key.find("/") > 0:  # Find tag and add it to the dict
                tag, key = key.split("/", maxsplit=1)
                tag = tag + "/"
                key2str[(tag, self._truncate(tag))] = ""
                # Remove tag from key
                key = "   " + key

            truncated_key = self._truncate(key)
            if (tag, truncated_key) in key2str:
                raise ValueError(
                    f"Key '{key}' truncated to '{truncated_key}' that already exists. Consider increasing `max_length`."
                )
            key2str[(tag, truncated_key)] = self._truncate(value_str)

        if len(key2str) == 0:
            warnings.warn("Tried to write empty key-value dict")
            return

        # Find max widths
        tagless_keys = map(lambda x: x[1], key2str.keys())
        key_width = max(map(len, tagless_keys))
        val_width = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (key_width + val_width + 7)
        lines = [dashes]
        for (_, key), value in key2str.items():
            key_space = " " * (key_width - len(key))
            val_space = " " * (val_width - len(value))
            lines.append(f"| {key}{key_space} | {value}{val_space} |")
        lines.append(dashes)

        if tqdm is not None and hasattr(self.file, "name") and self.file.name == "<stdout>":
            # Do not mess up with progress bar
            tqdm.write("\n".join(lines) + "\n", file=sys.stdout, end="")
        else:
            self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, string: str) -> str:
        if len(string) > self.max_length:
            string = string[: self.max_length - 3] + "..."
        return string

    def write_message(self, sequence: List) -> None:
        sequence = list(sequence)
        for i, elem in enumerate(sequence):
            self.file.write(elem)
            if i < len(sequence) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """
        self.file.close()


class StdOutWriter(TxtFileWriter): # must be created explicitly
    def __init__(self, max_length: int = 36):
        self.max_length = max_length
        self.file = sys.stdout

    def close(self) -> None:
        # do not try to close sys.stdout
        pass


class JSONWriter(KeyValueWriter, LogWriter, name="json"):
    """
    Log to a file, in the JSON format

    :param filename: the file to write the log to
    """

    def __init__(self, filename: PathLike, name: Optional[str] = None):
        self.file = open(filename, "wt")

    def write(self, key_values: Dict[str, Any], step: int = 0) -> None:
        def cast_to_json_serializable(value: Any):
            if isinstance(value, (Video, Figure, Image, HParam)):
                raise FormatUnsupportedError(["csv"], type(value).__name__)
            if hasattr(value, "dtype"):
                if value.shape == () or len(value) == 1:
                    # if value is a dimensionless numpy array or of length 1, serialize as a float
                    return float(value)
                else:
                    # otherwise, a value is a numpy array, serialize as a list or nested lists
                    return value.tolist()
            return value

        key_values = {
            key: cast_to_json_serializable(value)
            for key, value in key_values.items()
        }
        self.file.write(json.dumps(key_values) + "\n")
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """
        self.file.close()


class CSVWriter(KeyValueWriter, LogWriter, name="csv"):
    """
    Log to a file, in a CSV format

    :param filename: the file to write the log to
    """

    def __init__(self, filename: PathLike, name: Optional[str] = None):
        self.file = open(filename, "w+t")
        self.keys = []
        self.separator = ","
        self.quotechar = '"'

    def write(self, key_values: Dict[str, Any], step: int = 0) -> None:
        # Add our current row to the history
        extra_keys = key_values.keys() - self.keys
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, key) in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(key)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.separator * len(extra_keys))
                self.file.write("\n")
        for i, key in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            value = key_values.get(key)

            if isinstance(value, (Video, Figure, Image, HParam)):
                raise FormatUnsupportedError(["csv"], type(value).__name__)

            elif isinstance(value, str):
                # escape quotechars by prepending them with another quotechar
                value = value.replace(self.quotechar, self.quotechar + self.quotechar)

                # additionally wrap text with quotechars so that any delimiters in the text are ignored by csv readers
                self.file.write(self.quotechar + value + self.quotechar)

            elif value is not None:
                self.file.write(str(value))
            
        self.file.write("\n")
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """
        self.file.close()


class TensorBoardWriter(KeyValueWriter, LogWriter, name="tensorboard"):
    """
    Dumps key/value pairs into TensorBoard's numeric format.

    :param folder: the folder to write the log to
    """

    def __init__(self, folder: PathLike, name: Optional[str] = None):
        assert SummaryWriter is not None, "tensorboard is not installed, you can use " "pip install tensorboard to do so"
        self.writer = SummaryWriter(log_dir=folder)

    def write(self, key_values: Dict[str, Any], step: int = 0) -> None:

        for key, value in key_values.items():

            if isinstance(value, np.ScalarType):
                if isinstance(value, str):
                    # str is considered a np.ScalarType
                    self.writer.add_text(key, value, step)
                else:
                    self.writer.add_scalar(key, value, step)

            elif torch is not None and isinstance(value, torch.Tensor):
                self.writer.add_histogram(key, value, step)

            elif isinstance(value, Video):
                self.writer.add_video(key, value.frames, step, value.fps)

            elif isinstance(value, Figure):
                self.writer.add_figure(key, value.figure, step, close=value.close)

            elif isinstance(value, Image):
                self.writer.add_image(key, value.image, step, dataformats=value.dataformats)

            elif isinstance(value, HParam):
                # we don't use `self.writer.add_hparams` to have control over the log_dir
                experiment, session_start_info, session_end_info = hparams(value.hparam_dict, metric_dict=value.metric_dict)
                self.writer.file_writer.add_summary(experiment)
                self.writer.file_writer.add_summary(session_start_info)
                self.writer.file_writer.add_summary(session_end_info)

        # Flush the output to the file
        self.writer.flush()

    def close(self) -> None:
        """
        closes the file
        """
        if self.writer:
            self.writer.close()
            self.writer = None


class WandBWriter(KeyValueWriter, VideoWriter, LogWriter):
    def __init__(self) -> None:
        if wandb is None:
            raise RuntimeError(
                "WandB is not installed. Install with `pip install wandb`"
            )
        if wandb.run is None:
            raise ValueError(
                "WandB is not initialized. Call `wandb.init` with arguments."
            )

    def write(self, key_values: Dict[str, Any], step: int = 0) -> None:
        for key, value in key_values.items():
            if isinstance(value, np.ScalarType):
                pass
            elif isinstance(value, Image):
                key_values[key] = wandb.Image(
                    value.image,
                    mode=value.dataformats,
                )

        wandb.log(key_values, step=step)

    def write_video(self, key: str, video: Video, step: int) -> Optional[Path]:
        if not video.is_file:
            assert video.channel_format == "CHW"

        wandb_video = wandb.Video(
            data_or_path=video.data_or_path,
            caption=f"step_{step}",
            fps=video.fps,
            format="mp4" if video.is_file else None,
        )
        wandb.log({key: wandb_video}, commit=False)


class MP4Writer(VideoWriter, LogWriter, name="mp4"):
    def __init__(self, folder: PathLike, name: Optional[str] = None) -> None:
        self.output_dir = Path(folder)
    
    def write_video(self, key: str, video: Video, step: int) -> Optional[Path]:
        path = self.output_dir / f"{key}_step_{step}.mp4"

        assert not video.is_file, "MP4Writer is for writing numpy arrays to disk"

        video_data = video.data_or_path

        # TODO: make more general?
        if video.channel_format == "CHW":
            video_data = video_data.transpose(0, 2, 3, 1)

        recorder = video_recorder.ImageEncoder(
            output_path=str(path),
            frame_shape=video.frame_shape,
            frames_per_sec=video.fps,
            output_frames_per_sec=video.output_fps,
        )

        for frame in video_data:
            recorder.capture_frame(frame)

        recorder.close()
        
        return video.as_video_file(path=path, file_format="mp4")
