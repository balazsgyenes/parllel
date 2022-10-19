# because this module is full of types from 3rd party packages, treat type
# annotations as strings and do not evaluate them
from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
import sys
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union

import numpy as np
import torch as torch
# from matplotlib import pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.tensorboard.summary import hparams
except ImportError:
    SummaryWriter = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

from parllel.logging import log_config
from parllel.handlers.agent import Agent


# logger API
_logger = None
record = None
record_mean = None
dump = None
save_model = None
log = None
debug = None
info = None
warn = None
error = None
set_verbosity = None
close = None

# logger verbosity levels
DISABLED = 0
ERROR = 1
WARN = 2
INFO = 3
DEBUG = 4


class Video:
    """
    Video data class storing the video frames and the frame per seconds

    :param frames: frames to create the video from
    :param fps: frames per second
    """

    def __init__(self, frames: torch.Tensor, fps: Union[float, int]):
        self.frames = frames
        self.fps = fps


class Figure:
    """
    Figure data class storing a matplotlib figure and whether to close the figure after logging it

    :param figure: figure to log
    :param close: if true, close the figure after logging it
    """

    def __init__(self, figure: plt.figure, close: bool):
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

    def __init__(self, image: Union[torch.Tensor, np.ndarray, str], dataformats: str):
        self.image = image
        self.dataformats = dataformats


class HParam:
    """
    Hyperparameter data class storing hyperparameters and metrics in dictionaries

    :param hparam_dict: key-value pairs of hyperparameters to log
    :param metric_dict: key-value pairs of metrics to log
        A non-empty metrics dict is required to display hyperparameters in the corresponding Tensorboard section.
    """

    def __init__(self, hparam_dict: Dict[str, Union[bool, str, float, int, None]], metric_dict: Dict[str, Union[float, int]]):
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

    def __init_subclass__(cls, /, **kwargs) -> None:
        name = kwargs.pop("name", None)
        super().__init_subclass__(**kwargs)
        # register subclass if defined with a name
        if name is not None:
            cls._writer_classes[name] = cls

    def __new__(cls, *args, **kwargs):
        # if instantiating a subclass directly, just create that class
        if cls != LogWriter:
            return super().__new__(cls)
        # otherwise look up name in dictionary of registered subclasses
        try:
            name = kwargs["name"]
        except KeyError:
            raise ValueError("Missing required keyword-only argument 'name'")
        try:
            return super().__new__(cls._writer_classes[name])
        except KeyError:
            raise RuntimeError(f"No writer registered under name {name}")

    def __init__(self, file: PathLike, **kwargs) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """
        Close owned resources
        """
        raise NotImplementedError


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
        filename_or_file: Union[str, Path, TextIO],
        max_length: int = 36,
        **kwargs,
    ) -> None:
        self.max_length = max_length
        # TODO: this can be made safer, since we know what we expect
        if isinstance(filename_or_file, (str, Path)):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "write"), f"Expected file or str, got {filename_or_file}"
            self.file = filename_or_file
            self.own_file = False

    def write(self, key_values: Dict, step: int = 0) -> None:
        # Create strings for printing
        key2str = {}
        tag = None
        for key, value in key_values.items():

            if isinstance(value, (Video, Figure, Image, HParam)):
                raise FormatUnsupportedError(["stdout", "log"], type(value).__name__)

            elif isinstance(value, float):
                # Align left
                value_str = f"{value:<8.3g}"
            else:
                value_str = str(value)

            if key.find("/") > 0:  # Find tag and add it to the dict
                tag = key[: key.find("/") + 1]
                key2str[(tag, self._truncate(tag))] = ""
            # Remove tag from key
            if tag is not None and tag in key:
                key = str("   " + key[len(tag) :])

            truncated_key = self._truncate(key)
            if (tag, truncated_key) in key2str:
                raise ValueError(
                    f"Key '{key}' truncated to '{truncated_key}' that already exists. Consider increasing `max_length`."
                )
            key2str[(tag, truncated_key)] = self._truncate(value_str)

        # Find max widths
        if len(key2str) == 0:
            warnings.warn("Tried to write empty key-value dict")
            return
        else:
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
        if self.own_file:
            self.file.close()


class StdOutWriter(TxtFileWriter): # must be created explicitly
    def __init__(self, max_length: int = 36, **kwargs):
        super().__init__(sys.stdout, max_length, **kwargs)


class JSONWriter(KeyValueWriter, LogWriter, name="json"):
    """
    Log to a file, in the JSON format

    :param filename: the file to write the log to
    """

    def __init__(self, filename: PathLike, **kwargs):
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

    def __init__(self, filename: PathLike, **kwargs):
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

    def __init__(self, folder: PathLike, **kwargs):
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

            if isinstance(value, torch.Tensor):
                self.writer.add_histogram(key, value, step)

            if isinstance(value, Video):
                self.writer.add_video(key, value.frames, step, value.fps)

            if isinstance(value, Figure):
                self.writer.add_figure(key, value.figure, step, close=value.close)

            if isinstance(value, Image):
                self.writer.add_image(key, value.image, step, dataformats=value.dataformats)

            if isinstance(value, HParam):
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


class Logger:
    """
    The logger class.

    :param output_files: the list of output formats
    :param verbosity: the logging level (can be DEBUG=4, INFO=3, WARN=2, ERROR=1, DISABLED=0)
    """
    def __init__(self,
        output_files: Dict[str, Path] = None,
        stdout: bool = True,
        verbosity: int = INFO,
        model_save_path: Path = None,
        use_wandb: bool = False,
    ):
        self.writers: Dict[str, LogWriter] = {}
        for _format, path in output_files.items():
            # TODO: how to set other parameters like max_length for stdout?
            self.writers[_format] = LogWriter(path, name=_format)

        if stdout:
            self.writers["stdout"] = StdOutWriter()

        if not output_files:
            # TODO: maybe using warnings module
            print("WARNING: no output will be logged")

        self.verbosity = verbosity # TODO: each writer has its verbosity level
        self.model_save_path = model_save_path
        self.use_wandb = use_wandb

        self.values = defaultdict(float)  # values this iteration
        self.counts = defaultdict(int)
        self.excluded_writers = defaultdict(str)

        # TODO: print info about where logs are saved

    def record(self,
        key: str,
        value: Any,
        do_not_write_to: Optional[Union[str, Tuple[str, ...]]] = "",
    ) -> None:
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.

        :param key: save to log this key
        :param value: save to log this value
        :param do_not_write_to: outputs to be excluded
        """
        self.values[key] = value
        self.excluded_writers[key] = do_not_write_to

    def record_mean(self,
        key: str,
        value: Union[np.ndarray, List[float], float],
        do_not_write_to: Optional[Union[str, Tuple[str, ...]]] = "",
    ) -> None:
        """
        The same as record(), but if called many times, values averaged.

        :param key: save to log this key
        :param value: save to log this value
        :param do_not_write_to: outputs to be excluded
        """
        if isinstance(value, list):
            value = np.array(value)
        if isinstance(value, np.ndarray):
            n = len(value)
            batch_mean = np.mean(value)
        else:
            n = 1
            batch_mean = value
        old_val, count = self.values[key], self.counts[key]
        delta = batch_mean - old_val
        new_count = count + n
        self.values[key] = old_val + delta * n / new_count
        self.counts[key] = new_count
        self.excluded_writers[key] = do_not_write_to

    def dump(self, step: int = 0) -> None:
        """
        Write all of the diagnostics from the current iteration
        """
        if self.verbosity == DISABLED:
            return
        for writer_name, writer in self.writers.items():
            if isinstance(writer, KeyValueWriter):
                values = {
                    key: value
                    for key, value in self.values.items()
                    if writer_name not in self.excluded_writers[key]
                }
                writer.write(values, step)

        self.values.clear()
        self.counts.clear()
        self.excluded_writers.clear()

    def save_model(self, agent: Agent):
        if self.model_save_path is not None:
            agent.save_model(self.model_save_path)
            if self.use_wandb:
                # sync model with wandb server
                wandb.save(str(self.model_save_path), base_path=self.model_save_path.parent)

    def log(self, *args, level: int = INFO) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).

        level: int. (see logger.py docs) If the global logger level is higher than
                    the level argument here, don't print to stdout.

        :param args: log the arguments
        :param level: the logging level (can be DEBUG=4, INFO=3, WARN=2, ERROR=1, DISABLED=0)
        """
        if self.verbosity >= level:
            for writer in self.writers.values():
                if isinstance(writer, MessageWriter):
                    writer.write_message(map(str, args))

    def debug(self, *args) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).
        Using the DEBUG level.

        :param args: log the arguments
        """
        self.log(*args, level=DEBUG)

    def info(self, *args) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).
        Using the INFO level.

        :param args: log the arguments
        """
        self.log(*args, level=INFO)

    def warn(self, *args) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).
        Using the WARN level.

        :param args: log the arguments
        """
        self.log(*args, level=WARN)
        # TODO: throw warning, maybe with warnings module

    def error(self, *args) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).
        Using the ERROR level.

        :param args: log the arguments
        """
        self.log(*args, level=ERROR)
        # TODO: throw runtime exception here

    # Configuration
    # ----------------------------------------
    def set_verbosity(self, verbosity: int) -> None:
        """
        Set logging threshold on current logger.

        :param verbosity: the logging level (can be DEBUG=4, INFO=3, WARN=2, ERROR=1, DISABLED=0)
        """
        self.verbosity = verbosity

    def close(self) -> None:
        """
        closes the file
        """
        for writer in self.writers.values():
            writer.close()


class DefaultLogger(Logger):
    # TODO: is this class necessary, or is the default logger just the logger
    # with default arguments?
    def __init__(self) -> None:
        self.warned = False

    def _warn_no_logger(self) -> None:
        if not self.warned:
            print("Logging is not enabled! Call `parllel.logger.init()`")
            self.warned = True

    def log(self, *args, **kwargs) -> None:
        self._warn_no_logger()

    record = log
    record_mean = log
    dump = log
    debug = log
    info = log
    warn = log
    error = log
    set_verbosity = log
    close = log


def _set_logger(new_logger: Logger):
    # TODO: is there a cleaner paradigm for global resources?
    globals()["_logger"] = new_logger

    # API calls need to point to bound methods of new logger object
    globals()["record"] = _logger.record
    globals()["record_mean"] = _logger.record_mean
    globals()["dump"] = _logger.dump
    globals()["save_model"] = _logger.save_model
    globals()["log"] = _logger.log
    globals()["debug"] = _logger.debug
    globals()["info"] = _logger.info
    globals()["warn"] = _logger.warn
    globals()["error"] = _logger.error
    globals()["set_verbosity"] = _logger.set_verbosity
    globals()["close"] = _logger.close


def init(
    log_dir: Optional[PathLike] = None,
    tensorboard: bool = False, # TODO: add passing tensorboard dir explicitly
    wandb: Optional[wandb.Run] = None,
    stdout: bool = True,
    output_files: Dict[str, PathLike] = None,
    config: Dict[str, Any] = None,
    config_path: Optional[PathLike] = None,
    model_save_path: Optional[PathLike] = None,
    verbosity: int = INFO,
) -> None:
    if not isinstance(_logger, DefaultLogger):
        raise RuntimeError("Logging has already been initialized!")

    # TODO: can the presence of a wandb run be automatically detected?
    # also want to prevent the user from initializing parllel logging
    # before wandb logging

    # log_dir defaults to wandb folder if using wandb
    if wandb is not None:
        log_dir = Path(wandb.dir)
    elif log_dir is not None:
        # if log_dir set manually, create it
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True)
    else: # add outputs must have absolute paths
        # TODO: add option to specify all paths absolutely
        assert not tensorboard, "Not implemented yet"

    # if requested, add tensorboard to output files
    if tensorboard:
        output_files["tensorboard"] = ""

    # make relative paths absolute by prepending log_dir
    for name, path in output_files.items():
        path = Path(path)
        if not path.is_absolute():
            path = log_dir / path
        output_files[name] = path

    # make model_save_path absolute
    if model_save_path is not None:
        model_save_path = Path(model_save_path)
        if not model_save_path.is_absolute():
            model_save_path = log_dir / model_save_path

    # make Logger object and assign it to module globals
    logger = Logger(
        output_files=output_files,
        stdout=stdout,
        verbosity=verbosity,
        model_save_path=model_save_path,
        use_wandb=(wandb is not None),
    )
    _set_logger(logger)

    # make config_path absolute
    if config_path is None:
        config_path = Path("config.json")
    if not config_path.is_absolute():
        config_path = log_dir / config_path

    # if given, write config to file
    if config is not None:
        log_config(config=config, path=config_path)


# create default logger to alert user that logging has not been initialized
_set_logger(DefaultLogger())
