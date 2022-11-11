# because this module is full of types from 3rd party packages, treat type
# annotations as strings and do not evaluate them
from __future__ import annotations

from collections import defaultdict
from enum import IntEnum
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

from parllel.handlers.agent import Agent

from .serializers import JSONConfigSerializer
from .logwriters import LogWriter, KeyValueWriter, MessageWriter, StdOutWriter


class Verbosity(IntEnum):
    DISABLED = 0
    ERROR = 1
    WARN = 2
    INFO = 3
    DEBUG = 4


class Logger:
    """The logger class. This class is effectively a singleton, whose API is
    accessible through the parllel.logger module. To initialize the logger,
    call parllel.logger.init().
    """
    def __init__(self,
        stdout: bool = True,
        verbosity: Verbosity = Verbosity.INFO,
    ):
        self.writers: Dict[str, LogWriter] = {}

        if stdout:
            self.writers["stdout"] = StdOutWriter()

        self.verbosity = verbosity
        self.model_save_path = None
        self.use_wandb = False

        self.values = defaultdict(float)  # values this iteration
        self.counts = defaultdict(int)
        self.excluded_writers = defaultdict(str)
        self.initialized = False

    def init(self,
        log_dir: Optional[PathLike] = None,
        tensorboard: bool = False, # TODO: add passing tensorboard dir explicitly
        wandb_run: Optional["wandb.Run"] = None,
        stdout: bool = True,
        stdout_max_length: Optional[int] = None,
        output_files: Dict[str, PathLike] = None,
        config: Dict[str, Any] = None,
        config_path: PathLike = "config.json",
        model_save_path: Optional[PathLike] = None,
        verbosity: Verbosity = Verbosity.INFO,
    ) -> None:
        """Initialize logging.
        :param log_dir: folder where all outputs are saved by default. Outputs
            specified at absolute paths ignore this.
        :param tensorboard: save outputs additionally to a tensorboard file
            stored in the log_dir?
        :param wandb_run: WandB run. If given, the WandB log folder overrides
            the log_dir.
        :param stdout: output additionally to standard output?
        :param stdout_max_length: maximum width of the tabular standard output
        :param output_files: a Dict of files to write key-value pairs to, where
            the keys are the file type (txt, json, csv), and the values are the
            filepaths
        :param config: a Dict of config values to save to a json file in a form
            that can be reloaded later
        :param config_path: the filepath for saving the config (default:
            config.json)
        :param model_save_path: the filepath for saving the model
        :param verbosity: the minimum verbosity that should be written to text
            output and standard out
        """
        if self.initialized:
            # TODO: allow reinitialization
            raise RuntimeError("Logging has already been initialized!")

        self.close() # clean up resources
        self.initialized = True

        # TODO: if a wandb is detected but none was passed, should wandb be
        # used by default?

        # determine log_dir by checking options in order of preference
        if wandb_run is not None and not wandb_run.disabled:
            # wandb takes priority if given and not disabled
            log_dir = Path(wandb_run.dir)
        elif log_dir is not None:
            # if log_dir set manually, create it
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True)
        elif wandb_run is not None:
            # if wandb is disabled, use its log_dir instead of raising error
            log_dir = Path(wandb_run.dir)
        else: # all outputs must have absolute paths
            raise ValueError("Must specify either log_dir or use WandB")
            # TODO: add option to specify all paths absolutely

        if output_files is None:
            output_files = {}

        # if requested, add tensorboard to output files
        if tensorboard:
            # tensorboard is special because the path should be a directory
            output_files["tensorboard"] = ""

        # make relative paths absolute by prepending log_dir
        for name, path in output_files.items():
            path = Path(path)
            if not path.is_absolute():
                path = log_dir / path
            output_files[name] = path

        # TODO: print info about where logs are saved

        for _format, path in output_files.items():
            self.writers[_format] = LogWriter(path, name=_format)

        if not self.writers:
            # TODO: maybe using warnings module
            print("WARNING: no output will be saved")

        if stdout:
            if stdout_max_length is None:
                self.writers["stdout"] = StdOutWriter()
            else:
                self.writers["stdout"] = StdOutWriter(stdout_max_length)

        config_path = Path(config_path)
        # make config_path absolute
        if not config_path.is_absolute():
            config_path = log_dir / config_path
        self.config_path = config_path

        # if given, write config to file
        if config is not None:
            serializer = JSONConfigSerializer()
            serializer.dump(config=config, path=config_path)
        
        # make model_save_path absolute
        # TODO: add other model saving schemes (e.g. all, best, etc.)
        if model_save_path is not None:
            model_save_path = Path(model_save_path)
            if not model_save_path.is_absolute():
                model_save_path = log_dir / model_save_path
        else:
            print("WARNING: the model will not be saved")
        self.model_save_path = model_save_path

        # TODO: give each writer its own verbosity level
        self.verbosity = verbosity
        self.use_wandb = (wandb is not None)
        
        self.values.clear()
        self.counts.clear()
        self.excluded_writers.clear()

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
        if self.verbosity == Verbosity.DISABLED:
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
                wandb.save(
                    str(self.model_save_path),
                    base_path=self.model_save_path.parent,
                )

    def log(self, *args, level: Verbosity = Verbosity.INFO) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).

        level: Verbosity. (see logger.py docs) If the global logger level is
            higher than the level argument here, don't print to stdout.

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
        self.log(*args, level=Verbosity.DEBUG)

    def info(self, *args) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).
        Using the INFO level.

        :param args: log the arguments
        """
        self.log(*args, level=Verbosity.INFO)

    def warn(self, *args) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).
        Using the WARN level.

        :param args: log the arguments
        """
        self.log(*args, level=Verbosity.WARN)
        # TODO: throw warning, maybe with warnings module

    def error(self, *args) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).
        Using the ERROR level.

        :param args: log the arguments
        """
        self.log(*args, level=Verbosity.ERROR)
        # TODO: throw runtime exception here?

    def set_verbosity(self, verbosity: Verbosity) -> None:
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
        self.writers.clear()