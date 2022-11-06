# because this module is full of types from 3rd party packages, treat type
# annotations as strings and do not evaluate them
from __future__ import annotations

from os import PathLike
from pathlib import Path
from collections import defaultdict
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

# logger verbosity levels
DISABLED = 0
ERROR = 1
WARN = 2
INFO = 3
DEBUG = 4


class Logger:
    """
    The logger class.

    :param output_files: the list of output formats
    :param verbosity: the logging level (can be DEBUG=4, INFO=3, WARN=2, ERROR=1, DISABLED=0)
    """
    def __init__(self,
        stdout: bool = True,
        verbosity: int = INFO,
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
        wandb: Optional[wandb.Run] = None,
        stdout: bool = True,
        output_files: Dict[str, PathLike] = None,
        config: Dict[str, Any] = None,
        config_path: Optional[PathLike] = None,
        model_save_path: Optional[PathLike] = None,
        verbosity: int = INFO,
    ) -> None:

        if self.initialized:
            raise RuntimeError("Logging has already been initialized!")

        self.close() # clean up resources
        self.initialized = True

        # TODO: if a wandb is detected but none was passed, should wandb be
        # used by default?

        # log_dir defaults to wandb folder if using wandb
        if wandb is not None:
            log_dir = Path(wandb.dir)
        elif log_dir is not None:
            # if log_dir set manually, create it
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True)
        else: # all outputs must have absolute paths
            raise ValueError("Must specify either log_dir or use WandB")
            # TODO: add option to specify all paths absolutely

        if output_files is None:
            output_files = {}

        # if requested, add tensorboard to output files
        if tensorboard:
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
            # TODO: how to set other parameters like max_length for stdout?
            self.writers["stdout"] = StdOutWriter()

        # make config_path absolute
        if config_path is None:
            # default is log_dir/"config.json"
            config_path = Path("config.json")
        if not config_path.is_absolute():
            config_path = log_dir / config_path
        self.config_path = config_path

        # if given, write config to file
        if config is not None:
            serializer = JSONConfigSerializer()
            serializer.dump(config=config, path=config_path)
        
        # make model_save_path absolute
        if model_save_path is not None:
            model_save_path = Path(model_save_path)
            if not model_save_path.is_absolute():
                model_save_path = log_dir / model_save_path
        else:
            print("WARNING: the model will not be saved")
        self.model_save_path = model_save_path

        # TODO: each writer has its verbosity level
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
                wandb.save(
                    str(self.model_save_path),
                    base_path=self.model_save_path.parent,
                )

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
        # TODO: throw runtime exception here?

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
        self.writers.clear()


_logger = Logger(
    stdout=True, # by default, Logger only outputs to stdout
    verbosity=INFO,
)


# logger API
init = _logger.init
record = _logger.record
record_mean = _logger.record_mean
dump = _logger.dump
save_model = _logger.save_model
log = _logger.log
debug = _logger.debug
info = _logger.info
warn = _logger.warn
error = _logger.error
set_verbosity = _logger.set_verbosity
close = _logger.close