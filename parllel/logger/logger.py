# because this module is full of types from 3rd party packages, treat type
# annotations as strings and do not evaluate them
from __future__ import annotations

from collections import defaultdict
from enum import IntEnum
from os import PathLike
from pathlib import Path
from typing import Any
import warnings

import numpy as np

try:
    import wandb
    # catch if wandb is a local import, e.g. if a wandb folder exists in the
    # current directory
    has_wandb = wandb.__file__ is not None
except ImportError:
    has_wandb = False

import parllel

from .serializers import JSONConfigSerializer
from .logwriters import LogWriter, KeyValueWriter, MessageWriter, StdOutWriter


class Verbosity(IntEnum):
    DISABLED = 0
    ERROR = 1
    WARN = 2
    INFO = 3
    DEBUG = 4


# for now, all warnings from this module should be converted to errors
warnings.filterwarnings("error", module=__name__)
# except these ones, which should remain warnings
warnings.filterwarnings("default", message="No logger output will be saved to disk!", module=__name__)
warnings.filterwarnings("default", message="No trained models will be saved to disk!", module=__name__)
warnings.filterwarnings("default", message="No config information will be saved to disk!", module=__name__)


class Logger:
    """The logger class. This class is effectively a singleton, whose API is
    accessible through the parllel.logger module. To initialize the logger,
    call parllel.logger.init().
    """
    def __init__(self,
        stdout: bool,
        verbosity: Verbosity,
    ):
        self.writers: dict[str, LogWriter] = {}

        if stdout:
            self.writers["stdout"] = StdOutWriter()

        self.verbosity = verbosity
        self._model_save_path = None
        self.use_wandb = False
        self._log_dir = None

        self.values = defaultdict(float)  # values this iteration
        self.counts = defaultdict(int)
        self.excluded_writers = defaultdict(str)
        self.initialized = False

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    @log_dir.setter
    def log_dir(self, log_dir: Path) -> None:
        # after setting member variable, also set value in logger API
        self._log_dir = log_dir
        import parllel.logger as logger
        logger.log_dir = log_dir

    @property
    def model_save_path(self) -> Path:
        return self._model_save_path

    @model_save_path.setter
    def model_save_path(self, model_save_path: Path) -> None:
        self._model_save_path = model_save_path
        import parllel.logger as logger
        logger.model_save_path = model_save_path

    def init(self,
        log_dir: PathLike | None = None,
        tensorboard: bool = False, # TODO: add passing tensorboard dir explicitly
        wandb_run: "wandb.Run" | None = None,
        stdout: bool = True,
        stdout_max_length: int | None = None,
        output_files: dict[str, PathLike] | None = None,
        config: dict[str, Any] | None = None,
        config_path: PathLike = "config.json",
        model_save_path: PathLike | None = None,
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
        :param output_files: a dict of files to write key-value pairs to, where
            the keys are the file type (txt, json, csv), and the values are the
            filepaths
        :param config: a dict of config values to save to a json file in a form
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

        self.close() # clean default writers created by __init__

        # TODO: if a wandb is detected but none was passed, should wandb be
        # used by default?
        use_wandb = self.use_wandb = (has_wandb and wandb_run is not None)
        self.missing_wandb = False
        if use_wandb:
            # check if user did not call wandb.init with sync_tensorboard=True
            wandb_sync_tensorboard = len(wandb.patched['tensorboard']) > 0
            if not wandb_sync_tensorboard:
                warnings.warn(
                    "No data will be logged to WandB because WandB is not set "
                    "to copy data logged to Tensorboard. Please call "
                    "wandb.init with `sync_tensorboard=True`."
                )

            # check if user forgot to pass tensorboard=True
            if wandb_sync_tensorboard and not tensorboard:
                # TODO: just log to tensorboard even if not requested?
                warnings.warn(
                    "No data will be logged to WandB because parllel was not "
                    "requested to log to Tensorboard. Please call "
                    "parllel.logger.init with `tensorboard=True`."
                )

            # TODO: if no config was passed, take config from wandb
        elif has_wandb and wandb.run is not None:
            self.missing_wandb = True
            warnings.warn(
                "No data will be logged to WandB because the run was not "
                "passed to parllel logger. Please call parllel.logger.init "
                "with `wandb_run=wandb.run`."
            )

        # determine log_dir by checking options in order of preference
        if use_wandb and not wandb_run.disabled:
            # wandb takes priority if given and not disabled
            log_dir = Path(wandb_run.dir)
        elif log_dir is not None:
            # if log_dir set manually, create it
            log_dir = Path(log_dir)
            # error if specified log_dir already exists
            log_dir.mkdir(parents=True)
        elif use_wandb:
            # if wandb is disabled, use its log_dir instead of raising error
            log_dir = Path(wandb_run.dir)
        else: # all outputs must have absolute paths
            raise ValueError("Must specify either log_dir or use WandB")
            # TODO: add option to specify all paths absolutely
        self.log_dir = log_dir

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
            warnings.warn("No logger output will be saved to disk!")

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
        else:
            warnings.warn("No config information will be saved to disk!")
        
        # make model_save_path absolute
        # if using wandb, set up live syncing of model
        # TODO: add other model saving schemes (e.g. all, best, etc.)
        if model_save_path is not None:
            model_save_path = Path(model_save_path)
            model_base_path = None
            if not model_save_path.is_absolute():
                model_save_path = log_dir / model_save_path
                # on web UI, model will be at original relative path
                model_base_path = str(log_dir)
            self.model_base_path = model_base_path

            if use_wandb:
                wandb.save(
                    str(model_save_path),
                    base_path=model_base_path,
                )

            # create directory if necessary
            model_save_path.parent.mkdir(parents=True, exist_ok=True)

        else:
            warnings.warn("No trained models will be saved to disk!")
        self.model_save_path = model_save_path

        # TODO: give each writer its own verbosity level
        self.verbosity = verbosity
        
        self.values.clear()
        self.counts.clear()
        self.excluded_writers.clear()

        self.initialized = True

    def check_init(self) -> None:
        if self.initialized:
            # check if user initialized wandb after initializing parllel
            if (
                has_wandb and wandb.run is not None # wandb run exists
                and not self.use_wandb # everything called properly
                and not self.missing_wandb # warning was already issued    
            ):
                warnings.warn(
                    "No data will be logged to WandB because wandb.init was "
                    "called after parllel.logger.init. Please call wandb.init "
                    "first and pass the result to parllel.logger.init with "
                    "`wandb_run=wandb.run`."
                )
        else:
            # check if user initialized wandb without initializing parllel
            if has_wandb and wandb.run is not None:
                # TODO: should we just initialize parllel logging with defaults?
                warnings.warn(
                    "Even though a WandB run exists, no data will be logged "
                    "at all because parllel logging has not been initialized. "
                    "Please call parllel.logger.init."
                )

    def record(self,
        key: str,
        value: Any,
        do_not_write_to: str | tuple[str, ...] | None = "",
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
        value: np.ndarray | list[float] | float,
        do_not_write_to: str | tuple[str, ...] | None = "",
    ) -> None:
        """
        The same as record(), but if called many times, values averaged.

        :param key: save to log this key
        :param value: save to log this value
        :param do_not_write_to: outputs to be excluded

        TODO: also log std deviation using RunningMeanStd class
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

    def save_model(self, agent: "parllel.handlers.agent.Agent"):
        if self.model_save_path is not None:
            agent.save_model(self.model_save_path)

            # # sync model immediately with wandb server
            # # TODO: do we need to save immediately, or let wandb sync slowly?
            # if self.use_wandb:
            #     wandb.save(
            #         str(self.model_save_path),
            #         base_path=self.model_base_path,
            #         policy="now",
            #     )

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
