from pathlib import Path
from typing import Optional

from .logger import Logger, Verbosity
from .serializers import JSONConfigSerializer


# create singleton Logger instance
_logger = Logger(
    stdout=True, # by default, Logger only outputs to stdout
    verbosity=Verbosity.INFO,
)


# logger API
init = _logger.init
check_init = _logger.check_init
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
log_dir: Optional[Path] = None # updated by logger in log_dir.setter
model_save_path: Optional[Path] = None # updated by logger in model_save_path.setter


__all__ = [
    "init",
    "check_init",
    "record",
    "record_mean",
    "dump",
    "save_model",
    "log",
    "debug",
    "info",
    "warn",
    "error",
    "set_verbosity",
    "close",
    "log_dir",
    JSONConfigSerializer,
]
