from .logger import Logger, Verbosity


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
