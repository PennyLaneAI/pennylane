import logging, logging.config
import os

# Define a more verbose mode for the messages. Not currently controlled by internal log configurations.
TRACE = 1
path = os.path.dirname(__file__)


def enable_logging(use_yaml=False):
    """
    This method allows top selectively enable logging throughout PennyLane.
    All configurations are read through the `log_config.toml` or `log_config.yaml` files, selectively controlled via the `use_yaml` argument.
    """
    if use_yaml:
        import yaml

        with open(os.path.join(path, "log_config.yaml"), "r") as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
    else:
        import pytoml

        with open(os.path.join(path, "log_config.toml"), "r") as f:
            config = pytoml.load(f)
            logging.config.dictConfig(config)

    # Enable a more verbose mode than DEBUG.
    # Used to enable inspection of function definitions in log messages.
    def trace(self, message, *args, **kws):
        self._log(TRACE, message, args, **kws)

    logging.addLevelName(TRACE, "TRACE")
    lc = logging.getLoggerClass()
    lc.trace = trace
