import logging, logging.config
import os

# Define a more verbose mode for the messages. Not currently controlled by internal log configurations.
TRACE = 1
path = os.path.dirname(__file__)


def configure_logging(config_file):
    """
    This method allows custom logging configuration throughout PennyLane.
    All configurations are read through config_file `toml` or `yaml` files.
    """
    if config_file.endswith(".yaml"):
        import yaml

        with open(os.path.join(path, config_file), "r") as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
    elif config_file.endswith(".toml"):
        import pytoml

        with open(os.path.join(path, config_file), "r") as f:
            config = pytoml.load(f)
            logging.config.dictConfig(config)
    else:
        raise NotImplementedError("Logging configuration expects a yaml or toml file.")

    # Enable a more verbose mode than DEBUG.
    # Used to enable inspection of function definitions in log messages.
    def trace(self, message, *args, **kws):
        self._log(TRACE, message, args, **kws)

    logging.addLevelName(TRACE, "TRACE")
    lc = logging.getLoggerClass()
    lc.trace = trace


def enable_logging(use_yaml=False):
    """
    This method allows top selectively enable logging throughout PennyLane.
    All configurations are read through the `log_config.toml` or `log_config.yaml` files, selectively controlled via the `use_yaml` argument.
    """
    if use_yaml:
        configure_logging("log_config.yaml")
    else:
        configure_logging("log_config.toml")
