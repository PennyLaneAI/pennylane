import logging, logging.config
import os

path = os.path.dirname(__file__)
# from .formatters.formatter import LogFormatter


def enable_logging(use_yaml=False):
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
