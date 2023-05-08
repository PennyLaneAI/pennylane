import logging
import os
from logging import Filter


class LocalProcessFilter(Filter):
    """
    Filters logs not originating from the current executing Python process ID.
    """

    def __init__(self):
        self._pid = os.getpid()

    def filter(self, record):
        if record.process == self._pid:
            return True
        return False


class DebugOnlyFilter(Filter):
    """
    Filters logs that are less verbose than the DEBUG level (CRITICAL, ERROR, WARN & INFO).
    """

    def filter(self, record):
        if record.levelno > logging.DEBUG:
            return False
        return True
