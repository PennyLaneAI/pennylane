"""This module enables support for log-level messaging throughout PennyLane, following the native Python logging framework interface. Please see the official documentation for details on usage https://docs.python.org/3/library/logging.html"""

from .configure import configure_logging
from .configure import enable_logging
from .configure import TRACE
