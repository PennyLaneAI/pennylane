"""

"""

from typing import Union, Callable
from dataclasses import dataclass


@dataclass
class ExecutionConfig:
    """Configuration dataclass to support runtime execution of given workloads.

    Args:
        shots (bool): Indicate whether finite-shots are enabled.
        grad (bool): Indicate whether gradients are enabled.
        preproc (Union[None, Callable]): Provides support for a preprocessing function outside of the gradient tracking logic.
        postproc (Union[None, Callable]): Provides support for a postprocessing function outside of the gradient tracking logic.
        interface (Union[None, InterfaceType]): Defines the expected autodifferentiation framework explicitly.
    """

    shots: int = 0
    interface: Union[None, str] = None
    diff_method: Union[None, str, Callable] = None
    cache_size: int = 10000  # Set to 0 to disable cache#
    max_expansion: int = 10
    max_diff: int = 1
    grad_args: Union[None, dict] = None
