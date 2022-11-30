from typing import Callable, Union
from enum import IntEnum, auto
from dataclasses import dataclass, field


class FnType(IntEnum):
    """Enum datatype to aid organisation of registration function types"""

    PREPROCESS = auto()
    EXECUTE = auto()
    POSTPROCESS = auto()
    GRADIENT = auto()
    VJP = auto()
    UNKNOWN = auto()


class InterfaceType(IntEnum):
    """Enum to specify supported interface types"""

    AUTO = auto()
    AUTOGRAD = auto()
    JAX = auto()
    TF = auto()
    TORCH = auto()


class DiffType(IntEnum):
    """Enum to specify supported interface types"""

    NONE = auto()
    DEVICE = auto()  # Device-native registered method (adjoint, backprop, etc)
    PARAMSHIFT = auto()  # Always guaranteed to work
    FINITEDIFF = auto()  # Always guaranteed to work


class ExpansionStrategy(IntEnum):
    """Enum to specify supported interface types"""

    DEVICE = auto()  # Device-native registered method (adjoint, backprop, etc)
    GRADIENT = auto()


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
    interface: Union[None, InterfaceType] = InterfaceType.AUTOGRAD
    diff_method: Union[None, DiffType] = DiffType.DEVICE
    cache_size: int = 10000  # Set to 0 to disable cache#
    max_expansion: int = 10
    max_diff: int = 1
    grad_args: dict = field(default_factory=dict)
