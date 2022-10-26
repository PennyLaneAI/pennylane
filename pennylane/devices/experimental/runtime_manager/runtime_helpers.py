from typing import Callable, Union
from enum import IntEnum, auto
from ..device_interface.device_helpers import DeviceType


class FnType(IntEnum):
    """Enum datatype to aid organisation of registration function types"""

    PREPROCESS = auto()
    EXECUTE = auto()
    POSTPROCESS = auto()
    GRADIENT = auto()
    VJP = auto()
    UNKNOWN = auto()


class ExecutionConfig:
    """Configuration dataclass to support execution of given device workloads.

    Args:
        device_type (DeviceType): Indicate the type of the given device.
        shots (bool): Indicate whether finite-shots are enabled.
        grad (bool): Indicate whether gradients are enabled.
        preproc (Union[None, Callable]): Provides support for a preprocessing function outside of the gradient tracking logic.
        postproc (Union[None, Callable]): Provides support for a postprocessing function outside of the gradient tracking logic.
    """

    device_type: DeviceType = DeviceType.UNKNOWN
    shots: int = 0
    grad: Union[None, Callable] = None
    preproc: Union[None, Callable] = None
    postproc: Union[None, Callable] = None
