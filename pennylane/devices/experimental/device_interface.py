from enum import IntEnum, auto
from abc import ABC, abstractmethod
from typing import Callable, List, Union
from dataclasses import dataclass

import pennylane as qml
from pennylane.tape import QuantumScript
import sys

#########################################################################################
# Utility functionality set-up
#########################################################################################

class FnType(IntEnum):
    """Enum datatype to aid organisation of registration function types"""
    PREPROCESS = auto()
    PREPROCESS_TRACED = auto()
    EXECUTE = auto()
    POSTPROCESS_TRACED = auto()
    POSTPROCESS = auto()
    GRADIENT = auto()
    VJP = auto()
    UNKNOWN = auto()

class DeviceType(IntEnum):
    "Easily distinguish between physical and virtual devices"
    VIRTUAL = auto()
    PHYSICAL = auto()
    UNKNOWN = auto()

#########################################################################################
# Configuration data-classes for device setup and execution
#########################################################################################

# Use slots if available
dc_wrapper = dataclass
if sys.version_info.minor > 9:
    dc_wrapper = dataclass(slots=True)

@dataclass
class DeviceConfig:
    """Configuration dataclass to aid in device setup and initialization.

    Args:
        shots (bool): Indicate whether finite-shots are enabled.
        grad (bool): Indicate whether gradients are enabled.
        device_type (DeviceType): Indicate the type of the given device.
    """
    shots: bool = False
    grad: bool = False
    device_type: DeviceType = DeviceType.UNKNOWN


@dc_wrapper
class ExecutionConfig:
    """Configuration dataclass to support execution of given device workloads.

    Args:
        device_type (DeviceType): Indicate the type of the given device.
        shots (bool): Indicate whether finite-shots are enabled.
        grad (bool): Indicate whether gradients are enabled.
        preproc (Union[None, Callable]): Provides support for a preprocessing function outside of the gradient tracking logic.
        preproc_traced (Union[None, Callable]): Provides support for a preprocessing function inside of the gradient tracking logic.
        postproc (Union[None, Callable]): Provides support for a postprocessing function outside of the gradient tracking logic.
        postproc_traced (Union[None, Callable]): Provides support for a postprocessing function inside of the gradient tracking logic.
    """
    device_type: DeviceType = DeviceType.UNKNOWN
    shots: int = 0
    grad: Union[None, Callable] = None
    preproc: Union[None, Callable] = None
    preproc_traced: Union[None, Callable] = None
    postproc: Union[None, Callable] = None
    postproc_traced: Union[None, Callable] = None

#########################################################################################
# Device metaclass and abstract class setup
#########################################################################################

class RegistrationsMetaclass(type, ABC):
    def __new__(cls, name, bases, name_space):
        if not bases:
            return type.__new__(cls, name, bases, name_space)
        return super().__new__(cls, name, bases, dict(name_space, registrations={}))

class AbstractDevice(metaclass=RegistrationsMetaclass):
    """
    This abstract device interface enables direct and dynamic function registration for pre-processing, post-processing, gradients, VJPs, and arbitrary functionality.
    """

    def __init__(self, config: DeviceConfig = None, *args, **kwargs) -> None:
        self._config = config if config else DeviceConfig()

    @classmethod
    def register_gradient(cls, order=1) -> Callable:
        """Decorator to register gradient methods of a device
        (contain developer-facing details here).
        """
        def wrapper(fn):
            grad_registrations = cls.registrations.setdefault(FnType.GRADIENT, {})
            grad_registrations.update({order: fn})
            cls.registrations[FnType.GRADIENT] = grad_registrations
        return wrapper
    
    @classmethod
    def register_execute(cls, ) -> Callable:
        """Decorator to register gradient methods of a device
        (contain developer-facing details here).
        """
        def wrapper(fn):
            exe_registrations = cls.registrations.setdefault(FnType.EXECUTE, fn)
            cls.registrations[FnType.EXECUTE] = exe_registrations
        return wrapper
    
    @classmethod
    def register_fn(cls, fn_label: str, fn_type: FnType = FnType.UNKNOWN, **kwargs) -> Callable:
        """Decorator to register arbitrary pre or post processing functions
        """
        def wrapper(fn):
            fn_registrations = cls.registrations.setdefault(fn_type, {})
            fn_registrations.update({fn_label : fn})
            if len(kwargs) > 0:
                fn_registrations.update(kwargs)
            cls.registrations[fn_type] = fn_registrations
        return wrapper

    def gradient(self, qscript: QuantumScript, order: int=1):
        """Main gradient method, contains validation and post-processing
        so that device developers do not need to replicate all the
        internal pieces. Contain 'user' facing details here."""

        if FnType.GRADIENT not in self.registrations:
            raise ValueError("Device does not support derivatives")

        if order not in self.registrations[FnType.GRADIENT]:
            raise ValueError(f"Device does not support {order} order derivatives")

        grad = self.registrations[FnType.GRADIENT][order](self, qscript)

        # perform post-processing
        return grad
    
    def vjp(self, qscript: QuantumScript):
        """VJP method. Added through registration"""

        if FnType.VJP not in self.registrations:
            raise ValueError("Device does not support VJP")

        vjp = self.registrations[FnType.VJP](self, qscript)

        # perform post-processing
        return vjp

    def execute(self, qscript: Union[QuantumScript, List[QuantumScript]]):
        if FnType.EXECUTE not in self.registrations:
            raise ValueError("Device does not have an execute method.")

        return self.registrations[FnType.EXECUTE](self, qscript)
