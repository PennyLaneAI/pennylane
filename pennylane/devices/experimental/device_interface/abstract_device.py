from typing import Callable, List, Union

from pennylane.tape.qscript import QuantumScript

# Explicit device utility functions
from .device_helpers import RegistrationsMetaclass
from .device_config import DeviceConfig

# Runtime specific utilities, such as pre and postprocessing annotations
from ..runtime_manager.runtime_helpers import FnType


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
    def register_execute(
        cls,
    ) -> Callable:
        """Decorator to register gradient methods of a device
        (contain developer-facing details here).
        """

        def wrapper(fn):
            exe_registrations = cls.registrations.setdefault(FnType.EXECUTE, fn)
            cls.registrations[FnType.EXECUTE] = exe_registrations

        return wrapper

    @classmethod
    def register_fn(cls, fn_label: str, fn_type: FnType = FnType.UNKNOWN, **kwargs) -> Callable:
        """Decorator to register arbitrary pre or post processing functions"""

        def wrapper(fn):
            fn_registrations = cls.registrations.setdefault(fn_type, {})
            fn_registrations.update({fn_label: fn})
            if len(kwargs) > 0:
                fn_registrations.update(kwargs)
            cls.registrations[fn_type] = fn_registrations

        return wrapper

    def gradient(self, qscript: QuantumScript, order: int = 1):
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
