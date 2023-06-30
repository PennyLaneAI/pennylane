import dataclasses
import inspect
import warnings

from cachetools import LRUCache
from typing import Callable, Optional, Union, Iterable

import pennylane as qml
from pennylane.tape import make_qscript
from pennylane.interfaces import SUPPORTED_INTERFACES
from pennylane.transforms.core import TransformProgram, TransformContainer
from pennylane.measurements import Shots
from pennylane.typing import Result, ResultBatch


from pennylane.devices.experimental import ExecutionConfig as DeviceConfig

from .build_workflow import build_workflow
from .core_transforms import get_default_core_transforms

_gradient_transform_map = {
    "parameter-shift": qml.gradients.param_shift,
    "finite-diff": qml.gradients.finite_diff,
    "spsa": qml.gradients.spsa_grad,
    "hadamard": qml.gradients.hadamard_grad,
}


class QNode:

    _qfunc_uses_shots_arg = False  # bool: whether or not the qfunc has shots as a keyword argument

    def __init__(
        self,
        func: Callable,
        device: qml.devices.experimental.Device,
        interface="auto",
        diff_method: Optional[Union[str, qml.gradients.gradient_transform]] = "best",
        grad_on_execution="best",
        cache: int = True,
        cachesize: int = 10000,
        max_diff: int = 1,
        gradient_kwargs: Optional[dict] = None,
        shots: Optional[Union[int, Iterable[Union[int, Iterable[int]]]]] = None,
    ):
        # Setting initialization properties
        self._func = func
        self._device = device
        self._interface = interface
        self._diff_method = diff_method
        self._grad_on_execution = grad_on_execution
        self._max_diff = max_diff
        self._gradient_kwargs = gradient_kwargs
        self._shots = Shots(shots)
        self._cache = cache
        self._cachesize = cachesize

        # initialization
        self._transform_program = TransformProgram(get_default_core_transforms(self._interface))
        self._inner_transform_program = TransformProgram()
        if self.gradient_transform:

            def gradient_preprocessing(tape):
                def dummy_post_processing(results: ResultBatch) -> Result:
                    return results[0]

                return (self.gradient_transform.expand_fn(tape),), dummy_post_processing

            gradient_preprocessing = TransformContainer(gradient_preprocessing)
            self._transform_program.append(gradient_preprocessing)

    def __repr__(self):
        return (
            f"<QNode( {self.func},\n\t dev={self.device},\n\t shots={self.shots}"
            f",\n\t interface={self.interface}"
            f",\n\t diff_method={self.diff_method},\n\t grad_on_execution={self.grad_on_execution}"
            f",\n\t max_diff={self.max_diff},\n\t gradient_kwargs={self.gradient_kwargs})"
        )

    @property
    def func(self) -> Callable:
        """
        User provided quantum function.
        """
        return self._func

    @property
    def device(self) -> qml.devices.experimental.Device:
        """
        Device to perform the execution on.
        """
        return self._device

    @property
    def interface(self) -> Optional[str]:
        """ """
        return self._interface

    @property
    def diff_method(self) -> Optional[Union[str, qml.gradients.gradient_transform]]:
        """ """
        return self._diff_method

    @property
    def grad_on_execution(self) -> Union[str, bool]:
        """ """
        return self._grad_on_execution

    @property
    def max_diff(self) -> int:
        """ """
        return self._max_diff

    @property
    def gradient_kwargs(self) -> dict:
        """ """
        return self._gradient_kwargs

    @property
    def shots(self) -> Shots:
        """ """
        return self._shots

    def _validate_state(self):

        if self._interface not in SUPPORTED_INTERFACES:
            raise qml.QuantumFunctionError(
                f"Unknown interface {self.interface}. Interface must be "
                f"one of {SUPPORTED_INTERFACES}."
            )

        if not isinstance(self._device, qml.devices.experimental.Device):
            raise qml.QuantumFunctionError(
                "Invalid device. Device must be a valid PennyLane device."
            )

        if "shots" in inspect.signature(self._func).parameters:
            warnings.warn(
                "Detected 'shots' as an argument to the given quantum function. "
                "The 'shots' argument name is reserved for overriding the number of shots "
                "taken by the device. Its use outside of this context should be avoided.",
                UserWarning,
            )
            self._qfunc_uses_shots_arg = True

    @property
    def gradient_transform(self) -> Optional[qml.gradients.gradient_transform]:
        if isinstance(self.diff_method, qml.gradients.gradient_transform):
            return self._diff_method
        return _gradient_transform_map.get(self.diff_method, None)

    @property
    def transform_program(self) -> TransformProgram:
        return self._transform_program

    @property
    def inner_transform_program(self) -> TransformProgram:
        return self._inner_transform_program

    def construct_workflow(self, circuit):

        initial_device_config = DeviceConfig(
            grad_on_execution=None if self._grad_on_execution == "best" else self.grad_on_execution,
            gradient_method=self.diff_method,
            derivative_order=self.max_diff,
        )
        processed_device_config = self.device.setup_configuration((circuit,), initial_device_config)

        if not processed_device_config.use_device_gradient and self.diff_method == "best":
            gradient_method = qml.gradients.param_shift
        else:
            gradient_method = self.gradient_transform or self.diff_method

        interface = self.interface
        if self.interface == "auto":
            interface = qml.math.get_interface(*circuit.get_parameters())
        if processed_device_config.gradient_method == "backprop":
            interface = None

        return build_workflow(
            self.device,
            processed_device_config,
            ml_interface=interface,
            gradient_method=gradient_method,
            gradient_kwargs=self.gradient_kwargs,
            inner_program=self.inner_transform_program,
            outer_program=self.transform_program,
            cache=self._cache,
            max_diff=self._max_diff,
        )

    def __call__(self, *args, **kwargs):

        shots = Shots(None) if self._qfunc_uses_shots_arg else kwargs.pop("shots", self.shots)

        circuit = make_qscript(self.func, shots=shots)(*args, **kwargs)
        workflow = self.construct_workflow(circuit)
        result_batch = workflow((circuit,))
        return result_batch[0]
