# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Defines a LegacyDeviceFacade class for converting legacy devices to the
new interface.
"""

import warnings

# pylint: disable=not-callable
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import replace

from pennylane import math, ops
from pennylane.devices.capabilities import DeviceCapabilities
from pennylane.exceptions import (
    DecompositionUndefinedError,
    DeviceError,
    PennyLaneDeprecationWarning,
    QuantumFunctionError,
)
from pennylane.math import Interface, requires_grad
from pennylane.measurements import ExpectationMP, Shots
from pennylane.operation import Operator
from pennylane.ops import MidMeasure
from pennylane.tape import QuantumScript
from pennylane.transforms import broadcast_expand, defer_measurements, dynamic_one_shot
from pennylane.transforms.core import CompilePipeline, transform
from pennylane.wires import Wires

from ._legacy_device import Device as LegacyDevice
from .device_api import Device
from .execution_config import ExecutionConfig
from .modifiers import single_tape_support
from .preprocess import (
    decompose,
    no_sampling,
    validate_adjoint_trainable_params,
    validate_measurements,
)


def _requests_adjoint(execution_config):
    return execution_config.gradient_method == "adjoint" or (
        execution_config.gradient_method == "device"
        and execution_config.gradient_keyword_arguments.get("method", None) == "adjoint_jacobian"
    )


@contextmanager
def _set_shots(device, shots):
    """Context manager to temporarily change the shots
    of a device.

    This context manager can be used in two ways.

    As a standard context manager:

    >>> with _set_shots(dev, shots=100):
    ...     print(dev.shots)
    100
    >>> print(dev.shots)
    None

    Or as a decorator that acts on a function that uses the device:

    >>> _set_shots(dev, shots=100)(lambda: dev.shots)()
    100
    """
    shots = Shots(shots)
    shots = shots.shot_vector if shots.has_partitioned_shots else shots.total_shots
    if shots == device.shots:
        yield
        return

    original_shots = device.shots
    original_shot_vector = device._shot_vector  # pylint: disable=protected-access

    try:
        device.shots = shots
        yield
    finally:
        device.shots = original_shots
        device._shot_vector = original_shot_vector  # pylint: disable=protected-access


def null_postprocessing(results):
    """A postprocessing function with null behavior."""
    return results[0]


@transform
def legacy_device_expand_fn(tape, device):
    """Turn the ``expand_fn`` from the legacy device interface into a transform."""
    new_tape = _set_shots(device, tape.shots)(device.expand_fn)(tape)
    return (new_tape,), null_postprocessing


@transform
def legacy_device_batch_transform(tape, device):
    """Turn the ``batch_transform`` from the legacy device interface into a transform."""
    return _set_shots(device, tape.shots)(device.batch_transform)(tape)


def adjoint_ops(op: Operator) -> bool:
    """Specify whether or not an Operator is supported by adjoint differentiation."""
    if isinstance(op, ops.QubitUnitary) and not any(requires_grad(d) for d in op.data):
        return True
    return not isinstance(op, MidMeasure) and (
        op.num_params == 0 or (op.num_params == 1 and op.has_generator)
    )


def _add_adjoint_transforms(pipeline: CompilePipeline, name="adjoint"):
    """Add the adjoint specific transforms to the transform pipeline."""
    pipeline.add_transform(no_sampling, name=name)
    pipeline.add_transform(
        decompose,
        stopping_condition=adjoint_ops,
        name=name,
    )

    def accepted_adjoint_measurements(mp):
        return isinstance(mp, ExpectationMP)

    pipeline.add_transform(
        validate_measurements,
        analytic_measurements=accepted_adjoint_measurements,
        name=name,
    )
    pipeline.add_transform(broadcast_expand)
    pipeline.add_transform(validate_adjoint_trainable_params)


@single_tape_support
class LegacyDeviceFacade(Device):
    """
    A Facade that converts a device from the old ``qp.Device`` interface into the new interface.

    Args:
        device (qp.device.LegacyDevice): a device that follows the legacy device interface.

    >>> from pennylane.devices import DefaultQutrit, LegacyDeviceFacade
    >>> legacy_dev = DefaultQutrit(wires=2)
    >>> new_dev = LegacyDeviceFacade(legacy_dev)
    >>> new_dev.preprocess()
    (CompilePipeline(legacy_device_batch_transform, legacy_device_expand_fn, defer_measurements),
    ExecutionConfig(grad_on_execution=None, use_device_gradient=None, use_device_jacobian_product=None,
    gradient_method=None, gradient_keyword_arguments={}, device_options={}, interface=<Interface.NUMPY: 'numpy'>,
    derivative_order=1, mcm_config=MCMConfig(mcm_method=None, postselect_mode=None)))
    >>> new_dev.shots
    Shots(total_shots=None, shot_vector=())
    >>> tape = qp.tape.QuantumScript([], [qp.sample(wires=0)], shots=5)
    >>> new_dev.execute(tape)
    array([[0],
       [0],
       [0],
       [0],
       [0]])

    """

    # pylint: disable=super-init-not-called
    def __init__(self, device: LegacyDevice):
        if isinstance(device, type(self)):
            raise RuntimeError("An already-facaded device can not be wrapped in a facade again.")

        if not isinstance(device, LegacyDevice):
            raise ValueError(
                "The LegacyDeviceFacade only accepts a device of type qp.devices.LegacyDevice."
            )

        self._device = device
        self.capabilities = None

        _config_filepath = getattr(self._device, "config_filepath", None)
        if _config_filepath:
            self.capabilities = DeviceCapabilities.from_toml_file(_config_filepath)
            self.config_filepath = _config_filepath

        if self._device.shots:
            warnings.warn(
                "Setting shots on device is deprecated. Please use the `set_shots` transform on the respective QNode instead.",
                PennyLaneDeprecationWarning,
            )

    @property
    def tracker(self):
        """A :class:`~pennylane.Tracker` that can store information about device executions, shots, batches,
        intermediate results, or any additional device dependent information.
        """
        return self._device.tracker

    @tracker.setter
    def tracker(self, new_tracker):
        self._device.tracker = new_tracker

    @property
    def name(self) -> str:
        return self._device.short_name

    def __repr__(self):
        return f"<LegacyDeviceFacade: {repr(self._device)}>"

    def __getattr__(self, name):
        return getattr(self._device, name)

    # These custom copy methods are needed for Catalyst
    def __copy__(self):
        return type(self)(copy(self.target_device))

    def __deepcopy__(self, memo):
        return type(self)(deepcopy(self.target_device, memo))

    @property
    def target_device(self) -> LegacyDevice:
        """The device wrapped by the facade."""
        return self._device

    @property
    def wires(self) -> Wires:
        return self._device.wires

    # pylint: disable=protected-access
    @property
    def shots(self) -> Shots:
        if self._device._shot_vector:
            return Shots(self._device._raw_shot_sequence)
        return Shots(self._device.shots)

    @property
    def _debugger(self):
        return self._device._debugger

    @_debugger.setter
    def _debugger(self, new_debugger):
        self._device._debugger = new_debugger

    def preprocess_transforms(
        self, execution_config: ExecutionConfig | None = None
    ) -> CompilePipeline:
        pipeline = CompilePipeline()

        if not execution_config:
            execution_config = ExecutionConfig()

        if execution_config.mcm_config.mcm_method == "deferred":
            pipeline.add_transform(
                defer_measurements,
                allow_postselect=False,
            )

        pipeline.add_transform(legacy_device_batch_transform, device=self._device)
        pipeline.add_transform(legacy_device_expand_fn, device=self._device)

        if _requests_adjoint(execution_config):
            _add_adjoint_transforms(pipeline, name=f"{self.name} + adjoint")

        if execution_config.mcm_config.mcm_method == "one-shot":
            pipeline.add_transform(
                dynamic_one_shot,
                postselect_mode=execution_config.mcm_config.postselect_mode,
            )

        return pipeline

    def _setup_backprop_config(self, execution_config, tape):
        if not self._validate_backprop_method(tape):
            raise DeviceError("device does not support backprop.")
        if execution_config.use_device_gradient is None:
            return replace(execution_config, use_device_gradient=True)
        return execution_config

    def _setup_adjoint_config(self, execution_config, tape):
        if not self._validate_adjoint_method(tape):
            raise DeviceError("device does not support device derivatives")
        updated_values = {
            "gradient_keyword_arguments": {"use_device_state": True, "method": "adjoint_jacobian"}
        }

        if execution_config.use_device_gradient is None:
            updated_values["use_device_gradient"] = True
        if execution_config.grad_on_execution is None:
            updated_values["grad_on_execution"] = True
        return replace(execution_config, **updated_values)

    def _setup_device_config(self, execution_config, tape):
        if not self._validate_device_method(tape):
            raise DeviceError("device does not support device derivatives")

        updated_values = {}
        if execution_config.use_device_gradient is None:
            updated_values["use_device_gradient"] = True
        if execution_config.grad_on_execution is None:
            updated_values["grad_on_execution"] = True
        return replace(execution_config, **updated_values)

    def setup_execution_config(
        self, config: ExecutionConfig | None = None, circuit: QuantumScript | None = None
    ) -> ExecutionConfig:
        """Sets up an ``ExecutionConfig`` that configures the execution behaviour."""

        config = config or ExecutionConfig()

        if config.gradient_method == "best":
            if self._validate_device_method(circuit):
                config = replace(config, gradient_method="device")
                return self.setup_execution_config(config, circuit)

            if self._validate_backprop_method(circuit):
                config = replace(config, gradient_method="backprop")
                return self._setup_backprop_config(config, circuit)

        if config.gradient_method == "backprop":
            return self._setup_backprop_config(config, circuit)
        if _requests_adjoint(config):
            return self._setup_adjoint_config(config, circuit)
        if config.gradient_method == "device":
            return self._setup_device_config(config, circuit)

        shots_present = bool(circuit and bool(circuit.shots))
        self._validate_mcm_method(config.mcm_config.mcm_method, shots_present)
        if config.mcm_config.mcm_method is None:
            default_mcm_method = self._default_mcm_method(shots_present)
            new_mcm_config = replace(config.mcm_config, mcm_method=default_mcm_method)
            config = replace(config, mcm_config=new_mcm_config)

        return config

    def supports_derivatives(self, execution_config=None, circuit=None) -> bool:
        circuit = QuantumScript([], [], shots=self.shots) if circuit is None else circuit

        if execution_config is None or execution_config.gradient_method == "best":
            validation_methods = (
                self._validate_backprop_method,
                self._validate_device_method,
            )
            return any(validate(circuit) for validate in validation_methods)

        if execution_config.gradient_method == "backprop":
            return self._validate_backprop_method(circuit)
        if _requests_adjoint(execution_config):
            return self._validate_adjoint_method(circuit)
        if execution_config.gradient_method == "device":
            return self._validate_device_method(circuit)

        return False

    def _validate_backprop_method(self, tape):
        if tape is None:
            tape = QuantumScript()
        if tape.shots:
            return False
        params = tape.get_parameters(trainable_only=False)
        if (interface := math.get_interface(*params)) != "numpy":
            interface = Interface(interface).value

        if tape and any(isinstance(m.obs, ops.SparseHamiltonian) for m in tape.measurements):
            return False

        # determine if the device supports backpropagation
        backprop_interface = self._device.capabilities().get("passthru_interface", None)

        if backprop_interface is not None:
            # device supports backpropagation natively
            return interface in [backprop_interface, "numpy"]
        # determine if the device has any child devices that support backpropagation
        backprop_devices = self._device.capabilities().get("passthru_devices", None)

        if backprop_devices is None:
            return False
        return interface in backprop_devices or interface == "numpy"

    def _validate_adjoint_method(self, tape):
        # The conditions below provide a minimal set of requirements that we can likely improve upon in
        # future, or alternatively summarize within a single device capability. Moreover, we also
        # need to inspect the circuit measurements to ensure only expectation values are taken. This
        # cannot be done here since we don't yet know the composition of the circuit.

        required_attrs = ["_apply_operation", "_apply_unitary", "adjoint_jacobian"]
        supported_device = all(hasattr(self._device, attr) for attr in required_attrs)
        supported_device = supported_device and self._device.capabilities().get("returns_state")

        if tape is None:
            tape = QuantumScript()
        if not supported_device or bool(tape.shots):
            return False
        pipeline = CompilePipeline()
        _add_adjoint_transforms(pipeline, name=f"{self.name} + adjoint")
        try:
            pipeline((tape,))
        except (
            DecompositionUndefinedError,
            DeviceError,
            AttributeError,
        ):
            return False
        return True

    def _validate_device_method(self, _):
        # determine if the device provides its own jacobian method
        return self._device.capabilities().get("provides_jacobian", False)

    def _validate_mcm_method(self, mcm_method: str, shots_present: bool):
        """Validates an MCM method against the device's capabilities."""

        if mcm_method in (None, "deferred"):
            return  # No need to validate because "deferred" is always supported.

        if mcm_method == "one-shot" and not shots_present:
            raise QuantumFunctionError('mcm_method="one-shot" is only supported with finite shots.')

        supported_mcm_methods = ("deferred",)
        if self.capabilities:
            supported_mcm_methods += tuple(self.capabilities.supported_mcm_methods)
        elif self._device.capabilities().get("supports_mid_measure", False):
            supported_mcm_methods += ("one-shot",)

        if mcm_method not in supported_mcm_methods:
            raise QuantumFunctionError(
                f'The requested MCM method "{mcm_method}" unsupported by the device. '
                f"Supported methods are: {supported_mcm_methods}."
            )

    def _default_mcm_method(self, shots_present: bool) -> str:
        """Simple strategy to find the best match for the default mcm method."""

        supports_one_shot = False
        if self.capabilities and "one-shot" in self.capabilities.supported_mcm_methods:
            supports_one_shot = True
        elif self._device.capabilities().get("supports_mid_measure", False):
            supports_one_shot = True

        if supports_one_shot and shots_present:
            return "one-shot"

        return "deferred"

    def execute(self, circuits, execution_config: ExecutionConfig | None = None):
        if execution_config is None:
            execution_config = ExecutionConfig()

        dev = self.target_device

        kwargs = {}
        if dev.capabilities().get("supports_mid_measure", False):
            kwargs["postselect_mode"] = execution_config.mcm_config.postselect_mode

        first_shot = circuits[0].shots
        if all(t.shots == first_shot for t in circuits):
            return _set_shots(dev, first_shot)(dev.batch_execute)(circuits, **kwargs)
        return tuple(
            _set_shots(dev, t.shots)(dev.batch_execute)((t,), **kwargs)[0] for t in circuits
        )

    def execute_and_compute_derivatives(
        self, circuits, execution_config: ExecutionConfig | None = None
    ):
        if execution_config is None:
            execution_config = ExecutionConfig(gradient_method="device")

        first_shot = circuits[0].shots
        if all(t.shots == first_shot for t in circuits):
            return _set_shots(self._device, first_shot)(self._device.execute_and_gradients)(
                circuits, **execution_config.gradient_keyword_arguments
            )
        batched_res = tuple(
            self.execute_and_compute_derivatives((c,), execution_config) for c in circuits
        )
        return tuple(zip(*batched_res))

    def compute_derivatives(self, circuits, execution_config: ExecutionConfig | None = None):
        if execution_config is None:
            execution_config = ExecutionConfig(gradient_method="device")

        first_shot = circuits[0].shots
        if all(t.shots == first_shot for t in circuits):
            return _set_shots(self._device, first_shot)(self._device.gradients)(
                circuits, **execution_config.gradient_keyword_arguments
            )
        return tuple(self.compute_derivatives((c,), execution_config) for c in circuits)
