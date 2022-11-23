# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This module contains the QNode class and qnode decorator.
"""
# pylint: disable=too-many-instance-attributes,too-many-arguments,protected-access,unnecessary-lambda-assignment
import functools
import inspect
import warnings
from collections.abc import Sequence

import autograd

import pennylane as qml
from pennylane import Device
from pennylane.interfaces import INTERFACE_MAP, SUPPORTED_INTERFACES, set_shots
from pennylane.tape import QuantumTape
from .qnode_utils import *
from .runtime_config import *


class QNode_STD_22:
    """Represents a quantum node in the hybrid computational graph following the 2022 device API standard."""

    def __init__(
        self,
        func,
        device,
        diff_method=None,
        exec_config: ExecutionConfig = ExecutionConfig(
            # interface="autograd",
            # diff_method="best",
            # expansion_strategy="gradient",
            # max_expansion=10,
            # cache=True,
        ),
    ):

        if not isinstance(device, (qml.devices.experimental.AbstractDevice, Device)):
            raise qml.QuantumFunctionError(
                "Invalid device. Device must be a valid PennyLane device."
            )

        # input arguments
        self.exec_config = exec_config
        self.func = func
        self.device = device
        self.diff_method = diff_method
        self._interface = (
            self.exec_config.interface.name.lower()
        )  # In future, replace all string checks to enum checks
        #self.diff_method = self.exec_config.diff_method.name.lower() if self.exec_config.diff_method else None # As above
        self.expansion_strategy = (
            self.exec_config.expansion_strategy.name.lower()
        )  # Expansion should happen as a preproc step defined to run before execution pipeline
        self.max_expansion = (
            self.exec_config.max_expansion
        )  # This should also be part of preproc stage

        # execution keyword arguments
        self.execute_kwargs = {
            "cache": bool(self.exec_config.cache_size),
            "cachesize": self.exec_config.cache_size,
            "max_diff": self.exec_config.max_diff,
            "max_expansion": self.exec_config.max_expansion,
        }

        if self.exec_config.expansion_strategy.name.lower() == "device":
            self.execute_kwargs["expand_fn"] = None

        # internal data attributes
        self._tape = None
        self._qfunc_output = None
        self._user_gradient_kwargs = self.exec_config.grad_args
        self._original_device = device
        self.gradient_fn = None
        self.gradient_kwargs = None
        self._tape_cached = False

        self._update_gradient_fn()
        functools.update_wrapper(self, func)

    def __repr__(self):
        """String representation."""
        detail = f"<QNode: wires={None}, device='{self.device.short_name}', interface='{self.interface}', diff_method='{self.diff_method}'>"
        return detail

    @property
    def interface(self):
        """The interface used by the QNode"""
        return self._interface

    @interface.setter
    def interface(self, value):
        if value not in SUPPORTED_INTERFACES:
            raise qml.QuantumFunctionError(
                f"Unknown interface {value}. Interface must be one of {SUPPORTED_INTERFACES}."
            )

        self._interface = INTERFACE_MAP[value]
        self._update_gradient_fn()

    def _update_gradient_fn(self):
        if self.diff_method is None:
            self._interface = None
            self.gradient_fn = None
            self.gradient_kwargs = {}
            return
        if self.interface == "auto":
            return

        self.gradient_fn, self.gradient_kwargs, self.device = get_gradient_fn(
            self._original_device, self.interface, self.diff_method
        )
        self.gradient_kwargs.update(self._user_gradient_kwargs or {})

    def _update_original_device(self):
        # FIX: If the qnode swapped the device, increase the num_execution value on the original device.
        # In the long run, we should make sure that the user's device is the one
        # actually run so she has full control. This could be done by changing the class
        # of the user's device before and after executing the tape.

        if self.device is not self._original_device:

            if not self._tape_cached:
                self._original_device._num_executions += 1  # pylint: disable=protected-access

            # Update for state vector simulators that have the _pre_rotated_state attribute
            if hasattr(self._original_device, "_pre_rotated_state"):
                self._original_device._pre_rotated_state = self.device._pre_rotated_state

            # Update for state vector simulators that have the _state attribute
            if hasattr(self._original_device, "_state"):
                self._original_device._state = self.device._state

    @property
    def tape(self) -> QuantumTape:
        """The quantum tape"""
        return self._tape

    qtape = tape  # for backwards compatibility

    def construct(self, args, kwargs):
        """Call the quantum function with a tape context, ensuring the operations get queued."""

        self._tape = qml.tape.QuantumTape()

        with self.tape:
            self._qfunc_output = self.func(*args, **kwargs)
        self._tape._qfunc_output = self._qfunc_output

        params = self.tape.get_parameters(trainable_only=False)
        self.tape.trainable_params = qml.math.get_trainable_indices(params)

        if isinstance(self._qfunc_output, qml.numpy.ndarray):
            measurement_processes = tuple(self.tape.measurements)
        elif not isinstance(self._qfunc_output, Sequence):
            measurement_processes = (self._qfunc_output,)
        else:
            measurement_processes = self._qfunc_output

        if not all(
            isinstance(m, qml.measurements.MeasurementProcess) for m in measurement_processes
        ):
            raise qml.QuantumFunctionError(
                "A quantum function must return either a single measurement, "
                "or a nonempty sequence of measurements."
            )

        terminal_measurements = [
            m for m in self.tape.measurements if m.return_type != qml.measurements.MidMeasure
        ]
        if any(ret != m for ret, m in zip(measurement_processes, terminal_measurements)):
            raise qml.QuantumFunctionError(
                "All measurements must be returned in the order they are measured."
            )

        for obj in self.tape.operations + self.tape.observables:

            if (
                getattr(obj, "num_wires", None) is qml.operation.WiresEnum.AllWires
                and len(obj.wires) != self.device.num_wires
            ):
                # check here only if enough wires
                raise qml.QuantumFunctionError(f"Operator {obj.name} must act on all wires")

            # pylint: disable=no-member
            if isinstance(obj, qml.ops.qubit.SparseHamiltonian) and self.gradient_fn == "backprop":
                raise qml.QuantumFunctionError(
                    "SparseHamiltonian observable must be used with the parameter-shift"
                    " differentiation method"
                )

        # Apply the deferred measurement principle if the device doesn't
        # support mid-circuit measurements natively
        # TODO:
        # 1. Change once mid-circuit measurements are not considered as tape
        # operations
        # 2. Move this expansion to Device (e.g., default_expand_fn or
        # batch_transform method)
        if any(
            getattr(obs, "return_type", None) == qml.measurements.MidMeasure
            for obs in self.tape.operations
        ):
            self._tape = qml.defer_measurements(self._tape)

        if self.expansion_strategy == "device":
            self._tape = self.device.expand_fn(self.tape, max_expansion=self.max_expansion)

        # If the gradient function is a transform, expand the tape so that
        # all operations are supported by the transform.
        if isinstance(self.gradient_fn, qml.gradients.gradient_transform):
            self._tape = self.gradient_fn.expand_fn(self._tape)

    def __call__(self, *args, **kwargs):  # pylint: disable=too-many-branches
        override_shots = False
        old_interface = self.interface
        if old_interface == "auto":
            self.interface = qml.math.get_interface(*args, *list(kwargs.values()))

        # If shots specified in call but not in qfunc signature,
        # interpret it as device shots value for this call.
        override_shots = kwargs.pop("shots", False)

        if override_shots is not False:
            # Since shots has changed, we need to update the preferred gradient function.
            # This is because the gradient function chosen at initialization may
            # no longer be applicable.

            # store the initialization gradient function
            original_grad_fn = [self.gradient_fn, self.gradient_kwargs, self.device]

            # pylint: disable=not-callable
            # update the gradient function
            set_shots(self._original_device, override_shots)(self._update_gradient_fn)()

        # construct the tape
        self.construct(args, kwargs)

        cache = self.execute_kwargs.get("cache", False)
        using_custom_cache = (
            hasattr(cache, "__getitem__")
            and hasattr(cache, "__setitem__")
            and hasattr(cache, "__delitem__")
        )
        self._tape_cached = using_custom_cache and self.tape.hash in cache

        if qml.active_return():
            res = qml.execute(
                [self.tape],
                device=self.device,
                gradient_fn="device",#self.gradient_fn,
                interface=self.interface,
                gradient_kwargs=self.gradient_kwargs,
                override_shots=override_shots,
                **self.execute_kwargs,
            )

            res = res[0]

            # Autograd or tensorflow: they do not support tuple return with backpropagation
            backprop = False
            if not isinstance(
                self._qfunc_output, qml.measurements.MeasurementProcess
            ) and self.interface in ("tf", "autograd"):
                backprop = any(qml.math.in_backprop(x) for x in res)

            if old_interface == "auto":
                self.interface = "auto"

            # Special case of single Measurement in a list
            if isinstance(self._qfunc_output, list) and len(self._qfunc_output) == 1:
                return [res]

            if self.gradient_fn == "backprop" and backprop:
                res = self.device._asarray(res)

            # If the return type is not tuple (list or ndarray) (Autograd and TF backprop removed)
            if (
                not isinstance(self._qfunc_output, (tuple, qml.measurements.MeasurementProcess))
                and not backprop
            ):
                if self.device._shot_vector:
                    res = [type(self.tape._qfunc_output)(r) for r in res]
                    res = tuple(res)

                else:
                    res = type(self.tape._qfunc_output)(res)

            if override_shots is not False:
                # restore the initialization gradient function
                self.gradient_fn, self.gradient_kwargs, self.device = original_grad_fn

            self._update_original_device()

            return res

        print("QNODE22")
        from IPython import embed
        #embed()

        res = qml.execute(
            [self.tape],
            device=self.device,
            gradient_fn=self.gradient_fn,
            interface=self.interface,
            gradient_kwargs=self.gradient_kwargs,
            override_shots=override_shots,
            **self.execute_kwargs,
        )

        if old_interface == "auto":
            self.interface = "auto"

        if autograd.isinstance(res, (tuple, list)) and len(res) == 1:
            # If a device batch transform was applied, we need to 'unpack'
            # the returned tuple/list to a float.
            #
            # Note that we use autograd.isinstance, because on the backwards pass
            # with Autograd, lists and tuples are converted to autograd.box.SequenceBox.
            # autograd.isinstance is a 'safer' isinstance check that supports
            # autograd backwards passes.
            #
            # TODO: find a more explicit way of determining that a batch transform
            # was applied.

            res = res[0]

        if not isinstance(self._qfunc_output, Sequence) and self._qfunc_output.return_type in (
            qml.measurements.Counts,
            qml.measurements.AllCounts,
        ):
            if self.device._has_partitioned_shots():
                return tuple(res)

            # return a dictionary with counts not as a single-element array
            return res[0]

        if isinstance(self._qfunc_output, Sequence) and any(
            m.return_type in (qml.measurements.Counts, qml.measurements.AllCounts)
            for m in self._qfunc_output
        ):

            # If Counts was returned with other measurements, then apply the
            # data structure used in the qfunc
            qfunc_output_type = type(self._qfunc_output)
            return qfunc_output_type(res)

        if override_shots is not False:
            # restore the initialization gradient function
            self.gradient_fn, self.gradient_kwargs, self.device = original_grad_fn

        self._update_original_device()

        if isinstance(self._qfunc_output, Sequence) or (
            self.tape.is_sampled and self.device._has_partitioned_shots()
        ):
            return res

        if self._qfunc_output.return_type is qml.measurements.Shadow:
            # if classical shadows is returned, then don't squeeze the
            # last axis corresponding to the number of qubits
            return qml.math.squeeze(res, axis=0)

        # Squeeze arraylike outputs
        return qml.math.squeeze(res)


qnode_std_22 = lambda device, **kwargs: functools.partial(QNode_STD_22, device=device, **kwargs)
qnode_std_22.__doc__ = QNode_STD_22.__doc__
qnode_std_22.__signature__ = inspect.signature(QNode_STD_22)
