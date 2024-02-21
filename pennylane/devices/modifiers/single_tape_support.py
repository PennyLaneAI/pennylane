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
"""Defines the ``single_tape_support`` device modifier.

"""
from functools import wraps

from pennylane.devices import Device, DefaultExecutionConfig
from pennylane.tape import QuantumScript


def _make_execute(batch_execute):
    """Allows an ``execute`` function to handle individual circuits."""

    @wraps(batch_execute)
    def execute(self, circuits, execution_config=DefaultExecutionConfig):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = (circuits,)
        results = batch_execute(self, circuits, execution_config)
        return results[0] if is_single_circuit else results

    return execute


def _make_compute_derivatives(batch_derivatives):
    """Allows an ``compute_derivatives`` method to handle individual circuits."""

    @wraps(batch_derivatives)
    def compute_derivatives(self, circuits, execution_config=DefaultExecutionConfig):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = (circuits,)
        jacs = batch_derivatives(self, circuits, execution_config)
        return jacs[0] if is_single_circuit else jacs

    return compute_derivatives


def _make_execute_and_compute_derivatives(batch_execute_and_compute_derivatives):
    """Allows an ``execute_and_compute_derivatives`` method to handle individual circuits."""

    @wraps(batch_execute_and_compute_derivatives)
    def execute_and_compute_derivatives(self, circuits, execution_config=DefaultExecutionConfig):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = (circuits,)
        results, jacs = batch_execute_and_compute_derivatives(self, circuits, execution_config)
        return (results[0], jacs[0]) if is_single_circuit else (results, jacs)

    return execute_and_compute_derivatives


def _make_compute_jvp(batch_compute_jvp):
    """Allows an ``compute_jvp`` method to handle individual circuits."""

    @wraps(batch_compute_jvp)
    def compute_jvp(self, circuits, tangents, execution_config=DefaultExecutionConfig):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]
            tangents = [tangents]

        res = batch_compute_jvp(self, circuits, tangents, execution_config)
        return res[0] if is_single_circuit else res

    return compute_jvp


def _make_execute_and_compute_jvp(batch_execute_and_compute_jvp):
    """Allows an ``execute_and_compute_jvp`` method to handle individual circuits."""

    @wraps(batch_execute_and_compute_jvp)
    def execute_and_compute_jvp(self, circuits, tangents, execution_config=DefaultExecutionConfig):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]
            tangents = [tangents]

        results, jvps = batch_execute_and_compute_jvp(self, circuits, tangents, execution_config)

        return (results[0], jvps[0]) if is_single_circuit else (results, jvps)

    return execute_and_compute_jvp


def _make_compute_vjp(batch_compute_vjp):
    """Allows an ``execute_and_compute_vjp`` method to handle individual circuits."""

    @wraps(batch_compute_vjp)
    def compute_vjp(self, circuits, cotangents, execution_config=DefaultExecutionConfig):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]
            cotangents = [cotangents]

        res = batch_compute_vjp(self, circuits, cotangents, execution_config)

        return res[0] if is_single_circuit else res

    return compute_vjp


def _make_execute_and_compute_vjp(batch_execute_and_compute_vjp):
    """Allows an ``execute_and_compute_vjp`` method to handle individual circuits."""

    @wraps(batch_execute_and_compute_vjp)
    def execute_and_compute_vjp(
        self, circuits, cotangents, execution_config=DefaultExecutionConfig
    ):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]
            cotangents = [cotangents]

        results, vjps = batch_execute_and_compute_vjp(self, circuits, cotangents, execution_config)

        return (results[0], vjps[0]) if is_single_circuit else (results, vjps)

    return execute_and_compute_vjp


# pylint: disable=protected-access
def single_tape_support(cls: type) -> type:
    """Modifies all functions to accept single tapes in addition to batches. This allows the definition
    of the device class to purely focus on executing batches.

    Args:
        cls (type): a subclass of :class:`pennylane.devices.Device`

    Returns
        type: The inputted class that has now been modified to accept single circuits as well as batches.

    .. code-block:: python

        @single_tape_support
        class MyDevice(qml.devices.Device):

            def execute(self, circuits, execution_config = qml.devices.DefaultExecutionConfig):
                return tuple(0.0 for _ in circuits)

    >>> dev = MyDevice()
    >>> t = qml.tape.QuantumScript()
    >>> dev.execute(t)
    0.0
    >>> dev.execute((t, ))
    (0.0,)

    In this situation, ``MyDevice.execute`` only needs to handle the case where ``circuits`` is an iterable
    of :class:`~pennylane.tape.QuantumTape` objects, not a single value.

    """
    if not issubclass(cls, Device):
        raise ValueError("single_tape_support only accepts subclasses of pennylane.devices.Device")

    if hasattr(cls, "_applied_modifiers"):
        cls._applied_modifiers.append(single_tape_support)
    else:
        cls._applied_modifiers = [single_tape_support]

    # execute must be defined
    cls.execute = _make_execute(cls.execute)

    modifier_map = {
        "compute_derivatives": _make_compute_derivatives,
        "execute_and_compute_derivatives": _make_execute_and_compute_derivatives,
        "compute_jvp": _make_compute_jvp,
        "execute_and_compute_jvp": _make_execute_and_compute_jvp,
        "compute_vjp": _make_compute_vjp,
        "execute_and_compute_vjp": _make_execute_and_compute_vjp,
    }

    for name, modifier in modifier_map.items():
        if getattr(cls, name) != getattr(Device, name):
            original = getattr(cls, name)
            setattr(cls, name, modifier(original))

    return cls
