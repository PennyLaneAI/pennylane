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
"""Defines the ``simulator_tracking`` device modifier.

"""
from functools import wraps

from pennylane.devices import DefaultExecutionConfig, Device

from ..qubit.sampling import get_num_shots_and_executions


def _track_execute(untracked_execute):
    """Adds default tracking to an execute method."""

    @wraps(untracked_execute)
    def execute(self, circuits, execution_config=DefaultExecutionConfig):
        results = untracked_execute(self, circuits, execution_config)
        if self.tracker.active:
            self.tracker.update(batches=1)
            self.tracker.record()
            for r, c in zip(results, circuits):
                qpu_executions, shots = get_num_shots_and_executions(c)
                if c.shots:
                    self.tracker.update(
                        simulations=1,
                        executions=qpu_executions,
                        results=r,
                        shots=shots,
                        resources=c.specs["resources"],
                    )
                else:
                    self.tracker.update(
                        simulations=1,
                        executions=qpu_executions,
                        results=r,
                        resources=c.specs["resources"],
                    )
                self.tracker.record()
        return results

    return execute


def _track_compute_derivatives(untracked_compute_derivatives):
    """Adds default tracking to a ``compute_derivatives`` method."""

    @wraps(untracked_compute_derivatives)
    def compute_derivatives(self, circuits, execution_config=DefaultExecutionConfig):
        if self.tracker.active:
            self.tracker.update(derivative_batches=1, derivatives=len(circuits))
            self.tracker.record()
        return untracked_compute_derivatives(self, circuits, execution_config)

    return compute_derivatives


def _track_execute_and_compute_derivatives(untracked_execute_and_compute_derivatives):
    """Adds default tracking to a ``execute_and_compute_derivatives`` method."""

    @wraps(untracked_execute_and_compute_derivatives)
    def execute_and_compute_derivatives(self, circuits, execution_config=DefaultExecutionConfig):
        if self.tracker.active:
            for c in circuits:
                self.tracker.update(resources=c.specs["resources"])
            self.tracker.update(
                execute_and_derivative_batches=1,
                executions=len(circuits),
                derivatives=len(circuits),
            )
            self.tracker.record()
        return untracked_execute_and_compute_derivatives(self, circuits, execution_config)

    return execute_and_compute_derivatives


def _track_compute_jvp(untracked_compute_jvp):
    """Adds default tracking to a ``compute_jvp`` method."""

    @wraps(untracked_compute_jvp)
    def compute_jvp(self, circuits, tangents, execution_config=DefaultExecutionConfig):
        if self.tracker.active:
            self.tracker.update(jvp_batches=1, jvps=len(circuits))
            self.tracker.record()
        return untracked_compute_jvp(self, circuits, tangents, execution_config)

    return compute_jvp


def _track_execute_and_compute_jvp(untracked_execute_and_compute_jvp):
    """Adds default tracking to a ``execute_and_compute_jvp`` method."""

    @wraps(untracked_execute_and_compute_jvp)
    def execute_and_compute_jvp(self, circuits, tangents, execution_config=DefaultExecutionConfig):
        if self.tracker.active:
            for c in circuits:
                self.tracker.update(resources=c.specs["resources"])
            self.tracker.update(
                execute_and_jvp_batches=1, executions=len(circuits), jvps=len(circuits)
            )
            self.tracker.record()

        return untracked_execute_and_compute_jvp(self, circuits, tangents, execution_config)

    return execute_and_compute_jvp


def _track_compute_vjp(untracked_compute_vjp):
    """Adds default tracking to a ``compute_vjp`` method."""

    @wraps(untracked_compute_vjp)
    def compute_vjp(self, circuits, cotangents, execution_config=DefaultExecutionConfig):
        if self.tracker.active:
            self.tracker.update(vjp_batches=1, vjps=len(circuits))
            self.tracker.record()

        return untracked_compute_vjp(self, circuits, cotangents, execution_config)

    return compute_vjp


def _track_execute_and_compute_vjp(untracked_execute_and_compute_vjp):
    """Adds default trakcing to a ``execute_and_compute_vjp`` method."""

    @wraps(untracked_execute_and_compute_vjp)
    def execute_and_compute_vjp(
        self, circuits, cotangents, execution_config=DefaultExecutionConfig
    ):
        if self.tracker.active:
            for c in circuits:
                self.tracker.update(resources=c.specs["resources"])
            self.tracker.update(
                execute_and_vjp_batches=1, executions=len(circuits), vjps=len(circuits)
            )
            self.tracker.record()
        return untracked_execute_and_compute_vjp(self, circuits, cotangents, execution_config)

    return execute_and_compute_vjp


# pylint: disable=protected-access
def simulator_tracking(cls: type) -> type:
    """Modifies all methods to add default simulator style tracking.

    Args:
        cls (type): a subclass of :class:`pennylane.devices.Device`

    Returns
        type: The inputted class that has now been modified to update the tracker upon function calls.

    .. warning::

        If used together with :func:`~.convert_single_circuit_to_batch`, that modifier must be applied first.

    Simulator style tracking updates:

        * ``executions``: the number of unique circuits that would be required on quantum hardware
        * ``shots``: the number of shots
        * ``resources``: the :class:`~.resource.Resources` for the executed circuit.
        * ``simulations``: the number of simulations performed. One simulation can cover multiple QPU executions, such as for non-commuting measurements and batched parameters.
        * ``batches``: The number of times :meth:`~.execute` is called.
        * ``results``: The results of each call of :meth:`~.execute`
        * ``derivative_batches``: How many times :meth:`~.compute_derivatives` is called.
        * ``execute_and_derivative_batches``: How many times :meth:`~.execute_and_compute_derivatives` is called
        * ``vjp_batches``: How many times :meth:`~.compute_vjp` is called
        * ``execute_and_vjp_batches``: How many times :meth:`~.execute_and_compute_vjp` is called
        * ``jvp_batches``: How many times :meth:`~.compute_jvp` is called
        * ``execute_and_jvp_batches``: How many times :meth:`~.execute_and_compute_jvp` is called
        * ``derivatives``: How many circuits are submitted to :meth:`~.compute_derivatives` or :meth:`~.execute_and_compute_derivatives`.
        * ``vjps``: How many circuits are submitted to :meth:`~.compute_vjp` or :meth:`~.execute_and_compute_vjp`
        * ``jvps``: How many circuits are submitted to :meth:`~.compute_jvp` or :meth:`~.execute_and_compute_jvp`


    .. code-block:: python

        @simulator_tracking
        @convert_single_circuit_to_batch
        class MyDevice(qml.devices.Device):

            def execute(self, circuits, execution_config = qml.devices.DefaultExecutionConfig):
                return tuple(0.0 for c in circuits)

    >>> dev = MyDevice()
    >>> ops = [qml.S(0)]
    >>> measurements = [qml.expval(qml.X(0)), qml.expval(qml.Z(0))]
    >>> t = qml.tape.QuantumScript(ops, measurements,shots=50)
    >>> with dev.tracker:
    ...     dev.execute((t, ) )
    >>> dev.tracker.history
    {'batches': [1],
    'simulations': [1],
    'executions': [2],
    'results': [0.0],
    'shots': [100],
    'resources': [Resources(num_wires=1, num_gates=1, gate_types=defaultdict(<class 'int'>, {'S': 1}),
    gate_sizes=defaultdict(<class 'int'>, {1: 1}), depth=1, shots=Shots(total_shots=50,
    shot_vector=(ShotCopies(50 shots x 1),)))]}

    """
    if not issubclass(cls, Device):
        raise ValueError(
            "convert_single_circuit_to_batch only accepts subclasses of pennylane.devices.Device"
        )

    if hasattr(cls, "_applied_modifiers"):
        cls._applied_modifiers.append(simulator_tracking)
    else:
        cls._applied_modifiers = [simulator_tracking]

    if cls.execute != Device.execute:
        cls._untracked_execute = cls.execute
        cls.execute = _track_execute(cls._untracked_execute)

    if cls.compute_derivatives != Device.compute_derivatives:
        cls._untracked_compute_derivatives = cls.compute_derivatives
        cls.compute_derivatives = _track_compute_derivatives(cls._untracked_compute_derivatives)

    if cls.execute_and_compute_derivatives != Device.execute_and_compute_derivatives:
        cls._untracked_execute_and_compute_derivatives = cls.execute_and_compute_derivatives
        cls.execute_and_compute_derivatives = _track_execute_and_compute_derivatives(
            cls._untracked_execute_and_compute_derivatives
        )

    if cls.compute_jvp != Device.compute_jvp:
        cls._untracked_compute_jvp = cls.compute_jvp
        cls.compute_jvp = _track_compute_jvp(cls._untracked_compute_jvp)

    if cls.execute_and_compute_jvp != Device.execute_and_compute_jvp:
        cls._untracked_execute_and_compute_jvp = cls.execute_and_compute_jvp
        cls.execute_and_compute_jvp = _track_execute_and_compute_jvp(
            cls._untracked_execute_and_compute_jvp
        )

    if cls.compute_vjp != Device.compute_vjp:
        cls._untracked_compute_vjp = cls.compute_vjp
        cls.compute_vjp = _track_compute_vjp(cls._untracked_compute_vjp)

    if cls.execute_and_compute_vjp != Device.execute_and_compute_vjp:
        cls._untracked_execute_and_compute_vjp = cls.execute_and_compute_vjp
        cls.execute_and_compute_vjp = _track_execute_and_compute_vjp(
            cls._untracked_execute_and_compute_vjp
        )

    return cls
