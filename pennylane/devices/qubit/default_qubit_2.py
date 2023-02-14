# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This module contains the next generation successor to default qubit
"""

from typing import Union, Callable, Tuple, Optional, Sequence

from pennylane.tape import QuantumTape, QuantumScript

from .. import QuantumDevice
from ..execution_config import ExecutionConfig, DefaultExecutionConfig
from .simulate import simulate
from .preprocess import preprocess

QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]

Config_or_Batch = Union[ExecutionConfig, Sequence[ExecutionConfig]]


class DefaultQubit2(QuantumDevice):
    """A PennyLane device written in Python and capable of backpropagation derivatives.

    This class currently has no arguments.

    **Example:**

    .. code-block:: python

        n_layers = 5
        n_wires = 10
        num_qscripts = 5

        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
        rng = qml.numpy.random.default_rng(seed=42)

        qscripts = []
        for i in range(num_qscripts):
            params = rng.random(shape)
            op = qml.StronglyEntanglingLayers(params, wires=range(n_wires))
            qs = qml.tape.QuantumScript([op], [qml.expval(qml.PauliZ(0))])
            qscripts.append(qs)

    >>> dev = DefaultQubit2()
    >>> new_batch, post_processing_fn = dev.preprocess(qscripts)
    >>> results = dev.execute(new_batch)
    >>> post_processing_fn(results)
    [-0.0006888975950537501,
    0.025576307134457577,
    -0.0038567269892757494,
    0.1339705146860149,
    -0.03780669772690448]

    This device currently supports backpropogation derivatives:

    >>> from pennylane.devices import ExecutionConfig
    >>> dev.supports_derivatives(ExecutionConfig(gradient_method="backprop"))
    True

    For example, we can use jax to jit computing the derivative:

    .. code-block:: python

        @jax.jit
        def f(x):
            qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
            new_batch, post_processing_fn = dev.preprocess(qs)
            results = dev.execute(new_batch)
            return post_processing_fn(results)

    >>> f(jax.numpy.array(1.2))
    [DeviceArray(0.36235774, dtype=float32)]
    >>> jax.jacobian(f)(jax.numpy.array(1.2))
    [DeviceArray(-0.93203914, dtype=float32, weak_type=True)]

    """

    def __init__(self) -> None:
        # each instance should have its own Tracker.
        super().__init__()

    @property
    def name(self):
        return "default.qubit.2"

    def supports_derivatives(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """Check whether or not derivatives are available for a given configuration and circuit.

        ``DefaultQubit2`` supports backpropogation derivatives with analytic results.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calcualtion
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            Bool: Whether or not a derivative can be calculated provided the given information

        """
        # requesting information about device derivatives. Adjoint jacobian not added yet
        if execution_config is None:
            return False
        # backpropogation currently supported for all supported circuits
        return execution_config.gradient_method == "backprop" and execution_config.shots is None

    def preprocess(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: Config_or_Batch = DefaultExecutionConfig,
    ) -> Tuple[QuantumTapeBatch, Callable]:
        if isinstance(circuits, QuantumScript):
            circuits = [circuits]
        return preprocess(circuits, execution_config=execution_config)

    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: Config_or_Batch = DefaultExecutionConfig,
    ):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]

        if self.tracker.active:
            self.tracker.update(batches=1, executions=len(circuits))
            self.tracker.record()

        results = tuple(simulate(c) for c in circuits)
        return results[0] if is_single_circuit else results
