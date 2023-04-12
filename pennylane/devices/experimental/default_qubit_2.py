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

from . import Device
from .execution_config import ExecutionConfig, DefaultExecutionConfig
from ..qubit.simulate import simulate
from ..qubit.preprocess import preprocess

QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]


class DefaultQubit2(Device):
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

    This device currently supports backpropagation derivatives:

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
    DeviceArray(0.36235774, dtype=float32)
    >>> jax.grad(f)(jax.numpy.array(1.2))
    DeviceArray(-0.93203914, dtype=float32, weak_type=True)

    """

    @property
    def name(self):
        """The name of the device."""
        return "default.qubit.2"

    def supports_derivatives(
        self,
        execution_config: ExecutionConfig,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """Check whether or not derivatives are available for a given configuration and circuit.

        ``DefaultQubit2`` supports backpropagation derivatives with analytic results.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            Bool: Whether or not a derivative can be calculated provided the given information

        """
        # backpropagation currently supported for all supported circuits
        # will later need to add logic if backprop requested with finite shots
        # do once device accepts finite shots
        return execution_config.gradient_method == "backprop"

    def preprocess(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Tuple[QuantumTapeBatch, Callable]:
        """Converts an arbitrary circuit or batch of circuits into a batch natively executable by the :meth:`~.execute` method.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): The circuit or a batch of circuits to preprocess
                before execution on the device
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure describing the parameters needed to fully describe
                the execution. Includes such information as shots.

        Returns:
            Sequence[QuantumTape], Callable: QuantumTapes that the device can natively execute
            and a postprocessing function to be called after execution.

        This device:

        * Supports any qubit operations that provide a matrix
        * Currently does not support finite shots
        * Currently does not intrinsically support parameter broadcasting

        """
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            circuits = [circuits]
            is_single_circuit = True

        batch, post_processing_fn = preprocess(circuits, execution_config=execution_config)

        if is_single_circuit:

            def convert_batch_to_single_output(results):
                """Unwraps a dimension so that executing the batch of circuits looks like executing a single circuit."""
                return post_processing_fn(results)[0]

            return batch, convert_batch_to_single_output

        return batch, post_processing_fn

    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        """Execute a circuit or a batch of circuits and turn it into results.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the quantum circuits to be executed
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfg]]): a datastructure with additional information required for execution

        Returns:
            TensorLike, tuple[TensorLike], tuple[tuple[TensorLike]]: A numeric result of the computation.
        """
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]

        if self.tracker.active:
            self.tracker.update(batches=1, executions=len(circuits))
            self.tracker.record()

        results = tuple(simulate(c) for c in circuits)
        return results[0] if is_single_circuit else results
