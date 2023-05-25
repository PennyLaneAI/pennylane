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

import typing

import pennylane as qml
from pennylane import tape
from pennylane.measurements import MidMeasureMP, StateMeasurement, SampleMeasurement, ExpectationMP
from pennylane.typing import ResultBatch, Result

from .execution_config import ExecutionConfig, DefaultExecutionConfig

PostprocessingFn = typing.Callable[[ResultBatch], typing.Union[Result, ResultBatch]]


def _operator_decomposition_gen(
    op: qml.operation.Operator, accepted_operator: typing.Callable[[qml.operation.Operator], bool]
) -> typing.Generator[qml.operation.Operator, None, None]:
    """A generator that yields the next operation that is accepted by DefaultQubit2."""
    if accepted_operator(op):
        yield op
    else:
        try:
            decomp = op.decomposition()
        except qml.operation.DecompositionUndefinedError as e:
            raise qml.DeviceError(
                f"Operator {op} not supported on DefaultQubit2. Must provide either a matrix or a decomposition."
            ) from e

        for sub_op in decomp:
            yield from _operator_decomposition_gen(sub_op, accepted_operator)


class Preprocessor:
    """A convienient way to construct standard preprocessing for an arbitrary PennyLane device.

    # option: should we make these all class properties instead?

    Args:
        device_name (str): the name of the device. Used when raising error messages.
        supported_operation (set[str]): a set of the names of supported operations
        supported_observables (set[str]): a set of the names of supported observables
        supported_measurements (set[type]): a set of supported measurement types

    Keyword Args:
        supports_non_commuting_measurements (bool): whether or not the device supports measurements
            in more than one measurement basis at the same time
        supports_broadcasting (bool): whether or not the device supports native parameter broadcasting
        supports_midcircuit_measurements

    **Example:**

    >>> ops = {"PauliX", "PauliY", "PauliZ", "RX", "RY", "RZ"}
    >>> obs = {"PauliX", "PauliY", "PauliZ"}
    >>> ms = {qml.measurements.ExpectationMP}
    >>> p = Preprocessor("temp", ops, obs, ms)
    >>> circuit = qml.tape.QuantumScript([qml.RX(1.0, wires=0)], [qml.expval(qml.PauliZ(0))])
    >>> p((circuit,))
    ([<QuantumScript: wires=[0], params=1>],
    <function pennylane.transforms.batch_transform.map_batch_transform.<locals>.processing_fn(res: Tuple[~Result]) -> Tuple[~Result]>,
    ExecutionConfig(grad_on_execution=None, use_device_gradient=None, gradient_method=None, gradient_keyword_arguments={}, device_options={}, interface='autograd', derivative_order=1))
    >>> p((circuit,))[0][0].circuit
    [RX(1.0, wires=[0]), expval(PauliZ(wires=[0]))]


    The arguments and keyword arguments to this class can be used to custom most of the behavior for class.

    For additional control, developers can inherit from this class and override individual methods instead.

    For example, if the device supports any :class:`~.SampleMeasurement`, the developer can create a custom class by:

    .. code-block:: python

        class CustomPreprocessor(Preprocessor):

            def supports_measurement(self, m: qml.measurements.MeasurementProcess) -> bool:
                return isinstance(m, qml.measurements.SampleMeasurement)

    """

    def __init__(
        self,
        device_name: str,
        supported_operations: typing.Set[str],
        supported_observables: typing.Set[str],
        supported_measurements: typing.Set[type],
        supports_non_commuting_measurements: bool = False,
        supports_broadcasting: bool = False,
        supports_midcircuit_measurements: bool = False,
    ):

        self.device_name = device_name
        self.supported_operations = supported_operations
        self.supported_observables = supported_observables
        self.supported_measurements = supported_measurements
        self.supports_non_commuting_measurements = supports_non_commuting_measurements
        self.supports_broadcasting = supports_broadcasting
        self.supports_midcircuit_measurements = supports_midcircuit_measurements

    def supports_operator(self, op: qml.operation.Operator) -> bool:
        return op.name in self.supported_operations

    def supports_measurement(self, m: qml.measurements.MeasurementProcess) -> bool:
        return type(m) in self.supported_measurements

    def expand_fn(self, circuit: tape.QuantumScript) -> tape.QuantumScript:
        """ """
        if not self.supports_midcircuit_measurements and any(
            isinstance(o, MidMeasureMP) for o in circuit.operations
        ):
            circuit = qml.defer_measurements(circuit)

        for m in circuit.measurements:
            if not self.supports_measurement(m):
                raise qml.DeviceError(
                    f"Measurement {m} not supported on device {self.device_name}."
                )

            if m.obs is not None:
                if isinstance(m.obs, qml.operation.Tensor):
                    if any(o.name not in self.supported_observables for o in m.obs.obs):
                        raise qml.DeviceError(
                            f"Observable {m.obs} not supported on device {self.device_name}."
                        )
                elif m.obs.name not in self.supported_observables:
                    raise qml.DeviceError(
                        f"Observable {m.obs} not suppport on device {self.device_name}."
                    )

        if not all(self.supports_operator(op) for op in circuit._ops):
            try:
                new_ops = [
                    final_op
                    for op in circuit._ops
                    for final_op in _operator_decomposition_gen(op, self.supports_operator)
                ]
            except RecursionError as e:
                raise qml.DeviceError(
                    "Reached recursion limit trying to decompose operations. "
                    "Operator decomposition may have entered an infinite loop."
                ) from e
            circuit = qml.tape.QuantumScript(
                new_ops, circuit.measurements, circuit._prep, shots=circuit.shots
            )
        return circuit

    def batch_transform(
        self, circuit: tape.QuantumScript
    ) -> typing.Tuple[typing.Tuple[tape.QuantumScript], PostprocessingFn]:
        """ """
        circuits = (circuit,)

        def post_processing_fn(results):
            """null post processing."""
            return results

        if "Hamiltonian" not in self.supported_observables:

            if isinstance(circuit.measurements[0].obs, qml.Hamiltonian):
                circuits, post_processing_fn = qml.transforms.hamiltonian_expand(circuit)
        elif "Sum" not in self.supported_observables:
            if any(
                isinstance(m.obs, qml.ops.Sum) and isinstance(m, ExpectationMP)
                for m in circuit.measurements
            ):
                circuits, post_processing_fn = qml.transforms.sum_expand(circuit)
        elif not self.supports_non_commuting_measurements:
            if len(circuit._obs_sharing_wires) > 0 and all(
                not isinstance(
                    m,
                    (
                        qml.measurements.SampleMP,
                        qml.measurements.ProbabilityMP,
                        qml.measurements.CountsMP,
                    ),
                )
                for m in circuit.measurements
            ):
                circuits, post_processing_fn = qml.transforms.split_non_commuting(circuit)

        if circuit.batch_size is None or self.supports_broadcasting:
            return circuits, post_processing_fn

        # Expand each of the broadcasted Hamiltonian-expanded circuits
        expanded_tapes, broadcast_expand_fn = qml.transforms.map_batch_transform(
            qml.transforms.broadcast_expand, circuits
        )

        # Chain the postprocessing functions of the broadcasted-tape expansions and the Hamiltonian
        # expansion. Note that the application order is reversed compared to the expansion order,
        # i.e. while we first applied `hamiltonian_expand` to the tape, we need to process the
        # results from the broadcast expansion first.
        def total_processing(results):
            return post_processing_fn(broadcast_expand_fn(results))

        return expanded_tapes, total_processing

    def __call__(
        self,
        circuits: typing.Tuple[tape.QuantumScript],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> typing.Tuple[typing.Tuple[tape.QuantumScript], PostprocessingFn, ExecutionConfig]:
        """ """
        circuits = tuple(self.expand_fn(c) for c in circuits)

        circuits, batch_fn = qml.transforms.map_batch_transform(self.batch_transform, circuits)
        return circuits, batch_fn, execution_config
