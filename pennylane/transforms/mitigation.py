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
Contains the mitigation transform
"""
import pennylane as qml
from pennylane.operation import Operation

cirq_operation_map = None


def _load_operation_map():
    """If not done already, this function loads the mapping between PennyLane operations and Cirq
    operations.
    """
    global cirq_operation_map
    if cirq_operation_map is None:
        from pennylane_cirq.cirq_device import CirqOperation, CirqDevice

        operation_map = CirqDevice._operation_map
        inv_operation_map = {}

        for key in operation_map:
            if not operation_map[key]:
                continue

            inverted_operation = CirqOperation(operation_map[key].parametrization)
            inverted_operation.inv()

            inv_operation_map[key + Operation.string_for_inverse] = inverted_operation

        cirq_operation_map = {**operation_map, **inv_operation_map}

    return cirq_operation_map


def _tape_to_cirq(tape):
    """Converts a PennyLane :class:`~.QuantumTape` to a Cirq ``Circuit``.

    This function replicates the action of ``_apply_operation`` in
    ``pennylane_cirq.cirq_device.CirqDevice``. This functionality would ideally be factored-out
    into a standalone function in ``pennylane_cirq``, which we could just import here.

    Args:
        tape (.QuantumTape): input quantum tape

    Returns:
        cirq.Circuit: the corresponding circuit in Cirq (not including measurements or state
        preparations)
    """
    try:
        import cirq
        from cirq import Circuit
    except ImportError as e:
        raise ImportError("The Cirq package is required") from e

    cirq_operation_map = _load_operation_map()

    circuit = Circuit()
    wires_to_qubits = {wire: cirq.LineQubit(i) for i, wire in enumerate(tape.wires)}

    for operation in tape.operations:

        cirq_operation = cirq_operation_map[operation.name]

        if cirq_operation:
            cirq_operation.parametrize(*operation.parameters)

            q = [wires_to_qubits[wire] for wire in operation.wires]
            circuit.append(
                cirq_operation.apply(*q)
            )

    return circuit


def _cirq_to_tape(circuit, measurements=None):
    """Converts a Cirq ``Circuit`` to a PennyLane :class:`~.QuantumTape`.

    This function first converts the Cirq circuit to a Qiskit circuit using Mitiq. The Qiskit
    circuit is then converted to a :class:`~.QuantumTape` using :func:`~.from_qiskit`.

    Args:
        circuit (cirq.Circuit): input Cirq circuit
        measurements (Iterable[.MeasurementProcess]): measurements to be appended to tape

    Returns:
        .QuantumTape: the corresponding :class:`~.QuantumTape`
    """
    try:
        from mitiq.mitiq_qiskit import to_qiskit
    except ImportError as e:
        raise ImportError("The mitiq package is required") from e

    measurements = measurements or []
    qiskit_circuit = to_qiskit(circuit)

    with qml.tape.QuantumTape() as tape:
        qml.from_qiskit(qiskit_circuit)()

        for m in measurements:
            m.queue()

    return tape


def mitigate(tape, factory=None, scale_noise=None):
    """Returns a list of tapes to be executed, and a classical postprocessing function, for error
    mitigation.

    The `zero noise extrapolation <https://mitiq.readthedocs.io/en/stable/guide/guide-zne.html>`__
    method is used by harnessing the `mitiq <https://mitiq.readthedocs.io/en/stable/index.html>`__
    package.

    Args:
        tape (.QuantumTape): the tape whose output should be error mitigated
        factory (mitiq.zne.inference.Factory): the ``mitiq`` factory used to specify the inference
            method for zero noise extrapolation
        scale_noise (Callable): A transformation that folds the circuit for a given scale factor. If
            unspecified, defaults to ``mitiq.zne.scaling.fold_gates_at_random``.

    Returns:
        tuple[list[.QuantumTape], func]: The collection of quantum tapes to be executed, and the
        classical postprocessing function that should be applied to the executed tape results and
        will return the zero noise extrapolated values.
    """
    try:
        import mitiq
        from mitiq.zne.inference import RichardsonFactory
        from mitiq.zne.scaling import fold_gates_at_random
    except ImportError as e:
        raise ImportError("The mitiq package is required") from e

    if factory is None:
        factory = RichardsonFactory

    if scale_noise is None:
        scale_noise = fold_gates_at_random

    cirq_circuit = _tape_to_cirq(tape)
    factory._batch_populate_instack()
    circuits = factory._generate_circuits(cirq_circuit, scale_noise=scale_noise)
    tapes = [_cirq_to_tape(circuit, measurements=tape.measurements) for circuit in circuits]

    return tapes, lambda: None
