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
"""Simulate a quantum script."""
# pylint: disable=protected-access
from numpy.random import default_rng

import pennylane as qml
from pennylane.typing import Result

from .initialize_state import create_initial_state
from .apply_operation import apply_operation
from .measure import measure
from .sampling import measure_with_samples


def simulate(circuit: qml.tape.QuantumScript, rng=None, debugger=None) -> Result:
    """Simulate a single quantum script.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        debugger (._Debugger): The debugger to use

    Returns:
        tuple(TensorLike): The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.

    It does currently not support sampling or observables without diagonalizing gates.

    This function assumes that all operations provide matrices.

    >>> qs = qml.tape.QuantumScript([qml.RX(1.2, wires=0)], [qml.expval(qml.PauliZ(0)), qml.probs(wires=(0,1))])
    >>> simulate(qs)
    (0.36235775447667357,
    tensor([0.68117888, 0.        , 0.31882112, 0.        ], requires_grad=True))

    """
    if set(circuit.wires) != set(range(circuit.num_wires)):
        wire_map = {w: i for i, w in enumerate(circuit.wires)}
        circuit = qml.map_wires(circuit, wire_map)

    state = create_initial_state(circuit.wires, circuit._prep[0] if circuit._prep else None)

    # initial state is batched only if the state preparation (if it exists) is batched
    is_state_batched = False
    if circuit._prep and circuit._prep[0].batch_size is not None:
        is_state_batched = True

    for op in circuit._ops:
        state = apply_operation(op, state, is_state_batched=is_state_batched, debugger=debugger)

        # new state is batched if i) the old state is batched, or ii) the new op adds a batch dim
        is_state_batched = is_state_batched or op.batch_size is not None

    if not circuit.shots:
        # analytic case

        if len(circuit.measurements) == 1:
            return measure(circuit.measurements[0], state)

        return tuple(
            measure(mp, state, is_state_batched=is_state_batched) for mp in circuit.measurements
        )

    # finite-shot case

    if len(circuit.measurements) == 1:
        return measure_with_samples(circuit.measurements[0], state, shots=circuit.shots, rng=rng)

    rng = default_rng(rng)
    results = tuple(
        measure_with_samples(
            mp, state, shots=circuit.shots, is_state_batched=is_state_batched, rng=rng
        )
        for mp in circuit.measurements
    )

    # no shot vector
    if not circuit.shots.has_partitioned_shots:
        return results

    # shot vector case: move the shot vector axis before the measurement axis
    return tuple(zip(*results))
