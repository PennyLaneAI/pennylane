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
from pennylane.wires import Wires

from .initialize_state import create_initial_state
from .apply_operation import apply_operation
from .measure import measure
from .sampling import measure_with_samples


def expand_state_over_wires(state, state_wires, all_wires, is_state_batched):
    """
    Expand and re-order a state given some initial and target wire orders, setting
    all additional wires to the 0 state.

    Args:
        state (~pennylane.typing.TensorLike): The state to re-order and expand
        state_wires (.Wires): The wire order of the inputted state
        all_wires (.Wires): The desired wire order
        is_state_batched (bool): Whether the state has a batch dimension or not

    Returns:
        TensorLike: The state in the new desired size and order
    """
    is_torch = qml.math.get_interface(state) == "torch"
    pad_width = 2 ** len(all_wires) - 2 ** len(state_wires)
    pad = (pad_width, 0) if is_torch else (0, pad_width)
    shape = (2,) * len(all_wires)
    if is_state_batched:
        pad = ((0, 0), pad)
        batch_size = qml.math.shape(state)[0]
        shape = (batch_size,) + shape
        state = qml.math.reshape(state, (batch_size, -1))
    else:
        if is_torch:
            pad = (pad,)
        state = qml.math.flatten(state)

    state = qml.math.pad(state, pad)
    state = qml.math.reshape(state, shape)

    # re-order
    new_wire_order = Wires.unique_wires([all_wires, state_wires]) + state_wires
    desired_axes = [new_wire_order.index(w) for w in all_wires]
    if is_state_batched:
        desired_axes = [0] + [i + 1 for i in desired_axes]
    return qml.math.transpose(state, desired_axes)


def get_final_state(circuit, debugger=None):
    """
    Get the final state that results from executing the given quantum script.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        debugger (._Debugger): The debugger to use

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """
    circuit = circuit.map_to_standard_wires()

    prep = None
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        prep = circuit[0]

    state = create_initial_state(circuit.op_wires, prep)

    # initial state is batched only if the state preparation (if it exists) is batched
    is_state_batched = False
    if prep and prep.batch_size is not None:
        is_state_batched = True

    for op in circuit.operations[bool(prep) :]:
        state = apply_operation(op, state, is_state_batched=is_state_batched, debugger=debugger)

        # new state is batched if i) the old state is batched, or ii) the new op adds a batch dim
        is_state_batched = is_state_batched or op.batch_size is not None

    if set(circuit.op_wires) < set(circuit.wires):
        state = expand_state_over_wires(
            state,
            Wires(range(len(circuit.op_wires))),
            Wires(range(circuit.num_wires)),
            is_state_batched,
        )

    return state, is_state_batched


def measure_final_state(circuit, state, is_state_batched, rng=None) -> Result:
    """
    Perform the measurements required by the circuit on the provided state.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        state (TensorLike): The state to perform measurement on
        is_state_batched (bool): Whether the state has a batch dimension or not.
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.

    Returns:
        Tuple[TensorLike]: The measurement results
    """
    circuit = circuit.map_to_standard_wires()

    if not circuit.shots:
        # analytic case

        if len(circuit.measurements) == 1:
            return measure(circuit.measurements[0], state, is_state_batched=is_state_batched)

        return tuple(
            measure(mp, state, is_state_batched=is_state_batched) for mp in circuit.measurements
        )

    # finite-shot case

    rng = default_rng(rng)
    results = measure_with_samples(
        circuit.measurements, state, shots=circuit.shots, is_state_batched=is_state_batched, rng=rng
    )

    if len(circuit.measurements) == 1:
        if circuit.shots.has_partitioned_shots:
            return tuple(res[0] for res in results)

        return results[0]

    return results


def simulate(circuit: qml.tape.QuantumScript, rng=None, debugger=None) -> Result:
    """Simulate a single quantum script.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        debugger (_Debugger): The debugger to use

    Returns:
        tuple(TensorLike): The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.

    This function assumes that all operations provide matrices.

    >>> qs = qml.tape.QuantumScript([qml.RX(1.2, wires=0)], [qml.expval(qml.PauliZ(0)), qml.probs(wires=(0,1))])
    >>> simulate(qs)
    (0.36235775447667357,
    tensor([0.68117888, 0.        , 0.31882112, 0.        ], requires_grad=True))

    """
    state, is_state_batched = get_final_state(circuit, debugger=debugger)
    return measure_final_state(circuit, state, is_state_batched, rng=rng)
