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
"""Simulate a quantum script for a qutrit mixed state device."""
# pylint: disable=protected-access
from numpy.random import default_rng

import pennylane as qml
from pennylane.typing import Result

from .initialize_state import create_initial_state
from .apply_operation import apply_operation
from .measure import measure
from .sampling import measure_with_samples
from .utils import QUDIT_DIM


INTERFACE_TO_LIKE = {
    # map interfaces known by autoray to themselves
    None: None,
    "numpy": "numpy",
    "autograd": "autograd",
    "jax": "jax",
    "torch": "torch",
    "tensorflow": "tensorflow",
    # map non-standard interfaces to those known by autoray
    "auto": None,
    "scipy": "numpy",
    "jax-jit": "jax",
    "jax-python": "jax",
    "JAX": "jax",
    "pytorch": "torch",
    "tf": "tensorflow",
    "tensorflow-autograph": "tensorflow",
    "tf-autograph": "tensorflow",
}


def get_final_state(circuit, debugger=None, interface=None):
    """
    Get the final state that results from executing the given quantum script.

    This is an internal function that will be called by ``default.qutrit.mixed``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        debugger (._Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """
    circuit = circuit.map_to_standard_wires()

    prep = None
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        prep = circuit[0]

    state = create_initial_state(sorted(circuit.op_wires), prep, like=INTERFACE_TO_LIKE[interface])

    # initial state is batched only if the state preparation (if it exists) is batched
    is_state_batched = bool(prep and prep.batch_size is not None)
    for op in circuit.operations[bool(prep) :]:
        state = apply_operation(op, state, is_state_batched=is_state_batched, debugger=debugger)

        # new state is batched if i) the old state is batched, or ii) the new op adds a batch dim
        is_state_batched = is_state_batched or op.batch_size is not None

    num_operated_wires = len(circuit.op_wires)
    for i in range(len(circuit.wires) - num_operated_wires):
        # If any measured wires are not operated on, we pad the density matrix with zeros.
        # We know they belong at the end because the circuit is in standard wire-order
        # Since it is a dm, we must pad it with 0s on the last row and last column
        current_axis = num_operated_wires + i + is_state_batched
        state = qml.math.stack(
            ([state] + [qml.math.zeros_like(state)] * (QUDIT_DIM - 1)), axis=current_axis
        )
        state = qml.math.stack(([state] + [qml.math.zeros_like(state)] * (QUDIT_DIM - 1)), axis=-1)

    return state, is_state_batched


def measure_final_state(circuit, state, is_state_batched, rng=None, prng_key=None) -> Result:
    """
    Perform the measurements required by the circuit on the provided state.

    This is an internal function that will be called by ``default.qutrit.mixed``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        state (TensorLike): The state to perform measurement on
        is_state_batched (bool): Whether the state has a batch dimension or not.
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
            If None, the default ``sample_state`` function and a ``numpy.random.default_rng``
            will be for sampling.

    Returns:
        Tuple[TensorLike]: The measurement results
    """

    circuit = circuit.map_to_standard_wires()

    if not circuit.shots:
        # analytic case

        if len(circuit.measurements) == 1:
            return measure(circuit.measurements[0], state, is_state_batched)

        return tuple(measure(mp, state, is_state_batched) for mp in circuit.measurements)

    # finite-shot case
    rng = default_rng(rng)
    results = tuple(
        measure_with_samples(
            mp,
            state,
            shots=circuit.shots,
            is_state_batched=is_state_batched,
            rng=rng,
            prng_key=prng_key,
        )
        for mp in circuit.measurements
    )

    if len(circuit.measurements) == 1:
        return results[0]
    if circuit.shots.has_partitioned_shots:
        return tuple(zip(*results))
    return results


def simulate(
    circuit: qml.tape.QuantumScript, rng=None, prng_key=None, debugger=None, interface=None
) -> Result:
    """Simulate a single quantum script.

    This is an internal function that will be called by ``default.qutrit.mixed``.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. If None, a random key will be
            generated. Only for simulation using JAX.
        debugger (_Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with

    Returns:
        tuple(TensorLike): The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.

    This function assumes that all operations provide matrices.

    >>> qs = qml.tape.QuantumScript([qml.TRX(1.2, wires=0)], [qml.expval(qml.GellMann(0, 3)), qml.probs(wires=(0,1))])
    >>> simulate(qs)
    (0.36235775447667357,
    tensor([0.68117888, 0.        , 0.        , 0.31882112, 0.        , 0.        ], requires_grad=True))

    """
    state, is_state_batched = get_final_state(circuit, debugger=debugger, interface=interface)
    return measure_final_state(circuit, state, is_state_batched, rng=rng, prng_key=prng_key)
