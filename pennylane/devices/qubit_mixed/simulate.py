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
"""Simulate a quantum script for a qubit mixed state device."""
from typing import Optional

# pylint: disable=protected-access
from numpy.random import default_rng

import pennylane as qml
from pennylane.devices.qubit.sampling import jax_random_split
from pennylane.math.interface_utils import get_canonical_interface_name
from pennylane.typing import Result

from .apply_operation import apply_operation
from .initialize_state import create_initial_state
from .measure import measure
from .sampling import measure_with_samples


def get_final_state(circuit, debugger=None, **execution_kwargs):
    """
    Get the final state that results from executing the given quantum script.

    This is an internal function that will be called by ``default.mixed``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        debugger (._Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with
        rng (Optional[numpy.random._generator.Generator]): A NumPy random number generator.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
            If None, a ``numpy.random.default_rng`` will be used for sampling.

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """
    prng_key = execution_kwargs.pop("prng_key", None)
    interface = execution_kwargs.get("interface", None)

    prep = None
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        prep = circuit[0]

    interface = get_canonical_interface_name(interface)
    state = create_initial_state(sorted(circuit.op_wires), prep, like=interface.get_like())

    # initial state is batched only if the state preparation (if it exists) is batched
    is_state_batched = bool(prep and prep.batch_size is not None)
    key = prng_key

    for op in circuit.operations[bool(prep) :]:
        state = apply_operation(
            op,
            state,
            is_state_batched=is_state_batched,
            debugger=debugger,
            prng_key=key,
            tape_shots=circuit.shots,
            **execution_kwargs,
        )

        # new state is batched if i) the old state is batched, or ii) the new op adds a batch dim
        is_state_batched = is_state_batched or op.batch_size is not None

    num_operated_wires = len(circuit.op_wires)
    for i in range(len(circuit.wires) - num_operated_wires):
        # If any measured wires are not operated on, we pad the density matrix with zeros.
        # We know they belong at the end because the circuit is in standard wire-order
        # Since it is a dm, we must pad it with 0s on the last row and last column
        current_axis = num_operated_wires + i + is_state_batched
        state = qml.math.stack(([state] + [qml.math.zeros_like(state)]), axis=current_axis)
        state = qml.math.stack(([state] + [qml.math.zeros_like(state)]), axis=-1)

    return state, is_state_batched


# pylint: disable=too-many-arguments, too-many-positional-arguments, unused-argument
def measure_final_state(circuit, state, is_state_batched, **execution_kwargs) -> Result:
    """
    Perform the measurements required by the circuit on the provided state.

    This is an internal function that will be called by ``default.mixed``.

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
        readout_errors (List[Callable]): List of channels to apply to each wire being measured
        to simulate readout errors.

    Returns:
        Tuple[TensorLike]: The measurement results
    """

    rng = execution_kwargs.get("rng", None)
    prng_key = execution_kwargs.get("prng_key", None)
    readout_errors = execution_kwargs.get("readout_errors", None)

    if not circuit.shots:
        # analytic case
        if len(circuit.measurements) == 1:
            return measure(circuit.measurements[0], state, is_state_batched, readout_errors)

        return tuple(
            measure(mp, state, is_state_batched=is_state_batched, readout_errors=readout_errors)
            for mp in circuit.measurements
        )

    # finite-shot case
    rng = default_rng(rng)
    results = tuple(
        measure_with_samples(
            circuit.measurements,
            state,
            shots=circuit.shots,
            is_state_batched=is_state_batched,
            rng=rng,
            prng_key=prng_key,
            readout_errors=readout_errors,
        )
    )

    if len(circuit.measurements) == 1:
        if circuit.shots.has_partitioned_shots:
            return tuple(res[0] for res in results)
        return results[0]
    return results


# pylint: disable=too-many-arguments, too-many-positional-arguments
def simulate(
    circuit: qml.tape.QuantumScript,
    debugger=None,
    state_cache: Optional[dict] = None,
    **execution_kwargs,
) -> Result:
    """Simulate a single quantum script.

        This is an internal function that will be called by ``default.mixed``.

        Args:
            circuit (QuantumScript): The single circuit to simulate
            rng (Optional[Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]]): A
                seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
                If no value is provided, a default RNG will be used.
            prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
                the key to the JAX pseudo random number generator. If None, a random key will be
                generated. Only for simulation using JAX.
            debugger (_Debugger): The debugger to use
            interface (str): The machine learning interface to create the initial state with
            readout_errors (List[Callable]): List of channels to apply to each wire being measured
            to simulate readout errors.

        Returns:
            tuple(TensorLike): The results of the simulation

        Note that this function can return measurements for non-commuting observables simultaneously.

        This function assumes that all operations provide matrices.

    >>> qs = qml.tape.QuantumScript(
    ...     [qml.RX(1.2, wires=0)],
    ...     [qml.expval(qml.PauliX(0)), qml.probs(wires=(0, 1))]
    ... )
    >>> simulate(qs)
    (0.0, array([0.68117888, 0.        , 0.31882112, 0.        ]))


    """
    prng_key = execution_kwargs.pop("prng_key", None)
    circuit = circuit.map_to_standard_wires()

    ops_key, meas_key = jax_random_split(prng_key)
    state, is_state_batched = get_final_state(
        circuit, debugger=debugger, prng_key=ops_key, **execution_kwargs
    )
    if state_cache is not None:
        state_cache[circuit.hash] = state
    return measure_final_state(
        circuit, state, is_state_batched, prng_key=meas_key, **execution_kwargs
    )
