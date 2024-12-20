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
    """Get the final state resulting from executing the given quantum script.

    This is an internal function used by ``default.mixed`` to simulate
    the evolution of a quantum circuit.

    Args:
        circuit (.QuantumScript): The quantum script containing operations and measurements
            that define the quantum computation.
        debugger (._Debugger): Debugger instance used for tracking execution and debugging
            circuit operations.

    Keyword Args:
        interface (str): The machine learning interface used to create the initial state.
        rng (Optional[numpy.random._generator.Generator]): A NumPy random number generator.
        prng_key (Optional[jax.random.PRNGKey]): A key for the JAX pseudo-random number 
            generator. Used only for simulations with JAX. If None, a ``numpy.random.default_rng`` 
            is used for sampling.

    Returns:
        Tuple[TensorLike, bool]: A tuple containing:
            - Tensor-like final state of the quantum script.
            - A boolean indicating whether the final state includes a batch dimension.

    Raises:
        ValueError: If the circuit contains invalid or unsupported operations.

    .. seealso:: 
        :func:`~.apply_operation`, :class:`~.QuantumScript`

    **Example**

    Simulate a simple quantum script to obtain its final state:

    .. code-block:: python

        from pennylane.devices.qubit_mixed import get_final_state
        from pennylane.tape import QuantumScript
        from pennylane.ops import RX, CNOT

        circuit = QuantumScript([RX(0.5, wires=0), CNOT(wires=[0, 1])])
        final_state, is_batched = get_final_state(circuit)
        print(final_state, is_batched)

    .. details::
        :title: Usage Details

        This function supports multiple execution backends and random number generators,
        such as NumPy and JAX. It initializes the quantum state, applies all operations in
        the circuit, and returns the final state tensor and batching information.

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
    """Perform the measurements specified in the circuit on the provided state.

    This is an internal function called by the ``default.mixed`` device to simulate
    measurement processes in a quantum circuit.

    Args:
        circuit (.QuantumScript): The quantum script containing operations and measurements
            to be simulated.
        state (TensorLike): The quantum state on which measurements are performed.
        is_state_batched (bool): Indicates whether the quantum state has a batch dimension.
        
    Keyword Args:
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): 
            A seed-like parameter for ``numpy.random.default_rng``. If no value is provided,
            a default random number generator is used.
        prng_key (Optional[jax.random.PRNGKey]): A key for the JAX pseudo-random number generator,
            used for sampling during JAX-based simulations. If None, a default NumPy RNG is used.
        readout_errors (List[Callable]): A list of quantum channels (callable functions) applied 
            to each wire during measurement to simulate readout errors.

    Returns:
        Tuple[TensorLike]: The measurement results. If the circuit specifies only one measurement,
        the result is a single tensor-like object. If multiple measurements are specified, a tuple
        of results is returned.

    Raises:
        ValueError: If the circuit contains invalid or unsupported measurements.

    .. seealso:: 
        :func:`~.measure`, :func:`~.measure_with_samples`

    **Example**

    Simulate a circuit measurement process on a given state:

    .. code-block:: python

        import numpy as np
        import pennylane as qml

        from pennylane.devices.qubit_mixed import measure_final_state
        from pennylane.tape import QuantumScript
        from pennylane.ops import RX, CNOT, PauliZ

        # Define a circuit with a PauliZ measurement
        circuit = QuantumScript(
            ops=[RX(0.5, wires=0), CNOT(wires=[0, 1])],
            measurements=[qml.expval(PauliZ(wires=0))]
        )

        # Simulate measurement
        state = np.ones((2,2,2,2)) * 0.25  # Initialize or compute the state
        results = measure_final_state(circuit, state, is_state_batched=False)
        print(results)

    .. details::
        :title: Usage Details

        The function supports both analytic and finite-shot measurement processes. 
        - In the analytic case (no shots specified), the exact expectation values 
          are computed for each measurement in the circuit.
        - In the finite-shot case (with shots specified), random samples are drawn 
          according to the specified measurement process, using the provided RNG 
          or PRNG key. Readout errors, if provided, are applied during the simulation.
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
    r"""
    Simulate a single quantum script.

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
