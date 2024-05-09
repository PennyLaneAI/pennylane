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
from functools import partial
from typing import Optional

import numpy as np
from numpy.random import default_rng

import pennylane as qml
from pennylane.measurements import MidMeasureMP
from pennylane.typing import Result

from .apply_operation import apply_operation
from .initialize_state import create_initial_state
from .measure import measure
from .sampling import jax_random_split, measure_with_samples

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


class _FlexShots(qml.measurements.Shots):
    """Shots class that allows zero shots."""

    # pylint: disable=super-init-not-called
    def __init__(self, shots=None):
        if isinstance(shots, int):
            self.total_shots = shots
            self.shot_vector = (qml.measurements.ShotCopies(shots, 1),)
        else:
            self.__all_tuple_init__([s if isinstance(s, tuple) else (s, 1) for s in shots])

        self._frozen = True


def _postselection_postprocess(state, is_state_batched, shots, rng=None, prng_key=None):
    """Update state after projector is applied."""
    if is_state_batched:
        raise ValueError(
            "Cannot postselect on circuits with broadcasting. Use the "
            "qml.transforms.broadcast_expand transform to split a broadcasted "
            "tape into multiple non-broadcasted tapes before executing if "
            "postselection is used."
        )

    # The floor function is being used here so that a norm very close to zero becomes exactly
    # equal to zero so that the state can become invalid. This way, execution can continue, and
    # bad postselection gives results that are invalid rather than results that look valid but
    # are incorrect.
    norm = qml.math.norm(state)

    if not qml.math.is_abstract(state) and qml.math.allclose(norm, 0.0):
        norm = 0.0

    if shots:
        # Clip the number of shots using a binomial distribution using the probability of
        # measuring the postselected state.
        if prng_key is not None:
            # pylint: disable=import-outside-toplevel
            from jax.random import binomial

            binomial_fn = partial(binomial, prng_key)
        else:
            binomial_fn = np.random.binomial if rng is None else rng.binomial

        postselected_shots = (
            [int(binomial_fn(s, float(norm**2))) for s in shots]
            if not qml.math.is_abstract(norm)
            else shots
        )

        # _FlexShots is used here since the binomial distribution could result in zero
        # valid samples
        shots = _FlexShots(postselected_shots)

    state = state / norm
    return state, shots


def get_final_state(circuit, debugger=None, **execution_kwargs):
    """
    Get the final state that results from executing the given quantum script.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        debugger (._Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with
        mid_measurements (None, dict): Dictionary of mid-circuit measurements
        rng (Optional[numpy.random._generator.Generator]): A NumPy random number generator.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
            If None, a ``numpy.random.default_rng`` will be for sampling.

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """
    rng = execution_kwargs.get("rng", None)
    prng_key = execution_kwargs.get("prng_key", None)
    interface = execution_kwargs.get("interface", None)
    mid_measurements = execution_kwargs.get("mid_measurements", None)
    circuit = circuit.map_to_standard_wires()

    prep = None
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        prep = circuit[0]

    state = create_initial_state(sorted(circuit.op_wires), prep, like=INTERFACE_TO_LIKE[interface])

    # initial state is batched only if the state preparation (if it exists) is batched
    is_state_batched = bool(prep and prep.batch_size is not None)
    key = prng_key

    for op in circuit.operations[bool(prep) :]:
        if isinstance(op, MidMeasureMP):
            prng_key, key = jax_random_split(prng_key)
        state = apply_operation(
            op,
            state,
            is_state_batched=is_state_batched,
            debugger=debugger,
            mid_measurements=mid_measurements,
            rng=rng,
            prng_key=key,
        )
        # Handle postselection on mid-circuit measurements
        if isinstance(op, qml.Projector):
            prng_key, key = jax_random_split(prng_key)
            state, circuit._shots = _postselection_postprocess(
                state, is_state_batched, circuit.shots, rng=rng, prng_key=key
            )

        # new state is batched if i) the old state is batched, or ii) the new op adds a batch dim
        is_state_batched = is_state_batched or (op.batch_size is not None)

    for _ in range(len(circuit.wires) - len(circuit.op_wires)):
        # if any measured wires are not operated on, we pad the state with zeros.
        # We know they belong at the end because the circuit is in standard wire-order
        state = qml.math.stack([state, qml.math.zeros_like(state)], axis=-1)

    return state, is_state_batched


# pylint: disable=too-many-arguments
def measure_final_state(circuit, state, is_state_batched, **execution_kwargs) -> Result:
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
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
            If None, the default ``sample_state`` function and a ``numpy.random.default_rng``
            will be for sampling.
        mid_measurements (None, dict): Dictionary of mid-circuit measurements

    Returns:
        Tuple[TensorLike]: The measurement results
    """
    rng = execution_kwargs.get("rng", None)
    prng_key = execution_kwargs.get("prng_key", None)
    mid_measurements = execution_kwargs.get("mid_measurements", None)

    circuit = circuit.map_to_standard_wires()

    # analytic case

    if not circuit.shots:
        if mid_measurements is not None:
            raise TypeError("Native mid-circuit measurements are only supported with finite shots.")

        if len(circuit.measurements) == 1:
            return measure(circuit.measurements[0], state, is_state_batched=is_state_batched)

        return tuple(
            measure(mp, state, is_state_batched=is_state_batched) for mp in circuit.measurements
        )

    # finite-shot case

    rng = default_rng(rng)
    results = measure_with_samples(
        circuit.measurements,
        state,
        shots=circuit.shots,
        is_state_batched=is_state_batched,
        rng=rng,
        prng_key=prng_key,
        mid_measurements=mid_measurements,
    )

    if len(circuit.measurements) == 1:
        if circuit.shots.has_partitioned_shots:
            return tuple(res[0] for res in results)

        return results[0]

    return results


def simulate(
    circuit: qml.tape.QuantumScript,
    debugger=None,
    state_cache: Optional[dict] = None,
    **execution_kwargs,
) -> Result:
    """Simulate a single quantum script.

    This is an internal function that is used by``default.qubit``.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        debugger (_Debugger): The debugger to use
        state_cache=None (Optional[dict]): A dictionary mapping the hash of a circuit to
            the pre-rotated state. Used to pass the state between forward passes and vjp
            calculations.
        rng (Optional[numpy.random._generator.Generator]): A NumPy random number generator.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. If None, a random key will be
            generated. Only for simulation using JAX.
        interface (str): The machine learning interface to create the initial state with

    Returns:
        tuple(TensorLike): The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.

    This function assumes that all operations provide matrices.

    >>> qs = qml.tape.QuantumScript([qml.RX(1.2, wires=0)], [qml.expval(qml.Z(0)), qml.probs(wires=(0,1))])
    >>> simulate(qs)
    (0.36235775447667357,
    tensor([0.68117888, 0.        , 0.31882112, 0.        ], requires_grad=True))

    """
    rng = execution_kwargs.get("rng", None)
    prng_key = execution_kwargs.get("prng_key", None)
    interface = execution_kwargs.get("interface", None)

    has_mcm = any(isinstance(op, MidMeasureMP) for op in circuit.operations)
    if circuit.shots and has_mcm:
        return simulate_one_shot_native_mcm(
            circuit, debugger=debugger, rng=rng, prng_key=prng_key, interface=interface
        )

    ops_key, meas_key = jax_random_split(prng_key)
    state, is_state_batched = get_final_state(
        circuit, debugger=debugger, rng=rng, prng_key=ops_key, interface=interface
    )
    if state_cache is not None:
        state_cache[circuit.hash] = state
    return measure_final_state(circuit, state, is_state_batched, rng=rng, prng_key=meas_key)


def simulate_one_shot_native_mcm(
    circuit: qml.tape.QuantumScript, debugger=None, **execution_kwargs
) -> Result:
    """Simulate a single shot of a single quantum script with native mid-circuit measurements.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        debugger (_Debugger): The debugger to use
        rng (Optional[numpy.random._generator.Generator]): A NumPy random number generator.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. If None, a random key will be
            generated. Only for simulation using JAX.
        interface (str): The machine learning interface to create the initial state with

    Returns:
        tuple(TensorLike): The results of the simulation
        dict: The mid-circuit measurement results of the simulation
    """
    rng = execution_kwargs.get("rng", None)
    prng_key = execution_kwargs.get("prng_key", None)
    interface = execution_kwargs.get("interface", None)

    ops_key, meas_key = jax_random_split(prng_key)
    mid_measurements = {}
    state, is_state_batched = get_final_state(
        circuit,
        debugger=debugger,
        interface=interface,
        mid_measurements=mid_measurements,
        rng=rng,
        prng_key=ops_key,
    )
    return measure_final_state(
        circuit,
        state,
        is_state_batched,
        rng=rng,
        prng_key=meas_key,
        mid_measurements=mid_measurements,
    )
