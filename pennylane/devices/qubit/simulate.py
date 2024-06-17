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
import logging
import sys

# pylint: disable=protected-access
from collections import Counter
from functools import partial, singledispatch
from typing import Optional

import numpy as np
from numpy.random import default_rng

import pennylane as qml
from pennylane.logging import debug_logger
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    MidMeasureMP,
    ProbabilityMP,
    SampleMP,
    VarianceMP,
)
from pennylane.transforms.dynamic_one_shot import gather_mcm as dyn_gather_mcm
from pennylane.typing import Result

from .apply_operation import apply_operation
from .initialize_state import create_initial_state
from .measure import measure
from .sampling import jax_random_split, measure_with_samples

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

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
        elif isinstance(shots, self.__class__):
            return  # self already _is_ shots as defined by __new__
        else:
            self.__all_tuple_init__([s if isinstance(s, tuple) else (s, 1) for s in shots])

        self._frozen = True


def _postselection_postprocess(state, is_state_batched, shots, **execution_kwargs):
    """Update state after projector is applied."""
    if is_state_batched:
        raise ValueError(
            "Cannot postselect on circuits with broadcasting. Use the "
            "qml.transforms.broadcast_expand transform to split a broadcasted "
            "tape into multiple non-broadcasted tapes before executing if "
            "postselection is used."
        )

    rng = execution_kwargs.get("rng", None)
    prng_key = execution_kwargs.get("prng_key", None)
    postselect_mode = execution_kwargs.get("postselect_mode", None)

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
            shots
            if postselect_mode == "fill-shots" or qml.math.is_abstract(norm)
            else [int(binomial_fn(s, float(norm**2))) for s in shots]
        )

        # _FlexShots is used here since the binomial distribution could result in zero
        # valid samples
        shots = _FlexShots(postselected_shots)

    state = state / norm
    return state, shots


@debug_logger
def get_final_state(circuit, debugger=None, **execution_kwargs):
    """
    Get the final state that results from executing the given quantum script.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate. This circuit is assumed to have
            non-negative integer wire labels
        debugger (._Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with
        initial_state (TensorLike): Initial state vector
        mid_measurements (None, dict): Dictionary of mid-circuit measurements
        rng (Optional[numpy.random._generator.Generator]): A NumPy random number generator.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
            If None, a ``numpy.random.default_rng`` will be used for sampling.
        postselect_mode (str): Configuration for handling shots with mid-circuit measurement
            postselection. Use ``"hw-like"`` to discard invalid shots and ``"fill-shots"`` to
            keep the same number of shots. Default is ``None``.

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """
    prng_key = execution_kwargs.pop("prng_key", None)
    interface = execution_kwargs.get("interface", None)
    initial_state = execution_kwargs.get("initial_state", None)

    if initial_state is None:
        circuit = circuit.map_to_standard_wires()

    prep = None
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        prep = circuit[0]

    state = (
        create_initial_state(sorted(circuit.op_wires), prep, like=INTERFACE_TO_LIKE[interface])
        if initial_state is None
        else initial_state
    )

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
            prng_key=key,
            **execution_kwargs,
        )
        # Handle postselection on mid-circuit measurements
        if isinstance(op, qml.Projector):
            prng_key, key = jax_random_split(prng_key)
            state, new_shots = _postselection_postprocess(
                state, is_state_batched, circuit.shots, prng_key=key, **execution_kwargs
            )
            circuit._shots = new_shots

        # new state is batched if i) the old state is batched, or ii) the new op adds a batch dim
        is_state_batched = is_state_batched or (op.batch_size is not None)

    if initial_state is None:
        for _ in range(circuit.num_wires - len(circuit.op_wires)):
            # if any measured wires are not operated on, we pad the state with zeros.
            # We know they belong at the end because the circuit is in standard wire-order
            state = qml.math.stack([state, qml.math.zeros_like(state)], axis=-1)

    return state, is_state_batched


# pylint: disable=too-many-arguments
@debug_logger
def measure_final_state(
    circuit, state, is_state_batched, initial_state=None, **execution_kwargs
) -> Result:
    """
    Perform the measurements required by the circuit on the provided state.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate. This circuit is assumed to have
            non-negative integer wire labels
        state (TensorLike): The state to perform measurement on
        is_state_batched (bool): Whether the state has a batch dimension or not.
        initial_state (TensorLike): Initial state vector
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
            If None, the default ``sample_state`` function and a ``numpy.random.default_rng``
            will be used for sampling.
        mid_measurements (None, dict): Dictionary of mid-circuit measurements

    Returns:
        Tuple[TensorLike]: The measurement results
    """

    rng = execution_kwargs.get("rng", None)
    prng_key = execution_kwargs.get("prng_key", None)
    mid_measurements = execution_kwargs.get("mid_measurements", None)

    if initial_state is None:
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


@debug_logger
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
        postselect_mode (str): Configuration for handling shots with mid-circuit measurement
            postselection. Use ``"hw-like"`` to discard invalid shots and ``"fill-shots"`` to
            keep the same number of shots. Default is ``None``.

    Returns:
        tuple(TensorLike): The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.

    This function assumes that all operations provide matrices.

    >>> qs = qml.tape.QuantumScript([qml.RX(1.2, wires=0)], [qml.expval(qml.Z(0)), qml.probs(wires=(0,1))])
    >>> simulate(qs)
    (0.36235775447667357,
    tensor([0.68117888, 0.        , 0.31882112, 0.        ], requires_grad=True))

    """
    prng_key = execution_kwargs.pop("prng_key", None)
    circuit = circuit.map_to_standard_wires()

    has_mcm = any(isinstance(op, MidMeasureMP) for op in circuit.operations)
    if circuit.shots and has_mcm:
        if execution_kwargs.get("mcm_method", None) == "tree-traversal":
            n_mcms = sum(isinstance(op, MidMeasureMP) for op in circuit.operations)
            if 2 * n_mcms + 100 > sys.getrecursionlimit():
                sys.setrecursionlimit(2 * n_mcms + 100)
            return simulate_tree_mcm(circuit, **execution_kwargs)

        results = []
        aux_circ = qml.tape.QuantumScript(
            circuit.operations,
            circuit.measurements,
            shots=[1],
            trainable_params=circuit.trainable_params,
        )
        keys = jax_random_split(prng_key, num=circuit.shots.total_shots)
        if qml.math.get_deep_interface(circuit.data) == "jax" and prng_key is not None:
            # pylint: disable=import-outside-toplevel
            import jax

            def simulate_partial(k):
                return simulate_one_shot_native_mcm(
                    aux_circ, debugger=debugger, prng_key=k, **execution_kwargs
                )

            results = jax.vmap(simulate_partial, in_axes=(0,))(keys)
            results = tuple(zip(*results))
        else:
            for i in range(circuit.shots.total_shots):
                results.append(
                    simulate_one_shot_native_mcm(
                        aux_circ, debugger=debugger, prng_key=keys[i], **execution_kwargs
                    )
                )
        return tuple(results)

    ops_key, meas_key = jax_random_split(prng_key)
    state, is_state_batched = get_final_state(
        circuit, debugger=debugger, prng_key=ops_key, **execution_kwargs
    )
    if state_cache is not None:
        state_cache[circuit.hash] = state
    return measure_final_state(
        circuit, state, is_state_batched, prng_key=meas_key, **execution_kwargs
    )


# pylint: disable=too-many-arguments, dangerous-default-value
def simulate_tree_mcm(
    circuit: qml.tape.QuantumScript,
    debugger=None,
    initial_state=None,
    mcm_active=None,
    mcm_samples=None,
    **execution_kwargs,
) -> Result:
    """Simulate a single quantum script with native mid-circuit measurements.

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
        initial_state (TensorLike): Initial state vector
        mcm_active (dict): Mid-circuit measurement values or all parent circuits of ``circuit``
        mcm_samples (dict): Mid-circuit measurement samples or all parent circuits of ``circuit``

    Returns:
        tuple(TensorLike): The results of the simulation
    """
    interface = execution_kwargs.get("interface", None)
    postselect_mode = execution_kwargs.get("postselect_mode", None)

    samples_present = any(isinstance(mp, SampleMP) for mp in circuit.measurements)
    postselect_present = any(
        op.postselect is not None for op in circuit.operations if isinstance(op, MidMeasureMP)
    )
    if postselect_present and samples_present and circuit.batch_size is not None:
        raise ValueError(
            "Returning qml.sample is not supported when postselecting mid-circuit "
            "measurements with broadcasting"
        )

    #########################
    # shot vector treatment #
    #########################
    if circuit.shots.has_partitioned_shots:
        results = []
        for s in circuit.shots:
            aux_circuit = qml.tape.QuantumScript(
                circuit.operations,
                circuit.measurements,
                shots=qml.measurements.Shots(s),
                trainable_params=circuit.trainable_params,
            )
            results.append(simulate_tree_mcm(aux_circuit, debugger, **execution_kwargs))
        return tuple(results)

    #######################
    # main implementation #
    #######################

    mcms = tuple(op for op in circuit.operations if isinstance(op, MidMeasureMP))
    n_mcms = len(mcms)
    mcm_current = qml.math.zeros(n_mcms + 1, dtype=bool)
    mcm_active = dict((k, v) for k, v in zip(mcms, mcm_current[1:]))
    mcm_samples = dict((k, qml.math.empty(circuit.shots.total_shots, dtype=int)) for k in mcms)
    circuit_wires = circuit.wires
    depth = 0
    circuits = []
    circuit_right = circuit
    for _ in mcms:
        circuit_left, circuit_right, _ = circuit_up_to_first_mcm(circuit_right)
        circuits.append(circuit_left)
    circuits.append(circuit_right)
    states = [prep_initial_state(circuit_left, interface, None, circuit_wires)] + [None] * n_mcms
    initial_state = states[0]
    results_0 = [None] * (n_mcms + 1)
    results_1 = [None] * (n_mcms + 1)
    counts = [None] * (n_mcms + 1)

    # The goal is to obtain the measurements of the zero-branch and one-branch
    # and to combine them into the final result. Exit the loop once the
    # zero-branch and one-branch measurements are available.
    while results_0[1] is None or results_1[1] is None:

        # Combine two leaves once measurements are available
        if results_0[depth] is not None and results_1[depth] is not None:
            # We use `circuits[-1].measurements` since it contains the
            # target measurements (this is the only tape segment with
            # unmodified measurements)
            measurements = [{} for _ in circuits[-1].measurements]
            # Special treatment for single measurements
            single_measurement = len(circuits[-1].measurements) == 1
            # Store each measurement in a dictionary `{branch: (count, measure)}`
            for branch, count in counts[depth].items():
                meas = results_0[depth] if branch == 0 else results_1[depth]
                if single_measurement:
                    meas = [meas]
                for i, m in enumerate(meas):
                    measurements[i][branch] = (count, m)
            # Call `combine_measurements` to count-average measurements
            measurements = combine_measurements(circuits[-1], measurements, mcm_samples)
            # Clear stack
            counts[depth] = None
            results_0[depth] = None
            results_1[depth] = None
            mcm_current[depth] = False
            # Go up one level to explore alternate subtree of the same depth
            depth -= 1
            if not mcm_current[depth]:
                results_0[depth] = measurements
                mcm_current[depth] = True
            else:
                results_1[depth] = measurements
                mcm_current[depth] = False
                # depth -= 1
            if depth == 0:
                continue
            initial_state = branch_state(
                states[depth], mcm_current[depth], mcms[depth - 1].wires, mcms[depth - 1].reset
            )
            continue

        # Run the circuit segment at the current depth
        circtmp = circuits[depth].copy()
        # Make sure the state is the all-wire state
        circtmp._ops = [qml.StatePrep(initial_state.ravel(), wires=circuit_wires)] + circtmp._ops
        # Parse shots for the current branche
        if counts[depth]:
            shots = counts[depth][int(mcm_current[depth])]
        else:
            shots = circuits[depth].shots.total_shots
        # print(shots)
        # Update active branch dict
        mcm_active = dict((k, v) for k, v in zip(mcms, mcm_current[1:]))
        # If num_shots is zero, update measurements with empty tuple
        if not bool(shots):
            measurements = qml.math.array([], dtype=int)  # WARN: is this line necessary?
            # On the zero-branch, update measurements and switch to the one-branch
            if not mcm_current[depth]:
                results_0[depth] = tuple()
                mcm_current[depth] = True
                initial_state = branch_state(
                    states[depth], mcm_current[depth], mcms[depth - 1].wires, mcms[depth - 1].reset
                )
                continue
            # On the one-branch, update measurements and continue
            results_1[depth] = tuple()
            continue
        else:
            # If num_shots is non-zero, simulate circuit segment
            circtmp._shots = qml.measurements.shots.Shots(shots)
            # print(list(mcm_active.values()))
            # print(initial_state)
            state, is_state_batched = get_final_state(
                circtmp,
                debugger=debugger,
                initial_state=initial_state,
                mid_measurements=mcm_active,
                **execution_kwargs,
            )
            measurements = measure_final_state(
                circtmp, state, is_state_batched, initial_state=initial_state, **execution_kwargs
            )
            # if depth >= n_mcms:
            #     print(measurements)
        # If not at a leaf, project on the zero-branch and increase depth by one
        if depth < n_mcms:
            # Update the active branch samples with `update_mcm_samples`
            samples = qml.math.atleast_1d(measurements)
            update_mcm_samples(mcms[depth], samples, mcm_active, mcm_samples)
            # print(f"branching to {mcm_current[depth]}")
            initial_state = branch_state(
                state, mcm_current[depth + 1], mcms[depth].wires, mcms[depth].reset
            )
            depth += 1
            # Store a copy of the state-vector to project on the one-branch
            states[depth] = state
            counts[depth] = samples_to_counts(samples, all_outcomes=True)
            continue

        # If at a zero-branch leaf, update measurements and switch to the one-branch
        if not mcm_current[depth]:
            results_0[depth] = measurements
            mcm_current[depth] = True
            initial_state = branch_state(
                states[depth], mcm_current[depth], mcms[depth - 1].wires, mcms[depth - 1].reset
            )
            continue
        # If at a one-branch leaf, update measurements
        results_1[depth] = measurements

    # Combine first two branches
    measurements = [{} for _ in circuit.measurements]
    single_measurement = len(circuit.measurements) == 1
    for branch, count in counts[depth].items():
        meas = results_0[depth] if branch == 0 else results_1[depth]
        if single_measurement:
            meas = [meas]
        for i, m in enumerate(meas):
            measurements[i][branch] = (count, m)
    measurements = combine_measurements(circuit, measurements, mcm_samples)

    return measurements


def branch_state(state, branch, wire, reset):
    state = apply_operation(qml.Projector([branch], wire), state)
    if qml.math.norm(state) < 1e-16:
        pass
    state = state / qml.math.norm(state)
    if reset and branch == 1:
        state = apply_operation(qml.PauliX(wire), state)
    return state


def samples_to_counts(samples, all_outcomes=True):
    """Converts samples to counts.

    This function forces integer keys and values which are required by ``simulate_tree_mcm``.
    """
    counts = qml.math.unique(samples, return_counts=True)
    result = {0: 0, 1: 0} if all_outcomes else {}
    result.update(dict((int(x), int(y)) for x, y in zip(*counts)))
    return result


def prep_initial_state(circuit_base, interface, initial_state, wires):
    """Returns an initial state which will act on all wires.

    ``get_final_state`` executes a circuit on a subset of wires found in operations
    or measurements, unless an initial_state is passed as an optional argument.
    This function makes sure that an initial state with the correct size is passed
    on the first invocation of ``simulate_tree_mcm``. ``wires`` should be the wires attribute
    of the original circuit."""
    if initial_state is not None:
        return initial_state
    prep = None
    if len(circuit_base) > 0 and isinstance(circuit_base[0], qml.operation.StatePrepBase):
        prep = circuit_base[0]
    return create_initial_state(wires, prep, like=INTERFACE_TO_LIKE[interface])


def prune_mcm_samples(op, branch, mcm_active, mcm_samples):
    """Removes samples from mid-measurement sample dictionary given a MidMeasureMP and branch.

    Post-selection on a given mid-circuit measurement leads to ignoring certain branches
    of the tree and samples. The corresponding samples in all other mid-circuit measurement
    must be deleted accordingly. We need to find which samples are
    corresponding to the current branch by looking at all parent nodes.
    """
    mask = mcm_samples[op] == branch
    for k, v in mcm_active.items():
        if k == op:
            break
        mask = np.logical_and(mask, mcm_samples[k] == v)
    for k in mcm_samples.keys():
        mcm_samples[k] = mcm_samples[k][np.logical_not(mask)]


def get_active_sample_mask(op, mcm_active, mcm_samples):
    shape = next(iter(mcm_samples.values())).shape
    mask = np.ones(shape, dtype=bool)
    for k, v in mcm_active.items():
        if k == op:
            break
        mask = np.logical_and(mask, mcm_samples[k] == v)
    return mask


def get_active_samples(op, mcm_active, mcm_samples):
    mask = get_active_sample_mask(op, mcm_active, mcm_samples)
    return mcm_samples[op][mask]


def update_mcm_samples(op, samples, mcm_active, mcm_samples):
    """Updates the mid-measurement sample dictionary given a MidMeasureMP and samples.

    If the ``mcm_active`` dictionary is empty, we are at the root and ``mcm_samples`
    is simply updated with ``samples``.

    If the ``mcm_active`` dictionary is not empty, we need to find which samples are
    corresponding to the current branch by looking at all parent nodes. ``mcm_samples`
    is then updated with samples at indices corresponding to parent nodes.

    To illustrate how the function works, let's take an example. Suppose there are
    `2**20` shots in total and the computation is midway through the circuit at the
    7th MCM, the active branch is `[0,1,1,0,0,1]` and each MCM everything happened to
    split the counts 50/50 so there are `2**24` samples to update.
    These samples are not contiguous in general and they are correlated with the parent
    branches, so where do they go? They must update the `2**24` elements whose parent
    sequence corresponds to `[0,1,1,0,0,1]`.
    """
    if mcm_active:
        shape = next(iter(mcm_samples.values())).shape
        mask = np.ones(shape, dtype=bool)
        for k, v in mcm_active.items():
            if k == op:
                break
            mask = np.logical_and(mask, mcm_samples[k] == v)
        if op not in mcm_samples:
            mcm_samples[op] = np.empty(shape, dtype=samples.dtype)
        mcm_samples[op][mask] = samples
    else:
        mcm_samples[op] = samples


def circuit_up_to_first_mcm(circuit):
    """Returns two circuits; one that runs up-to the next mid-circuit measurement and one that runs beyond it.

    Measurement processes are computed on each branch, and then combined at the node.
    This can be done recursively until a single node is left.
    This is true for `counts`, `expval`, `probs` and `sample` but not `var` measurements.
    There is no way to recombine "partial variances" from two branches, so `var` measurements are replaced
    by `sample` measurements from which the variance is calculated (once samples from all branches are available).
    """
    if not has_mid_circuit_measurements(circuit):
        return circuit, None, None

    # find next MidMeasureMP
    def find_next_mcm(circuit):
        for i, op in enumerate(circuit.operations):
            if isinstance(op, MidMeasureMP):
                return i, op
        return len(circuit.operations) + 1, None

    i, op = find_next_mcm(circuit)
    # run circuit until next MidMeasureMP and sample
    circuit_base = qml.tape.QuantumScript(
        circuit.operations[0:i],
        [qml.sample(wires=op.wires) if op.obs is None else qml.sample(op=op.obs)],
        shots=circuit.shots,
        trainable_params=circuit.trainable_params,
    )
    # circuit beyond next MidMeasureMP with VarianceMP <==> SampleMP
    new_measurements = []
    for m in circuit.measurements:
        if not m.mv:
            if isinstance(m, VarianceMP):
                new_measurements.append(SampleMP(obs=m.obs))
            else:
                new_measurements.append(m)
    circuit_next = qml.tape.QuantumScript(
        circuit.operations[i + 1 :],
        new_measurements,
        shots=circuit.shots,
        trainable_params=circuit.trainable_params,
    )
    return circuit_base, circuit_next, op


def measurement_with_no_shots(measurement):
    """Returns a NaN scalar or array of the correct size when executing an all-invalid-shot circuit."""
    return (
        np.nan * np.ones_like(measurement.eigvals())
        if isinstance(measurement, ProbabilityMP)
        else np.nan
    )


def combine_measurements(circuit, measurements, mcm_samples):
    """Returns combined measurement values of various types."""
    empty_mcm_samples = len(next(iter(mcm_samples.values()))) == 0
    if empty_mcm_samples and any(len(m) != 0 for m in mcm_samples.values()):
        raise ValueError("mcm_samples have inconsistent shapes.")
    # loop over measurements
    final_measurements = []
    for circ_meas in circuit.measurements:
        if circ_meas.mv and empty_mcm_samples:
            comb_meas = measurement_with_no_shots(circ_meas)
        elif circ_meas.mv:
            mcm_samples = dict((k, v.reshape((-1, 1))) for k, v in mcm_samples.items())
            is_valid = qml.math.ones(list(mcm_samples.values())[0].shape[0], dtype=bool)
            comb_meas = dyn_gather_mcm(circ_meas, mcm_samples, is_valid)
        elif not measurements or not measurements[0]:
            if len(measurements) > 0:
                _ = measurements.pop(0)
            comb_meas = measurement_with_no_shots(circ_meas)
        else:
            comb_meas = combine_measurements_core(circ_meas, measurements.pop(0))
        if isinstance(circ_meas, SampleMP):
            comb_meas = qml.math.squeeze(comb_meas)
        final_measurements.append(comb_meas)
    # special treatment of var
    for i, (c, m) in enumerate(zip(circuit.measurements, final_measurements)):
        if not c.mv and isinstance(circuit.measurements[i], VarianceMP):
            final_measurements[i] = qml.math.var(m)
    return final_measurements[0] if len(final_measurements) == 1 else tuple(final_measurements)


@singledispatch
def combine_measurements_core(original_measurement, measures):  # pylint: disable=unused-argument
    """Returns the combined measurement value of a given type."""
    raise TypeError(
        f"Native mid-circuit measurement mode does not support {type(original_measurement).__name__}"
    )


@combine_measurements_core.register
def _(original_measurement: CountsMP, measures):  # pylint: disable=unused-argument
    keys = list(measures.keys())
    new_counts = Counter()
    for k in keys:
        if not measures[k][0]:
            continue
        new_counts.update(measures[k][1])
    return dict(sorted(new_counts.items()))


@combine_measurements_core.register
def _(original_measurement: ExpectationMP, measures):  # pylint: disable=unused-argument
    cum_value = 0
    total_counts = 0
    for v in measures.values():
        if not v[0]:
            continue
        cum_value += v[0] * v[1]
        total_counts += v[0]
    return cum_value / total_counts


@combine_measurements_core.register
def _(original_measurement: ProbabilityMP, measures):  # pylint: disable=unused-argument
    cum_value = 0
    total_counts = 0
    for v in measures.values():
        if not v[0]:
            continue
        cum_value += v[0] * v[1]
        total_counts += v[0]
    return cum_value / total_counts


@combine_measurements_core.register
def _(original_measurement: SampleMP, measures):  # pylint: disable=unused-argument
    new_sample = tuple(qml.math.atleast_1d(m[1]) for m in measures.values() if m[0])
    return np.squeeze(np.concatenate(new_sample))


@combine_measurements_core.register
def _(original_measurement: VarianceMP, measures):  # pylint: disable=unused-argument
    new_sample = tuple(qml.math.atleast_1d(m[1]) for m in measures.values() if m[0])
    return np.squeeze(np.concatenate(new_sample))


@debug_logger
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
        postselect_mode (str): Configuration for handling shots with mid-circuit measurement
            postselection. Use ``"hw-like"`` to discard invalid shots and ``"fill-shots"`` to
            keep the same number of shots. Default is ``None``.

    Returns:
        tuple(TensorLike): The results of the simulation
        dict: The mid-circuit measurement results of the simulation
    """
    prng_key = execution_kwargs.pop("prng_key", None)

    ops_key, meas_key = jax_random_split(prng_key)
    mid_measurements = {}
    state, is_state_batched = get_final_state(
        circuit,
        debugger=debugger,
        mid_measurements=mid_measurements,
        prng_key=ops_key,
        **execution_kwargs,
    )
    return measure_final_state(
        circuit,
        state,
        is_state_batched,
        prng_key=meas_key,
        mid_measurements=mid_measurements,
        **execution_kwargs,
    )


def has_mid_circuit_measurements(
    circuit: qml.tape.QuantumScript,
):
    """Returns True if the circuit contains a MidMeasureMP object and False otherwise.

    Args:
        circuit (QuantumTape): A QuantumScript

    Returns:
        bool: Whether the circuit contains a MidMeasureMP object
    """
    return any(isinstance(op, MidMeasureMP) for op in circuit.operations)
