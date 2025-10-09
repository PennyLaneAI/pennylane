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

# pylint: disable=protected-access
from collections import Counter
from functools import partial, singledispatch

import numpy as np
from numpy.random import default_rng

import pennylane as qml
from pennylane import math
from pennylane.logging import debug_logger
from pennylane.math.interface_utils import Interface
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    MidMeasureMP,
    ProbabilityMP,
    SampleMP,
    ShotCopies,
    Shots,
    VarianceMP,
    find_post_processed_mcms,
)
from pennylane.operation import StatePrepBase
from pennylane.tape import QuantumScript
from pennylane.transforms.dynamic_one_shot import gather_mcm
from pennylane.typing import Result

from .apply_operation import apply_operation
from .initialize_state import create_initial_state
from .measure import measure
from .sampling import jax_random_split, measure_with_samples

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TreeTraversalStack:
    """This class is used to record various data used during the
    depth-first tree-traversal procedure for simulating dynamic circuits."""

    counts: list
    probs: list
    results_0: list
    results_1: list
    states: list

    def __init__(self, max_depth):
        self.counts = [None] * max_depth
        self.probs = [None] * max_depth
        self.results_0 = [None] * max_depth
        self.results_1 = [None] * max_depth
        self.states = [None] * max_depth

    def any_is_empty(self, depth):
        """Return True if any result at ``depth`` is ``None`` and False otherwise."""
        return self.results_0[depth] is None or self.results_1[depth] is None

    def is_full(self, depth):
        """Return True if the results at ``depth`` are both not ``None`` and False otherwise."""
        return self.results_0[depth] is not None and self.results_1[depth] is not None

    def prune(self, depth):
        """Reset all stack entries at ``depth`` to ``None``."""
        self.counts[depth] = None
        self.probs[depth] = None
        self.results_0[depth] = None
        self.results_1[depth] = None
        self.states[depth] = None


class _FlexShots(Shots):
    """Shots class that allows zero shots."""

    # pylint: disable=super-init-not-called
    def __init__(self, shots=None):
        if isinstance(shots, int):
            self.total_shots = shots
            self.shot_vector = (ShotCopies(shots, 1),)
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
    norm = math.norm(state)

    if not math.is_abstract(state) and math.allclose(norm, 0.0):
        if postselect_mode == "fill-shots" and shots:
            raise RuntimeError(
                "The probability of the postselected mid-circuit measurement outcome is 0. "
                "This leads to invalid results when using postselect_mode='fill-shots'."
            )
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
            if postselect_mode == "fill-shots" or math.is_abstract(norm)
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

    prep = None
    if len(circuit) > 0 and isinstance(circuit[0], StatePrepBase):
        prep = circuit[0]

    state = create_initial_state(
        sorted(circuit.op_wires), prep, like=Interface(interface).get_like()
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
            tape_shots=circuit.shots,
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

    for _ in range(circuit.num_wires - len(circuit.op_wires)):
        # if any measured wires are not operated on, we pad the state with zeros.
        # We know they belong at the end because the circuit is in standard wire-order
        state = math.stack([state, math.zeros_like(state)], axis=-1)

    return state, is_state_batched


@debug_logger
def measure_final_state(circuit, state, is_state_batched, **execution_kwargs) -> Result:
    """
    Perform the measurements required by the circuit on the provided state.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate. This circuit is assumed to have
            non-negative integer wire labels
        state (TensorLike): The state to perform measurement on
        is_state_batched (bool): Whether the state has a batch dimension or not.
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
    circuit: QuantumScript,
    debugger=None,
    state_cache: dict | None = None,
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
        mcm_method (str): Strategy to use when executing circuits with mid-circuit measurements.
            ``"deferred"`` is ignored. If mid-circuit measurements are found in the circuit,
            the device will use ``"tree-traversal"`` if specified and the ``"one-shot"`` method
            otherwise. For usage details, please refer to the
            :doc:`dynamic quantum circuits page </introduction/dynamic_quantum_circuits>`.

    Returns:
        tuple(TensorLike): The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.

    This function assumes that all operations provide matrices.

    >>> qs = qml.tape.QuantumScript([qml.RX(1.2, wires=0)], [qml.expval(qml.Z(0)), qml.probs(wires=(0,1))])
    >>> simulate(qs)
    (0.36235775447667357,
    tensor([0.68117888, 0.        , 0.31882112, 0.        ], requires_grad=True))

    """
    circuit = circuit.copy()
    prng_key = execution_kwargs.pop("prng_key", None)
    circuit = circuit.map_to_standard_wires()

    has_mcm = any(isinstance(op, MidMeasureMP) for op in circuit.operations)
    if has_mcm:
        if execution_kwargs.get("mcm_method", None) == "tree-traversal":
            return simulate_tree_mcm(
                circuit, prng_key=prng_key, debugger=debugger, **execution_kwargs
            )

        results = []
        aux_circ = circuit.copy(shots=[1])
        keys = jax_random_split(prng_key, num=circuit.shots.total_shots)
        if math.get_deep_interface(circuit.data) == "jax" and prng_key is not None:
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


# pylint: disable=too-many-branches,too-many-statements
def simulate_tree_mcm(
    circuit: QuantumScript,
    debugger=None,
    **execution_kwargs,
) -> Result:
    """Simulate a single quantum script with native mid-circuit measurements using the tree-traversal algorithm.

    The tree-traversal algorithm recursively explores all combinations of mid-circuit measurement
    outcomes using a depth-first approach. The depth-first approach requires ``n_mcm`` copies
    of the state vector (``n_mcm + 1`` state vectors in total) and records ``n_mcm`` vectors
    of mid-circuit measurement samples. It is generally more efficient than ``one-shot`` because it takes all samples
    at a leaf at once and stops exploring more branches when a single shot is allocated to a sub-tree.

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
    """
    PROBS_TOL = 0.0
    interface = execution_kwargs.get("interface", None)

    ##########################
    # shot vector processing #
    ##########################
    if circuit.shots.has_partitioned_shots:
        prng_key = execution_kwargs.pop("prng_key", None)
        keys = jax_random_split(prng_key, num=circuit.shots.num_copies)
        results = []
        for k, s in zip(keys, circuit.shots):
            aux_circuit = circuit.copy(shots=s)
            results.append(simulate_tree_mcm(aux_circuit, debugger, prng_key=k, **execution_kwargs))
        return tuple(results)

    #######################
    # main implementation #
    #######################

    # `var` measurements cannot be aggregated on the fly as they require the global `expval`
    # variance_transform replaces `var` measurements with `expval` and `expval**2` measurements
    [circuit], variance_post_processing = variance_transform(circuit)
    finite_shots = bool(circuit.shots)

    ##################
    # Parse MCM info #
    ##################

    # mcms is the list of all mid-circuit measurement operations
    # mcms[d] is the parent MCM (node) of a circuit segment (edge) at depth `d`
    # The first element is None because there is no parent MCM at depth 0
    mcms = tuple([None] + [op for op in circuit.operations if isinstance(op, MidMeasureMP)])
    n_mcms = len(mcms) - 1
    # We obtain `measured_mcms_indices`, the list of MCMs which require post-processing:
    # either as requested by terminal measurements or post-selection
    measured_mcms = find_post_processed_mcms(circuit)
    measured_mcms_indices = [i for i, mcm in enumerate(mcms[1:]) if mcm in measured_mcms]
    # `mcm_samples` is a register of MCMs. It is necessary to correctly keep track of
    # correlated MCM values which may be requested by terminal measurements.
    mcm_samples = {
        k + 1: math.empty((circuit.shots.total_shots,), dtype=int) if finite_shots else None
        for k in measured_mcms_indices
    }

    #############################
    # Initialize tree-traversal #
    #############################

    # mcm_current[:d+1] is the active branch at depth `d`
    # The first entry is always 0 as the first edge does not stem from an MCM.
    # For example, if `d = 2` and `mcm_current = [0, 1, 1, 0]` we are on the 11-branch,
    # i.e. the first two MCMs had outcome 1. The last entry isn't meaningful until we are
    # at depth `d=3`.
    mcm_current = math.zeros(n_mcms + 1, dtype=int)
    # `mid_measurements` maps the elements of `mcm_current` to their respective MCMs
    # This is used by `get_final_state::apply_operation` for `Conditional` operations
    mid_measurements = dict(zip(mcms[1:], mcm_current[1:].tolist()))
    # Split circuit into segments
    circuits = split_circuit_at_mcms(circuit)
    circuits[0] = prepend_state_prep(circuits[0], None, interface, circuit.wires)
    terminal_measurements = circuits[-1].measurements if finite_shots else circuit.measurements
    # Initialize stacks
    cumcounts = [0] * (n_mcms + 1)
    stack = TreeTraversalStack(n_mcms + 1)
    # The goal is to obtain the measurements of the zero-branch and one-branch
    # and to combine them into the final result. Exit the loop once the
    # zero-branch and one-branch measurements are available.
    depth = 0

    while stack.any_is_empty(1):
        ###########################################
        # Combine measurements & step up the tree #
        ###########################################

        # Combine two leaves once measurements are available
        if stack.is_full(depth):
            # Call `combine_measurements` to count-average measurements
            measurement_dicts = get_measurement_dicts(terminal_measurements, stack, depth)
            measurements = combine_measurements(
                terminal_measurements, measurement_dicts, mcm_samples
            )
            mcm_current[depth:] = 0  # Reset current branch
            stack.prune(depth)  # Clear stacks
            # Go up one level to explore alternate subtree of the same depth
            depth -= 1
            if mcm_current[depth] == 1:
                stack.results_1[depth] = measurements
                mcm_current[depth] = 0
            else:
                stack.results_0[depth] = measurements
                mcm_current[depth] = 1
            # Update MCM values
            mid_measurements.update(
                (k, v) for k, v in zip(mcms[depth:], mcm_current[depth:].tolist())
            )
            continue

        ################################################
        # Determine whether to execute the active edge #
        ################################################

        # Parse shots for the current branch
        if finite_shots:
            if stack.counts[depth]:
                shots = stack.counts[depth][mcm_current[depth]]
            else:
                shots = circuits[depth].shots.total_shots
            skip_subtree = not bool(shots)
        else:
            shots = None
            skip_subtree = (
                stack.probs[depth] is not None
                and float(stack.probs[depth][mcm_current[depth]]) <= PROBS_TOL
            )
        # Update active branch dict
        invalid_postselect = (
            depth > 0
            and mcms[depth].postselect is not None
            and mcm_current[depth] != mcms[depth].postselect
        )

        ###########################################
        # Obtain measurements for the active edge #
        ###########################################

        # If num_shots is zero or postselecting on the wrong branch, update measurements with an empty tuple
        if skip_subtree or invalid_postselect:
            # Adjust counts if `invalid_postselect`
            if invalid_postselect:
                if finite_shots:
                    # Bump downstream cumulative counts before zeroing-out counts
                    for d in range(depth + 1, n_mcms + 1):
                        cumcounts[d] += stack.counts[depth][mcm_current[depth]]
                    stack.counts[depth][mcm_current[depth]] = 0
                else:
                    stack.probs[depth][mcm_current[depth]] = 0
            measurements = tuple()
        else:
            # If num_shots is non-zero, simulate the current depth circuit segment
            if depth == 0:
                initial_state = stack.states[0]
            else:
                initial_state = branch_state(stack.states[depth], mcm_current[depth], mcms[depth])
            circtmp = circuits[depth].copy(shots=Shots(shots))

            circtmp = prepend_state_prep(circtmp, initial_state, interface, sorted(circuit.wires))
            state, is_state_batched = get_final_state(
                circtmp,
                debugger=debugger,
                mid_measurements=mid_measurements,
                **execution_kwargs,
            )
            measurements = measure_final_state(circtmp, state, is_state_batched, **execution_kwargs)

        #####################################
        # Update stack & step down the tree #
        #####################################

        # If not at a leaf, project on the zero-branch and increase depth by one
        if depth < n_mcms and (not skip_subtree and not invalid_postselect):
            depth += 1
            # Update the active branch samples with `update_mcm_samples`
            if finite_shots:
                samples = math.atleast_1d(measurements)
                stack.counts[depth] = samples_to_counts(samples)
                stack.probs[depth] = counts_to_probs(stack.counts[depth])
            else:
                stack.probs[depth] = dict(zip([False, True], measurements))
                samples = None
            # Store a copy of the state-vector to project on the one-branch
            stack.states[depth] = state
            mcm_samples, cumcounts = update_mcm_samples(samples, mcm_samples, depth, cumcounts)
            continue

        ################################################
        # Update terminal measurements & step sideways #
        ################################################

        if not skip_subtree and not invalid_postselect:
            measurements = insert_mcms(circuit, measurements, mid_measurements)

        # If at a zero-branch leaf, update measurements and switch to the one-branch
        if mcm_current[depth] == 0:
            stack.results_0[depth] = measurements
            mcm_current[depth] = True
            mid_measurements[mcms[depth]] = True
            continue
        # If at a one-branch leaf, update measurements
        stack.results_1[depth] = measurements

    ##################################################
    # Finalize terminal measurements post-processing #
    ##################################################

    _finalize_debugger(debugger)
    measurement_dicts = get_measurement_dicts(terminal_measurements, stack, depth)
    if finite_shots:
        terminal_measurements = circuit.measurements
    mcm_samples = {mcms[i]: v for i, v in mcm_samples.items()}
    mcm_samples = prune_mcm_samples(mcm_samples)
    results = combine_measurements(terminal_measurements, measurement_dicts, mcm_samples)
    return variance_post_processing((results,))


def _finalize_debugger(debugger):
    """Ensures all snapshot results are wrapped in a list for consistency."""
    if not debugger or not debugger.active:
        return
    for tag, results in debugger.snapshots.items():
        if not isinstance(results, list):
            debugger.snapshots[tag] = [results]


def split_circuit_at_mcms(circuit):
    """Return a list of circuits segments (one for each mid-circuit measurement in the
    original circuit) where the terminal measurements probe the MCM statistics. Only
    the last segment retains the original terminal measurements.

    Args:
        circuit (QuantumTape): The circuit to simulate

    Returns:
        Sequence[QuantumTape]: Circuit segments.
    """

    mcm_gen = ((i, op) for i, op in enumerate(circuit) if isinstance(op, MidMeasureMP))
    circuits = []

    first = 0
    for last, op in mcm_gen:
        new_operations = circuit.operations[first:last]
        new_measurements = (
            [qml.sample(wires=op.wires)] if circuit.shots else [qml.probs(wires=op.wires)]
        )
        circuits.append(circuit.copy(operations=new_operations, measurements=new_measurements))
        first = last + 1

    last_circuit_operations = circuit.operations[first:]
    last_circuit_measurements = []

    for m in circuit.measurements:
        if m.mv is None:
            last_circuit_measurements.append(m)

    circuits.append(
        QuantumScript(last_circuit_operations, last_circuit_measurements, shots=circuit.shots)
    )
    return circuits


def prepend_state_prep(circuit, state, interface, wires):
    """Prepend a ``StatePrep`` operation with the prescribed ``wires`` to the circuit.

    ``get_final_state`` executes a circuit on a subset of wires found in operations
    or measurements. This function makes sure that an initial state with the correct size is created
    on the first invocation of ``simulate_tree_mcm``. ``wires`` should be the wires attribute
    of the original circuit (which included all wires)."""
    if len(circuit) > 0 and isinstance(circuit[0], StatePrepBase):
        return circuit

    interface = Interface(interface)
    state = create_initial_state(wires, None, like=interface.get_like()) if state is None else state
    new_ops = [
        qml.StatePrep(math.ravel(state), wires=wires, validate_norm=False)
    ] + circuit.operations
    return circuit.copy(operations=new_ops)


def insert_mcms(circuit, results, mid_measurements):
    """Inserts terminal measurements of MCMs if the circuit is evaluated in analytic mode."""
    if circuit.shots or all(m.mv is None for m in circuit.measurements):
        return results
    results = list(results)
    new_results = []
    mid_measurements = {k: math.array([[v]]) for k, v in mid_measurements.items()}
    for m in circuit.measurements:
        if m.mv is None:
            new_results.append(results.pop(0))
        else:
            new_results.append(gather_mcm(m, mid_measurements, math.array([[True]])))

    return new_results


def get_measurement_dicts(measurements, stack, depth):
    """Combine a probs dictionary and two tuples of measurements into a
    tuple of dictionaries storing the probs and measurements of both branches."""
    # We use `circuits[-1].measurements` since it contains the
    # target measurements (this is the only tape segment with
    # unmodified measurements)
    probs, results_0, results_1 = stack.probs[depth], stack.results_0[depth], stack.results_1[depth]
    measurement_dicts = [{} for _ in measurements]
    # Special treatment for single measurements
    single_measurement = len(measurements) == 1
    # Store each measurement in a dictionary `{branch: (prob, measure)}`
    for branch, prob in probs.items():
        meas = results_0 if branch == 0 else results_1
        if single_measurement:
            meas = [meas]
        for i, m in enumerate(meas):
            measurement_dicts[i][branch] = (prob, m)
    return measurement_dicts


def branch_state(state, branch, mcm):
    """Collapse the state on a given branch.

    Args:
        state (TensorLike): The initial state
        branch (int): The branch on which the state is collapsed
        mcm (MidMeasureMP): Mid-circuit measurement object used to obtain the wires and ``reset``

    Returns:
        TensorLike: The collapsed state
    """
    if isinstance(state, np.ndarray):
        # FASTER
        state = state.copy()
        slices = [slice(None)] * math.ndim(state)
        axis = mcm.wires.toarray()[0]
        slices[axis] = int(not branch)
        state[tuple(slices)] = 0.0
        state /= math.norm(state)
    else:
        # SLOWER
        state = apply_operation(qml.Projector([branch], mcm.wires), state)
        state = state / math.norm(state)

    if mcm.reset and branch == 1:
        state = apply_operation(qml.PauliX(mcm.wires), state)
    return state


def samples_to_counts(samples):
    """Converts samples to counts.

    This function forces integer keys and values which are required by ``simulate_tree_mcm``.
    """
    counts_1 = int(math.count_nonzero(samples))
    return {0: samples.size - counts_1, 1: counts_1}


def counts_to_probs(counts):
    """Converts counts to probs."""
    probs = math.array(list(counts.values()))
    probs = probs / math.sum(probs)
    return dict(zip(counts.keys(), probs))


def prune_mcm_samples(mcm_samples):
    """Removes invalid mid-measurement samples.

    Post-selection on a given mid-circuit measurement leads to ignoring certain branches
    of the tree and samples. The corresponding samples in all other mid-circuit measurement
    must be deleted accordingly. We need to find which samples are
    corresponding to the current branch by looking at all parent nodes.
    """
    if not mcm_samples or all(v is None for v in mcm_samples.values()):
        return mcm_samples
    mask = math.ones(list(mcm_samples.values())[0].shape, dtype=bool)
    for mcm, s in mcm_samples.items():
        if mcm.postselect is None:
            continue
        mask = math.logical_and(mask, s == mcm.postselect)
    return {k: v[mask] for k, v in mcm_samples.items()}


def update_mcm_samples(samples, mcm_samples, depth, cumcounts):
    """Updates the depth-th mid-measurement samples.

    To illustrate how the function works, let's take an example. Suppose there are
    ``2**20`` shots in total and the computation is midway through the circuit at the
    7th MCM, the active branch is ``[0,1,1,0,0,1]``, and at each MCM everything happened to
    split the counts 50/50, so there are ``2**14`` samples to update.
    These samples are correlated with the parent
    branches, so where do they go? They must update the ``2**14`` elements whose parent
    sequence corresponds to ``[0,1,1,0,0,1]``. ``cumcounts`` is used for this job and
    increased by the size of ``samples`` each time this function is called.
    """
    if depth not in mcm_samples or mcm_samples[depth] is None:
        return mcm_samples, cumcounts
    count1 = math.sum(samples)
    count0 = samples.size - count1
    mcm_samples[depth][cumcounts[depth] : cumcounts[depth] + count0] = 0
    cumcounts[depth] += count0
    mcm_samples[depth][cumcounts[depth] : cumcounts[depth] + count1] = 1
    cumcounts[depth] += count1
    return mcm_samples, cumcounts


@qml.transform
def variance_transform(circuit):
    """Replace variance measurements by expectation value measurements of both the observable and the observable square.

    This is necessary since computing the variance requires the global expectation value which is not available from measurements on subtrees.
    """
    skip_transform = not any(isinstance(m, VarianceMP) for m in circuit.measurements)
    if skip_transform:
        return (circuit,), lambda x: x[0]

    def variance_post_processing(results):
        """Compute the global variance from expectation value measurements of both the observable and the observable square."""
        new_results = list(results[0])
        offset = len(circuit.measurements)
        for i, (r, m) in enumerate(zip(new_results, circuit.measurements)):
            if isinstance(m, VarianceMP):
                expval = new_results.pop(offset)
                new_results[i] = r - expval**2
        return new_results[0] if len(new_results) == 1 else new_results

    new_measurements = []
    extra_measurements = []
    for m in circuit.measurements:
        if isinstance(m, VarianceMP):
            obs2 = m.mv * m.mv if m.mv is not None else m.obs @ m.obs
            new_measurements.append(ExpectationMP(obs=obs2))
            extra_measurements.append(ExpectationMP(obs=m.mv if m.mv is not None else m.obs))
        else:
            new_measurements.append(m)
    new_measurements.extend(extra_measurements)
    return (
        (
            QuantumScript(
                circuit.operations,
                new_measurements,
                shots=circuit.shots,
            ),
        ),
        variance_post_processing,
    )


def measurement_with_no_shots(measurement):
    """Returns a NaN scalar or array of the correct size when executing an all-invalid-shot circuit."""
    if isinstance(measurement, ProbabilityMP):
        return np.nan * math.ones(2 ** len(measurement.wires))
    return np.nan


def combine_measurements(terminal_measurements, results, mcm_samples):
    """Returns combined measurement values of various types."""
    empty_mcm_samples = False
    need_mcm_samples = not all(v is None for v in mcm_samples.values())
    need_mcm_samples = need_mcm_samples and any(
        circ_meas.mv is not None for circ_meas in terminal_measurements
    )
    if need_mcm_samples:
        empty_mcm_samples = len(next(iter(mcm_samples.values()))) == 0
        if empty_mcm_samples and any(len(m) != 0 for m in mcm_samples.values()):  # pragma: no cover
            raise ValueError("mcm_samples have inconsistent shapes.")
    final_measurements = []
    for circ_meas in terminal_measurements:
        if need_mcm_samples and circ_meas.mv is not None and empty_mcm_samples:
            comb_meas = measurement_with_no_shots(circ_meas)
        elif need_mcm_samples and circ_meas.mv is not None:
            mcm_samples = {k: v.reshape((-1, 1)) for k, v in mcm_samples.items()}
            is_valid = math.ones(list(mcm_samples.values())[0].shape[0], dtype=bool)
            comb_meas = gather_mcm(circ_meas, mcm_samples, is_valid)
        elif not results or not results[0]:
            if len(results) > 0:
                _ = results.pop(0)
            comb_meas = measurement_with_no_shots(circ_meas)
        else:
            comb_meas = combine_measurements_core(circ_meas, results.pop(0))
        final_measurements.append(comb_meas)
    return final_measurements[0] if len(final_measurements) == 1 else tuple(final_measurements)


@singledispatch
def combine_measurements_core(original_measurement, measures):
    """Returns the combined measurement value of a given type."""
    raise TypeError(
        f"Native mid-circuit measurement mode does not support {type(original_measurement).__name__}"
    )


@combine_measurements_core.register
def _(original_measurement: CountsMP, measures):
    """The counts are accumulated using a ``Counter`` object."""
    keys = list(measures.keys())
    new_counts = Counter()
    for k in keys:
        if not measures[k][0]:
            continue
        new_counts.update(measures[k][1])
    return dict(sorted(new_counts.items()))


@combine_measurements_core.register
def _(original_measurement: ExpectationMP, measures):
    """The expectation value of two branches is a weighted sum of expectation values."""
    cum_value = 0
    total_counts = 0
    for v in measures.values():
        if not v[0] or v[1] is tuple():
            continue
        cum_value += math.multiply(v[0], v[1])
        total_counts += v[0]
    return cum_value / total_counts


@combine_measurements_core.register
def _(original_measurement: ProbabilityMP, measures):
    """The combined probability of two branches is a weighted sum of the probabilities. Note the implementation is the same as for ``ExpectationMP``."""
    cum_value = 0
    total_counts = 0
    for v in measures.values():
        if not v[0] or v[1] is tuple():
            continue
        cum_value += math.multiply(v[0], v[1])
        total_counts += v[0]
    return cum_value / total_counts


@combine_measurements_core.register
def _(original_measurement: SampleMP, measures):
    """The combined samples of two branches is obtained by concatenating the sample of each branch."""
    new_sample = tuple(
        math.atleast_1d(m[1]) for m in measures.values() if m[0] and not m[1] is tuple()
    )
    return math.concatenate(new_sample)


@debug_logger
def simulate_one_shot_native_mcm(
    circuit: QuantumScript, debugger=None, **execution_kwargs
) -> Result:
    """Simulate a single shot of a single quantum script with native mid-circuit measurements.

    Assumes that the circuit has been transformed by `dynamic_one_shot`.

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
        Result: The results of the simulation

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
