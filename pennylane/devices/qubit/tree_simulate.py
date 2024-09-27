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
"""Simulate a quantum script with channels."""

# pylint: disable=protected-access
from collections import Counter
from functools import lru_cache, partial, singledispatch
from typing import Optional

import numpy as np

import pennylane as qml
from pennylane.logging import debug_logger
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    MidMeasureMP,
    ProbabilityMP,
    SampleMP,
    VarianceMP,
    find_post_processed_mcms,
)
from pennylane.operation import Channel
from pennylane.tape import QuantumScript
from pennylane.typing import Result

from .apply_operation import apply_operation
from .initialize_state import create_initial_state
from .sampling import jax_random_split, measure_with_samples
from .simulate import get_final_state, measure_final_state

NORM_TOL = 1e-10


class TreeTraversalStack:
    """This class is used to record various data used during the
    depth-first tree-traversal procedure for simulating circuits with channels."""

    probs: list
    results: list
    n_branches: list
    states: list

    def __init__(self, max_depth, n_branches, prob_threshold):
        self.n_branches = n_branches
        self.probs = [None] * max_depth
        self.results = [[None]] + [[None] * self.n_branches[d] for d in range(1, max_depth)]
        self.states = [None] * max_depth
        # current_branch[:d+1] is the active branch at depth `d`
        # The first entry is always 0 as the first edge does not stem from a channel.
        # For example, if `d = 2` and `current_branch = [0, 1, 1, 0]` we are on the 11-branch,
        # i.e. we're exploring the first two Channels at index 1 of their respective Kraus matrices.
        # The last entry isn't meaningful until we are at depth `d=3`.
        self.current_branch = np.zeros(max_depth, dtype=int)
        self.prob_threshold = prob_threshold

    def is_full(self, depth):
        """Return True if the results at ``depth`` are both not ``None`` and False otherwise."""
        return all(r is not None for r in self.results[depth])

    def prune(self, depth):
        """Reset all stack entries at ``depth`` to ``None``."""
        self.probs[depth] = None
        self.results[depth] = [None] * self.n_branches[depth]
        self.states[depth] = None
        self.current_branch[depth:] = 0  # Reset current branch


    def set_prob(self, prob, depth):
        self.probs[depth][self.current_branch[depth]] = prob

    def save_and_move(self, result, depth):
        self.results[depth][self.current_branch[depth]] = result
        self.current_branch[depth] = (self.current_branch[depth] + 1) % self.n_branches[depth]

    def init_probs(self, depth):
        if self.probs[depth] is None:
            # If probs list has not been initialized at the current depth, initialize a list
            # for storing the probabilities of each of the different possible branches at the
            # current depth
            self.probs[depth] = [None] * self.n_branches[depth]

    def threshold_test(self, depth, new_prob):
        if self.prob_threshold is None:
            return False
        subtree_prob = new_prob * np.prod([self.probs[d][self.current_branch[d]] for d in range(1, depth)])
        return subtree_prob < self.prob_threshold

    def branch_state(self, op, depth):
        """Collapse the state on the current branch.

        Args:
            state (TensorLike): The initial state
            op (Channel): Channel being applied to the state
            index (int): The index of the list of kraus matrices

        Returns:
            tuple[TensorLike, float]: The collapsed state and the probability
        """
        state = self.states[depth]
        matrix = _get_kraus_matrices(op)[self.current_branch[depth]]
        state = apply_operation(qml.QubitUnitary(matrix, wires=op.wires), state)

        norm = qml.math.norm(state)
        if norm < NORM_TOL:
            return state, 0.0

        state /= norm
        if isinstance(op, MidMeasureMP) and op.postselect is not None:
            norm = 1.0
        return state, norm**2



def tree_simulate(
    circuit: QuantumScript,
    prob_threshold: float or None,
    **execution_kwargs,
) -> Result:
    """Simulate a single quantum script using the tree-traversal algorithm.

    The tree-traversal algorithm recursively explores all combinations of Kraus matrices
    outcomes using a depth-first approach. The depth-first approach requires ``n_nodes`` copies
    of the state vector (``n_nodes + 1`` state vectors in total) and records ``n_nodes`` vectors
    of measurements after applying the Kraus matrix for a given branch.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        prob_threshold (Union[None, float]): A probability threshold below which subtrees are truncated.
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
    #######################
    # main implementation #
    #######################

    [circuit], variance_post_processing = variance_transform(circuit)

    ##################
    # Parse node info #
    ##################

    # nodes is the list of all channel operations. nodes[d] is the parent
    # node of a circuit segment (edge) at depth `d`. The first element
    # is None because there is no parent node at depth 0
    nodes: list[Channel] = [None] + [op for op in circuit.operations if isinstance(op, Channel)]
    mcm_nodes: list[tuple[int, MidMeasureMP]] = [
        (i, node) for i, node in enumerate(nodes) if isinstance(node, MidMeasureMP)
    ]
    mcm_value_modifiers: dict = {
        i: (int(ps) if (ps := node.postselect) is not None else 0) for i, node in mcm_nodes
    }
    n_nodes: int = len(nodes) - 1
    n_kraus: list[int] = [None] + [c.num_kraus for c in nodes[1:]]

    #############################
    # Initialize tree-traversal #
    #############################
    # Split circuit into segments
    circuits = split_circuit_at_nodes(circuit)
    circuits[0] = prepend_state_prep(circuits[0], None, circuit.wires)
    terminal_measurements = circuits[-1].measurements
    # Initialize stacks
    cumcounts = [0] * (n_nodes + 1)
    stack = TreeTraversalStack(n_nodes + 1, n_kraus, prob_threshold)
    # The goal is to obtain the measurements of the branches
    # and to combine them into the final result. Exit the loop once the
    # measurements for all branches are available.
    depth = 0

    def cast_to_mid_measurements(branch_current):
        """Take the information about the current tree branch and encode
        it in a mid_measurements dictionary to be used in Conditional ops."""
        return {node: branch_current[i] + mcm_value_modifiers[i] for i, node in mcm_nodes}

    while not stack.is_full(1):

        ###########################################
        # Combine measurements & step up the tree #
        ###########################################

        # Combine two leaves once measurements are available
        if stack.is_full(depth):
            # Call `combine_measurements` to count-average measurements
            measurements = combine_measurements(terminal_measurements, stack, depth)

            stack.prune(depth)  # Clear stacks

            # Go up one level to explore alternate subtree of the same depth
            depth -= 1
            stack.save_and_move(measurements, depth)
            continue

        ###########################################
        # Obtain measurements for the active edge #
        ###########################################

        # Simulate the current depth circuit segment
        if depth == 0:
            initial_state = stack.states[0]
        else:
            initial_state, p = stack.branch_state(nodes[depth], depth)
            if p == 0.0 or stack.threshold_test(depth, p):
                # Do not update probs. None-valued probs are filtered out in `combine_measurements`
                # Set results to a tuple of `None`s with the correct length, they will be filtered
                # out as well
                # The entire subtree has vanishing probability, move to next subtree immediately
                stack.save_and_move((None,) * len(terminal_measurements), depth)
                continue
            stack.set_prob(p, depth)

        circtmp = QuantumScript(circuits[depth].operations, circuits[depth].measurements)
        circtmp = prepend_state_prep(circtmp, initial_state, circuit.wires)
        state, is_state_batched = get_final_state(
            circtmp, mid_measurements=cast_to_mid_measurements(stack.current_branch), **execution_kwargs
        )

        ################################################
        # Update terminal measurements & step sideways #
        ################################################

        if depth == n_nodes:
            # Update measurements and switch to the next branch
            measurements = measure_final_state(circtmp, state, is_state_batched, **execution_kwargs)
            if len(terminal_measurements) == 1:
                measurements = (measurements,)
            stack.save_and_move(measurements, depth)
            continue

        #####################################
        # Update stack & step down the tree #
        #####################################

        # If not at a leaf, project on the zero-branch and increase depth by one
        depth += 1

        stack.init_probs(depth)
        # Store a copy of the state-vector to project on the next branch
        stack.states[depth] = state

    ##################################################
    # Finalize terminal measurements post-processing #
    ##################################################

    results = combine_measurements(terminal_measurements, stack, 1)
    if len(terminal_measurements) == 1:
        results = results[0]
    return variance_post_processing((results,))


def split_circuit_at_nodes(circuit):
    """Return a list of circuits segments (one for each channel in the
    original circuit) where the terminal measurements probe the state. Only
    the last segment retains the original terminal measurements.

    Args:
        circuit (QuantumTape): The circuit to simulate

    Returns:
        Sequence[QuantumTape]: Circuit segments.
    """

    split_gen = ((i, op) for i, op in enumerate(circuit) if isinstance(op, Channel))
    circuits = []

    first = 0
    for last, _ in split_gen:
        new_operations = circuit.operations[first:last]
        new_measurements = []
        circuits.append(
            QuantumScript(new_operations, new_measurements, shots=circuit.shots)
        )
        first = last + 1

    last_circuit_operations = circuit.operations[first:]
    last_circuit_measurements = circuit.measurements

    circuits.append(
        QuantumScript(
            last_circuit_operations, last_circuit_measurements, shots=circuit.shots
        )
    )
    return circuits


def prepend_state_prep(circuit, state, wires):
    """Prepend a ``StatePrep`` operation with the prescribed ``wires`` to the circuit.

    ``get_final_state`` executes a circuit on a subset of wires found in operations
    or measurements. This function makes sure that an initial state with the correct size is created
    on the first invocation of ``simulate_tree_mcm``. ``wires`` should be the wires attribute
    of the original circuit (which included all wires)."""
    has_prep = len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase)
    if has_prep:
        if len(circuit[0].wires) == len(wires):
            return circuit
        # If a state preparation op is present but does not act on all wires,
        # the state needs to be extended to act on all wires, and a new prep op is placed.
        state = circuit[0].state_vector(wire_order=wires)
    state = create_initial_state(wires, None) if state is None else state
    return QuantumScript(
        [qml.StatePrep(qml.math.ravel(state), wires=wires, validate_norm=False)]
        + circuit.operations[int(has_prep) :],
        circuit.measurements,
        shots=circuit.shots,
    )


@lru_cache
def _get_kraus_matrices(op):
    return op.kraus_matrices()


def combine_measurements(terminal_measurements, stack, depth):
    """Returns combined measurement values of various types."""
    final_measurements = []
    all_results = stack.results[depth]
    all_probs = [p for p in stack.probs[depth] if p is not None]
    if len(all_probs) == 0:
        stack.set_prob(None, depth-1)
        return (None,) * len(terminal_measurements)

    for i, mp in enumerate(terminal_measurements):
        all_mp_results = [res[i] for res in all_results if res[i] is not None]
        comb_meas = combine_measurements_core(mp, all_probs, all_mp_results)
        final_measurements.append(comb_meas)

    return tuple(final_measurements)


@singledispatch
def combine_measurements_core(
    original_measurement, measures, node_is_mcm
):  # pylint: disable=unused-argument
    """Returns the combined measurement value of a given type."""
    raise TypeError(f"tree_simulate does not support {type(original_measurement).__name__}")


@combine_measurements_core.register
def _(original_measurement: ExpectationMP, probs, results):
    """The expectation value of two branches is a weighted sum of expectation values."""
    return qml.math.dot(probs, results)


@combine_measurements_core.register
def _(original_measurement: ProbabilityMP, probs, results):
    """The combined probability of two branches is a weighted sum of the probabilities.
    Note the implementation is the same as for ``ExpectationMP``."""
    return qml.math.dot(probs, qml.math.stack(results))
