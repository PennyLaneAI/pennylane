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
from typing import Optional, Union

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
    r"""This class is used to record various data used during the
    depth-first tree-traversal procedure for simulating circuits with channels.


    Args:
        max_depth (int): The maximal depth of the tree, matching the length of
            all branches, or the number of nodes (counting the root as node).
        n_branches (list[int]): A list with the number of edges coming out of each
            node. The :math:`k`\ th entry sets the number of edges coming out of all
            :math:`k`\ -level nodes (counting the tree root as the unique :math:`0`\ -level node).
        prob_threshold (Union[None, float]): Threshold for probabilities below which subtrees
            are discarded in the simulation.
        shots (Union[None, int]): The total number of shots for the full circuit.

    Keyword Args:
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used. Only used for shot-based
            simulation.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
            This functionality is currently not supported yet.


    The main simulation algorithm in ``tree_simulate`` uses a while loop that traverses this tree
    data structure, with the stack memorizing all non-trivial data, including

    - The overall tree structure, consisting of a number of nodes (``max_depth-1``)
      and the number of branches coming out of each node (``n_branches``),

    - The currently active branch, encoded as an array of integers (``current_branch``),
      with the :math:`k`\ th integer determining the active edge coming out of the
      nodes at :math:`k`\ th level (where we count the root of the tree as the :math:`0`\ th level),

    - The probablities for the current and all previously explored edges, at all nodes
      that are part of the currently active branch,

    - The results at all nodes that are the root of maximal subtrees that have fully been explored.
      That is, the first stored result is one of the very first branch that is explored.
      Once all alternatives at maximal depth are explored, the result at the depth reduced by 1
      is set, and the maximal-depth result is deleted,

    - The quantum states that the quantum circuit produces at each node,

    - If run with finite shots, the allocated shots for the current and all previously explored
      edges, at all nodes that are part of the currently active branch (c.f. probabilities).

    In addition, the stack memorizes whether finite shots are being used at all, a random number
    generator in case that yes, as well as a probability threshold below which subtrees are
    discarded.

    **Encoding of the current branch**

    The variable ``current_branch`` caches the currently active branch/subtree of the tree.
    ``current_branch[:depth+1]`` is the active branch at depth ``depth``.
    The first entry is always 0 as the first edge stems from the root of the tree.
    For example, if ``d = 2`` and ``current_branch = [0, 2, 1, 0]`` we are on the ``21``-branch,
    i.e. we're exploring the first channel at index 2 (third Kraus matrix) and the second channel
    at index 1 (second Kraus matrix). The last entry of ``current_branch`` isn't meaningful
    until we are at ``depth=3``.

    **Movements across the tree**

    Whenever we need to move between branches at a fixed depth ("sideways"), this is done by the
    ``save_and_move`` method of the stack, by updating the ``current_branch``. Whenever
    we need to move between levels of nodes ("vertically"), this is done by updating the
    pointer ``depth`` in the ``while`` loop in ``tree_simulate``, not by the stack!
    """

    n_branches: int
    probs: list
    results: list
    states: list
    finite_shots: bool
    allocated_shots: list
    current_total: np.ndarray
    prob_threshold: Union[None, float]

    def __init__(self, max_depth, n_branches, prob_threshold, shots, **rand_kwargs):
        self.n_branches = n_branches
        self.probs = [None] * max_depth
        self.results = [[None]] + [[None] * self.n_branches[d] for d in range(1, max_depth)]
        self.states = [None] * max_depth
        self.finite_shots = shots.total_shots is not None
        if self.finite_shots:
            self.allocated_shots = [[shots.total_shots]] + [None] * (max_depth - 1)
            if rand_kwargs.get("prng_key", None) is not None:
                raise NotImplementedError("JAX randomness is not implemented yet.")
            self.rng = np.random.default_rng(rand_kwargs.get("rng", None))

        self.current_branch = np.zeros(max_depth, dtype=int)
        self.prob_threshold = prob_threshold

    def is_full(self, depth):
        """Return True if all results at ``depth`` are not ``None``, and False otherwise.

        Args:
            depth (int): The depth at which to query the stored results.
        """
        return all(r is not None for r in self.results[depth])

    def prune(self, depth):
        """Reset all stack entries at ``depth`` to ``None``-like values.

        Args:
            depth (int): The depth at which to reset.
        """
        self.probs[depth] = None
        self.results[depth] = [None] * self.n_branches[depth]
        self.states[depth] = None
        if self.finite_shots:
            self.allocated_shots[depth] = None

    def set_prob(self, prob, depth):
        """Store a probability at the specified depth and for the current branch.

        Args:
            prob (float): The probability to be stored.
            depth (int): The depth at which to store the probability.
        """
        self.probs[depth][self.current_branch[depth]] = prob

    def save_and_move(self, result, depth):
        """Store a result at the specified depth and for the current branch.
        Afterwards update the current branch to move to the next Kraus matrix at the specified
        depth.

        Args:
            result (Union[float, np.ndarray]): The result to be stored.
            depth (int): The depth at which to store the result and update the current branch.

        **Note:** If the current branch already was at the maximal index at the specified depth,
        it is reset to 0. The move to the next node at the parent level of the tree is performed
        within the while loop of ``tree_simulate``, not within this function.
        """
        self.results[depth][self.current_branch[depth]] = result
        self.current_branch[depth] = (self.current_branch[depth] + 1) % self.n_branches[depth]

    def init_probs(self, depth):
        """Initialize the list of probabilities at a specified depth if it has not been
        initialized before (or was reset by ``prune``). In this case all entries are set
        to ``None``.
        """
        if self.probs[depth] is None:
            # If probs list has not been initialized at the current depth, initialize a list
            # for storing the probabilities of each of the different possible branches at the
            # current depth
            self.probs[depth] = [None] * self.n_branches[depth]

    def threshold_test(self, depth, new_prob):
        """Test whether a given probability pushes the overall probability (product) for the
        current branch/subtree below the probability threshold.

        Args:
            depth (int): The depth at which the new probability occurred.
            new_prob (float): The new probability
        Returns:
            bool: Whether the product of all parent edge probabilities, up to the given depth,
            and the new probability lie below the probability threshold. If the stack has
            no threshold specified, returns ``False``.

        If the threshold is not specified, the result is trivially ``False``.
        Otherwise, the given probability is multiplied with all probabilities of the
        current branch up to the specified depth, and compared to the threshold.
        """
        if self.prob_threshold is None:
            return False
        prev_prob = np.prod([self.probs[d][self.current_branch[d]] for d in range(1, depth)])
        return new_prob * prev_prob < self.prob_threshold

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

    def discretize_prob(self, prob, depth):
        """Sample from a binomial distribution to obtain a discretized probability,
        and store the obtained sample as allocated shots for the current branch.

        Args:
            prob (float): Probability for the binomial distribution to sample from.
            depth (int): The depth at which the sampling happens, required to know how
                many shots to allocate from the parent node, and to store the sampling
                result in the ``allocated_shots`` of the stack.

        Returns:
            float: The discretized probability given by the sampled shots divided by the
            shot budget currently available at the parent node.
        """
        if not self.finite_shots:
            return prob

        # Initialize allocated shots if not happened before
        if self.allocated_shots[depth] is None:
            self.allocated_shots[depth] = [None] * self.n_branches[depth]

        # Obtain the shots allocated to the parent node. This is the current budget.
        current_total = self.allocated_shots[depth - 1][self.current_branch[depth - 1]]

        if self.current_branch[depth] == self.n_branches[depth] - 1:
            # If we're on the last branch we can only allocate exactly the remaining shots
            counts = current_total - sum(self.allocated_shots[depth][:-1])
        else:
            # If we're not on the last branch, sample a number of shots based on prob
            counts = self.sample(prob, current_total)

        # Store the sampled counts in `allocated_shots`
        self.allocated_shots[depth][self.current_branch[depth]] = counts
        return counts / current_total

    def sample(self, prob, n):
        """Sample from a binomial distribution.

        Args:
            prob (float): Probability, parameter of binomial distribution.
            n (int): Total number of trials, parameter of binomial distribution.

        Returns:
            int: Number of "yes" samples from binomial distribution.
        """
        prob = np.clip(prob, 0.0, 1.0)
        return self.rng.binomial(n=n, p=prob)


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
    postselect_mode = execution_kwargs.get("postselect_mode", None)
    if postselect_mode is not None:
        raise NotImplementedError(f"Postselect mode {postselect_mode} not implemented")

    ##########################
    # shot vector processing #
    ##########################
    if circuit.shots.has_partitioned_shots:
        prng_key = execution_kwargs.pop("prng_key", None)
        keys = jax_random_split(prng_key, num=circuit.shots.num_copies)
        results = []
        for k, s in zip(keys, circuit.shots):
            aux_circuit = QuantumScript(circuit.operations, circuit.measurements, shots=s)
            results.append(tree_simulate(
                aux_circuit,
                prob_threshold=prob_threshold,
                prng_key=k,
                **execution_kwargs
            ))
        return tuple(results)

    #######################
    # main implementation #
    #######################

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
    circuits, terminal_measurements = split_circuit_at_nodes(circuit)
    circuits[0] = prepend_state_prep(circuits[0], None, circuit.wires)
    # Initialize stacks
    cumcounts = [0] * (n_nodes + 1)
    stack = TreeTraversalStack(n_nodes + 1, n_kraus, prob_threshold, circuit.shots)
    # The goal is to obtain the measurements of the branches
    # and to combine them into the final result. Exit the loop once the
    # measurements for all branches are available.

    def cast_to_mid_measurements(current_branch):
        """Take the information about the current tree branch and encode
        it in a mid_measurements dictionary to be used in Conditional ops."""
        return {node: current_branch[i] + mcm_value_modifiers[i] for i, node in mcm_nodes}

    depth = 0

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
            initial_state = None
        else:
            initial_state, p = stack.branch_state(nodes[depth], depth)
            p = stack.discretize_prob(p, depth)
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
            circtmp,
            mid_measurements=cast_to_mid_measurements(stack.current_branch),
            **execution_kwargs,
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
        return results[0]
    return results


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
        circuits.append(QuantumScript(new_operations, new_measurements, shots=circuit.shots))
        first = last + 1

    last_circuit_operations = circuit.operations[first:]
    last_circuit_measurements = circuit.measurements

    circuits.append(
        QuantumScript(last_circuit_operations, last_circuit_measurements, shots=circuit.shots)
    )
    return circuits, last_circuit_measurements


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
        stack.set_prob(None, depth - 1)
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
def _(original_measurement: ExpectationMP, probs, results):  # pylint: disable=unused-argument
    """The expectation value of two branches is a weighted sum of expectation values."""
    return qml.math.dot(probs, results)


@combine_measurements_core.register
def _(original_measurement: ProbabilityMP, probs, results):  # pylint: disable=unused-argument
    """The combined probability of two branches is a weighted sum of the probabilities.
    Note the implementation is the same as for ``ExpectationMP``."""
    return qml.math.dot(probs, qml.math.stack(results))
