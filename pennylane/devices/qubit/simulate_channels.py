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
    find_post_processed_mcms,
)
from pennylane.operation import Channel
from pennylane.transforms.dynamic_one_shot import gather_mcm
from pennylane.typing import Result

from .apply_operation import apply_operation
from .initialize_state import create_initial_state
from .measure import measure
from .sampling import jax_random_split, measure_with_samples
from .simulate import get_final_state, measure_final_state


class TreeTraversalStack:
    """This class is used to record various data used during the
    depth-first tree-traversal procedure for simulating circuits with channels."""

    probs: list
    results: list
    n_kraus: list
    states: list

    def __init__(self, max_depth, n_kraus):
        self.probs = [None] * max_depth
        self.results = [[None] * max_depth for _ in range(max(n_kraus[1:]))]
        self.n_kraus = n_kraus
        self.states = [None] * max_depth

    def any_is_empty(self, depth):
        """Return True if any result at ``depth`` is ``None`` and False otherwise."""
        return any(r[depth] is None for r in self.results)

    def is_full(self, depth):
        """Return True if the results at ``depth`` are both not ``None`` and False otherwise."""
        return all(r[depth] is not None for r in self.results)

    def prune(self, depth):
        """Reset all stack entries at ``depth`` to ``None``."""
        self.counts[depth] = None
        self.probs[depth] = None
        for r in self.results:
            r[depth] = None
        self.states[depth] = None


def simulate_channels(
    circuit: qml.tape.QuantumScript,
    **execution_kwargs,
) -> Result:
    """Simulate a single quantum script with channels using the tree-traversal algorithm.

    The tree-traversal algorithm recursively explores all combinations of Kraus matrices
    outcomes using a depth-first approach. The depth-first approach requires ``n_channels`` copies
    of the state vector (``n_channels + 1`` state vectors in total) and records ``n_channels`` vectors
    of samples after applying the Kraus matrix for a given branch.

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
    PROBS_TOL = 1e-8

    #######################
    # main implementation #
    #######################

    ##################
    # Parse channel info #
    ##################

    # channels is the list of all channel operations. channels[d] is the parent
    # channel (node) of a circuit segment (edge) at depth `d`. The first element
    # is None because there is no parent channel at depth 0
    channels: tuple[Channel] = tuple(
        [None] + [op for op in circuit.operations if isinstance(op, Channel)]
    )
    n_channels: int = len(channels) - 1
    n_kraus: list[int] = [None] + [c.num_kraus for c in channels[1:]]

    #############################
    # Initialize tree-traversal #
    #############################
    branch_current = qml.math.zeros(n_channels + 1, dtype=int)
    # Split circuit into segments
    circuits = split_circuit_at_channels(circuit)
    circuits[0] = prepend_state_prep(circuits[0], None, circuit.wires)
    terminal_measurements = circuits[-1].measurements
    # Initialize stacks
    cumcounts = [0] * (n_channels + 1)
    stack = TreeTraversalStack(n_channels + 1, n_kraus=n_kraus)
    # The goal is to obtain the measurements of the branches
    # and to combine them into the final result. Exit the loop once the
    # measurements for all branches are available.
    depth = 0

    while stack.any_is_empty(1):

        ###########################################
        # Combine measurements & step up the tree #
        ###########################################

        # Combine two leaves once measurements are available
        if stack.is_full(depth):
            return 0

        ################################################
        # Determine whether to execute the active edge #
        ################################################

        skip_subtree = (
            stack.probs[depth] is not None
            and float(stack.probs[depth][branch_current[depth]]) <= PROBS_TOL
        )
        # Update active branch dict
        invalid_postselect = (
            depth > 0
            and channels[depth].postselect is not None
            and branch_current[depth] != channels[depth].postselect
        )

        ###########################################
        # Obtain measurements for the active edge #
        ###########################################

        # If num_shots is non-zero, simulate the current depth circuit segment
        if depth == 0:
            initial_state = stack.states[0]
        else:
            initial_state = branch_state(
                stack.states[depth], branch_current[depth], channels[depth]
            )
        circtmp = qml.tape.QuantumScript(
            circuits[depth].operations,
            circuits[depth].measurements,
        )
        circtmp = prepend_state_prep(circtmp, initial_state, circuit.wires)
        state, is_state_batched = get_final_state(
            circtmp, mid_measurements=branch_current, **execution_kwargs
        )
        measurements = measure_final_state(circtmp, state, is_state_batched, **execution_kwargs)

        #####################################
        # Update stack & step down the tree #
        #####################################

        # If not at a leaf, project on the zero-branch and increase depth by one
        if depth < n_channels and (not skip_subtree and not invalid_postselect):
            depth += 1
            stack.probs[depth] = dict(zip([False, True], measurements))
            samples = None
            # Store a copy of the state-vector to project on the one-branch
            stack.states[depth] = state
            continue

        ################################################
        # Update terminal measurements & step sideways #
        ################################################

        # If at a zero-branch leaf, update measurements and switch to the one-branch
        if branch_current[depth] == 0:
            stack.results[0][depth] = measurements
            branch_current[depth] = True
            continue
        # If at a one-branch leaf, update measurements
        stack.results[1][depth] = measurements

    ##################################################
    # Finalize terminal measurements post-processing #
    ##################################################

    results = combine_measurements(terminal_measurements)
    return results[0] if len(results) == 1 else results


def split_circuit_at_channels(circuit):
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
            qml.tape.QuantumScript(new_operations, new_measurements, shots=circuit.shots)
        )
        first = last + 1

    last_circuit_operations = circuit.operations[first:]
    last_circuit_measurements = circuit.measurements

    circuits.append(
        qml.tape.QuantumScript(
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
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        return circuit
    state = create_initial_state(wires, None) if state is None else state
    return qml.tape.QuantumScript(
        [qml.StatePrep(qml.math.ravel(state), wires=wires, validate_norm=False)]
        + circuit.operations,
        circuit.measurements,
        shots=circuit.shots,
    )


def branch_state(state, branch, mcm):
    """Collapse the state on a given branch.

    Args:
        state (TensorLike): The initial state
        branch (int): The branch on which the state is collapsed
        mcm (MidMeasureMP): Mid-circuit measurement object used to obtain the wires and ``reset``

    Returns:
        TensorLike: The collapsed state
    """
    state = state.copy()
    slices = [slice(None)] * qml.math.ndim(state)
    axis = mcm.wires.toarray()[0]
    slices[axis] = int(not branch)
    state[tuple(slices)] = 0.0
    state /= qml.math.norm(state)

    return state


def combine_measurements(*args, **kwargs):
    return None
