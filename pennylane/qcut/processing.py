# Copyright 2022 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processing functions for circuit cutting.
"""

import string
from typing import List, Sequence
from networkx import MultiDiGraph

import pennylane as qml
from pennylane import numpy as pnp

from .utils import MeasureNode, PrepareNode


def qcut_processing_fn(
    results: Sequence[Sequence],
    communication_graph: MultiDiGraph,
    prepare_nodes: Sequence[Sequence[PrepareNode]],
    measure_nodes: Sequence[Sequence[MeasureNode]],
    use_opt_einsum: bool = False,
):
    """Processing function for the :func:`cut_circuit() <pennylane.cut_circuit>` transform.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        results (Sequence[Sequence]): A collection of execution results generated from the
            expansion of circuit fragments over measurement and preparation node configurations.
            These results are processed into tensors and then contracted.
        communication_graph (nx.MultiDiGraph): the communication graph determining connectivity
            between circuit fragments
        prepare_nodes (Sequence[Sequence[PrepareNode]]): a sequence of size
            ``len(communication_graph.nodes)`` that determines the order of preparation indices in
            each tensor
        measure_nodes (Sequence[Sequence[MeasureNode]]): a sequence of size
            ``len(communication_graph.nodes)`` that determines the order of measurement indices in
            each tensor
        use_opt_einsum (bool): Determines whether to use the
            `opt_einsum <https://dgasmith.github.io/opt_einsum/>`__ package. This package is useful
            for faster tensor contractions of large networks but must be installed separately using,
            e.g., ``pip install opt_einsum``. Both settings for ``use_opt_einsum`` result in a
            differentiable contraction.

    Returns:
        float or tensor_like: the output of the original uncut circuit arising from contracting
        the tensor network of circuit fragments
    """
    # each tape contains only expval measurements or sample measurements, so
    # stacking won't create any ragged arrays
    results = [
        (
            qml.math.stack(tape_res)
            if isinstance(tape_res, tuple)
            else qml.math.reshape(tape_res, [-1])
        )
        for tape_res in results
    ]

    flat_results = qml.math.concatenate(results)

    tensors = _to_tensors(flat_results, prepare_nodes, measure_nodes)
    result = contract_tensors(
        tensors, communication_graph, prepare_nodes, measure_nodes, use_opt_einsum
    )
    return result


def qcut_processing_fn_sample(
    results: Sequence, communication_graph: MultiDiGraph, shots: int
) -> List:
    """
    Function to postprocess samples for the :func:`cut_circuit_mc() <pennylane.cut_circuit_mc>`
    transform. This removes superfluous mid-circuit measurement samples from fragment
    circuit outputs.

    .. note::

        This function is designed for use as part of the sampling-based circuit cutting workflow.
        Check out the :func:`qml.cut_circuit_mc() <pennylane.cut_circuit_mc>` transform for more details.

    Args:
        results (Sequence): a collection of sample-based execution results generated from the
            random expansion of circuit fragments over measurement and preparation node configurations
        communication_graph (nx.MultiDiGraph): the communication graph determining connectivity
            between circuit fragments
        shots (int): the number of shots

    Returns:
        List[tensor_like]: the sampled output for all terminal measurements over the number of shots given
    """
    results = _reshape_results(results, shots)
    res0 = results[0][0]
    out_degrees = [d for _, d in communication_graph.out_degree]

    samples = []
    for result in results:
        sample = []
        for fragment_result, out_degree in zip(result, out_degrees):
            sample.append(fragment_result[: -out_degree or None])
        samples.append(pnp.hstack(sample))
    return [qml.math.convert_like(pnp.array(samples), res0)]


def qcut_processing_fn_mc(
    results: Sequence,
    communication_graph: MultiDiGraph,
    settings: pnp.ndarray,
    shots: int,
    classical_processing_fn: callable,
):
    """
    Function to postprocess samples for the :func:`cut_circuit_mc() <pennylane.cut_circuit_mc>`
    transform. This takes a user-specified classical function to act on bitstrings and
    generates an expectation value.

    .. note::

        This function is designed for use as part of the sampling-based circuit cutting workflow.
        Check out the :func:`qml.cut_circuit_mc() <pennylane.cut_circuit_mc>` transform for more details.

    Args:
        results (Sequence): a collection of sample-based execution results generated from the
            random expansion of circuit fragments over measurement and preparation node configurations
        communication_graph (nx.MultiDiGraph): the communication graph determining connectivity
            between circuit fragments
        settings (np.ndarray): Each element is one of 8 unique values that tracks the specific
            measurement and preparation operations over all configurations. The number of rows is determined
            by the number of cuts and the number of columns is determined by the number of shots.
        shots (int): the number of shots
        classical_processing_fn (callable): A classical postprocessing function to be applied to
            the reconstructed bitstrings. The expected input is a bitstring; a flat array of length ``wires``
            and the output should be a single number within the interval :math:`[-1, 1]`.

    Returns:
        float or tensor_like: the expectation value calculated in accordance to Eq. (35) of
        `Peng et al. <https://arxiv.org/abs/1904.00102>`__
    """
    results = _reshape_results(results, shots)
    res0 = results[0][0]
    out_degrees = [d for _, d in communication_graph.out_degree]

    evals = (0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5)
    expvals = []
    for result, setting in zip(results, settings.T):
        sample_terminal = []
        sample_mid = []

        for fragment_result, out_degree in zip(result, out_degrees):
            sample_terminal.append(fragment_result[: -out_degree or None])
            sample_mid.append(fragment_result[-out_degree or len(fragment_result) :])

        sample_terminal = pnp.hstack(sample_terminal)
        sample_mid = pnp.hstack(sample_mid)

        assert set(sample_terminal).issubset({pnp.array(0), pnp.array(1)})
        assert set(sample_mid).issubset({pnp.array(-1), pnp.array(1)})
        # following Eq.(35) of Peng et.al: https://arxiv.org/abs/1904.00102
        f = classical_processing_fn(sample_terminal)
        if not -1 <= f <= 1:
            raise ValueError(
                "The classical processing function supplied must "
                "give output in the interval [-1, 1]"
            )
        sigma_s = pnp.prod(sample_mid)
        t_s = f * sigma_s
        c_s = pnp.prod([evals[s] for s in setting])
        K = len(sample_mid)
        expvals.append(8**K * c_s * t_s)

    return qml.math.convert_like(pnp.mean(expvals), res0)


def _reshape_results(results: Sequence, shots: int) -> List[List]:
    """
    Helper function to reshape ``results`` into a two-dimensional nested list whose number of rows
    is determined by the number of shots and whose number of columns is determined by the number of
    cuts.
    """
    # each tape contains only expval measurements or sample measurements, so
    # stacking won't create any ragged arrays
    results = [
        qml.math.stack(tape_res) if isinstance(tape_res, tuple) else tape_res
        for tape_res in results
    ]

    results = [qml.math.flatten(r) for r in results]
    results = [results[i : i + shots] for i in range(0, len(results), shots)]
    results = list(map(list, zip(*results)))  # calculate list-based transpose

    return results


def _get_symbol(i):
    """Finds the i-th ASCII symbol. Works for lowercase and uppercase letters, allowing i up to
    51."""
    if i >= len(string.ascii_letters):
        raise ValueError(
            "Set the use_opt_einsum argument to True when applying more than "
            f"{len(string.ascii_letters)} wire cuts to a circuit"
        )
    return string.ascii_letters[i]


# pylint: disable=too-many-branches
def contract_tensors(
    tensors: Sequence,
    communication_graph: MultiDiGraph,
    prepare_nodes: Sequence[Sequence[PrepareNode]],
    measure_nodes: Sequence[Sequence[MeasureNode]],
    use_opt_einsum: bool = False,
):
    r"""Contract tensors according to the edges specified in the communication graph.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Consider the three tensors :math:`T^{(1)}`, :math:`T^{(2)}`, and :math:`T^{(3)}`, along with
    their contraction equation

    .. math::

        \sum_{ijklmn} T^{(1)}_{ij,km} T^{(2)}_{kl,in} T^{(3)}_{mn,jl}

    Each tensor is the result of the tomography of a circuit fragment and has some indices
    corresponding to state preparations (marked by the indices before the comma) and some indices
    corresponding to measurements (marked by the indices after the comma).

    An equivalent representation of the contraction equation is to use a directed multigraph known
    as the communication/quotient graph. In the communication graph, each tensor is assigned a node
    and edges are added between nodes to mark a contraction along an index. The communication graph
    resulting from the above contraction equation is a complete directed graph.

    In the communication graph provided by :func:`fragment_graph`, edges are composed of
    :class:`PrepareNode` and :class:`MeasureNode` pairs. To correctly map back to the contraction
    equation, we must keep track of the order of preparation and measurement indices in each tensor.
    This order is specified in the ``prepare_nodes`` and ``measure_nodes`` arguments.

    Args:
        tensors (Sequence): the tensors to be contracted
        communication_graph (nx.MultiDiGraph): the communication graph determining connectivity
            between the tensors
        prepare_nodes (Sequence[Sequence[PrepareNode]]): a sequence of size
            ``len(communication_graph.nodes)`` that determines the order of preparation indices in
            each tensor
        measure_nodes (Sequence[Sequence[MeasureNode]]): a sequence of size
            ``len(communication_graph.nodes)`` that determines the order of measurement indices in
            each tensor
        use_opt_einsum (bool): Determines whether to use the
            `opt_einsum <https://dgasmith.github.io/opt_einsum/>`__ package. This package is useful
            for faster tensor contractions of large networks but must be installed separately using,
            e.g., ``pip install opt_einsum``. Both settings for ``use_opt_einsum`` result in a
            differentiable contraction.

    Returns:
        float or tensor_like: the result of contracting the tensor network

    **Example**

    We first set up the tensors and their corresponding :class:`~.PrepareNode` and
    :class:`~.MeasureNode` orderings:

    .. code-block:: python

        from pennylane.transforms import qcut
        import networkx as nx
        import numpy as np

        tensors = [np.arange(4), np.arange(4, 8)]
        prep = [[], [qcut.PrepareNode(wires=0)]]
        meas = [[qcut.MeasureNode(wires=0)], []]

    The communication graph describing edges in the tensor network must also be constructed.
    The nodes of the fragment graphs are formatted as ``WrappedObj(op)``, where ``WrappedObj.obj``
    is the operator, and the same format should be preserved in the pairs stored
    with the edge data of the communication graph:

    .. code-block:: python

        graph = nx.MultiDiGraph(
            [(0, 1, {"pair": (WrappedObj(meas[0][0]), WrappedObj(prep[1][0]))})]
        )

    The network can then be contracted using:

    >>> qml.qcut.contract_tensors(tensors, graph, prep, meas)
    38
    """
    # pylint: disable=import-outside-toplevel
    if use_opt_einsum:
        try:
            from opt_einsum import contract, get_symbol
        except ImportError as e:
            raise ImportError(
                "The opt_einsum package is required when use_opt_einsum is set to "
                "True in the contract_tensors function. This package can be "
                "installed using:\npip install opt_einsum"
            ) from e
    else:
        contract = qml.math.einsum
        get_symbol = _get_symbol

    ctr = 0
    tensor_indxs = [""] * len(communication_graph.nodes)

    meas_map = {}

    for i, (node, prep) in enumerate(zip(communication_graph.nodes, prepare_nodes)):
        predecessors = communication_graph.pred[node]

        for p in prep:
            for _, pred_edges in predecessors.items():
                for pred_edge in pred_edges.values():
                    meas_op, prep_op = pred_edge["pair"]

                    if p.id is prep_op.obj.id:
                        symb = get_symbol(ctr)
                        ctr += 1
                        tensor_indxs[i] += symb
                        meas_map[meas_op] = symb

    for i, (node, meas) in enumerate(zip(communication_graph.nodes, measure_nodes)):
        successors = communication_graph.succ[node]

        for m in meas:
            for _, succ_edges in successors.items():
                for succ_edge in succ_edges.values():
                    meas_op, _ = succ_edge["pair"]

                    if m.id is meas_op.obj.id:
                        symb = meas_map[meas_op]
                        tensor_indxs[i] += symb

    eqn = ",".join(tensor_indxs)
    kwargs = {} if use_opt_einsum else {"like": tensors[0]}

    return contract(eqn, *tensors, **kwargs)


CHANGE_OF_BASIS = qml.math.array(
    [[1.0, 1.0, 0.0, 0.0], [-1.0, -1.0, 2.0, 0.0], [-1.0, -1.0, 0.0, 2.0], [1.0, -1.0, 0.0, 0.0]]
)


def _process_tensor(results, n_prep: int, n_meas: int):
    """Convert a flat slice of an individual circuit fragment's execution results into a tensor.

    This function performs the following steps:

    1. Reshapes ``results`` into the intermediate shape ``(4,) * n_prep + (4**n_meas,)``
    2. Shuffles the final axis to follow the standard product over measurement settings. E.g., for
      ``n_meas = 2`` the standard product is: II, IX, IY, IZ, XI, ..., ZY, ZZ while the input order
      will be the result of ``qml.pauli.partition_pauli_group(2)``, i.e., II, IZ, ZI, ZZ, ...,
      YY.
    3. Reshapes into the final target shape ``(4,) * (n_prep + n_meas)``
    4. Performs a change of basis for the preparation indices (the first ``n_prep`` indices) from
       the |0>, |1>, |+>, |+i> basis to the I, X, Y, Z basis using ``CHANGE_OF_BASIS``.

    Args:
        results (tensor_like): the input execution results
        n_prep (int): the number of preparation nodes in the corresponding circuit fragment
        n_meas (int): the number of measurement nodes in the corresponding circuit fragment

    Returns:
        tensor_like: the corresponding fragment tensor
    """
    n = n_prep + n_meas
    dim_meas = 4**n_meas

    # Step 1
    intermediate_shape = (4,) * n_prep + (dim_meas,)
    intermediate_tensor = qml.math.reshape(results, intermediate_shape)

    # Step 2
    grouped = qml.pauli.partition_pauli_group(n_meas)
    grouped_flat = [term for group in grouped for term in group]
    order = qml.math.argsort(grouped_flat)

    if qml.math.get_interface(intermediate_tensor) == "tensorflow":
        # TensorFlow does not support slicing
        intermediate_tensor = qml.math.gather(intermediate_tensor, order, axis=-1)
    else:
        sl = [slice(None)] * n_prep + [order]
        intermediate_tensor = intermediate_tensor[tuple(sl)]

    # Step 3
    final_shape = (4,) * n
    final_tensor = qml.math.reshape(intermediate_tensor, final_shape)

    # Step 4
    change_of_basis = qml.math.convert_like(CHANGE_OF_BASIS, intermediate_tensor)

    for i in range(n_prep):
        axes = [[1], [i]]
        final_tensor = qml.math.tensordot(change_of_basis, final_tensor, axes=axes)

    axes = list(reversed(range(n_prep))) + list(range(n_prep, n))

    # Use transpose to reorder indices. We must do this because tensordot returns a tensor whose
    # indices are ordered according to the uncontracted indices of the first tensor, followed
    # by the uncontracted indices of the second tensor. For example, calculating C_kj T_ij returns
    # a tensor T'_ki rather than T'_ik.
    final_tensor = qml.math.transpose(final_tensor, axes=axes)

    final_tensor *= qml.math.power(2, -(n_meas + n_prep) / 2)
    return final_tensor


def _to_tensors(
    results,
    prepare_nodes: Sequence[Sequence[PrepareNode]],
    measure_nodes: Sequence[Sequence[MeasureNode]],
) -> List:
    """Process a flat list of execution results from all circuit fragments into the corresponding
    tensors.

    This function slices ``results`` according to the expected size of fragment tensors derived from
    the ``prepare_nodes`` and ``measure_nodes`` and then passes onto ``_process_tensor`` for further
    transformation.

    Args:
        results (tensor_like): A collection of execution results, provided as a flat tensor,
            corresponding to the expansion of circuit fragments in the communication graph over
            measurement and preparation node configurations. These results are processed into
            tensors by this function.
        prepare_nodes (Sequence[Sequence[PrepareNode]]): a sequence whose length is equal to the
            number of circuit fragments, with each element used here to determine the number of
            preparation nodes in a given fragment
        measure_nodes (Sequence[Sequence[MeasureNode]]): a sequence whose length is equal to the
            number of circuit fragments, with each element used here to determine the number of
            measurement nodes in a given fragment

    Returns:
        List[tensor_like]: the tensors for each circuit fragment in the communication graph
    """
    ctr = 0
    tensors = []

    for p, m in zip(prepare_nodes, measure_nodes):
        n_prep = len(p)
        n_meas = len(m)
        n = n_prep + n_meas

        dim = 4**n
        results_slice = results[ctr : dim + ctr]

        tensors.append(_process_tensor(results_slice, n_prep, n_meas))

        ctr += dim

    if results.shape[0] != ctr:
        raise ValueError(f"The results argument should be a flat list of length {ctr}")

    return tensors
