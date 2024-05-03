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
Functions for partitioning a graph using KaHyPar.
"""


from collections.abc import Sequence as SequenceType
from itertools import compress
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
from networkx import MultiDiGraph

import pennylane as qml
from pennylane.operation import Operation


def kahypar_cut(
    graph: MultiDiGraph,
    num_fragments: int,
    imbalance: int = None,
    edge_weights: List[Union[int, float]] = None,
    node_weights: List[Union[int, float]] = None,
    fragment_weights: List[Union[int, float]] = None,
    hyperwire_weight: int = 1,
    seed: int = None,
    config_path: Union[str, Path] = None,
    trial: int = None,
    verbose: bool = False,
) -> List[Tuple[Operation, Operation, Any]]:
    """Calls `KaHyPar <https://kahypar.org/>`__ to partition a graph.

    .. warning::
        Requires KaHyPar to be installed separately. For Linux and Mac users,
        KaHyPar can be installed using ``pip install kahypar``. Windows users
        can follow the instructions
        `here <https://kahypar.org>`__ to compile from source.

    Args:
        graph (nx.MultiDiGraph): The graph to be partitioned.
        num_fragments (int): Desired number of fragments.
        imbalance (int): Imbalance factor of the partitioning. Defaults to KaHyPar's determination.
        edge_weights (List[Union[int, float]]): Weights for edges. Defaults to unit-weighted edges.
        node_weights (List[Union[int, float]]): Weights for nodes. Defaults to unit-weighted nodes.
        fragment_weights (List[Union[int, float]]): Maximum size constraints by fragment. Defaults
            to no such constraints, with ``imbalance`` the only parameter affecting fragment sizes.
        hyperwire_weight (int): Weight on the artificially appended hyperedges representing wires.
            Setting it to 0 leads to no such insertion. If greater than 0, hyperedges will be
            appended with the provided weight, to encourage the resulting fragments to cluster gates
            on the same wire together. Defaults to 1.
        seed (int): KaHyPar's seed. Defaults to the seed in the config file which defaults to -1,
            i.e. unfixed seed.
        config_path (str): KaHyPar's ``.ini`` config file path. Defaults to its SEA20 paper config.
        trial (int): trial id for summary label creation. Defaults to ``None``.
        verbose (bool): Flag for printing KaHyPar's output summary. Defaults to ``False``.

    Returns:
        List[Union[int, Any]]: List of cut edges.

    **Example**

    Consider the following 2-wire circuit with one CNOT gate connecting the wires:

    .. code-block:: python

        ops = [
            qml.RX(0.432, wires=0),
            qml.RY(0.543, wires="a"),
            qml.CNOT(wires=[0, "a"]),
            qml.RZ(0.240, wires=0),
            qml.RZ(0.133, wires="a"),
            qml.RX(0.432, wires=0),
            qml.RY(0.543, wires="a"),
        ]
        measurements = [qml.expval(qml.Z(0))]
        tape = qml.tape.QuantumTape(ops, measurements)

    We can let KaHyPar automatically find the optimal edges to place cuts:

    >>> graph = qml.qcut.tape_to_graph(tape)
    >>> cut_edges = qml.qcut.kahypar_cut(
    ...     graph=graph,
    ...     num_fragments=2,
    ... )
    >>> cut_edges
    [(Wrapped(CNOT(wires=[0, 'a'])), Wrapped(RZ(0.24, wires=[0])), 0)]
    """
    # pylint: disable=too-many-arguments, import-outside-toplevel
    try:
        import kahypar
    except ImportError as e:
        raise ImportError(
            "KaHyPar must be installed to use this method for automatic "
            "cut placement. Try pip install kahypar or visit "
            "https://kahypar.org/ for installation instructions."
        ) from e

    adjacent_nodes, edge_splits, edge_weights = _graph_to_hmetis(
        graph=graph, hyperwire_weight=hyperwire_weight, edge_weights=edge_weights
    )

    trial = 0 if trial is None else trial
    ne = len(edge_splits) - 1
    nv = max(adjacent_nodes) + 1

    if edge_weights is not None or node_weights is not None:
        edge_weights = edge_weights or [1] * ne
        node_weights = node_weights or [1] * nv
        hypergraph = kahypar.Hypergraph(
            nv,
            ne,
            edge_splits,
            adjacent_nodes,
            num_fragments,
            edge_weights,
            node_weights,
        )

    else:
        hypergraph = kahypar.Hypergraph(nv, ne, edge_splits, adjacent_nodes, num_fragments)

    context = kahypar.Context()

    config_path = config_path or str(Path(__file__).parent / "_cut_kKaHyPar_sea20.ini")
    context.loadINIconfiguration(config_path)

    context.setK(num_fragments)

    if isinstance(imbalance, float):
        context.setEpsilon(imbalance)
    if isinstance(fragment_weights, SequenceType) and (len(fragment_weights) == num_fragments):
        context.setCustomTargetBlockWeights(fragment_weights)
    if not verbose:
        context.suppressOutput(True)

    # KaHyPar fixes seed to 42 by default, need to manually sample seed to randomize:
    kahypar_seed = np.random.default_rng(seed).choice(2**15)
    context.setSeed(kahypar_seed)

    kahypar.partition(hypergraph, context)

    cut_edge_mask = [hypergraph.connectivity(e) > 1 for e in hypergraph.edges()]

    # compress() ignores the extra hyperwires at the end if there is any.
    cut_edges = list(compress(graph.edges, cut_edge_mask))

    if verbose:
        fragment_sizes = [hypergraph.blockSize(p) for p in range(num_fragments)]
        print(len(fragment_sizes), fragment_sizes)

    return cut_edges


def _graph_to_hmetis(
    graph: MultiDiGraph,
    hyperwire_weight: int = 0,
    edge_weights: Sequence[int] = None,
) -> Tuple[List[int], List[int], List[Union[int, float]]]:
    """Converts a ``MultiDiGraph`` into the
    `hMETIS hypergraph input format <http://glaros.dtc.umn.edu/gkhome/fetch/sw/hmetis/manual.pdf>`__
    conforming to KaHyPar's calling signature.

    Args:
        graph (MultiDiGraph): The original (tape-converted) graph to be cut.
        hyperwire_weight (int): Weight on the artificially appended hyperedges representing wires.
            Defaults to 0 which leads to no such insertion. If greater than 0, hyperedges will be
            appended with the provided weight, to encourage the resulting fragments to cluster gates
            on the same wire together.
        edge_weights (Sequence[int]): Weights for regular edges in the graph. Defaults to ``None``,
            which leads to unit-weighted edges.

    Returns:
        Tuple[List,List,List]: The 3 lists representing an (optionally weighted) hypergraph:
        - Flattened list of adjacent node indices.
        - List of starting indices for edges in the above adjacent-nodes-list.
        - Optional list of edge weights. ``None`` if ``hyperwire_weight`` is equal to 0.
    """

    nodes = list(graph.nodes)
    edges = graph.edges(data="wire")
    wires = {w for _, _, w in edges}

    adj_nodes = [nodes.index(v) for ops in graph.edges(keys=False) for v in ops]
    edge_splits = qml.math.cumsum([0] + [len(e) for e in graph.edges(keys=False)]).tolist()
    edge_weights = (
        edge_weights if edge_weights is not None and len(edges) == len(edge_weights) else None
    )

    if hyperwire_weight:
        hyperwires = {w: set() for w in wires}
        num_wires = len(hyperwires)

        for v0, v1, wire in edges:
            hyperwires[wire].update([nodes.index(v0), nodes.index(v1)])

        for wire, nodes_on_wire in hyperwires.items():
            nwv = len(nodes_on_wire)
            edge_splits.append(nwv + edge_splits[-1])
            adj_nodes = adj_nodes + list(nodes_on_wire)
        assert len(edge_splits) == len(edges) + num_wires + 1

        if isinstance(hyperwire_weight, (int, float)):
            # assumes original edges having unit weights by default:
            edge_weights = edge_weights or ([1] * len(edges))
            wire_weights = [hyperwire_weight] * num_wires
            edge_weights = edge_weights + wire_weights

    return adj_nodes, edge_splits, edge_weights
