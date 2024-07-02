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

"""
This test file performs system-level tests with a PennyLane workload against Lightning, both with and without Catalyst.
The workload is running a QAOA workload from the Lightning benchmarks, and hits the following parts of the pipeline:

* Device creation: "lightning.qubit"
* Execution of a gradient-backed QAOA workload
* Application of the QCUT transform at the QNode layer
* Execution of a templated circuit with and without JITing

The workload is first run with default.qubit, and then compared against the others for consistency, rather than correctness.
"""
from typing import List, Optional, Tuple

import networkx as nx
import pytest

import pennylane as qml
from pennylane import numpy as pnp

jax = pytest.importorskip("jax")
catalyst = pytest.importorskip("catalyst")
pytestmark = [pytest.mark.catalyst, pytest.mark.external, pytest.mark.system, pytest.mark.slow]

###############################################################################
# Workload setup: define parameters, quantum circuit and utility decorator
###############################################################################


def clustered_chain_graph(
    n: int, r: int, k: int, q1: float, q2: float, seed: Optional[int] = None
) -> Tuple[nx.Graph, List[List[int]], List[List[int]]]:
    """
    Function to build clustered chain graph

    Args:
        n (int): number of nodes in each cluster
        r (int): number of clusters
        k (int): number of vertex separators between each cluster pair
        q1 (float): probability of an edge connecting any two nodes in a cluster
        q2 (float): probability of an edge connecting a vertex separator to any node in a cluster
        seed (Optional[int]=None): seed for fixing edge generation

    Returns:
        nx.Graph: clustered chain graph
    """

    if r <= 0 or not isinstance(r, int):
        raise ValueError("Number of clusters must be an integer greater than 0")

    clusters = []
    for i in range(r):
        _seed = seed * i if seed is not None else None
        cluster = nx.erdos_renyi_graph(n, q1, seed=_seed)
        nx.set_node_attributes(cluster, f"cluster_{i}", "subgraph")
        clusters.append(cluster)

    separators = []
    for i in range(r - 1):
        separator = nx.empty_graph(k)
        nx.set_node_attributes(separator, f"separator_{i}", "subgraph")
        separators.append(separator)

    G = nx.disjoint_union_all(clusters + separators)

    cluster_nodes = [
        [n[0] for n in G.nodes(data="subgraph") if n[1] == f"cluster_{i}"] for i in range(r)
    ]
    separator_nodes = [
        [n[0] for n in G.nodes(data="subgraph") if n[1] == f"separator_{i}"] for i in range(r - 1)
    ]

    rng = pnp.random.default_rng(seed)

    for i, separator in enumerate(separator_nodes):
        for s in separator:
            for c in cluster_nodes[i] + cluster_nodes[i + 1]:
                if rng.random() < q2:
                    G.add_edge(s, c)

    return G, cluster_nodes, separator_nodes


def cut_decorator(use_qcut=False):
    "Decorator for selective QCUT application to the QNode"

    def inner_func(func):
        if use_qcut:
            return qml.cut_circuit(func)
        return func

    return inner_func


def create_workload(dev_name, diff_method, shots, cut_circuit, layers):
    "Create QAOA workload for QCUT paper example targeting Lightning"
    r = 2  # number of clusters
    n = 2  # nodes in clusters
    k = 1  # vertex separators

    q1 = 0.7
    q2 = 0.3

    seed = 1967

    G, cluster_nodes, separator_nodes = clustered_chain_graph(n, r, k, q1, q2, seed=seed)

    wires = len(G)

    dev = qml.device(dev_name, wires=wires, shots=shots)

    r = len(cluster_nodes)
    cost_H, _ = qml.qaoa.maxcut(G)

    @cut_decorator(cut_circuit)
    @qml.qnode(dev, diff_method=diff_method)
    def circuit(params):
        for w in range(wires):
            qml.Hadamard(wires=w)

        for l in range(layers):
            for i, c in enumerate(cluster_nodes):
                if i == 0:
                    current_separator = []
                    next_separator = separator_nodes[0]
                elif i == r - 1:
                    current_separator = separator_nodes[-1]
                    next_separator = []
                else:
                    current_separator = separator_nodes[i - 1]
                    next_separator = separator_nodes[i]

                for cs in current_separator:
                    qml.WireCut(wires=cs)

                nodes = c + current_separator + next_separator
                subgraph = G.subgraph(nodes)

                for edge in subgraph.edges:
                    qml.IsingZZ(
                        2 * params[l][0], wires=edge
                    )  # multiply param by 2 for consistency with analytic cost

            # mixer layer
            for w in range(wires):
                qml.RX(2 * params[l][1], wires=w)

            # reset cuts
            if l < layers - 1:
                for s in separator_nodes:
                    qml.WireCut(wires=s)

        return qml.expval(cost_H)

    return circuit


###############################################################################
# Backend setup: define devices, working environment, and DQ comparator
###############################################################################


def workload_non_catalyst(params, valid_results, diff_method, shots, cut_circuit, layers=1):
    "Run gradient workload directly with PennyLane and Lightning, comparing results against the input"
    lq_qnode = create_workload("lightning.qubit", diff_method, shots, cut_circuit, layers)
    assert pnp.allclose(valid_results, qml.grad(lq_qnode)(params), rtol=1e-3)


def workload_catalyst(params, valid_results, diff_method, shots, cut_circuit, layers=1):
    "Run gradient workload with PennyLane, Lightning, and Catalyst, comparing results against the input"
    lq_qnode = create_workload("lightning.qubit", diff_method, shots, cut_circuit, layers)
    local_params = jax.numpy.array(params)
    assert pnp.allclose(valid_results, catalyst.grad(qml.qjit(lq_qnode))(params), rtol=1e-3)


def dq_workload(diff_method, shots, cut_circuit, layers=1):
    params = pnp.array([[7.20792567e-01, 1.02761748e-04]] * layers, requires_grad=True)
    dq_qnode = create_workload("default.qubit", diff_method, shots, cut_circuit, layers)
    dq_grad = qml.grad(dq_qnode)(params)
    return dq_grad, params


###############################################################################
# Test setup: choose pytest template parameters to run across
###############################################################################


@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("use_jit", [False, True])
@pytest.mark.parametrize(
    "diff_method, shots",
    [
        ("best", None),
        ("adjoint", None),
        ("adjoint", None),
        ("parameter-shift", None),
        ("parameter-shift", 10000),
    ],
)
@pytest.mark.parametrize("cut_ciruit", [False, True])
def test_QAOA_layers_scaling(layers, use_jit, diff_method, shots, cut_ciruit):
    "Run the example workload over the given parameters"

    dq_grad, params = dq_workload(diff_method, shots, cut_ciruit, layers)

    if use_jit:
        workload_catalyst(params, dq_grad, diff_method, shots, cut_ciruit, layers)
    else:
        workload_non_catalyst(params, dq_grad, diff_method, shots, cut_ciruit, layers)
