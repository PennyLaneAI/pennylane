# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
The main function for measurement reduction, ``optimize_measurements`` returns the partitions and
corresponding necessary circuit post-rotations for a given list of Pauli words.
"""

from pennylane.operation import Operator
from pennylane.pauli.utils import diagonalize_qwc_groupings
from pennylane.typing import Sequence

from .group_observables import compute_partition_indices


def optimize_measurements(
    observables: Sequence[Operator],
    coefficients: Sequence[float] = None,
    grouping: str = "qwc",
    colouring_method: str = "lf",
) -> tuple:
    """Partitions then diagonalizes a list of Pauli words, facilitating simultaneous measurement of
    all observables within a partition.

    The input list of observables are partitioned into mutually qubit-wise commuting (QWC) or
    mutually commuting partitions by approximately solving minimum clique cover on a graph where
    each observable represents a vertex. The unitaries which diagonalize the
    partitions are then found. See `arXiv:1907.03358
    <https://arxiv.org/abs/1907.03358>`_ and `arXiv:1907.09386
    <https://arxiv.org/abs/1907.09386>`_ for technical details of the QWC and
    fully-commuting measurement-partitioning approaches respectively.

    Args:
        observables (list[Operator]): a list of Pauli words (Pauli operation instances and tensors
            instances thereof)
        coefficients (list[float]): a list of float coefficients, for instance the weights of
            the Pauli words comprising a Hamiltonian
        grouping (str): the binary symmetric relation to use for operator partitioning,
            passed to :func:`~.pennylane.pauli.compute_partition_indices`.
        colouring_method (str): the graph-colouring heuristic to use in obtaining the operator
            partitions, passed to :func:`~pennylane.pauli.compute_partition_indices`.

    Returns:
        tuple:

            * list[callable]: a list of the post-rotation templates, one
              for each partition
            * list[list[Operator]]: A list of the obtained groupings. Each
              grouping is itself a list of Pauli words diagonal in the
              measurement basis.
            * list[list[float]]: A list of coefficient groupings. Each
              coefficient grouping is itself a list of the partitions
              corresponding coefficients.  Only output if coefficients are
              specified.

    **Example**

    >>> obs = [qml.Y(0), qml.X(0) @ qml.X(1), qml.Z(1)]
    >>> coeffs = [1.43, 4.21, 0.97]
    >>> rotations, groupings, grouped_coeffs = optimize_measurements(obs, coeffs, 'qwc', 'rlf')
    >>> print(rotations)
    [[RY(-1.5707963267948966, wires=[0]), RY(-1.5707963267948966, wires=[1])],
     [RX(1.5707963267948966, wires=[0])]]
    >>> print(groupings)
    [[Z(0) @ Z(1)], [Z(0), Z(1)]]
    >>> print(grouped_coeffs)
    [[4.21], [1.43, 0.97]]
    """

    partition_indices = compute_partition_indices(
        observables, grouping_type=grouping, method=colouring_method
    )
    grouped_obs = [[observables[idx] for idx in group] for group in partition_indices]

    if grouping.lower() == "qwc":
        (
            post_rotations,
            diagonalized_groupings,
        ) = diagonalize_qwc_groupings(grouped_obs)
    else:
        raise NotImplementedError(
            f"Measurement reduction by '{grouping.lower()}' grouping not implemented."
        )

    result = (post_rotations, diagonalized_groupings)

    if coefficients:
        grouped_coeffs = [[coefficients[idx] for idx in group] for group in partition_indices]
        result += (grouped_coeffs,)

    return result
