# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
The main function for measurement reduction, `optimize_measurements` returns the partitions and
corresponding necessary circuit post-rotations for a given list of Pauli words.
"""

from pennylane.grouping.group_observables import group_observables
from pennylane.grouping.transformations import obtain_qwc_post_rotations_and_diagonalized_groupings


def optimize_measurements(observables, coefficients=None, grouping="qwc", colouring_method="rlf"):
    """Partitions then diagonalizes a list of Pauli words, facilitating simultaneous measurement of
    all observables within a partition.

    The input list of observables are partitioned into mutually qubit-wise commuting (QWC) or
    mutually commuting partitions by approximately solving minimum clique cover on a graph where
    each observable represents a vertice. The unitaries which diagonalize the partitions are then
    found. See arXiv:1907.03358 and arXiv:1907.09386 for technical details of the QWC and fully
    commuting measurement partitioning approaches respectively.

    **Example usage:**

    >>> observables = [qml.PauliY(0), qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(1)]
    >>> coefficients = [1.43, 4.21, 0.97]
    >>> post_rotations,diagonalized_groupings,grouped_coeffs = optimize_measurements(
                                                                                     observables,
                                                                                     coefficients,
                                                                                     'qwc',
                                                                                     'rlf'
                                                                                     )
    >>> print(post_rotations)
    [[RY(-1.5707963267948966, wires=[0]), RY(-1.5707963267948966, wires=[1])],
     [RX(1.5707963267948966, wires=[0])]]
    >>> print(diagonalized_groupings)
    [[Tensor(PauliZ(wires=[0]), PauliZ(wires=[1]))],
     [Tensor(PauliZ(wires=[0])), Tensor(PauliZ(wires=[1]))]]
    >>> print(grouped_coeffs)
    [[4.21], [1.43, 0.97]]

    Args:
        observables (list[Observable]): a list of Pauli words (Pauli operation instances and Tensor
            instances thereof).

    Keyword args:
        coefficients (list[scalar]): a list of scalar coefficients.
        grouping (str): the binary symmetric relation to use for operator partitioning.
        colouring_method (str): the graph colouring heuristic to use in obtaining the operator
            partitions.

    Returns:
        post_rotations (list[Template]): a list of the post-rotation qml.Templates instances, one
            for each partition.
        diagonalized_groupings (list[list[Observable]]): a list of the obtained groupings. Each
            grouping is itself a list of Pauli words diagonal in the measurement basis.
        grouped_coeffs (list[list[scalar]]): a list of coefficient groupings. Each
            coefficient grouping is itself a list of the partitions corresponding coefficients.
            (Only output if coefficients are specified.)

    """

    if coefficients is None:
        grouped_obs = group_observables(
            observables, grouping_type=grouping, method=colouring_method
        )
    else:
        grouped_obs, grouped_coeffs = group_observables(
            observables, coefficients, grouping_type=grouping, method=colouring_method
        )

    if grouping.lower() == "qwc":
        (
            post_rotations,
            diagonalized_groupings,
        ) = obtain_qwc_post_rotations_and_diagonalized_groupings(grouped_obs)
    else:
        raise NotImplementedError(
            "Measurement reduction by '{}' grouping not implemented.".format(grouping.lower())
        )

    if coefficients is None:
        return post_rotations, diagonalized_groupings

    return post_rotations, diagonalized_groupings, grouped_coeffs
