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
"""Solovay-Kitaev implementation for approximate single-qubit unitary decomposition."""
import math
import warnings
import functools as ft

import scipy as sp

import pennylane as qml
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript
from pennylane.transforms.optimization import (
    cancel_inverses,
)

# Defining Clifford+T basis
_CLIFFORD_T_BASIS = {
    "I": qml.Identity(0),
    "X": qml.PauliX(0),
    "Y": qml.PauliY(0),
    "Z": qml.PauliZ(0),
    "H": qml.Hadamard(0),
    "T": qml.T(0),
    "T*": qml.adjoint(qml.T(0)),
    "S": qml.S(0),
    "S*": qml.adjoint(qml.S(0)),
}


def _SU2_transform(matrix):
    r"""Perform a U(2) to SU(2) transformation via a global phase addition.

    A general element of :math:`\text{SU}_2(\mathbb{C})` has the following form:

    .. math::
        \text{SU}_{2} = \begin{bmatrix} \alpha & \beta \\ -\beta^{*} & \alpha^{*} \end{bmatrix},

    where :math:`\alpha, \beta \in \mathbb{C}` and :math:`|\alpha|^2 + |\beta|^2 = 1`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        factor = qml.math.linalg.det(matrix)
    gphase = qml.math.mod(qml.math.angle(factor), 2 * math.pi) / 2
    s2_mat = matrix * qml.math.exp(-1j * qml.math.cast_like(gphase, 1j))
    return s2_mat, gphase


def _quaternion_transform(matrix):
    r"""Perform a SU(2) to quaternion transformation.

    Any element :math:`\text{SU}_2(\mathbb{C})` can be be written as a unique quaternion
    :math:`\alpha_0\mathbb{I} + \alpha_1\mathbf{i} + \alpha_2\mathbf{j} + \alpha_3\mathbf{k}`,
    where :math:`\mathbf{i}=-i\mathbf{X},\ \mathbf{j}=-i\mathbf{Y},\ \text{and}\ \mathbf{k}=-i\mathbf{Z}`.
    """
    return qml.math.array(
        [
            qml.math.real(matrix[0, 0]),
            -qml.math.imag(matrix[0, 1]),
            -qml.math.real(matrix[0, 1]),
            -qml.math.imag(matrix[0, 0]),
        ],
        dtype=float,
    )


def _contains_SU2(op_mat, ops_vecs, tol=1e-8):
    r"""Checks if a given SU(2) matrix is contained in a list of quaternions for a given tolerance.

    Args:
        op_mat (TensorLike): SU(2) matrix for the operation to be searched
        op_vecs (list(TensorLike)): List of quaternion for the operations that makes the search space.
        tol (float): Tolerance for the match to be considered ``True``.

    Returns:
        Tuple(bool, TensorLike): A bool that shows whether an operation similar to the given operations
        was found, and the quaternion representation of the searched operation.
    """
    node_points = qml.math.array(ops_vecs)
    gate_points = qml.math.array([_quaternion_transform(op_mat)])

    tree = sp.spatial.KDTree(node_points)
    dist = tree.query(gate_points, workers=-1)[0][0]

    return (dist < tol, gate_points[0])


@ft.lru_cache()
def _approximate_set(basis_gates, max_length=10):
    r"""Builds an approximate unitary set required for the `Solovay-Kitaev algorithm <https://arxiv.org/abs/quant-ph/0505030>`_.

    Args:
        basis_set (list(str)): Basis set to be used for Solovay-Kitaev decomposition build using
            following terms, ``['X', 'Y', 'Z', 'H', 'T', 'T*', 'S', 'S*']``, where `*` refers
            to the gate adjoint. Default is ``["T", "T*", "H"]``
        basis_length (int): Maximum expansion length of Clifford+T sequences in the approximation set. Default is `10`

    Returns:
        Tuple(list[list[~pennylane.operation.Operation]], list[TensorLike], list[TensorLike]): A tuple containing the list of
        Clifford+T sequences that will be used for approximating a matrix in the base case of recursive implementation of
        Solovay-Kitaev algorithm, with their corresponding SU(2) and quaternion representations.

    .. seealso:: :func:`~.sk_decomposition` for performing Solovay-Kitaev decomposition.
    """
    # Initial gate to begin with - Identity
    gate = _CLIFFORD_T_BASIS["I"]

    # Maintains a trie-like structure for each depth
    gtrie_ids = [[[gate]]]
    gtrie_mat = [[_SU2_transform(gate.matrix())[0]]]

    # Maintains the approximate set for gates' names, SU(2)s and quaternions
    approx_set_ids = [gtrie_ids[0][0]]
    approx_set_mat = [gtrie_mat[0][0]]
    approx_set_qat = [_quaternion_transform(gtrie_mat[0][0])]

    # Maintains basis gates and their SU(2)s
    basis = [_CLIFFORD_T_BASIS[gate.upper()] for gate in basis_gates]
    basis_su2 = {op: _SU2_transform(op.matrix())[0] for op in basis}

    # We will perform a breadth-first search (BFS) style set building for the set
    for depth in range(max_length):
        # Add the containers for next depth while we explore the current
        gtrie_id, gtrie_mt = [], []
        for node, su2m in zip(gtrie_ids[depth], gtrie_mat[depth]):
            # Get the last operation in the current node
            last_op = qml.adjoint(node[-1], lazy=False) if node else None

            # Now attempt extending the current node for each basis gate
            for op in basis:
                # If basis gate is adjoint of last op in the node, skip.
                if qml.equal(op, last_op):
                    continue

                # Extend and check if the node already exists in the approximate set.
                su2_op = basis_su2[op] @ su2m
                exists, quaternion = _contains_SU2(su2_op, approx_set_qat)
                if not exists:
                    # Add to the approximate set while removing Identity
                    approx_set_ids.append(node[1:] + [op])
                    approx_set_mat.append(su2_op)
                    approx_set_qat.append(quaternion)

                    # Add to the containers for next depth
                    gtrie_id.append(node + [op])
                    gtrie_mt.append(su2_op)

        # Add to the next depth for next iteration
        gtrie_ids.append(gtrie_id)
        gtrie_mat.append(gtrie_mt)

    return approx_set_ids, approx_set_mat, approx_set_qat


def _group_commutator_decompose(matrix, tol=1e-5):
    r"""Performs a group commutator decomposition :math:`U = V' \times W' \times V'^{\dagger}. \times W'^{\dagger}`
    as given in the Section 4.1 of `arXiv:0505030 <https://arxiv.org/abs/quant-ph/0505030>`_."""
    # Use the quaternion form to get the rotation axis and angle on the Bloch sphere.
    quaternion = _quaternion_transform(matrix)
    theta, axis = 2 * qml.math.arccos(quaternion[0]), quaternion[1:]

    # Early return for the case where matrix is I or -I, where I is Identity
    if qml.math.allclose(axis, 0.0, atol=tol) and qml.math.isclose(theta % math.pi, 0.0, atol=tol):
        return qml.math.eye(2, dtype=complex), qml.math.eye(2, dtype=complex)

    # The angle phi comes from the Eq. 10 in the Solovay-Kitaev algorithm paper (arXiv:0505030)
    phi = 2.0 * qml.math.arcsin(qml.math.sqrt(qml.math.sqrt((0.5 - 0.5 * qml.math.cos(theta / 2)))))

    # Begin decomposition by computing the rotation operations V and W
    v = qml.RX(phi, [0])
    w = qml.RY(2 * math.pi - phi, [0]) if axis[2] > 0 else qml.RY(phi, [0])

    # Get the similarity transormation matrices S and S.dag
    ud = qml.math.linalg.eig(matrix)[1]
    vwd = qml.math.linalg.eig(qml.matrix(v @ w @ v.adjoint() @ w.adjoint()))[1]
    s = ud @ qml.math.conj(qml.math.transpose(vwd))
    sdg = vwd @ qml.math.conj(qml.math.transpose(ud))

    # Get the required matrices V' and W'
    v_hat = s @ v.matrix() @ sdg
    w_hat = s @ w.matrix() @ sdg

    return w_hat, v_hat


def sk_decomposition(op, depth, basis_set=(), basis_length=10):
    r"""Approximate an arbitrary single-qubit gate in the Clifford+T basis using the `Solovay-Kitaev algorithm <https://arxiv.org/abs/quant-ph/0505030>`_.

    This method implements a recursive Solovay-Kitaev decomposition that approximates any :math:`U \in \text{SU}(2)`
    operation with :math:`\epsilon > 0` error up to a global phase. Increasing the recursion ``depth`` should
    reduce this error. In general, this algorithm runs in :math:`O(\text{log}^{2.71}(1/\epsilon))` time and produces
    a decomposition with :math:`O(\text{log}^{3.97}(1/\epsilon))` operations.

    Args:
        op (~pennylane.operation.Operation): A single-qubit gate operation
        depth (int): Depth until which the recursion occurs
        basis_set (list(str)): Basis set to be used for the decomposition and building an approximate set internally.
            It accepts the following gate terms: ``['X', 'Y', 'Z', 'H', 'T', 'T*', 'S', 'S*']``, where `*` refers
            to the gate adjoint. Default value is ``['T', 'T*', 'H']``
        basis_length (int): Maximum expansion length of Clifford+T sequences in the internally built approximate set.
            Default is `10`

    Returns:
        list(~pennylane.operation.Operation): A list of gates in the Clifford+T basis set that approximates the given operation

    Raises:
        ValueError: If the given operator acts on more than one wires

    **Example**

    Suppose one would like to decompose :class:`~.RZ` with rotation angle :math:`\phi = \pi/3`:

    .. code-block:: python3

        import numpy as np
        import pennylane as qml

        op  = qml.RZ(np.pi/3, wires=0)

        # Get the gate decomposition in ['T', 'T*', 'H']
        ops = qml.transforms.decompositions.sk_decomposition(op, depth=5)

        # Get SU2 matrix from the ops
        op_matrix = qml.prod(*reversed(ops)).matrix()
        su2_matrix = op_matrix / np.sqrt((1 + 0j) * np.linalg.det(op_matrix))

    When the function is run for a sufficient ``depth`` with a good enough approximate set,
    the output gate sequence should implement the same operation approximately up to a global phase.

    >>> qml.math.allclose(op.matrix(), su2_matrix, atol=1e-3)
    True

    """
    with QueuingManager.stop_recording():
        # Check for length of wires in the operation
        if len(op.wires) != 1:
            raise ValueError(
                f"Operator must be a single qubit operation, got {op} acting on {op.wires} wires."
            )

        # Build the approximate set with caching
        basis_gates = (gate for gate in basis_set) if basis_set else ("T", "T*", "H")
        approx_set_ids, approx_set_mat, approx_set_qat = _approximate_set(
            basis_gates, max_length=basis_length
        )

        # Build the k-d tree with the current approximation set for querying in the base case
        kd_tree = sp.spatial.KDTree(qml.math.array(approx_set_qat))

        # Obtain the SU(2) and quaternion for the operation
        gate_mat, _ = _SU2_transform(op.matrix())
        gate_qat = _quaternion_transform(gate_mat)

        def _solovay_kitaev(umat, n):
            """Recursive method as given in the Section 3 of arXiv:0505030"""

            if not n:
                # Check the approximate gate in our approximate set
                seq_node = qml.math.array([_quaternion_transform(umat)])
                _, index = kd_tree.query(seq_node, workers=-1)
                return approx_set_ids[index[0]], approx_set_mat[index[0]]

            # get the approximation for the given unitary: U --> U'
            u_n1_ids, u_n1_mat = _solovay_kitaev(umat, n - 1)

            # get the decomposition for the remaining unitary: U @ U'.dag()
            v_n, w_n = _group_commutator_decompose(
                umat @ qml.math.conj(qml.math.transpose(u_n1_mat))
            )

            # get the approximation for the remaining unitaries: V, W --> V', W'
            v_n1_ids, v_n1_mat = _solovay_kitaev(v_n, n - 1)
            w_n1_ids, w_n1_mat = _solovay_kitaev(w_n, n - 1)

            # get the adjoints for return: V', W' --> V'.dag(), W'.dag()
            v_n1_ids_adj = [qml.adjoint(gate, lazy=False) for gate in reversed(v_n1_ids)]
            v_n1_mat_adj = qml.math.conj(qml.math.transpose(v_n1_mat))
            w_n1_ids_adj = [qml.adjoint(gate, lazy=False) for gate in reversed(w_n1_ids)]
            w_n1_mat_adj = qml.math.conj(qml.math.transpose(w_n1_mat))

            # build the operations and their SU(2) for return
            approx_ids = u_n1_ids + w_n1_ids_adj + v_n1_ids_adj + w_n1_ids + v_n1_ids
            approx_mat = v_n1_mat @ w_n1_mat @ v_n1_mat_adj @ w_n1_mat_adj @ u_n1_mat

            return approx_ids, approx_mat

        # If we have it already, use that, otherwise proceed for deocomposition
        dist, index = kd_tree.query(qml.math.array([gate_qat]), workers=-1)
        if qml.math.isclose(dist, 0.0, atol=1e-8):
            decomposition = approx_set_ids[index[0]]
        else:
            decomposition, _ = _solovay_kitaev(gate_mat, depth)

        # Remove inverses if any in the decomposition and handle trivial case
        [new_tape], _ = cancel_inverses(QuantumScript(decomposition or [qml.Identity(0)]))

    # Map the wires to that of the operation and queue
    [map_tape], _ = qml.map_wires(new_tape, wire_map={0: op.wires[0]}, queue=True)

    # Return the gates from the mapped tape
    return map_tape.operations
