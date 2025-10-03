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
from functools import lru_cache

import scipy as sp

import pennylane as qml
from pennylane.queuing import QueuingManager


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

    # We put the phase first in [0, \pi] and then convert it to [0, \pi)
    gphase = qml.math.mod(qml.math.angle(factor), 2 * math.pi) / 2
    rphase = (-1) ** qml.math.isclose(gphase, math.pi)

    # Get the final matrix form using the phase information
    s2_mat = rphase * matrix * qml.math.exp(-1j * qml.math.cast_like(gphase, 1j))
    return s2_mat, gphase if rphase == 1 else 0.0


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


def _contains_SU2(op_mat, ops_vecs=None, kd_tree=None, tol=1e-8):
    r"""Checks if a given SU(2) matrix is contained in a list of quaternions for a given tolerance.

    Args:
        op_mat (TensorLike): SU(2) matrix for the operation to be searched
        op_vecs (list(TensorLike)): list of quaternion for the operations that makes the search space.
        kd_tree (scipy.spatial.KDTree): kd-tree built from the list of quaternions. Default is ``None``.
        tol (float): tolerance for the match to be considered ``True``.

    Returns:
        tuple(bool, TensorLike, int): A tuple including ``True``/``False`` for whether an operation similar to the
        given operations was found, the quaternion representation of the searched operations, and its index in
        the ``op_vecs`` or ``kd_tree``.
    """
    gate_points = qml.math.array([_quaternion_transform(op_mat)])

    tree = kd_tree or sp.spatial.KDTree(qml.math.array(ops_vecs))
    dist, indx = tree.query(gate_points, workers=-1)

    return (dist[0] < tol, gate_points[0], indx[0])


def _prune_approximate_set(
    approx_set_ids, approx_set_mat, approx_set_gph, approx_set_qat, approx_set_sum
):
    """Prune the approximate set for equivalent gate sequences with higher T-gate counts.

    Args:
        approx_set_ids (list[list[~pennylane.operation.Operation]]): list of gate sequences
        approx_set_mat (list[TensorLike]): list of SU(2) matrices
        approx_set_gph (list[float]): list of global phases
        approx_set_qat (list[TensorLike]): list of quaternion representations
        approx_set_sum (list[int]): list of numbers of the ``T`` and ``Adjoint(T)`` gates in the sequences

    Returns:
        tuple[list[list[~pennylane.operation.Operation]], list[TensorLike], list[float], list[TensorLike]]:
        A tuple containing the pruned approximate set.
    """
    if approx_set_qat:
        tree, tsum = sp.spatial.KDTree(approx_set_qat), qml.math.array(approx_set_sum)
        dists, indxs = tree.query(approx_set_qat, workers=-1, k=10)

        prune_ixs = []
        for dist, indx in zip(dists, indxs):
            eq_idx = qml.math.sort(indx[qml.math.where(dist.round(8) == 0.0)])
            prune_ixs.extend(eq_idx[qml.math.argsort(tsum[eq_idx])][1:])

        for ix in sorted(set(prune_ixs), reverse=True):
            del approx_set_ids[ix]
            del approx_set_mat[ix]
            del approx_set_gph[ix]
            del approx_set_qat[ix]

    return approx_set_ids, approx_set_mat, approx_set_gph, approx_set_qat


@lru_cache
def _approximate_set(basis_gates, max_length=10):
    r"""Builds an approximate unitary set required for the `Solovay-Kitaev algorithm <https://arxiv.org/abs/quant-ph/0505030>`_.

    Args:
        basis_gates (tuple[str]): Basis set to be used for Solovay-Kitaev decomposition build using the following
        terms, ``('X', 'Y', 'Z', 'H', 'T', 'T*', 'S', 'S*')``, where ``*`` refers to the gate adjoint.
        max_length (int): Maximum expansion length of Clifford+T sequences in the approximation set. Default is ``10``

    Returns:
        tuple(list[list[~pennylane.operation.Operation]], list[TensorLike], list[float], list[TensorLike]): A tuple containing the list of
        Clifford+T sequences that will be used for approximating a matrix in the base case of recursive implementation of
        Solovay-Kitaev algorithm, with their corresponding SU(2) representations, global phases, and quaternion representations.
    """
    # Defining Clifford+T basis
    _CLIFFORD_T_BASIS = {
        "I": qml.Identity(0),
        "X": qml.X(0),
        "Y": qml.Y(0),
        "Z": qml.Z(0),
        "H": qml.Hadamard(0),
        "T": qml.T(0),
        "T*": qml.adjoint(qml.T(0)),
        "S": qml.S(0),
        "S*": qml.adjoint(qml.S(0)),
    }
    # Maintains the basis gates
    basis = [_CLIFFORD_T_BASIS[gate.upper()] for gate in basis_gates]
    t_set = {qml.T(0), qml.adjoint(qml.T(0))}

    # Get the SU(2) data for the gates in basis set
    basis_mat, basis_gph = {}, {}
    for gate in basis:
        su2_mat, su2_gph = _SU2_transform(gate.matrix())
        basis_mat.update({gate: su2_mat})
        basis_gph.update({gate: su2_gph})

    # Maintain a trie-like structure that consists of -
    # gtrie_<ids / mat / gph / sum> stores <gates / SU2s / global phase / T-gate sum>
    # each of them are list of lists, where each inner list stores the data at a depth D.
    gtrie_ids = [[[gate] for gate in basis]]
    gtrie_mat = [list(basis_mat.values())]
    gtrie_gph = [list(basis_gph.values())]
    gtrie_sum = [[int(gate in t_set) for gate in basis]]

    # Maintains the approximate set for gates, SU2s, global phases, T-gate sums and quaternions,
    # where each of the approx_set_<name> is the corresponding flattened verison of gtrie_<name>.
    # We store the quaternion representations for the SU2 matrices to build a KDTree. This allows us to
    # query for nearest neighbours of any newly built gate sequence and test for its prior existence.
    approx_set_ids = list(gtrie_ids[0])
    approx_set_mat = list(gtrie_mat[0])
    approx_set_gph = list(gtrie_gph[0])
    approx_set_sum = list(gtrie_sum[0])
    approx_set_qat = [_quaternion_transform(mat) for mat in approx_set_mat]

    # We will perform a breadth-first search (BFS)-style trie building, starting from basis gates:
    # We attempt to extend every gate sequence at previous depth (defined by a node) with all
    # basis gates. We add the extended sequence and its corresponding data to the next depth by
    # comparing its quaternion representation with the gate sequences already added to the trie.
    for depth in range(max_length - 1):
        # Build a KDTree for the quaternions stored up to the current depth for querying.
        kdtree = sp.spatial.KDTree(qml.math.array(approx_set_qat))

        # Add the local containers for extending the trie to the next depth while traversing current one.
        ltrie_id, ltrie_mt, ltrie_gp, ltrie_sm, ltrie_qt = [], [], [], [], []
        for node, su2m, gphase, tgsum in zip(
            gtrie_ids[depth], gtrie_mat[depth], gtrie_gph[depth], gtrie_sum[depth]
        ):
            # Get the last operation in the current node
            last_op = qml.adjoint(node[-1], lazy=False) if node else None

            # Now attempt extending the current node with each gate in the basis set.
            for op in basis:
                # If the op is the adjoint of last operation in the node, skip.
                if qml.equal(op, last_op):
                    continue

                # Extend and check if the node already exists in the approximate set in two steps:
                # 1. (local check) => within the gate sequences built in the current iteration.
                # 2. (global check) => within the gate sequences built in the previous iterations.
                su2_gp = basis_gph[op] + gphase
                su2_op = (-1.0) ** bool(su2_gp >= math.pi) * (basis_mat[op] @ su2m)

                exists, quaternion, global_index, local_index = False, None, -1, -1
                if ltrie_qt:  # local check
                    exists, quaternion, local_index = _contains_SU2(su2_op, ops_vecs=ltrie_qt)

                if exists:  # get the global index from the local index
                    global_index = local_index + len(approx_set_qat) - len(ltrie_qt)
                else:  # global check
                    exists, quaternion, global_index = _contains_SU2(su2_op, kd_tree=kdtree)

                # Add the sequence if it is unique, i.e., new SU(2) representation or global phase.
                global_phase = qml.math.mod(su2_gp, math.pi)  # Get the global phase in [0, \pi)
                if not exists or global_phase != approx_set_gph[global_index]:
                    # Add the data to the approximate set
                    approx_set_ids.append(node + [op])
                    approx_set_mat.append(su2_op)
                    approx_set_gph.append(global_phase)
                    approx_set_qat.append(quaternion)

                    # Add the data to the containers for next depth
                    ltrie_id.append(node + [op])
                    ltrie_mt.append(su2_op)
                    ltrie_gp.append(global_phase)
                    ltrie_qt.append(quaternion)

                    # Add the T gate sum data
                    tbool = int(op in t_set)
                    approx_set_sum.append(tgsum + tbool)
                    ltrie_sm.append(tgsum + tbool)

        # Add to the next depth for new iteration
        gtrie_ids.append(ltrie_id)
        gtrie_mat.append(ltrie_mt)
        gtrie_gph.append(ltrie_gp)
        gtrie_sum.append(ltrie_sm)

    # Prune the approximate set for equivalent operations with higher T-gate counts and return
    return _prune_approximate_set(
        approx_set_ids, approx_set_mat, approx_set_gph, approx_set_qat, approx_set_sum
    )


def _group_commutator_decompose(matrix, tol=1e-5):
    r"""Performs a group commutator decomposition :math:`U = V' \times W' \times V'^{\dagger} \times W'^{\dagger}`
    as given in the Section 4.1 of `arXiv:0505030 <https://arxiv.org/abs/quant-ph/0505030>`_."""
    # Use the quaternion form to get the rotation axis and angle on the Bloch sphere,
    # while using clipping for dealing with floating point precision errors.
    quaternion = _quaternion_transform(matrix)
    theta, axis = 2 * qml.math.arccos(qml.math.clip(quaternion[0], -1.0, 1.0)), quaternion[1:]

    # Early return for the case where matrix is I or -I, where I is Identity.
    if qml.math.allclose(axis, 0.0, atol=tol) and qml.math.isclose(theta % math.pi, 0.0, atol=tol):
        return qml.math.eye(2, dtype=complex), qml.math.eye(2, dtype=complex)

    # The angle phi comes from the Eq. 10 in the Solovay-Kitaev algorithm paper (arXiv:0505030).
    phi = 2.0 * qml.math.arcsin(qml.math.sqrt(qml.math.sqrt(0.5 - 0.5 * qml.math.cos(theta / 2))))

    # Begin decomposition by computing the rotation operations V and W.
    v = qml.RX(phi, [0])
    w = qml.RY(2 * math.pi - phi, [0]) if axis[2] > 0 else qml.RY(phi, [0])

    # Get the similarity transormation matrices S and S.adjoint().
    ud = qml.math.linalg.eig(matrix)[1]
    vwd = qml.math.linalg.eig(qml.matrix(v @ w @ v.adjoint() @ w.adjoint()))[1]
    s = ud @ qml.math.conj(qml.math.transpose(vwd))
    sdg = vwd @ qml.math.conj(qml.math.transpose(ud))

    # Get the required matrices V' and W'.
    v_hat = s @ v.matrix() @ sdg
    w_hat = s @ w.matrix() @ sdg

    return w_hat, v_hat


def sk_decomposition(op, epsilon, *, max_depth=5, basis_set=("H", "S", "T"), basis_length=10):
    r"""Approximate an arbitrary single-qubit gate in the Clifford+T basis using the `Solovay-Kitaev algorithm <https://arxiv.org/abs/quant-ph/0505030>`_.

    This method implements the Solovay-Kitaev decomposition algorithm that approximates any single-qubit
    operation with :math:`\epsilon > 0` error. The procedure exits when the approximation error
    becomes less than :math:`\epsilon`, or when ``max_depth`` approximation passes have been made. In the
    latter case, the approximation error could be :math:`\geq \epsilon`.

    This algorithm produces a decomposition with :math:`O(\text{log}^{3.97}(1/\epsilon))` operations.

    Args:
        op (~pennylane.operation.Operation): A single-qubit gate operation.
        epsilon (float): The maximum permissible error.

    Keyword Args:
        max_depth (int): The maximum number of approximation passes. A smaller :math:`\epsilon` would generally require
            a greater number of passes. Default is ``5``.
        basis_set (tuple[str]): Basis set to be used for the decomposition and building an approximate set internally.
            It accepts the following gate terms: ``('X', 'Y', 'Z', 'H', 'T', 'T*', 'S', 'S*')``, where ``*`` refers
            to the gate adjoint. Default value is ``('H', 'S', 'T')``.
        basis_length (int): Maximum expansion length of Clifford+T sequences in the internally-built approximate set.
            Default is ``10``.

    Returns:
        list[~pennylane.operation.Operation]: A list of gates in the Clifford+T basis set that approximates the given
        operation along with a final global phase operation. The operations are in the circuit-order.

    Raises:
        ValueError: If the given operator acts on more than one wires.

    **Example**

    Suppose one would like to decompose :class:`~.RZ` with rotation angle :math:`\phi = \pi/3`:

    .. code-block:: python

        op  = qml.RZ(np.pi/3, wires=0)

        # Get the gate decomposition in ['H', 'S', 'T']
        ops = qml.ops.sk_decomposition(op, epsilon=1e-3)

        # Get the approximate matrix from the ops
        matrix_sk = qml.prod(*reversed(ops)).matrix()

    When the function is run for a sufficient ``depth`` with a good enough approximate set,
    the output gate sequence should implement the same operation approximately.

    >>> qml.math.allclose(op.matrix(), matrix_sk, atol=1e-3)
    True

    """
    # Check for length of wires in the operation
    if len(op.wires) != 1:
        raise ValueError(
            f"Operator must be a single qubit operation, got {op} acting on {op.wires} wires."
        )

    with QueuingManager.stop_recording():
        # Build the approximate set with caching
        approx_set_ids, approx_set_mat, approx_set_gph, approx_set_qat = _approximate_set(
            tuple(basis_set), max_length=basis_length
        )

        # Build the k-d tree with the current approximation set for querying in the base case
        kd_tree = sp.spatial.KDTree(qml.math.array(approx_set_qat))

        # Obtain the SU(2) and quaternion for the operation
        op_matrix = op.matrix()
        interface = qml.math.get_deep_interface(op_matrix)
        gate_mat, gate_gph = _SU2_transform(qml.math.unwrap(op_matrix))
        gate_qat = _quaternion_transform(gate_mat)

        def _solovay_kitaev(umat, n, u_n1_ids, u_n1_mat):
            """Recursive method as given in the Section 3 of arXiv:0505030"""

            if not n:
                # Check the approximate gate in our approximate set
                seq_node = qml.math.array([_quaternion_transform(umat)])
                _, [index] = kd_tree.query(seq_node, workers=-1)
                return approx_set_ids[index], approx_set_mat[index]

            # Get the decomposition for the remaining unitary: U @ U'.adjoint()
            v_n, w_n = _group_commutator_decompose(
                umat @ qml.math.conj(qml.math.transpose(u_n1_mat))
            )

            # Get the approximation for the residual commutator unitaries: V and W
            c_ids_mats = []
            for c_n in [v_n, w_n]:
                # Get the approximation for each commutator iteratively: C --> C'
                c_n1_ids, c_n1_mat = None, None
                for i in range(n):
                    c_n1_ids, c_n1_mat = _solovay_kitaev(c_n, i, c_n1_ids, c_n1_mat)

                # Get the adjoints C' --> C'.adjoint()
                c_n1_ids_adj = [qml.adjoint(gate, lazy=False) for gate in reversed(c_n1_ids)]
                c_n1_mat_adj = qml.math.conj(qml.math.transpose(c_n1_mat))

                # Store the decompositions and matrices for C'
                c_ids_mats.append([c_n1_ids, c_n1_mat, c_n1_ids_adj, c_n1_mat_adj])

            # Get the V' and W'
            v_n1_ids, v_n1_mat, v_n1_ids_adj, v_n1_mat_adj = c_ids_mats[0]
            w_n1_ids, w_n1_mat, w_n1_ids_adj, w_n1_mat_adj = c_ids_mats[1]

            # Build the operations and their SU(2) for return
            approx_ids = u_n1_ids + w_n1_ids_adj + v_n1_ids_adj + w_n1_ids + v_n1_ids
            approx_mat = v_n1_mat @ w_n1_mat @ v_n1_mat_adj @ w_n1_mat_adj @ u_n1_mat

            return approx_ids, approx_mat

        # If we have it already, use that, otherwise proceed for decomposition
        _, [index] = kd_tree.query(qml.math.array([gate_qat]), workers=-1)
        decomposition, u_prime = approx_set_ids[index], approx_set_mat[index]

        # Iterate until max_depth while doing an epsilon-error comparision
        for depth in range(max_depth):
            # For a SU(2) matrix Hilbert-Schmidt norm is √(|α|^2 + |β|^2),
            # which is simply the L2-norm for the first row of the matrix.
            if qml.math.norm(gate_mat[0] - u_prime[0]) <= epsilon:
                break
            # Approximate the residual with the approximation from previous iteration
            decomposition, u_prime = _solovay_kitaev(gate_mat, depth + 1, decomposition, u_prime)

        # Remove inverses if any in the decomposition and handle trivial case
        [new_tape], _ = qml.transforms.cancel_inverses(
            qml.tape.QuantumScript(decomposition or [qml.Identity(0)])
        )

    # Map the wires to that of the operation and queue
    if queuing := QueuingManager.recording():
        QueuingManager.remove(op)

    if (op_wire := op.wires[0]) != 0:
        [new_tape], _ = qml.map_wires(new_tape, wire_map={0: op_wire}, queue=True)
    else:
        if queuing:
            _ = [qml.apply(op) for op in new_tape.operations]

    # Get phase information based on the decomposition effort
    phase = approx_set_gph[index] - gate_gph
    global_phase = qml.GlobalPhase(qml.math.array(phase, like=interface))

    # Return the gates from the mapped tape and global phase
    return new_tape.operations + [global_phase]
