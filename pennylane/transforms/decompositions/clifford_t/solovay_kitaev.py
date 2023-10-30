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
import scipy as sp
import pennylane as qml

from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumTape
from pennylane.transforms.optimization import (
    cancel_inverses,
)

# Defining Clifford+T basis
_CLIFFORD_T_BASIS = {
    "i": qml.Identity(0),
    "x": qml.PauliX(0),
    "y": qml.PauliY(0),
    "z": qml.PauliZ(0),
    "h": qml.Hadamard(0),
    "t": qml.T(0),
    "tdg": qml.adjoint(qml.T(0)),
    "s": qml.S(0),
    "sdg": qml.adjoint(qml.S(0)),
}


class GateSet:
    """Implements a gate set for storing a sequence of gates and their SU(2) and SO(3) representations
    for the approximation step in the `Solovay-Kitaev algorithm <https://arxiv.org/abs/quant-ph/0505030>`_.

    Args:
        operations (Iterable): list of Clifford gates
    """

    def __init__(self, operations=None):
        """Instantiate a GateSet object"""
        self.gates = list(operations) if operations is not None else []
        self.labels = [op.name for op in self.gates]

        self.matrix = qml.prod(*(self.gates or [qml.Identity(0)])).matrix()
        self.su2_matrix, self.global_phase = self.get_SU2_matrix(self.matrix)
        self.so3_matrix = self.get_SO3_matrix(self.su2_matrix)

    @property
    def name(self):
        """Return name property for GateSet object"""
        return "".join(self.labels)

    def __len__(self):
        """Returns length of gates in GateSet"""
        return len(self.gates)

    def __getitem__(self, index):
        """Returns the operation at the given ``index`` in GateSet."""
        return self.gates[index]

    def __repr__(self):
        """Returns string representation of GateSet"""

        name = "Gates: ["
        name += ", ".join(self.labels)
        name += "], Matrix: " + str(self.matrix)
        return name

    def __eq__(self, gateset):
        """Checks equivalence of two GateSet objects"""

        if len(self.gates) != len(gateset.gates):
            return False

        if self.labels != gateset.labels:
            return False

        if not qml.math.allclose(self.matrix, gateset.matrix):
            return False

        if not qml.math.allclose(self.so3_matrix, gateset.so3_matrix):
            return False

        return True

    @staticmethod
    def get_SU2_matrix(matrix):
        """Performs a U(2) to SU(2) transformation via a global phase addition."""

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            factor = qml.math.sqrt((1 + 0j) * qml.math.linalg.det(matrix)) ** -1
            gphase = qml.math.arctan2(qml.math.imag(factor), qml.math.real(factor))
            s2_mat = factor * matrix
            return s2_mat, gphase

    @staticmethod
    def get_SO3_matrix(matrix):
        """Performs a SU(2) to SO(3) transformation."""

        a = qml.math.real(matrix[0, 0])
        b = qml.math.imag(matrix[0, 0])
        c = -qml.math.real(matrix[0, 1])
        d = -qml.math.imag(matrix[0, 1])

        rotation = qml.math.array(
            [
                [a**2 - b**2 - c**2 + d**2, 2 * a * b + 2 * c * d, -2 * a * c + 2 * b * d],
                [-2 * a * b + 2 * c * d, a**2 - b**2 + c**2 - d**2, 2 * a * d + 2 * b * c],
                [2 * a * c + 2 * b * d, 2 * b * c - 2 * a * d, a**2 + b**2 - c**2 - d**2],
            ],
            dtype=float,
        )

        return rotation

    @classmethod
    def from_matrix(cls, matrix):
        """Initialize the GateSet from a matrix"""
        gateset = cls()
        if matrix.shape == (2, 2):
            gateset.matrix = matrix
            gateset.su2_matrix, gateset.global_phase = gateset.get_SU2_matrix(matrix)
            gateset.so3_matrix = gateset.get_SO3_matrix(gateset.su2_matrix)
        else:
            raise ValueError(f"Matrix should be of shape (2, 2), got {matrix.shape}.")

        return gateset

    def append(self, operation):
        """Add a new operation to GateSet object"""

        self.gates.append(operation)
        self.labels.append(operation.name)

        self.matrix = self.matrix @ qml.matrix(operation)
        self.su2_matrix, self.global_phase = self.get_SU2_matrix(self.matrix)
        self.so3_matrix = self.get_SO3_matrix(self.su2_matrix)

        return self

    def copy(self):
        """Copy a GateSet object"""

        new_gateset = GateSet()
        new_gateset.gates = self.gates.copy()
        new_gateset.labels = self.labels.copy()
        new_gateset.matrix = self.matrix.copy()
        new_gateset.su2_matrix = self.su2_matrix.copy()
        new_gateset.global_phase = self.global_phase
        new_gateset.so3_matrix = self.so3_matrix.copy()
        return new_gateset

    def adjoint(self):
        """Compute the adjoint of the GateSet object"""

        adjoint = GateSet()
        adjoint.gates = [qml.adjoint(gate, lazy=False) for gate in reversed(self.gates)]
        adjoint.labels = [op.name for op in adjoint.gates]
        adjoint.matrix = qml.math.conj(qml.math.transpose(self.matrix))
        adjoint.su2_matrix = qml.math.conj(qml.math.transpose(self.su2_matrix))
        adjoint.so3_matrix = qml.math.conj(qml.math.transpose(self.so3_matrix))
        adjoint.global_phase = -self.global_phase
        return adjoint

    def dot(self, gateset):
        """Compute the dot-product with another GateSet."""

        dotset = GateSet()
        dotset.gates = self.gates + gateset.gates
        dotset.labels = self.labels + gateset.labels
        dotset.matrix = qml.math.dot(self.matrix, gateset.matrix)
        dotset.su2_matrix = qml.math.dot(self.su2_matrix, gateset.su2_matrix)
        dotset.so3_matrix = qml.math.dot(self.so3_matrix, gateset.so3_matrix)
        dotset.global_phase = gateset.global_phase + self.global_phase
        return dotset


class TreeSet:
    """Implements a tree structure using GateSet as nodes for the approximation step
    in the `Solovay-Kitaev algorithm <https://arxiv.org/abs/quant-ph/0505030>`_.

    Args:
        operations (GateSet): GateSet for the current node
        basis (list(str)): list of names of basis set for the approximation
        children (TreeSet): TreeSet for the child node
    """

    def __init__(self, operations, basis_gate, children):
        """Initializing TreeSet object"""
        self.operations = operations
        self.basis_gate = set(gate.lower() for gate in basis_gate)
        self.children = children

    def basic_approximation(self, depth=10):
        """Generate list of GateSet by unrolling the basis"""

        tree = TreeSet(GateSet(), self.basis_gate, [])
        curr = [tree]
        seqs = [tree.operations]

        for _ in range(depth):
            _next = []
            for node in curr:
                _next.extend(self.process_node(node, seqs))
            curr = _next

        return seqs

    def process_node(self, node, seqs):
        """Adds nodes to the tree structure"""

        last_op = (
            qml.adjoint(node.operations.gates[-1], lazy=False) if node.operations.gates else None
        )

        for label in self.basis_gate:
            op = _CLIFFORD_T_BASIS[label]
            if qml.equal(op, last_op):
                continue

            ops = node.operations.copy()
            ops.append(op)

            if self.check_nodes(ops, seqs):
                seqs.append(ops)
                node.children.append(TreeSet(ops, self.basis_gate, []))

        return node.children

    @staticmethod
    def check_nodes(seqs1, seqs2, tol=1e-10):
        """Check if there's an existing GateSet (node) that implements the same unitary
        matrix in the given sequence up to ``tol``."""

        if any(seqs1.labels == seq.labels for seq in seqs2):
            return False

        # using SO(3) instead of SU(2) since KDTrees doesn't support complex datatype
        node_points = qml.math.array([qml.math.flatten(seq.so3_matrix) for seq in seqs2])
        seq1_points = qml.math.array([qml.math.flatten(seqs1.so3_matrix)])

        tree = sp.spatial.KDTree(node_points)
        dist = tree.query(seq1_points, workers=-1)[0][0]

        return dist > tol


def _unitary_bloch(mat, eps=1e-10):
    """Computes angle and axis for any given single-qubit unitary matrix in the Bloch sphere."""

    angle = qml.math.real(qml.math.arccos(qml.math.trace(mat) / 2))
    sine = qml.math.sin(angle)

    if sine < eps:
        angle = 0.0  # put back even multiple of pi in the domain [0, 2*pi]
        axis = [0, 0, 1]
    else:
        axis = [
            -qml.math.imag((mat[1, 0] + mat[0, 1]) / (2 * sine)),  # nx
            qml.math.real((mat[1, 0] - mat[0, 1]) / (2 * sine)),  # ny
            qml.math.imag((mat[1, 1] - mat[0, 0]) / (2 * sine)),  # nz
        ]

    return axis, 2 * angle


def _group_commutator_decompose(mat):
    r"""Performs a group commutator decomposition :math:`U = V' \times W' \times V'^{\dagger}. \times W'^{\dagger}`
    as given in the Section 4.1 of `arXiv:0505030 <https://arxiv.org/abs/quant-ph/0505030>`_."""
    # Get axis and angle theta for the operator.
    axis, theta = _unitary_bloch(mat)

    # The angle phi comes from the Eq. 10 in the Solovay-Kitaev algorithm paper (arXiv:0505030)
    phi = 2.0 * qml.math.arcsin(qml.math.sqrt(qml.math.sqrt((0.5 - 0.5 * qml.math.cos(theta / 2)))))

    # Begin decomposition by computing the rotation matrices
    v = qml.RX(phi, [0])
    w = qml.RY(2 * math.pi - phi, [0]) if axis[2] > 0 else qml.RY(phi, [0])

    # Early return for the case where matrix is I or -I
    if qml.math.isclose(theta, 0.0) and qml.math.allclose(axis, [0, 0, 1]):
        return qml.math.eye(2, dtype=complex), qml.math.eye(2, dtype=complex)

    # Get similarity transormation matrix S and S.dag
    ud = qml.math.linalg.eig(mat)[1]
    vwd = qml.math.linalg.eig(qml.matrix(v @ w @ v.adjoint() @ w.adjoint()))[1]
    s = ud @ qml.math.conj(qml.math.transpose(vwd))
    sdg = vwd @ qml.math.conj(qml.math.transpose(ud))

    # Get the required matrices V' and W'
    v_hat = s @ v.matrix() @ sdg
    w_hat = s @ w.matrix() @ sdg

    return w_hat, v_hat


def _approximate_umat(seqs, basic_approximations, KDTree=None):
    """Approximates a given GateSet using the TreeSet structure"""

    if KDTree is None:
        # Use the built-in min function for comparision - should be at least O(n^2),
        # where `n` is number of GateSet in the `basic_approximations``
        def key(x):
            return qml.math.linalg.norm(qml.math.subtract(x.so3_matrix, seqs.so3_matrix))

        return min(basic_approximations, key=key)

    # Make use of the KD-Tree - should be at least O(log n),
    # where `n` is number of GateSet in the `basic_approximations``
    seq_node = qml.math.array([qml.math.flatten(seqs.so3_matrix)])
    _, index = KDTree.query(seq_node, workers=-1)

    return basic_approximations[index[0]]


def sk_approximate_set(basis_set=(), basis_depth=10, kd_tree=False):
    r"""Builds an approximate unitary set required for the `Solovay-Kitaev algorithm <https://arxiv.org/abs/quant-ph/0505030>`_.

    Args:
        basis_set (list(str)): Basis set to be used for Solovay-Kitaev decomposition build using
            following terms, ``['x', 'y', 'z', 'h', 't', 'tdg', 's', 'sdg']``, where `dg` refers
            to the gate adjoint. Default is ``["t", "tdg", "h"]``
        basis_depth (int): Maximum expansion length of Clifford+T sequences in the approximation set. Default is `10`
        kd_tree (bool): Return a KD-Tree corresponding to the gates in the approximated set. Default is ``False``

    Returns:
        list or tuple(list, scipy.spatial.KDTree): A list of Clifford+T sequences that will be used for approximating
        a matrix in the base case of recursive implementation of Solovay-Kitaev algorithm. If the ``kd_tree`` argument
        is ``True``, it also returns a KD-Tree built based on this list

    .. seealso:: :func:`~.sk_decomposition` for performing Solovay-Kitaev decomposition.
    """
    # Define a default basis set if not provided
    if not basis_set:
        basis_set = ["t", "tdg", "h"]

    # Build an approximate set using the TreeSet object
    approximate_set = TreeSet(GateSet(), basis_set, ()).basic_approximation(depth=basis_depth)
    if not kd_tree:
        return approximate_set

    # Build the KD-Tree for the approximate set (if requested)
    gnodes = qml.math.array([qml.math.flatten(seq.so3_matrix) for seq in approximate_set])
    return (approximate_set, sp.spatial.KDTree(gnodes))


# pylint: disable=too-many-arguments
def sk_decomposition(op, depth, basis_set=(), basis_depth=10, approximate_set=None, kd_tree=None):
    r"""Approximate an arbitrary single-qubit gate in the Clifford+T basis using the `Solovay-Kitaev algorithm <https://arxiv.org/abs/quant-ph/0505030>`_.

    This method implements a recursive Solovay-Kitaev decomposition that approximates any :math:`U \in \text{SU}(2)`
    operation with :math:`\epsilon > 0` error. The error depends on the recursion ``depth``. In general, this
    algorithm runs in :math:`O(\text{log}^{2.71}(1/\epsilon))` time and produces a decomposition with
    :math:`O(\text{log}^{3.97}(1/\epsilon))` operations.

    Args:
        op (~pennylane.operation.Operation): A single-qubit gate operation
        depth (int): Depth until which the recursion occurs
        basis_set (list(str)): Basis set to be used for the decomposition and building the ``approximate_set``. It
            accepts the following gate terms: ``['x', 'y', 'z', 'h', 't', 'tdg', 's', 'sdg']``, where `dg` refers
            to the gate adjoint. Default value is ``['h', 't', 'tdg']``
        basis_depth (int): Maximum expansion length of Clifford+T sequences in the ``approximate_set``. Default is `10`
        approximate_set (list): A list of gate sequences that are used to find an approximation of unitaries in the
            base case of recursion. This can be precomputed using the :func:`~.sk_approximate_set`
            method, otherwise will be built using the default/given values of ``basis_set`` and ``basis_depth``
        kd_tree (scipy.spatial.KDTree): A KD-Tree corresponding to the ``approximate_set``. This can also be
            precomputed using :func:`~.sk_approximate_set`, otherwise will be built internally.

    Returns:
        list(~pennylane.operation.Operation): A list of gates in the Clifford+T basis set that approximates the given operation

    **Example**

    Suppose one would like to decompose :class:`~.RZ` with :math:`\phi = \pi/3`:

    .. code-block:: python3

        import numpy as np
        import pennylane as qml

        op  = qml.RZ(np.pi/3, wires=0)

        # Get the gate decomposition in ['t', 'tdg', 'h']
        ops = qml.transforms.decompositions.sk_decomposition(op, depth=4)

        # Get SU2 matrix from the ops
        op_matrix = qml.prod(*ops).matrix()
        su2_matrix = op_matrix / np.sqrt((1 + 0j) * np.linalg.det(op_matrix))

    When the function is run for a sufficient ``depth`` with a good enough ``approximate_set``,
    the output gate sequence should implement the same operation approximately.

    >>> qml.math.allclose(op.matrix(), su2_matrix, atol=1e-3)
    True

    .. seealso:: :func:`~.sk_approximate_set` for precomputing the ``approximate_set`` and ``kd_tree``.
    """
    with QueuingManager.stop_recording():
        # Check for length of wires in the operation
        if len(op.wires) != 1:
            raise ValueError(
                f"Operator must be a single qubit operation, got {op} acting on {op.wires} wires."
            )

        # Check if we need to build the approximation set manually
        if (
            approximate_set is None
            or basis_depth != 10
            or (basis_set and sorted(basis_set) != ["h", "t", "tdg"])
        ):
            # Warn the user in case we have to perform rebuilding
            if approximate_set is not None:
                warnings.warn(
                    "Ignoring provided approximate set and recomputing it for given basis_set and basis_depth",
                    UserWarning,
                )
            approximate_set, kd_tree = sk_approximate_set(basis_set, basis_depth, True)

        # Build the KDTree with the current approximation set for querying in the base case
        if kd_tree is None:
            gnodes = qml.math.array([qml.math.flatten(seq.so3_matrix) for seq in approximate_set])
            kd_tree = sp.spatial.KDTree(gnodes)

        # Recursive implementation for Solovay-Kitaev algorithm
        def _solovay_kitaev(gateset, n):
            """Recursive method as given in the Section 3 of arXiv:0505030"""

            if not n:
                return _approximate_umat(gateset, approximate_set, kd_tree)

            u_n1 = _solovay_kitaev(gateset, n - 1)
            u_n1dg = u_n1.adjoint()

            v_n, w_n = _group_commutator_decompose(gateset.dot(u_n1dg).su2_matrix)

            v_n1 = _solovay_kitaev(GateSet.from_matrix(v_n), n - 1)
            w_n1 = _solovay_kitaev(GateSet.from_matrix(w_n), n - 1)

            v_n1dg = v_n1.adjoint()
            w_n1dg = w_n1.adjoint()

            return v_n1.dot(w_n1).dot(v_n1dg).dot(w_n1dg).dot(u_n1)

        # Build a GateSet object
        gate_set_op = GateSet([op])

        # Get the decomposition for the given operation
        decomposition = _solovay_kitaev(gate_set_op, depth)

        # Remove inverses if any in the decomposition and handle trivial case
        [new_tape], _ = cancel_inverses(QuantumTape(decomposition.gates or [qml.Identity(0)]))

    # Map the wires to that of the operation and queue
    [map_tape], _ = qml.map_wires(new_tape, wire_map={0: op.wires[0]}, queue=True)

    # Return the gates from the mapped tape
    return map_tape.operations
