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
r"""
Experimental simulator plugin based on tensor network contractions
"""
import math
import cmath
import warnings
from itertools import product

import numpy as np
from numpy.linalg import eigh

try:
    import tensornetwork as tn
except ImportError as e:
    raise ImportError("default.tensor device requires TensorNetwork>=0.2")

from pennylane._device import Device

# tolerance for numerical errors
tolerance = 1e-10


# ========================================================
#  utilities
# ========================================================


def spectral_decomposition(A):
    r"""Spectral decomposition of a Hermitian matrix.

    Args:
        A (array): Hermitian matrix

    Returns:
        (vector[float], list[array[complex]]): (a, P): eigenvalues and hermitian projectors
            such that :math:`A = \sum_k a_k P_k`.
    """
    d, v = eigh(A)
    P = []
    for k in range(d.shape[0]):
        temp = v[:, k]
        P.append(np.outer(temp, temp.conj()))
    return d, P


# ========================================================
#  fixed gates
# ========================================================

I = np.eye(2)
# Pauli matrices
X = np.array([[0, 1], [1, 0]])  #: Pauli-X matrix
Y = np.array([[0, -1j], [1j, 0]])  #: Pauli-Y matrix
Z = np.array([[1, 0], [0, -1]])  #: Pauli-Z matrix

H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)  #: Hadamard gate
# Two qubit gates
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])  #: CNOT gate
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])  #: SWAP gate
CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])  #: CZ gate
S = np.array([[1, 0], [0, 1j]])  #: Phase Gate
T = np.array([[1, 0], [0, cmath.exp(1j * np.pi / 4)]])  #: T Gate
# Three qubit gates
CSWAP = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
)  #: CSWAP gate

Toffoli = np.diag([1 for i in range(8)])
Toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])

# ========================================================
#  parametrized gates
# ========================================================


def Rphi(phi):
    r"""One-qubit phase shift.

    Args:
        phi (float): phase shift angle
    Returns:
        array: unitary 2x2 phase shift matrix
    """
    return np.array([[1, 0], [0, cmath.exp(1j * phi)]])


def Rotx(theta):
    r"""One-qubit rotation about the x axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}`
    """
    return math.cos(theta / 2) * I + 1j * math.sin(-theta / 2) * X


def Roty(theta):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    return math.cos(theta / 2) * I + 1j * math.sin(-theta / 2) * Y


def Rotz(theta):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    return math.cos(theta / 2) * I + 1j * math.sin(-theta / 2) * Z


def Rot3(a, b, c):
    r"""Arbitrary one-qubit rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:
        array: unitary 2x2 rotation matrix ``rz(c) @ ry(b) @ rz(a)``
    """
    return Rotz(c) @ (Roty(b) @ Rotz(a))


def CRotx(theta):
    r"""Two-qubit controlled rotation about the x axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_x(\theta)`
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, math.cos(theta / 2), -1j * math.sin(theta / 2)],
            [0, 0, -1j * math.sin(theta / 2), math.cos(theta / 2)],
        ]
    )


def CRoty(theta):
    r"""Two-qubit controlled rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_y(\theta)`
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, math.cos(theta / 2), -math.sin(theta / 2)],
            [0, 0, math.sin(theta / 2), math.cos(theta / 2)],
        ]
    )


def CRotz(theta):
    r"""Two-qubit controlled rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_z(\theta)`
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, cmath.exp(-1j * theta / 2), 0],
            [0, 0, 0, cmath.exp(1j * theta / 2)],
        ]
    )


def CRot3(a, b, c):
    r"""Arbitrary two-qubit controlled rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R(a,b,c)`
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [
                0,
                0,
                cmath.exp(-1j * (a + c) / 2) * math.cos(b / 2),
                -cmath.exp(1j * (a - c) / 2) * math.sin(b / 2),
            ],
            [
                0,
                0,
                cmath.exp(-1j * (a - c) / 2) * math.sin(b / 2),
                cmath.exp(1j * (a + c) / 2) * math.cos(b / 2),
            ],
        ]
    )


# ========================================================
#  Arbitrary states and operators
# ========================================================


def unitary(*args):
    r"""Input validation for an arbitary unitary operation.

    Args:
        args (array): square unitary matrix

    Raises:
        ValueError: if the matrix is not unitary or square

    Returns:
        array: square unitary matrix
    """
    U = np.asarray(args[0])

    if U.shape[0] != U.shape[1]:
        raise ValueError("Operator must be a square matrix.")

    if not np.allclose(U @ U.conj().T, np.identity(U.shape[0])):
        raise ValueError("Operator must be unitary.")

    return U


def hermitian(*args):
    r"""Input validation for an arbitary Hermitian expectation.

    Args:
        args (array): square hermitian matrix

    Raises:
        ValueError: if the matrix is not Hermitian or square

    Returns:
        array: square hermitian matrix
    """
    A = np.asarray(args[0])
    if A.shape[0] != A.shape[1]:
        raise ValueError("Expectation must be a square matrix.")

    if not np.allclose(A, A.conj().T):
        raise ValueError("Expectation must be Hermitian.")

    return A


def identity(*_):
    """Identity matrix observable.

    Returns:
        array: 2x2 identity matrix
    """
    return np.identity(2)


# ========================================================
#  device
# ========================================================


class DefaultTensor(Device):
    """Experimental Tensor Network simulator device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
    """

    name = "PennyLane TensorNetwork simulator plugin"
    short_name = "default.tensor"
    pennylane_requires = "0.9"
    version = "0.9.0"
    author = "Xanadu Inc."
    _capabilities = {"model": "qubit", "tensor_observables": True}

    _operation_map = {
        "BasisState": None,
        "QubitStateVector": None,
        "QubitUnitary": unitary,
        "PauliX": X,
        "PauliY": Y,
        "PauliZ": Z,
        "Hadamard": H,
        "S": S,
        "T": T,
        "CNOT": CNOT,
        "SWAP": SWAP,
        "CSWAP": CSWAP,
        "Toffoli": Toffoli,
        "CZ": CZ,
        "PhaseShift": Rphi,
        "RX": Rotx,
        "RY": Roty,
        "RZ": Rotz,
        "Rot": Rot3,
        "CRX": CRotx,
        "CRY": CRoty,
        "CRZ": CRotz,
        "CRot": CRot3,
    }

    _observable_map = {
        "PauliX": X,
        "PauliY": Y,
        "PauliZ": Z,
        "Hadamard": H,
        "Hermitian": hermitian,
        "Identity": identity,
    }

    backend = "numpy"
    _reshape = staticmethod(np.reshape)
    _array = staticmethod(np.array)
    _asarray = staticmethod(np.asarray)
    _real = staticmethod(np.real)
    _imag = staticmethod(np.imag)
    _abs = staticmethod(np.abs)

    C_DTYPE = np.complex128
    R_DTYPE = np.float64

    def __init__(self, wires, shots=1000, analytic=True):
        super().__init__(wires, shots)
        self.analytic = True
        self._nodes = []
        self._edges = []
        self._state_node = None
        self._free_edges = []
        self.reset()

    @staticmethod
    def _create_basis_state(state, wires):
        """Helper function to create a basis state with the correct shape.

        Args:
            state (array[int]): array of 0s and 1s of size ``(wires,)`` representing
                the basis state
            wires (list[int]): the wires the basis state should
                be prepared on

        Returns:
            array[int]: state array of size ``[2]*len(wires)``
        """
        state_node = np.zeros(tuple([2] * len(wires)))
        state_node[tuple(state)] = 1
        return state_node

    def _add_node(self, A, wires, name="UnnamedNode"):
        """Adds a node to the underlying tensor network.

        The node is also added to ``self._nodes`` for bookkeeping.

        Args:
            A (array): numerical data values for the operator (i.e., matrix form)
            wires (list[int]): wires that this operator acts on
            name (str): optional name for the node

        Returns:
            tn.Node: the newly created node
        """
        name = "{}{}".format(name, tuple(w for w in wires))
        if isinstance(A, tn.Node):
            A.set_name(name)
            node = A
        else:
            node = tn.Node(A, name=name, backend=self.backend)
        self._nodes.append(node)
        return node

    def _add_edge(self, node1, idx1, node2, idx2):
        """Adds an edge to the underlying tensor network.

        The edge is also added to ``self._edges`` for bookkeeping.

        Args:
            node1 (tn.Node): first node to connect
            idx1 (int): index of node1 to add the edge to
            node2 (tn.Node): second node to connect
            idx2 (int): index of node2 to add the edge to

        Returns:
            tn.Edge: the newly created edge
        """
        edge = tn.connect(node1[idx1], node2[idx2])
        self._edges.append(edge)

        return edge

    def pre_apply(self):
        self.reset()

    def apply(self, operation, wires, par):
        if operation == "QubitStateVector":
            state = self._array(par[0], dtype=self.C_DTYPE)
            if state.ndim == 1 and state.shape[0] == 2 ** self.num_wires:
                self._state_node.tensor = self._reshape(state, [2] * self.num_wires)
            else:
                raise ValueError("State vector must be of length 2**wires.")
            if wires is not None and wires != [] and list(wires) != list(range(self.num_wires)):
                raise ValueError(
                    "The default.tensor plugin can apply QubitStateVector only to all of the {} wires.".format(
                        self.num_wires
                    )
                )
            return
        if operation == "BasisState":
            n = len(par[0])
            if n == 0 or n > self.num_wires or not set(par[0]).issubset({0, 1}):
                raise ValueError(
                    "BasisState parameter must be an array of 0 or 1 integers of length at most {}.".format(
                        self.num_wires
                    )
                )
            if wires is not None and wires != [] and list(wires) != list(range(self.num_wires)):
                raise ValueError(
                    "The default.tensor plugin can apply BasisState only to all of the {} wires.".format(
                        self.num_wires
                    )
                )
            state_node = self._create_basis_state(par[0], wires)
            self._state_node.tensor = self._asarray(state_node, dtype=self.C_DTYPE)
            return

        A = self._get_operator_matrix(operation, par)
        num_mult_idxs = len(wires)
        A = self._reshape(A, [2] * num_mult_idxs * 2)
        op_node = self._add_node(A, wires=wires, name=operation)
        for idx, w in enumerate(wires):
            self._add_edge(op_node, num_mult_idxs + idx, self._state_node, w)
            self._free_edges[w] = op_node[idx]
        # TODO: can be smarter here about collecting contractions?
        self._state_node = tn.contract_between(
            op_node, self._state_node, output_edge_order=self._free_edges
        )

    def create_nodes_from_tensors(self, tensors: list, wires: list, observable_names: list):
        """Helper function for creating tensornetwork nodes based on tensors.

        Args:
          tensors (np.ndarray, tf.Tensor, torch.Tensor): tensors of the observables
          wires (Sequence[Sequence[int]]): measured subsystems for each observable
          observable_names (Sequence[str]): name of the operation/observable

        Returns:
          list[tn.Node]: the observables as tensornetwork Nodes
        """
        return [self._add_node(A, w, name=o) for A, w, o in zip(tensors, wires, observable_names)]

    def expval(self, observable, wires, par):

        if not isinstance(observable, list):
            observable, wires, par = [observable], [wires], [par]

        tensors = []
        for o, p, w in zip(observable, par, wires):
            A = self._get_operator_matrix(o, p)
            num_mult_idxs = len(w)
            tensors.append(self._reshape(A, [2] * num_mult_idxs * 2))

        nodes = self.create_nodes_from_tensors(tensors, wires, observable)
        return self.ev(nodes, wires)

    def var(self, observable, wires, par):

        if not isinstance(observable, list):
            observable, wires, par = [observable], [wires], [par]

        matrices = [self._get_operator_matrix(o, p) for o, p in zip(observable, par)]

        tensors = [self._reshape(A, [2] * len(wires) * 2) for A, wires in zip(matrices, wires)]
        tensors_of_squared_matrices = [
            self._reshape(A @ A, [2] * len(wires) * 2) for A, wires in zip(matrices, wires)
        ]

        obs_nodes = self.create_nodes_from_tensors(tensors, wires, observable)
        obs_nodes_for_squares = self.create_nodes_from_tensors(
            tensors_of_squared_matrices, wires, observable
        )

        return self.ev(obs_nodes_for_squares, wires) - self.ev(obs_nodes, wires) ** 2

    def sample(self, observable, wires, par):

        if not isinstance(observable, list):
            observable, wires, par = [observable], [wires], [par]

        matrices = [self._get_operator_matrix(o, p) for o, p in zip(observable, par)]

        decompositions = [spectral_decomposition(A) for A in matrices]
        eigenvalues, projector_groups = list(zip(*decompositions))
        eigenvalues = list(eigenvalues)

        # Matching each projector with the wires it acts on
        # while preserving the groupings
        projectors_with_wires = [
            [(proj, wires[idx]) for proj in proj_group]
            for idx, proj_group in enumerate(projector_groups)
        ]

        # The eigenvalue - projector maps are preserved as product() preserves
        # the previous ordering by creating a lexicographic ordering
        joint_outcomes = list(product(*eigenvalues))
        projector_tensor_products = list(product(*projectors_with_wires))

        joint_probabilities = []

        for projs in projector_tensor_products:
            obs_nodes = []
            obs_wires = []
            for proj, proj_wires in projs:

                tensor = proj.reshape([2] * len(proj_wires) * 2)
                obs_nodes.append(self._add_node(tensor, proj_wires))
                obs_wires.append(proj_wires)

            joint_probabilities.append(self.ev(obs_nodes, obs_wires))

        outcomes = np.array([np.prod(p) for p in joint_outcomes])
        return np.random.choice(outcomes, self.shots, p=joint_probabilities)

    def _get_operator_matrix(self, operation, par):
        """Get the operator matrix for a given operation or observable.

        Args:
          operation    (str): name of the operation/observable
          par (tuple[float]): parameter values
        Returns:
          array: matrix representation.
        """
        A = {**self._operation_map, **self._observable_map}[operation]
        if not callable(A):
            return self._array(A, dtype=self.C_DTYPE)
        return self._asarray(A(*par), dtype=self.C_DTYPE)

    def ev(self, obs_nodes, wires):
        r"""Expectation value of observables on specified wires.

         Args:
            obs_nodes (Sequence[tn.Node]): the observables as tensornetwork Nodes
            wires (Sequence[Sequence[int]]): measured subsystems for each observable
         Returns:
            float: expectation value :math:`\expect{A} = \bra{\psi}A\ket{\psi}`
        """

        all_wires = tuple(w for w in range(self.num_wires))
        ket = self._add_node(self._state_node, wires=all_wires, name="Ket")
        bra = self._add_node(tn.conj(ket), wires=all_wires, name="Bra")
        meas_wires = []
        # We need to build up <psi|A|psi> step-by-step.
        # For wires which are measured, we need to connect edges between
        # bra, obs_node, and ket.
        # For wires which are not measured, we need to connect edges between
        # bra and ket.
        # We use the convention that the indices of a tensor are ordered like
        # [output_idx1, output_idx2, ..., input_idx1, input_idx2, ...]
        for obs_node, obs_wires in zip(obs_nodes, wires):
            meas_wires.extend(obs_wires)
            for idx, w in enumerate(obs_wires):
                output_idx = idx
                input_idx = len(obs_wires) + idx
                self._add_edge(obs_node, input_idx, ket, w)  # A|psi>
                self._add_edge(bra, w, obs_node, output_idx)  # <psi|A
        for w in set(all_wires) - set(meas_wires):
            self._add_edge(bra, w, ket, w)  # |psi[w]|**2

        # At this stage, all nodes are connected, and the contraction yields a
        # scalar value.
        contracted_ket = ket
        for obs_node in obs_nodes:
            contracted_ket = tn.contract_between(obs_node, contracted_ket)
        expval = tn.contract_between(bra, contracted_ket).tensor
        if self._abs(self._imag(expval)) > tolerance:
            warnings.warn(
                "Nonvanishing imaginary part {} in expectation value.".format(expval.imag),
                RuntimeWarning,
            )
        return self._real(expval)

    @property
    def _state(self):
        """The numerical value of the current state vector.

        This attribute cannot be manually overwritten.

        Returns:
            (array, tf.Tensor, torch.Tensor): the numerical tensor
        """

        return self._state_node.tensor

    def reset(self):
        """Reset the device"""
        self._nodes = []
        self._edges = []

        state = self._create_basis_state([0] * self.num_wires, range(self.num_wires))
        state = self._array(state, dtype=self.C_DTYPE)

        # TODO: since this state is separable, can be more intelligent about not making a dense matrix
        self._state_node = self._add_node(
            state, wires=tuple(w for w in range(self.num_wires)), name="AllZeroState"
        )
        self._free_edges = self._state_node.edges[
            :
        ]  # we need this list to be distinct from self._state_node.edges

    @property
    def operations(self):
        return set(self._operation_map.keys())

    @property
    def observables(self):
        return set(self._observable_map.keys())
