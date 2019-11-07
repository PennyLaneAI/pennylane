# Copyright 2018 Xanadu Quantum Technologies Inc.

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

import warnings

import numpy as np
import tensornetwork as tn

from pennylane import Device
from pennylane.plugins.default_qubit import (
    X,
    Y,
    Z,
    H,
    CNOT,
    SWAP,
    CZ,
    S,
    T,
    CSWAP,
    Rphi,
    Rotx,
    Roty,
    Rotz,
    Rot3,
    CRotx,
    CRoty,
    CRotz,
    CRot3,
    hermitian,
    identity,
    unitary,
)

# tolerance for numerical errors
tolerance = 1e-10

# ========================================================
#  device
# ========================================================


class TensorNetwork(Device):
    """Experimental Tensor Network simulator device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
    """

    name = "PennyLane TensorNetwork simulator plugin"
    short_name = "expt.tensornet"
    pennylane_requires = "0.7"
    version = "0.7.0"
    author = "Xanadu Inc."
    _capabilities = {"model": "qubit", "tensor_observables": True}

    _operation_map = {
        "QubitStateVector": None,
        "BasisState": None,
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

    def __init__(self, wires):
        super().__init__(wires, shots=1)
        self.eng = None
        self.analytic = True
        self._nodes = []
        self._edges = []
        self._zero_state = np.zeros([2] * wires)
        self._zero_state[tuple([0] * wires)] = 1.0
        # TODO: since this state is separable, can be more intelligent about not making a dense matrix
        self._state_node = self._add_node(
            self._zero_state, wires=tuple(w for w in range(wires)), name="AllZeroState"
        )
        self._free_edges = self._state_node.edges[:]  # we need this list to be distinct from self._state_node.edges

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
            node = tn.Node(A, name=name)
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
            state = np.asarray(par[0], dtype=np.complex128)
            if state.ndim == 1 and state.shape[0] == 2 ** self.num_wires:
                self._state_node.tensor = np.reshape(state, [2] * self.num_wires)
            else:
                raise ValueError("State vector must be of length 2**wires.")
            if wires is not None and wires != [] and list(wires) != list(range(self.num_wires)):
                raise ValueError(
                    "The expt.tensornet plugin can apply QubitStateVector only to all of the {} wires.".format(
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
                    "The expt.tensornet plugin can apply BasisState only to all of the {} wires.".format(
                        self.num_wires
                    )
                )

            self._state_node.tensor[tuple([0] * len(wires))] = 0
            self._state_node.tensor[tuple(par[0])] = 1
            return

        A = self._get_operator_matrix(operation, par)
        num_mult_idxs = len(wires)
        A = np.reshape(A, [2] * num_mult_idxs * 2)
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
            tensors.append(np.reshape(A, [2] * num_mult_idxs * 2))

        nodes = self.create_nodes_from_tensors(tensors, wires, observable)
        return self.ev(nodes, wires)

    def var(self, observable, wires, par):

        if not isinstance(observable, list):
            observable, wires, par = [observable], [wires], [par]

        matrices = [self._get_operator_matrix(o, p) for o, p in zip(observable, par)]

        tensors = [np.reshape(A, [2] * len(wires) * 2) for A, wires in zip(matrices, wires)]
        tensors_of_squared_matrices = [np.reshape(A@A, [2] * len(wires) * 2) for A, wires in zip(matrices, wires)]

        obs_nodes = self.create_nodes_from_tensors(tensors, wires, observable)
        obs_nodes_for_squares = self.create_nodes_from_tensors(tensors_of_squared_matrices, wires, observable)

        return self.ev(obs_nodes_for_squares, wires) - self.ev(obs_nodes, wires)**2

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
            return A
        return A(*par)

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
        if np.abs(expval.imag) > tolerance:
            warnings.warn(
                "Nonvanishing imaginary part {} in expectation value.".format(expval.imag),
                RuntimeWarning,
            )
        return expval.real

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
        self.__init__(self.num_wires)

    @property
    def operations(self):
        return set(self._operation_map.keys())

    @property
    def observables(self):
        return set(self._observable_map.keys())
