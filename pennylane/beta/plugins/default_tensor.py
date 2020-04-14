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

import warnings
from itertools import product

from . import numpy_ops as ops

import numpy as np

try:
    import tensornetwork as tn

    v = tn.__version__.split(".")
    if int(v[0]) != 0 and int(v[1]) != 3:
        raise ImportError("default.tensor device requires TensorNetwork==0.3")
except ImportError as e:
    raise ImportError("default.tensor device requires TensorNetwork==0.3")

from pennylane._device import Device

# tolerance for numerical errors
TOL = 1e-10

contract_fns = {
    "greedy": tn.contractors.greedy,
    "branch": tn.contractors.branch,
    "optimal": tn.contractors.optimal,
    "auto": tn.contractors.auto,
}


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
        "QubitUnitary": ops.unitary,
        "PauliX": ops.X,
        "PauliY": ops.Y,
        "PauliZ": ops.Z,
        "Hadamard": ops.H,
        "S": ops.S,
        "T": ops.T,
        "CNOT": ops.CNOT,
        "SWAP": ops.SWAP,
        "CSWAP": ops.CSWAP,
        "Toffoli": ops.Toffoli,
        "CZ": ops.CZ,
        "PhaseShift": ops.Rphi,
        "RX": ops.Rotx,
        "RY": ops.Roty,
        "RZ": ops.Rotz,
        "Rot": ops.Rot3,
        "CRX": ops.CRotx,
        "CRY": ops.CRoty,
        "CRZ": ops.CRotz,
        "CRot": ops.CRot3,
    }

    _observable_map = {
        "PauliX": ops.X,
        "PauliY": ops.Y,
        "PauliZ": ops.Z,
        "Hadamard": ops.H,
        "Hermitian": ops.hermitian,
        "Identity": ops.identity,
    }

    _operation_and_observable_map = {**_operation_map, **_observable_map}

    backend = "numpy"
    _reshape = staticmethod(np.reshape)
    _array = staticmethod(np.array)
    _asarray = staticmethod(np.asarray)
    _real = staticmethod(np.real)
    _imag = staticmethod(np.imag)
    _abs = staticmethod(np.abs)

    C_DTYPE = np.complex128
    R_DTYPE = np.float64

    _zero_state = np.array([1.0, 0.0], dtype=C_DTYPE)

    def __init__(self, wires, shots=1000, analytic=True, representation="mps"):
        super().__init__(wires, shots)
        self.analytic = analytic
        self._rep = representation
        self.reset()

    def reset(self):
        """Reset the device."""
        self._clear_network_data()

        # prepare a factorized all-zeros state
        self._add_initial_state_nodes(
            [self._zero_state] * self.num_wires,
            [[w] for w in range(self.num_wires)],
            ["ZeroState"] * self.num_wires,
        )

    def _clear_network_data(self):
        """Remove all data representing the current network from internal cache."""
        self._nodes = {}
        if self._rep == "exact":
            self._contracted = False
            self._terminal_edges = []
            self.mps = None

    def _add_initial_state_nodes(self, tensors, wires, names):
        """Create the nodes representing the initial input state circuit.
        
           Input states can be factorized or entangled. If a state can be factorized
           into :math:`k` subsystems, then ``tensors``, ``wires``, and ``names`` should be
           sequences of length :math:`k`.

           If self._rep is "exact", then self._terminal_edges is updated with the dangling
           edges from the prepared state nodes.

           If self._rep is "mps", then the self.mps attribute is replaced with a new
           matrix product state object representing the prepared initial states.

          Args:
              tensors (Sequence[np.array, tf.Tensor, torch.Tensor]): the numerical tensors for each
               factorized component of the state (in the computational basis)
              wires (Sequence(list[int])): the wires for each factorized component of the state
              names (Sequence[str]): name for each factorized component of the state
        """
        if not (len(tensors) == len(wires) == len(names)):
            raise ValueError("tensors, wires, and names must all be the same length.")

        if self._rep == "exact":
            for tensor, wires_seq, name in zip(tensors, wires, names):
                node = self._add_node(tensor, wires=wires_seq, name=name)
                self._terminal_edges.extend(node.edges)

        elif self._rep == "mps":
            nodes = []
            for tensor, wires_seq, name in zip(tensors, wires, names):
                tensor = np.expand_dims(tensor, [0, -1])
                if tensor.shape == (1, 2, 1):
                    # MPS form
                    node = self._add_node(tensor, wires=wires_seq, name=name)
                    nodes.append(node)
                else:
                    # break down non-factorized tensors into MPS form
                    DV = tensor
                    for idx, wire in enumerate(wires_seq):
                        if idx < len(wires_seq) - 1:
                            node = tn.Node(DV)
                            U, DV, _error = tn.split_node(node, node[:2], node[2:])
                            node = self._add_node(U, wires=[wire], name=name)
                        else:
                            # final wire; no need to split further
                            node = self._add_node(DV, wires=[wire], name=name)
                        nodes.append(node)
            # TODO: might want to set canonicalize=False
            self.mps = tn.matrixproductstates.finite_mps.FiniteMPS(nodes, canonicalize=False)

    def _add_node(self, A, wires, name="UnnamedNode", key="state"):
        """Adds a node to the underlying tensor network.

        For bookkeeping, the dictionary ``self._nodes`` is updated. The created node is
        appended to the list found under ``key``.

        Args:
            A (array): numerical data values for the operator (i.e., matrix form)
            wires (list[int]): wires that this operator acts on
            name (str): optional name for the node
            key (str): which list of nodes to add new node to

        Returns:
            tn.Node: the newly created node
        """
        name = "{}{}".format(name, tuple(w for w in wires))
        if isinstance(A, tn.Node):
            A.set_name(name)
            node = A
        else:
            node = tn.Node(A, name=name, backend=self.backend)

        if key not in self._nodes:
            self._nodes[key] = []
        self._nodes[key].append(node)

        return node

    def _add_edge(self, node1, idx1, node2, idx2):
        """Adds an edge between two nodes.

        Args:
            node1 (tn.Node): first node to connect
            idx1 (int): index of node1 to connect the edge to
            node2 (tn.Node): second node to connect
            idx2 (int): index of node2 to connect the edge to

        Returns:
            tn.Edge: the newly created edge
        """
        edge = tn.connect(node1[idx1], node2[idx2])
        return edge

    def apply(self, operation, wires, par):
        if operation == "QubitStateVector":
            state = self._array(par[0], dtype=self.C_DTYPE)
            if state.ndim == 1 and state.shape[0] == 2 ** self.num_wires:
                tensor = self._reshape(state, [2] * self.num_wires)
                self._clear_network_data()
                self._add_initial_state_nodes(
                    [tensor], [list(range(self.num_wires))], ["QubitStateVector"]
                )
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
            full_wires_list = list(range(self.num_wires))
            if wires is not None and wires != [] and list(wires) != full_wires_list:
                raise ValueError(
                    "The default.tensor plugin can apply BasisState only to all of the {} wires.".format(
                        self.num_wires
                    )
                )
            zero_vec = self._array(self._zero_state, dtype=self.C_DTYPE)
            one_vec = zero_vec[::-1]
            tensors = [zero_vec if par[0][wire] == 0 else one_vec for wire in range(self.num_wires)]
            self._clear_network_data()
            self._add_initial_state_nodes(
                tensors,
                [[w] for w in range(self.num_wires)],
                ["BasisState"] * self.num_wires,
            )
            return

        A = self._get_operator_matrix(operation, par)
        num_wires = len(wires)
        A = self._reshape(A, [2] * num_wires * 2)
        op_node = self._add_node(A, wires=wires, name=operation)

        if self._rep == "exact":
            for idx, w in enumerate(wires):
                tn.connect(op_node[num_wires + idx], self._terminal_edges[w])
                self._terminal_edges[w] = op_node[idx]
        elif self._rep == "mps":
            if len(wires) == 1:
               self.mps.apply_one_site_gate(op_node, wires[0])
            elif len(wires) == 2:
               if abs(wires[1]-wires[0]) == 1:
                   ret = self.mps.apply_two_site_gate(op_node, *wires)
                   # TODO: determine what ``ret`` is and if it is useful for anything
                   # TODO: pass ``max_singular_values`` or ``max_truncation_error``
               else:
                   # only nearest-neighbours are natively supported
                   print("ruh roh")
            else:
               raise NotImplementedError

    def create_nodes_from_tensors(self, tensors, wires, observable_names, key):
        """Helper function for creating TensorNetwork nodes based on tensors.

        Args:
          tensors (Sequence[np.ndarray, tf.Tensor, torch.Tensor]): tensors of the observables
          wires (Sequence[Sequence[int]]): measured subsystems for each observable
          observable_names (Sequence[str]): names of the operation/observable
          key (str): which subset of nodes to add the nodes to

        Returns:
          list[tn.Node]: the observables as TensorNetwork Nodes
        """
        return [
            self._add_node(A, w, name=o, key=key)
            for A, w, o in zip(tensors, wires, observable_names)
        ]

    def expval(self, observable, wires, par):

        if not isinstance(observable, list):
            observable, wires, par = [observable], [wires], [par]

        tensors = []
        for o, p, w in zip(observable, par, wires):
            A = self._get_operator_matrix(o, p)
            offset = len(w)
            tensors.append(self._reshape(A, [2] * offset * 2))

        nodes = self.create_nodes_from_tensors(tensors, wires, observable, key="observables")
        return self.ev(nodes, wires)

    def var(self, observable, wires, par):

        if not isinstance(observable, list):
            observable, wires, par = [observable], [wires], [par]

        matrices = [self._get_operator_matrix(o, p) for o, p in zip(observable, par)]

        tensors = [self._reshape(A, [2] * len(wires) * 2) for A, wires in zip(matrices, wires)]
        tensors_of_squared_matrices = [
            self._reshape(A @ A, [2] * len(wires) * 2) for A, wires in zip(matrices, wires)
        ]

        obs_nodes = self.create_nodes_from_tensors(tensors, wires, observable, key="observables")
        obs_nodes_for_squares = self.create_nodes_from_tensors(
            tensors_of_squared_matrices, wires, observable, key="observables"
        )

        return self.ev(obs_nodes_for_squares, wires) - self.ev(obs_nodes, wires) ** 2

    def sample(self, observable, wires, par):

        if not isinstance(observable, list):
            observable, wires, par = [observable], [wires], [par]

        matrices = [self._get_operator_matrix(o, p) for o, p in zip(observable, par)]

        decompositions = [ops.spectral_decomposition(A) for A in matrices]
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
                obs_nodes.append(self._add_node(tensor, proj_wires, key="observables"))
                obs_wires.append(proj_wires)

            joint_probabilities.append(self.ev(obs_nodes, obs_wires))

        outcomes = np.array([np.prod(p) for p in joint_outcomes])
        return np.random.choice(outcomes, self.shots, p=joint_probabilities)

    def _get_operator_matrix(self, operation, par):
        """Get the operator matrix for a given operation or observable.

        Args:
          operation (str): name of the operation/observable
          par (tuple[float]): parameter values
        Returns:
          array: matrix representation.
        """
        A = self._operation_and_observable_map[operation]
        if not callable(A):
            return self._array(A, dtype=self.C_DTYPE)
        return self._asarray(A(*par), dtype=self.C_DTYPE)

    def ev(self, obs_nodes, wires, contraction_method="auto"):
        r"""Expectation value of observables on specified wires.

         Args:
            obs_nodes (Sequence[tn.Node]): the observables as TensorNetwork Nodes
            wires (Sequence[Sequence[int]]): measured subsystems for each observable
            contraction_method (str): The contraction method to be employed.
                Possible choices are "auto", "greedy", "branch", or "optimal".
                See TensorNetwork library documentation for more details.
         Returns:
            float: expectation value :math:`\expect{A} = \bra{\psi}A\ket{\psi}`
        """
        if self._rep == "exact":
            self._contract_to_ket(contraction_method)
            ket = self._nodes["contracted_state"]
            bra = tn.conj(ket, name="Bra")

            all_wires = tuple(range(self.num_wires))
            meas_wires = []
            # For wires which are measured, add edges between
            # the ket node, the observable nodes, and the bra node
            for obs_node, obs_wires in zip(obs_nodes, wires):
                meas_wires.extend(obs_wires)
                for idx, w in enumerate(obs_wires):
                    # Use convention that the indices of a tensor are ordered like
                    # [output_idx1, output_idx2, ..., input_idx1, input_idx2, ...]
                    output_idx = idx
                    input_idx = len(obs_wires) + idx
                    self._add_edge(obs_node, input_idx, ket, w)  # A|psi>
                    self._add_edge(bra, w, obs_node, output_idx)  # <psi|A
            # unmeasured wires are contracted directly between bra and ket
            for w in set(all_wires) - set(meas_wires):
                self._add_edge(bra, w, ket, w)

            # At this stage, all nodes are connected, and the contraction yields a
            # scalar value.
            ket_and_observable_node = ket
            for obs_node in obs_nodes:
                ket_and_observable_node = tn.contract_between(obs_node, ket_and_observable_node)
            expval = tn.contract_between(bra, ket_and_observable_node).tensor

        elif self._rep == "mps":
            if any(len(wires_seq) > 2 for wires_seq in wires):
                raise NotImplementedError
            else:
                if len(obs_nodes) == 1 and len(wires[0]) == 1:
                    # TODO: can measure multiple local expectation values at once,
                    # but this would require change of `expval` behaviour and
                    # refactor of `execute` logic
                    expval = self.mps.measure_local_operator(obs_nodes, wires[0])[0]
                else:
                    conj_nodes = [tn.conj(node) for node in self.mps.nodes]
                    meas_wires = []
                    # connect measured bra and ket nodes with observables
                    for obs_node, wire_seq in zip(obs_nodes, wires):
                        if len(wire_seq) > 2 or (len(wire_seq) == 2 and np.abs(wire_seq[0]-wire_seq[1]) > 1):
                            raise NotImplementedError("Multi-wire measurement only supported for nearest-neighbour qubit pairs.")
                        offset = len(wire_seq)
                        for idx, wire in enumerate(wire_seq):
                            tn.connect(conj_nodes[wire][1], obs_node[idx])
                            tn.connect(obs_node[offset + idx], self.mps.nodes[wire][1])
                        meas_wires.extend(wire_seq)
                    for wire in range(self.num_wires):
                        # connect unmeasured ket nodes with bra nodes
                        if wire not in meas_wires:
                            tn.connect(conj_nodes[wire][1], self.mps.nodes[wire][1])
                        # connect local nodes of MPS (not connected by default in tn)
                        if wire != self.num_wires - 1:
                            tn.connect(self.mps.nodes[wire][2], self.mps.nodes[wire + 1][0])
                            tn.connect(conj_nodes[wire][2], conj_nodes[wire + 1][0])

                    # contract MPS bonds first
                    bra_node = conj_nodes[0]
                    ket_node = self.mps.nodes[0]
                    for wire in range(self.num_wires - 1):
                        bra_node = tn.contract_between(bra_node, conj_nodes[wire + 1])
                        ket_node = tn.contract_between(ket_node, self.mps.nodes[wire + 1])
                    # contract observables into ket
                    for obs_node in obs_nodes:
                        ket_node = tn.contract_between(obs_node, ket_node)
                    # contract bra into observables/ket
                    expval_node = tn.contract_between(bra_node, ket_node)
                    # remove dangling singleton edges
                    expval = np.squeeze(expval_node.tensor)

        if self._abs(self._imag(expval)) > TOL:
            warnings.warn(
                "Nonvanishing imaginary part {} in expectation value.".format(expval.imag),
                RuntimeWarning,
            )
        return self._real(expval)

    def _contract_to_ket(self, contraction_method="auto"):
        """Contract the nodes which represent the state preparation and gate applications to get the pre-measurement state.

        The contracted tensor is stored in the ``_nodes`` dictionary under the key ``"contracted_state"``.

        Args:
            contraction_method (str): The contraction method to be employed.
                Possible choices are "auto", "greedy", "branch", or "optimal".
                See TensorNetwork library documentation for more details.
        """
        if "contracted_state" not in self._nodes:
            if self._rep == "exact":
                contract = contract_fns[contraction_method]
                ket = contract(self._nodes["state"], output_edge_order=self._terminal_edges)
            elif self._rep == "mps":
                # contract any mutual edges
                # remove superfluous edges from first and last sites
                # return tensor
                for idx, node in enumerate(self.mps.nodes):
                    if idx == 0:
                        prev_node = node
                    else:
                        tn.connect(prev_node[-1], node[0])
                        prev_node = tn.contract_between(prev_node, node)
                ket = prev_node
            ket.set_name("Ket")
            self._nodes["contracted_state"] = ket

    def _state(self, contraction_method="auto"):
        """The numerical quantum state tensor.

        The state is obtained by contracting all the gates in the tensor network.
        An optional contraction method can be specified.

        Args:
            contraction_method (str): The contraction method to be employed.
                Possible choices are "auto", "greedy", "branch", or "optimal".
                See TensorNetwork library documentation for more details.
        Returns:
            (array, tf.Tensor, torch.Tensor): the numerical tensor
        """
        self._contract_to_ket(contraction_method)
        ket = self._nodes["contracted_state"]
        return np.squeeze(ket.tensor)

    @property
    def operations(self):
        return set(self._operation_map.keys())

    @property
    def observables(self):
        return set(self._observable_map.keys())
