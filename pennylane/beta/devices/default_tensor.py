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
r"""
Experimental simulator plugin based on tensor network contractions
"""
# pylint: disable=too-many-instance-attributes
import warnings
from itertools import product

import numpy as np

from pennylane._device import Device
from pennylane.beta.devices import numpy_ops as ops
from pennylane.wires import Wires
from ..._version import __version__

try:
    import tensornetwork as tn

    v = tn.__version__.split(".")
    if int(v[0]) != 0 and int(v[1]) < 3:
        raise ImportError("default.tensor device requires TensorNetwork>=0.3")
except ImportError as e:
    raise ImportError("default.tensor device requires TensorNetwork>=0.3") from e


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

    **Short name:** ``default.tensor``

    This experimental device uses the
    `TensorNetwork <https://github.com/google/tensornetwork>`_ library
    to provide a basic tensor-network-based simulator backend for PennyLane.
    Tensor network simulators can faster or more efficient for certain types of
    circuit structures.

    To use this device, you will need to install TensorNetwork version 0.3:

    .. code-block:: bash

        pip install tensornetwork==0.3

    The ``default.tensor`` device supports two types of tensor networks: ``"exact"`` and ``"mps"``.

    The (default) ``"exact"`` representation does not make any approximations, using exact dense tensors for
    the simulator's quantum states and for the matrices of quantum gates and observables.

    The ``"mps"`` representation (standing for "matrix product state") approximates the quantum state
    using a one-dimensional grid of qubits with nearest-neighbour connectivity. As such, it does not support
    multi-qubit gates/observables that do not act on nearest-neighbour qubits.

    The preferred contraction method can also be specified when using the ``"exact"`` representation.
    Available options are "auto", "greedy", "branch", or "optimal".
    See the `TensorNetwork documentation <https://tensornetwork.readthedocs.io/en/latest/copy_contract.html>`_
    for more details.

    **Example**

      >>> exact_tensornet = qml.device("default.tensor", wires=2, contraction_method="greedy")
      >>> mps_tensornet = qml.device("default.tensor", wires=2, representation="mps")

    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (None, int): Number of circuit evaluations/random samples to return when sampling from the device.
            Defaults to ``None`` if not specified, which means that the device returns analytical results.
        representation (str): Underlying representation used for the tensor network simulation.
            Valid options are "exact" (no approximations made) or "mps" (simulated quantum
            state is approximated as a Matrix Product State).
        contraction_method (str): Method used to perform tensor network contractions. Only applicable
            for the "exact" representation. Valid options are "auto", "greedy", "branch", or "optimal".
            See documentation of the `TensorNetwork library <https://tensornetwork.readthedocs.io/en/latest/>`_
            for more information about contraction methods.
    """

    # pylint: disable=attribute-defined-outside-init
    name = "PennyLane TensorNetwork simulator plugin"
    short_name = "default.tensor"
    pennylane_requires = __version__
    version = __version__
    author = "Xanadu Inc."

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

    backend = "numpy"
    _reshape = staticmethod(np.reshape)
    _array = staticmethod(np.array)
    _asarray = staticmethod(np.asarray)
    _real = staticmethod(np.real)
    _imag = staticmethod(np.imag)
    _abs = staticmethod(np.abs)
    _squeeze = staticmethod(np.squeeze)
    _expand_dims = staticmethod(np.expand_dims)

    C_DTYPE = np.complex128
    R_DTYPE = np.float64

    _zero_state = np.array([1.0, 0.0], dtype=C_DTYPE)

    def __init__(self, wires, shots=None, representation="exact", contraction_method="auto"):
        super().__init__(wires, shots)
        if representation not in ["exact", "mps"]:
            raise ValueError("Invalid representation. Must be one of 'exact' or 'mps'.")
        self._operation_and_observable_map = {
            **self._operation_map,
            **self._observable_map,
        }
        self._rep = representation
        self._contraction_method = contraction_method
        self.reset()

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qubit",
            supports_analytic_computation=True,
            supports_finite_shots=False,
            supports_tensor_observables=True,
            returns_state=False,
            returns_probs=False,
        )
        return capabilities

    def reset(self):
        """Reset the device."""
        self._clear_network_data()

        # prepare a factorized all-zeros state
        self._add_initial_state_nodes(
            [self._zero_state] * self.num_wires,
            [Wires(w) for w in range(self.num_wires)],
            ["ZeroState"] * self.num_wires,
        )

    def _clear_network_data(self):
        """Remove all data representing the current network from internal cache."""
        self._nodes = {}
        self._free_wire_edges = []
        self.mps = None
        self._contracted_state_node = None

    def _add_node(self, A, wires, name="UnnamedNode", key="state"):
        """Adds a node to the underlying tensor network.

        For bookkeeping, the dictionary ``self._nodes`` is updated. The created node is
        appended to the list found under ``key``.

        Args:
            A (array): numerical data values for the operator (i.e., matrix form)
            wires (Wires): wires that this operator acts on
            name (str): optional name for the node
            key (str): which list of nodes to add new node to

        Returns:
            tn.Node: the newly created node
        """
        name = "{}{}".format(name, (tuple([wires]) if isinstance(wires, int) else wires.labels))
        if isinstance(A, tn.Node):
            A.set_name(name)
            A.backend = self.backend
            node = A
        else:
            node = tn.Node(A, name=name, backend=self.backend)

        if key not in self._nodes:
            self._nodes[key] = []
        self._nodes[key].append(node)

        return node

    def _add_initial_state_nodes(self, tensors, tensor_wires, names):
        """Create the nodes representing the initial input state circuit.

         Input states can be factorized or entangled. If a state can be factorized
         into :math:`k` subsystems, then ``tensors``, ``wires``, and ``names`` should be
         sequences of length :math:`k`.

         ``self._free_wire_edges`` is updated with the dangling edges from the prepared state nodes.

         If ``self._rep == "mps"``, then the ``self.mps`` attribute is replaced with a new
         matrix product state object representing the prepared initial states.

        Args:
            tensors (Sequence[np.array, tf.Tensor, torch.Tensor]): the numerical tensors for each
              factorized component of the state (in the computational basis)
            tensor_wires (Sequence(Wires)): wires for each factorized component of the state
            names (Sequence[str]): name for each factorized component of the state
        """
        # pylint: disable=too-many-branches
        if not len(tensors) == len(tensor_wires) == len(names):
            raise ValueError("tensors, wires, and names must all be the same length.")

        if self._rep == "exact":
            self._free_wire_edges = []
            for tensor, wires, name in zip(tensors, tensor_wires, names):
                if len(tensor.shape) != len(wires):
                    raise ValueError(
                        "Tensor provided has shape={}, which is incompatible "
                        "with provided wires {}.".format(tensor.shape, wires.tolist())
                    )
                node = self._add_node(tensor, wires=wires, name=name)
                self._free_wire_edges.extend(node.edges)

        elif self._rep == "mps":
            nodes = []
            for tensor, wires, name in zip(tensors, tensor_wires, names):
                if len(tensor.shape) != len(wires):
                    raise ValueError(
                        "Tensor provided has shape={}, which is incompatible "
                        "with provided wires {}.".format(tensor.shape, wires.tolist())
                    )
                tensor = self._expand_dims(tensor, 0)
                tensor = self._expand_dims(tensor, -1)
                if tensor.shape == (1, 2, 1):
                    # MPS form
                    node = self._add_node(tensor, wires=wires, name=name)
                    nodes.append(node)
                else:

                    # translate to wire labels used by device
                    wire_indices = self.map_wires(wires)

                    # break down non-factorized tensors into MPS form
                    if max(wire_indices.labels) - min(wire_indices.labels) != len(wire_indices) - 1:
                        raise NotImplementedError(
                            "Multi-wire state initializations only supported for tensors on consecutive wires."
                        )
                    DV = tensor
                    for idx, wire in enumerate(wires):
                        if idx < len(wires) - 1:
                            node = tn.Node(DV, name=name, backend=self.backend)
                            U, DV, _error = tn.split_node(node, node[:2], node[2:])
                            node = self._add_node(U, wires=wire, name=name)
                        else:
                            # final wire; no need to split further
                            node = self._add_node(DV, wires=wire, name=name)
                        nodes.append(node)
            self.mps = tn.matrixproductstates.finite_mps.FiniteMPS(
                [node.tensor for node in nodes],
                canonicalize=False,
                backend=self.backend,
            )
            self._free_wire_edges = [node[1] for node in self.mps.nodes]

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

    def apply(self, operation, wires, par):

        if operation in ("QubitStateVector", "BasisState"):
            if (
                wires is not None
                and wires != Wires([])
                and wires.tolist() != list(range(self.num_wires))
            ):
                raise ValueError(
                    "The default.tensor plugin can apply {} only to all of the {} wires.".format(
                        operation, self.num_wires
                    )
                )
            self._clear_network_data()
            self._add_state_prep_nodes(operation, par)
        else:
            self._add_gate_nodes(operation, wires, par)

    def _add_state_prep_nodes(self, operation, par):
        """Add tensor network nodes related to the state preparations ``QubitStateVector`` and
        ``BasisState`` operations.

        Args:
            operation (str): name of the state preparation operation
            par (tuple): parameter values for the state preparation
        """
        if operation == "QubitStateVector":
            state_vector = self._array(par[0], dtype=self.C_DTYPE)
            if state_vector.ndim == 1 and state_vector.shape[0] == 2 ** self.num_wires:
                tensors = [self._reshape(state_vector, [2] * self.num_wires)]
                tensor_wires = [Wires(range(self.num_wires))]
                name = [operation]
            else:
                raise ValueError("State vector must be of length 2**wires.")

        elif operation == "BasisState":
            n = len(par[0])
            elements = set(par[0].tolist())
            if n == 0 or n > self.num_wires or not elements.issubset({0, 1}):
                raise ValueError(
                    "BasisState parameter must be an array of 0 or 1 integers of length at most {}.".format(
                        self.num_wires
                    )
                )
            zero_vec = self._array(self._zero_state, dtype=self.C_DTYPE)
            one_vec = zero_vec[::-1]
            tensors = [zero_vec if par[0][wire] == 0 else one_vec for wire in range(self.num_wires)]
            tensor_wires = [Wires(w) for w in range(self.num_wires)]
            name = [operation] * self.num_wires

        self._add_initial_state_nodes(tensors, tensor_wires, name)

    def _add_gate_nodes(self, operation, wires, par):
        """Add tensor network nodes and edges related to the quantum gates.

        Args:
            operation (str): name of the gate operation
            wires (Wires): wires that the gate is applied to
            par (tuple): parameter values for the gate
        """

        A = self._get_operator_matrix(operation, par)
        num_wires = len(wires)
        A = self._reshape(A, [2] * num_wires * 2)
        op_node = self._add_node(A, wires=wires, name=operation)

        # translate to wire labels used by device
        wires = self.map_wires(wires)

        if self._rep == "exact":
            for idx, l in enumerate(wires.labels):
                tn.connect(op_node[num_wires + idx], self._free_wire_edges[l])
                self._free_wire_edges[l] = op_node[idx]
        elif self._rep == "mps":
            if len(wires) == 1:
                reg = wires.labels[0]
                self.mps.apply_one_site_gate(op_node, reg)
                self._free_wire_edges[reg] = self.mps.nodes[reg][1]
            elif len(wires) == 2:
                if abs(wires.labels[1] - wires.labels[0]) == 1:
                    # TODO: set ``max_singular_values`` or ``max_truncation_error``
                    self.mps.apply_two_site_gate(op_node, *wires.labels)
                    for reg in wires.labels:
                        self._free_wire_edges[reg] = self.mps.nodes[reg][1]
                else:
                    raise NotImplementedError(
                        "Multi-wire gates only supported for nearest-neighbour wire pairs."
                    )
            else:
                raise NotImplementedError(
                    "Multi-wire gates only supported for nearest-neighbour wire pairs."
                )

    def _create_nodes_from_tensors(self, tensors, tensor_wires, observable_names, key):
        """Helper function for creating TensorNetwork nodes based on tensors.

        Args:
          tensors (Sequence[np.ndarray, tf.Tensor, torch.Tensor]): tensors of the observables
          tensor_wires (Sequence[Wires]): measured subsystems for each observable
          observable_names (Sequence[str]): names of the operation/observable
          key (str): which subset of nodes to add the nodes to

        Returns:
          list[tn.Node]: the observables as TensorNetwork Nodes
        """
        return [
            self._add_node(A, wires, name=o, key=key)
            for A, wires, o in zip(tensors, tensor_wires, observable_names)
        ]

    def expval(self, observable, wires, par):

        if not isinstance(observable, list):
            observable, wires, par = [observable], [wires], [par]

        tensors = []
        for o, p, w in zip(observable, par, wires):

            A = self._get_operator_matrix(o, p)
            offset = len(w)
            tensors.append(self._reshape(A, [2] * offset * 2))

        nodes = self._create_nodes_from_tensors(tensors, wires, observable, key="observables")
        return self.ev(nodes, wires)

    def var(self, observable, wires, par):

        if not isinstance(observable, list):
            observable, wires, par = [observable], [wires], [par]

        matrices = [self._get_operator_matrix(o, p) for o, p in zip(observable, par)]

        tensors = [self._reshape(A, [2] * len(w) * 2) for A, w in zip(matrices, wires)]
        tensors_of_squared_matrices = [
            self._reshape(A @ A, [2] * len(w) * 2) for A, w in zip(matrices, wires)
        ]

        obs_nodes = self._create_nodes_from_tensors(tensors, wires, observable, key="observables")
        obs_nodes_for_squares = self._create_nodes_from_tensors(
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

    def ev(self, obs_nodes, obs_wires):
        r"""Expectation value of observables on specified wires.

        Args:
           obs_nodes (Sequence[tn.Node]): the observables as TensorNetwork Nodes
           obs_wires (Sequence[Wires]): measured wires for each observable

        Returns:
           float: expectation value :math:`\expect{A} = \bra{\psi}A\ket{\psi}`
        """
        if self._rep == "exact":
            expval = self._ev_exact(obs_nodes, obs_wires)
        elif self._rep == "mps":
            expval = self._ev_mps(obs_nodes, obs_wires)

        if self._abs(self._imag(expval)) > TOL:
            warnings.warn(
                "Nonvanishing imaginary part {} in expectation value.".format(expval.imag),
                RuntimeWarning,
            )
        return self._real(expval)

    def _ev_exact(self, obs_nodes, obs_wires):
        r"""Expectation value of observables on specified wires using an exact representation.

        Args:
           obs_nodes (Sequence[tn.Node]): the observables as TensorNetwork Nodes
           obs_wires (Sequence[Wires]): measured wires for each observable

        Returns:
           complex: expectation value :math:`\expect{A} = \bra{\psi}A\ket{\psi}`
        """
        self._contract_premeasurement_network()
        ket = self._contracted_state_node
        bra = tn.conj(ket, name="Bra")

        all_device_wires = Wires(range(self.num_wires))
        meas_device_wires = []
        # For wires which are measured, add edges between
        # the ket node, the observable nodes, and the bra node
        for obs_node, wires in zip(obs_nodes, obs_wires):

            # translate to consecutive wire labels used by device
            device_wires = self.map_wires(wires)

            meas_device_wires.append(device_wires)
            for idx, l in enumerate(device_wires.labels):
                # Use convention that the indices of a tensor are ordered like
                # [output_idx1, output_idx2, ..., input_idx1, input_idx2, ...]
                output_idx = idx
                input_idx = len(device_wires) + idx
                tn.connect(obs_node[input_idx], ket[l])  # A|psi>
                tn.connect(bra[l], obs_node[output_idx])  # <psi|A

        meas_device_wires = Wires.all_wires(meas_device_wires)

        # unmeasured wires are contracted directly between bra and ket
        unmeasured_device_wires = Wires.unique_wires([all_device_wires, meas_device_wires])
        for w in unmeasured_device_wires.labels:
            tn.connect(bra[w], ket[w])

        # At this stage, all nodes are connected, and the contraction yields a
        # scalar value.
        ket_and_observable_node = ket
        for obs_node in obs_nodes:
            ket_and_observable_node = tn.contract_between(obs_node, ket_and_observable_node)
        return tn.contract_between(bra, ket_and_observable_node).tensor

    def _ev_mps(self, obs_nodes, obs_wires):
        r"""Expectation value of observables on specified wires using a MPS representation.

        Args:
           obs_nodes (Sequence[tn.Node]): the observables as TensorNetwork Nodes
           obs_wires (Sequence[Wires]): measured wires for each observable
        Returns:
           complex: expectation value :math:`\expect{A} = \bra{\psi}A\ket{\psi}`
        """
        if any(len(wires) > 2 for wires in obs_wires):
            raise NotImplementedError(
                "Multi-wire measurement only supported for nearest-neighbour wire pairs."
            )
        if len(obs_nodes) == 1 and len(obs_wires[0]) == 1:
            # TODO: can measure multiple local expectation values at once,
            # but this would require change of `expval` behaviour and
            # refactor of `execute` logic from parent class

            # translate to consecutive wire labels used by device
            device_wires = self.map_wires(obs_wires[0])
            expval = self.mps.measure_local_operator(obs_nodes, device_wires.labels)[0]
        else:
            conj_nodes = [tn.conj(node) for node in self.mps.nodes]
            meas_wires = []
            # connect measured bra and ket nodes with observables
            for obs_node, wires in zip(obs_nodes, obs_wires):

                # translate to consecutive wire labels used by device
                device_wires = self.map_wires(wires)
                wire_labels = device_wires.labels

                if len(device_wires) == 2 and abs(wire_labels[0] - wire_labels[1]) > 1:
                    raise NotImplementedError(
                        "Multi-wire measurement only supported for nearest-neighbour wire pairs."
                    )
                offset = len(wire_labels)
                for idx, l in enumerate(wire_labels):
                    tn.connect(conj_nodes[l][1], obs_node[idx])
                    tn.connect(obs_node[offset + idx], self.mps.nodes[l][1])
                meas_wires.extend(wire_labels)
            for l in range(self.num_wires):
                # connect unmeasured ket nodes with bra nodes
                if l not in meas_wires:
                    tn.connect(conj_nodes[l][1], self.mps.nodes[l][1])
                # connect local nodes of MPS (not connected by default in tn)
                if l != self.num_wires - 1:
                    tn.connect(self.mps.nodes[l][2], self.mps.nodes[l + 1][0])
                    tn.connect(conj_nodes[l][2], conj_nodes[l + 1][0])

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
            expval = self._squeeze(expval_node.tensor)
        return expval

    def _contract_premeasurement_network(self):
        """Contract the nodes which represent the state preparation and gate applications to get the pre-measurement state.

        The contracted tensor is cached in the attribute ``_contracted_state_node``.
        """
        if self._contracted_state_node is None:
            if self._rep == "exact":
                contract_fn = contract_fns[self._contraction_method]
                ket = contract_fn(self._nodes["state"], output_edge_order=self._free_wire_edges)
            elif self._rep == "mps":
                # contract all mutual edges
                for idx, node in enumerate(self.mps.nodes):
                    if idx == 0:
                        prev_node = node
                    else:
                        tn.connect(prev_node[-1], node[0])
                        prev_node = tn.contract_between(prev_node, node)
                ket = prev_node
                # remove dangling singleton edges
                ket.tensor = self._squeeze(ket.tensor)
            ket.set_name("Ket")
            self._contracted_state_node = ket

    def _state(self):
        """The numerical quantum state tensor.

        The state is obtained by contracting all the gates in the tensor network.

        Returns:
            (array, tf.Tensor, torch.Tensor): the numerical tensor
        """
        # TODO: determine if there is an edge case where we can apply gates, pull out _state,
        # then apply more gates and try to access _state again (second call will bring out earlier
        # cached state)
        self._contract_premeasurement_network()
        ket = self._contracted_state_node
        return self._squeeze(ket.tensor)

    @property
    def contraction_method(self):
        """The contraction method used by the tensor network.
        Available options are "auto", "greedy", "branch", or "optimal".
        See TensorNetwork library documentation for more details.
        """
        return self._contraction_method

    @contraction_method.setter
    def contraction_method(self, method):
        """Changes the contraction method used by the tensor network.

        Args:
            method (str): The contraction method to be employed.
                Available options are "auto", "greedy", "branch", or "optimal".
                See TensorNetwork library documentation for more details.

        Raises:
            ValueError: if ``method`` is not one of the supported options
        """
        if method not in contract_fns:
            raise ValueError(
                "The contraction method ``{}`` was not found. Supported methods are"
                "'auto', 'greedy', 'branch', or 'optimal'.".format(method)
            )

        self._contraction_method = method

    @property
    def operations(self):
        return set(self._operation_map.keys())

    @property
    def observables(self):
        return set(self._observable_map.keys())
