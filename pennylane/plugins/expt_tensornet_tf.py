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
Experimental simulator plugin based on tensor network contractions,
using the TensorFlow backend for Jacobian computations.
"""
import numpy as np
import tensorflow as tf
import tensornetwork as tn

from pennylane.qnode import QuantumFunctionError
from pennylane.operation import Expectation, Variance, Sample
from .expt_tensornet import TensorNetwork

from pennylane.plugins.default_qubit import (CNOT, CSWAP, CZ, SWAP, I, H, S, T, X,
                                             Y, Z, hermitian, identity,
                                             unitary)

tn.set_default_backend("tensorflow")

I = tf.constant(I, dtype=tf.complex128)
X = tf.constant(X, dtype=tf.complex128)


def Rphi(phi):
    r"""One-qubit phase shift.

    Args:
        phi (float): phase shift angle
    Returns:
        array: unitary 2x2 phase shift matrix
    """
    return np.array([[1, 0], [0, np.exp(1j*phi)]])


def Rotx(theta):
    r"""One-qubit rotation about the x axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}`
    """
    theta = tf.cast(theta, dtype=tf.complex128)
    return tf.cos(theta/2) * I + 1j * tf.sin(-theta/2) * X


def Roty(theta):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    theta = tf.cast(theta, dtype=tf.complex128)
    return tf.cos(theta/2) * I + 1j * tf.sin(-theta/2) * Y


def Rotz(theta):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    return np.cos(theta/2) * I + 1j * np.sin(-theta/2) * Z


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
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.cos(theta/2), -1j*np.sin(theta/2)], [0, 0, -1j*np.sin(theta/2), np.cos(theta/2)]])


def CRoty(theta):
    r"""Two-qubit controlled rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_y(\theta)`
    """
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.cos(theta/2), -np.sin(theta/2)], [0, 0, np.sin(theta/2), np.cos(theta/2)]])


def CRotz(theta):
    r"""Two-qubit controlled rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_z(\theta)`
    """
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.exp(-1j*theta/2), 0], [0, 0, 0, np.exp(1j*theta/2)]])


def CRot3(a, b, c):
    r"""Arbitrary two-qubit controlled rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R(a,b,c)`
    """
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.exp(-1j*(a+c)/2)*np.cos(b/2), -np.exp(1j*(a-c)/2)*np.sin(b/2)], [0, 0, np.exp(-1j*(a-c)/2)*np.sin(b/2), np.exp(1j*(a+c)/2)*np.cos(b/2)]])


class TensorNetworkTF(TensorNetwork):
    """Experimental TensorFlow Tensor Network simulator device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
    """
    name = "PennyLane TensorNetwork (TensorFlow) simulator plugin"
    short_name = "expt.tensornet.tf"
    _capabilities = {"model": "qubit", "tensor_observables": True, "provides_jacobian": True}

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

    def __init__(self, wires, shots=1000, analytic=True):
        super().__init__(wires, shots)
        self.eng = None
        self.analytic = True
        self._nodes = []
        self._edges = []
        zero_state = np.zeros([2] * wires)
        zero_state[tuple([0] * wires)] = 1.0
        self._zero_state = tf.constant(zero_state, dtype=tf.complex128)
        # TODO: since this state is separable, can be more intelligent about not making a dense matrix
        self._state_node = self._add_node(
            self._zero_state, wires=tuple(w for w in range(wires)), name="AllZeroState"
        )
        self._free_edges = self._state_node.edges[:]  # we need this list to be distinct from self._state_node.edges

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
            return tf.constant(A, dtype=tf.complex128)
        return tf.convert_to_tensor(A(*par), dtype=tf.complex128)

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
        A = tf.reshape(A, [2] * num_mult_idxs * 2)
        op_node = self._add_node(A, wires=wires, name=operation)
        for idx, w in enumerate(wires):
            self._add_edge(op_node, num_mult_idxs + idx, self._state_node, w)
            self._free_edges[w] = op_node[idx]
        # TODO: can be smarter here about collecting contractions?
        self._state_node = tn.contract_between(
            op_node, self._state_node, output_edge_order=self._free_edges
        )

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

        # TODO: add complex non-vanishing part warning

        return tf.math.real(expval)

    def execute(self, queue, observables, parameters=None):
        self.check_validity(queue, observables)
        self._op_queue = queue
        self._obs_queue = observables
        self._parameters = {} if parameters is None else parameters
        self._parameters.update(parameters)

        results = []

        self.tape = tf.GradientTape()

        with self.tape:
            self.pre_apply()

            self.variables = {}

            for operation in queue:
                self.variables[operation] = operation.params

            for idx, par_dep_list in parameters.items():
                first = par_dep_list[0]
                v = tf.Variable(first.op.params[first.par_idx].val, dtype=tf.float64)
                self.tape.watch(v)

                for p in par_dep_list:
                    self.variables[p.op][p.par_idx] = v*p.op.params[p.par_idx].mult

            for operation in queue:
                self.apply(operation.name, operation.wires, self.variables[operation])

            for i, obs in enumerate(observables):
                if obs.return_type is Expectation:
                    results.append(self.expval(obs.name, obs.wires, obs.parameters))

                elif obs.return_type is Variance:
                    results.append(self.var(obs.name, obs.wires, obs.parameters))

                elif obs.return_type is Sample:
                    results.append(np.array(self.sample(obs.name, obs.wires, obs.parameters)))

                elif obs.return_type is not None:
                    raise QuantumFunctionError("Unsupported return type specified for observable {}".format(obs.name))

            self._op_queue = None
            self._obs_queue = None
            self._parameters = None

            return tf.stack(results)

    def jacobian(self, queue, observables, parameters):
        res = self.execute(queue, observables, parameters=parameters)
        var = tf.nest.flatten(list(self.variables.values()))
        jac = tf.stack(self.tape.jacobian(res, var)).numpy().T
        return jac
