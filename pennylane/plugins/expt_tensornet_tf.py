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

from pennylane import Device
from pennylane.operation import Expectation, Variance, Sample
from pennylane.qnode import QuantumFunctionError
from pennylane.variable import Variable

from pennylane.plugins.default_qubit import (CNOT, CSWAP, CZ, SWAP, I, H, S, T, X,
                                             Y, Z, unitary)

from .expt_tensornet import TensorNetwork


C_DTYPE = tf.complex128
R_DTYPE = tf.float64


I = tf.constant(I, dtype=C_DTYPE)
X = tf.constant(X, dtype=C_DTYPE)

II = tf.eye(4, dtype=C_DTYPE)
ZZ = tf.constant(np.kron(Z, Z), dtype=C_DTYPE)

IX = tf.constant(np.kron(I, X), dtype=C_DTYPE)
IY = tf.constant(np.kron(I, Y), dtype=C_DTYPE)
IZ = tf.constant(np.kron(I, Z), dtype=C_DTYPE)

ZI = tf.constant(np.kron(Z, I), dtype=C_DTYPE)
ZX = tf.constant(np.kron(Z, X), dtype=C_DTYPE)
ZY = tf.constant(np.kron(Z, Y), dtype=C_DTYPE)


def Rphi(phi):
    r"""One-qubit phase shift.

    Args:
        phi (float): phase shift angle

    Returns:
        array: unitary 2x2 phase shift matrix
    """
    phi = tf.cast(phi, dtype=C_DTYPE)
    return ((1+tf.exp(1j*phi)) * I + (1-tf.exp(1j*phi)) * Z)/2


def Rotx(theta):
    r"""One-qubit rotation about the x axis.

    Args:
        theta (float): rotation angle

    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}`
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
    return tf.cos(theta/2) * I + 1j * tf.sin(-theta/2) * X


def Roty(theta):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle

    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
    return tf.cos(theta/2) * I + 1j * tf.sin(-theta/2) * Y


def Rotz(theta):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle

    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
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
        array: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_x(\theta)`
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
    return tf.cos(theta/4)**2 * II - 1j*tf.sin(theta/2)/2 * IX + tf.sin(theta/4)**2 * ZI + 1j*tf.sin(theta/2)/2 * ZX


def CRoty(theta):
    r"""Two-qubit controlled rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_y(\theta)`
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
    return tf.cos(theta/4)**2 * II - 1j*tf.sin(theta/2)/2 * IY + tf.sin(theta/4)**2 * ZI + 1j*tf.sin(theta/2)/2 * ZY


def CRotz(theta):
    r"""Two-qubit controlled rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_z(\theta)`
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
    return tf.cos(theta/4)**2 * II - 1j*tf.sin(theta/2)/2 * IZ + tf.sin(theta/4)**2 * ZI + 1j*tf.sin(theta/2)/2 * ZZ


def CRot3(a, b, c):
    r"""Arbitrary two-qubit controlled rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:
        array: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R(a,b,c)`
    """
    return CRotz(c) @ (CRoty(b) @ CRotz(a))


class TensorNetworkTF(TensorNetwork):
    """Experimental TensorFlow Tensor Network simulator device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
    """
    # pylint: disable=too-many-instance-attributes
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

    # observable mapping is inherited from expt.tensornet

    def __init__(self, wires, shots=1000, analytic=True):
        Device.__init__(self, wires, shots)
        self.backend = "tensorflow"

        self.variables = []
        """List[tf.Variable]: Free parameters, cast to TensorFlow variables,
        for this circuit."""

        self.res = None
        """tf.tensor[R_DTYPE]: result from the last circuit execution"""

        self.eng = None
        self.analytic = True
        self._nodes = []
        self._edges = []
        zero_state = np.zeros([2] * wires)
        zero_state[tuple([0] * wires)] = 1.0
        self._zero_state = tf.constant(zero_state, dtype=C_DTYPE)
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
            return tf.constant(A, dtype=C_DTYPE)
        return tf.convert_to_tensor(A(*par), dtype=C_DTYPE)

    def apply(self, operation, wires, par):
        if operation == "QubitStateVector":
            state = tf.constant(par[0], dtype=np.complex128)
            if state.ndim == 1 and state.shape[0] == 2 ** self.num_wires:
                self._state_node.tensor = tf.reshape(state, [2] * self.num_wires)
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
            state_node = np.zeros(tuple([2] * len(wires)))
            state_node[tuple(par[0])] = 1
            self._state_node.tensor = tf.constant(state_node, dtype=C_DTYPE)
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
        # pylint: disable=attribute-defined-outside-init, pointless-string-statement
        self.check_validity(queue, observables)

        results = []

        self.tape = tf.GradientTape(persistent=True)

        with self.tape:
            self.pre_apply()

            op_params = {}
            """dict[Operation, List[Any, tf.Variable]]: a mapping from each operation
            in the queue, to the corresponding list of parameter values. These
            values can be Python numeric types, NumPy arrays, or TensorFlow variables."""

            for operation in queue:
                # Copy the operation parameters to the op_params dictionary.
                # Note that these are the unwrapped parameters, so PennyLane
                # free parameters will be represented as Variable instances.
                op_params[operation] = operation.params[:]

            # Loop through the free parameter reference dictionary
            for _, par_dep_list in parameters.items():
                # get the first parameter dependency for each free parameter
                first = par_dep_list[0]

                # For the above parameter dependency, get the corresponding
                # operation parameter variable, and get the numeric value.
                # Convert the resulting value to a TensorFlow tensor.
                v = tf.Variable(first.op.params[first.par_idx].val, dtype=R_DTYPE)

                # Mark the variable to be watched by the gradient tape,
                # and append it to the variable list.
                self.tape.watch(v)
                self.variables.append(v)

                for p in par_dep_list:
                    # Replace the existing Variable free parameter in the op_params dictionary
                    # with the corresponding tf.Variable parameter.
                    # Note that the free parameter might be scaled by the
                    # variable.mult scaling factor.
                    op_params[p.op][p.par_idx] = v*p.op.params[p.par_idx].mult

            # check that no Variables remain in the op_params dictionary
            values = [item for sublist in op_params.values() for item in sublist]
            assert not any(isinstance(v, Variable) for v in values)

            for operation in queue:
                # Apply each operation, but instead of passing operation.parameters
                # (which contains the evaluated numeric parameter values),
                # pass op_params[operation], which contains numeric values
                # for fixed parameters, and tf.Variable objects for free parameters.
                self.apply(operation.name, operation.wires, op_params[operation])

            for obs in observables:
                if obs.return_type is Expectation:
                    results.append(self.expval(obs.name, obs.wires, obs.parameters))

                elif obs.return_type is Variance:
                    results.append(self.var(obs.name, obs.wires, obs.parameters))

                elif obs.return_type is Sample:
                    results.append(np.array(self.sample(obs.name, obs.wires, obs.parameters)))

                elif obs.return_type is not None:
                    raise QuantumFunctionError("Unsupported return type specified for observable {}".format(obs.name))

            # convert the results list into a single tensor
            self.res = tf.stack(results)
            # flatten the variables list in case of nesting
            self.variables = tf.nest.flatten(self.variables)

            # return the results as a NumPy array
            return self.res.numpy()

    def jacobian(self, queue, observables, parameters):
        """Calculates the Jacobian of the device circuit using TensorFlow
        backpropagation.

        Args:
            queue (list[Operation]): operations to be applied to the device
            observables (list[Observable]): observables to be measured
            parameters (dict[int, ParameterDependency]): reference dictionary
                mapping free parameter values to the operations that
                depend on them

        Returns:
            array[float]: Jacobian matrix of size (``num_params``, ``num_wires``)
        """
        self.execute(queue, observables, parameters=parameters)
        jac = tf.stack(self.tape.jacobian(self.res, self.variables, experimental_use_pfor=False)).numpy().T
        return jac
