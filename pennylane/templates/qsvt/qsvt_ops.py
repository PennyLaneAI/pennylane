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
"""
Contains core operations used in the Quantum Singular Value Transform framework.
"""
import numpy as np
from scipy.linalg import sqrtm, norm

from pennylane.operation import Operation, AnyWires
from pennylane.ops.qubit.non_parametric_ops import PauliX
from pennylane.ops import PhaseShift, ctrl
from pennylane import QuantumFunctionError


class BlockEncode(Operation):
    """General Block Encoding"""
    num_params = 1
    num_wires = AnyWires

    def __init__(self, a, wires, do_queue=True, id=None):
        if isinstance(a, int) or isinstance(a, float):
            normalization = a if a > 1 else 1
            subspace = (1, 1, 2**len(wires))
        else:
            normalization = np.max([norm(a @ np.conj(a).T, ord=np.inf), norm(np.conj(a).T @ a, ord=np.inf)])
            subspace = (*a.shape, 2**len(wires))

        a = a / normalization if normalization > 1 else a

        if subspace[2] < (subspace[0] + subspace[1]):
            raise QuantumFunctionError(f"Block encoding an {subspace[0]} x {subspace[1]} matrix requires a hilbert soace"
                                       f" of size atleast {subspace[0] + subspace[1]} x {subspace[0] + subspace[1]}."
                                       f"Cannot be embedded in a {len(wires)} qubit system.")

        super().__init__(a, wires=wires, do_queue=do_queue, id=id)
        self.hyperparameters["norm"] = normalization
        self.hyperparameters["subspace"] = subspace

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        """Get the matrix representation of block encoded unitary."""
        a = params[0]
        n,m,k = hyperparams["subspace"]

        if isinstance(a,int) or isinstance(a,float):
            u = np.block([[a,np.sqrt(1-a*np.conj(a))],
                        [np.sqrt(1-a*np.conj(a)), -np.conj(a)]])
        else:
            d1, d2 = a.shape
            u = np.block([[a, sqrtm(np.eye(d1) - a @ np.conj(a).T)],
                        [sqrtm(np.eye(d2) - np.conj(a).T @ a), -np.conj(a).T]])        
        if n+m < k:
            r = k - (n + m)
            u = np.block([[u, np.zeros((n+m, r))],
                          [np.zeros((r, n+m)), np.eye(r)]])
        return u


class PCPhase(Operation):
    """A Pi-Controlled Phase gate"""
    num_wires = AnyWires

    def __init__(self, phi, dim, wires, do_queue=True, id=None):
        """Pi-Controlled phase gate.

        Args:
            phi (float): The phase we wish to apply.
            dim (int): Represents the first dim entries to apply phase along
        """
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)
        self.hyperparameters["dimension"] = (dim, 2**len(wires))

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        """Get the matrix representation of Pi-controlled phase unitary."""
        phi = params[0]
        d, t = hyperparams["dimension"]
        diag_vals = [np.exp(1j * phi) if index < d else 1 for index in range(t)]
        return np.diag(diag_vals)

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. note::

            Operations making up the decomposition should be queued within the
            ``compute_decomposition`` method.

        .. seealso:: :meth:`~.Operator.decomposition`.

        Args:
            params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            wires (Iterable[Any], Wires): wires that the operator acts on
            hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute

        Returns:
            list[Operator]: decomposition of the operator
        """
        phi = params[0]
        k, n = hyperparameters["dimension"]

        def _get_base_ops(theta, wire):
            return PauliX(wire) @ PhaseShift(theta, wire) @ PauliX(wire), PhaseShift(theta, wire)

        def _get_op_from_binary_rep(binary_rep, theta, wires):
            if len(binary_rep) == 1:
                op = _get_base_ops(theta, wire=wires[0])[int(binary_rep)]
            else:
                base_op = _get_base_ops(theta, wire=wires[-1])[int(binary_rep[-1])]
                op = ctrl(base_op, control=wires[:-1], control_values=[int(i) for i in binary_rep[:-1]])
            return op

        n_log2 = int(np.log2(n))
        binary_reps = [bin(_k)[2:].zfill(n_log2) for _k in range(k)]

        return [_get_op_from_binary_rep(br, phi, wires=wires) for br in binary_reps]
