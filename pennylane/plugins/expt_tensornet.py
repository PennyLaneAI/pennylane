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
import functools

import numpy as np
import tensornetwork as tn

from pennylane import Device
from pennylane.plugins.default_qubit import I, X, Y, Z, H, CNOT, SWAP, CZ, S, T, CSWAP, \
    Rphi, Rotx, Roty, Rotz, Rot3, CRot3, CRotx, CRoty, CRotz, CRot3

# tolerance for numerical errors
tolerance = 1e-10

#========================================================
#  device
#========================================================


class TensorNetwork(Device):
    """Experimental Tensor Network simulator device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
    """
    name = 'PennyLane TensorNetwork simulator plugin'
    short_name = 'expt.tensornet'
    pennylane_requires = '0.7'
    version = '0.7.0'
    author = 'Xanadu Inc.'
    _capabilities = {"model": "qubit", "tensor_observables": True}

    _operation_map = {
        'PauliX': X,
        'PauliY': Y,
        'PauliZ': Z,
        'Hadamard': H,
        'S': S,
        'T': T,
        'CNOT': CNOT,
        'SWAP': SWAP,
        'CSWAP':CSWAP,
        'CZ': CZ,
        'PhaseShift': Rphi,
        'RX': Rotx,
        'RY': Roty,
        'RZ': Rotz,
        'Rot': Rot3,
        'CRX': CRotx,
        'CRY': CRoty,
        'CRZ': CRotz,
        'CRot': CRot3
    }

    _observable_map = {
        'PauliX': X,
        'PauliY': Y,
        'PauliZ': Z,
        'Hadamard': H
    }

    def __init__(self, wires):
        super().__init__(wires, shots=1)
        self.eng = None
        self.analytic = True
        self._zero_state = np.zeros([2] * wires)
        self._zero_state[[0] * wires] = 1.0
        self._input_state_node = tn.Node(self._zero_state)
        self._nodes = [(self._input_state_node, tuple(idx for idx in range(wires)))]

    def pre_apply(self):
        self.reset()

    def apply(self, operation, wires, par):
        A = self._get_operator_matrix(operation, par)
        op_node = tn.Node(A)
        self._nodes.append((op_node, tuple(wires)))
        # TODO: modify for multi-wire ops
        wires = wires[0]
        # TODO: can be smarter here about collecting contractions
        # TODO: confirm op_node[0] or op_node[1] for matrix-vector multiplication
        self._state = tn.contract(tn.connect(op_node[1],  self._state[wires]))

    def expval(self, observable, wires, par):
        if isinstance(observable, list):
            A = self._get_tensor_operator_matrix(observable, par)
        else:
            A = self._get_operator_matrix(observable, par)
        return self.ev(A, wires)

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

    def _get_tensor_operator_matrix(self, obs, par):
        """Get the operator matrix for a given tensor product of operations.

        Args:
            obs (list[str]): list of observable names to tensor
            par (list[list[Any]]): parameter values

        Returns:
            array: matrix representation.
        """
        ops = [self._get_operator_matrix(o, p) for o, p in zip(obs, par)]
        return functools.reduce(np.kron, ops)

    def ev(self, A, wires):
        r"""Expectation value of observable on specified wires.

         Args:
            A (array[float]): the observable matrix as array
            wires (Sequence[int]): target subsystems
         Returns:
            float: expectation value :math:`\expect{A} = \bra{\psi}A\ket{\psi}`
        """
        obs_node = tn.Node(A)
        self._nodes.append((obs_node, tuple(wires)))
        ket = self._state
        bra = tn.conj(ket)
        self._nodes.append((bra, tuple(idx for idx in range(self.num_wires))))
        for w in range(self.num_wires):
            if w in wires:
                tn.connect(obs_node[1], ket[w])
                tn.connect(bra[w], obs_node[0])
            else:
                tn.connect(bra[w], ket[w])
        obs_ket = tn.contract_between(obs_node, ket)
        expval = tn.contract_between(bra, obs_ket).tensor
        if np.abs(expval.imag) > tolerance:
            warnings.warn('Nonvanishing imaginary part {} in expectation value.'.format(expval.imag), RuntimeWarning)
        return expval.real

    def reset(self):
        """Reset the device"""
        self._state = self._input_state_node

    @property
    def operations(self):
        return set(self._operation_map.keys())

    @property
    def observables(self):
        return set(self._observable_map.keys())

