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
"""
This module contains the Identity operation that is common to both
cv and qubit computing paradigms in PennyLane.
"""
import numpy as np

from pennylane.operation import CVObservable, Operation


class Identity(CVObservable, Operation):
    r"""pennylane.Identity(wires)
    The identity observable :math:`\I`.

    The expectation of this observable

    .. math::
        E[\I] = \text{Tr}(\I \rho)

    corresponds to the trace of the quantum state, which in exact
    simulators should always be equal to 1.
    """
    num_wires = 1
    grad_method = None

    ev_order = 1
    eigvals = np.array([1, 1])

    @property
    def num_params(self):
        return 0

    def label(self, decimals=None, base_label=None):
        return base_label or "I"

    @classmethod
    def _eigvals(cls, *params):
        return cls.eigvals

    @classmethod
    def _matrix(cls, *params):
        return np.eye(2)

    @staticmethod
    def _heisenberg_rep(p):
        return np.array([1, 0, 0])

    def diagonalizing_gates(self):
        return []

    @staticmethod
    def identity_op(*params):
        """Returns the matrix representation of the identity operator."""
        return Identity._matrix(*params)

    def adjoint(self):  # pylint:disable=arguments-differ
        return Identity(wires=self.wires)
