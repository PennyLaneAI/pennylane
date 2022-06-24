# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This submodule contains the discrete-variable quantum observables,
excepting the Pauli gates and Hadamard gate in ``non_parametric_ops.py``.
"""

import numpy as np

import pennylane as qml
from pennylane.operation import AllWires, AnyWires, Observable
from pennylane.wires import Wires
from .matrix_ops import QubitUnitary

class THermitian(Observable):
    r"""
    An arbitrary Ternary Hermitian observable.

    For a Hermitian matrix :math:`A`, the expectation command returns the value

    .. math::
        \braket{A} = \braketT{\psi}{\cdots \otimes I\otimes A\otimes I\cdots}{\psi}

    where :math:`A` acts on the requested wires.

    If acting on :math:`N` wires, then the matrix :math:`A` must be of size
    :math:`3^N\times 3^N`.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        A (array): square hermitian matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = AnyWires
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    # TODO: Add grad_method
    _eigs = {}

    def __init__(self, A, wires, do_queue=True, id=None):
        super().__init__(A, wires=wires, do_queue=do_queue, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "𝓗", cache=cache)