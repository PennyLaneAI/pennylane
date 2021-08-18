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
"""Contains transforms and helpers functions for decomposing arbitrary two-qubit
unitary operations into elementary gates.
"""
import pennylane as qml
from pennylane import math

def two_qubit_decomposition(U, wires):
    r"""Recover the decomposition of a two-qubit matrix :math:`U` in terms of
    elementary operations.

    The work of https://arxiv.org/abs/quant-ph/0308033 presents a fixed-form
    decomposition of U in terms of single-qubit gates and CNOTs. Multiple such
    decompositions are possible (by choosing two of {``RX``, ``RY``, ``RZ``}),
    here we choose the ``RY``, ``RZ`` case (fig. 2 in the above) to match with
    the default decomposition of the single-qubit ``Rot`` operations as ``RZ RY
    RZ``. The form of the decomposition is:

     0: ─C─X─RZ─C───X─A─|
     1: ─D─C─RY─X─RY─C─B─|

    where A, B, C, D are SU(2) gates.

    Args:
        U (tensor): A 4 x 4 unitary matrix.
        wires (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.

    Returns:
        list[qml.Operation]: A list of operations that represent the decomposition
        of the matrix U.
    """


    return
