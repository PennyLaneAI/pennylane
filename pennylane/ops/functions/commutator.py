# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This file contains the implementation of the commutator function in PennyLane
"""
# import itertools
# from copy import copy
# from typing import List

# import numpy as np

import pennylane as qml
from pennylane.pauli import PauliWord, PauliSentence

# from pennylane import math
# from pennylane.operation import Operator
# from pennylane.ops.qubit import Hamiltonian
# from pennylane.queuing import QueuingManager

# from .composite import CompositeOp


def commutator(op1, op2, simplify=True, pauli=False):
    r"""Compute commutator between two operators in PennyLane

    .. math:: [O_1, O_2] = O_1 O_2 - O_2 O_1

    Args:
        op1 (Union[Operator, PauliWord, PauliSentence]): First operator
        op2 (Union[Operator, PauliWord, PauliSentence]): Second operator
        pauli (bool): When ``True``, all results are passed as a ``PauliSentence`` instance. Else, results are always returned as ``Operator`` instances.

    Returns:
        ~Operator or ~PauliSentence: The commutator

    **Examples**

    """
    if pauli:
        if not isinstance(op1, (PauliSentence)):
            op1 = qml.pauli.pauli_sentence(op1)
        if not isinstance(op2, (PauliSentence)):
            op2 = qml.pauli.pauli_sentence(op2)
        return op1 @ op2 - op2 @ op1

    if isinstance(op1, (PauliWord, PauliSentence)):
        op1 = op1.operation()
    if isinstance(op2, (PauliWord, PauliSentence)):
        op2 = op2.operation()

    res = qml.prod(op1, op2) - qml.prod(op2, op1)
    return res.simplify()
