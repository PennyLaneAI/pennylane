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
"""This file contains the implementation of the commutator and anti-commutator classes."""

import itertools
from copy import copy
from functools import reduce
from itertools import combinations
from typing import List, Tuple, Union

from scipy.sparse import kron as sparse_kron

import pennylane as qml
from pennylane import math
from pennylane.operation import Operator
from pennylane.ops.op_math.prod import Prod
from pennylane.ops.op_math.sprod import SProd
from pennylane.ops.op_math.sum import Sum
from pennylane.ops.qubit.non_parametric_ops import PauliX, PauliY, PauliZ
from pennylane.ops import Identity

from .composite import CompositeOp


def commutator(*operands, lazy=False, do_queue=True, id=None):
    commutator_op = Commutator(*operands, do_queue=do_queue, id=id)

    if not lazy:
        A, B = commutator_op.operands
        if Identity in (type(A), type(B)):
            return B if isinstance(A, Identity) else A
        if A in (PauliX, PauliY, PauliZ)

    return


class Commutator(CompositeOp):
    """Symbolic class representing the commutator of two operators.

    The commutator is given by: $[A, B] = AB - BA$
    """
    _op_symbol = None

    def __init__(self, *operands: Operator, do_queue=True, id=None):
        if len(operands) != 2:
            raise ValueError(f"Expected two operands got {operands}")

        super().__init__(*operands, do_queue=do_queue, id=id)
        A, B = operands
        self.simplified_rep = Sum(Prod(A, B), SProd(-1, Prod(B, A)))

    def is_hermitian(self):
        """This property determines if the composite operator is hermitian."""
        return False  # no quick way to check for this without knowing if they commute

    def eigvals(self):
        """Return the eigenvalues of the specified operator."""
        return self.eigendecomposition["eigval"]

    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis."""
        return qml.matrix(self.simplified_rep, wire_order=None)

    def _sort(cls, op_list, wire_map: dict = None) -> List[Operator]:
        """Can't change the order of operands in this operation."""
        return op_list


class Anti_Commutator(CompositeOp):
    """Symbolic class representing the anti-commutator of two operators.

    The anti-commutator is given by: ${A, B} = AB + BA$
    """
    _op_symbol = None