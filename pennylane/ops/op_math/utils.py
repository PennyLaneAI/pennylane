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
This file contains the implementation of some utility functions to extend
the capabilities of operator arithmetic to more complicated operators.
"""
from pennylane.operation import Operator
from pennylane.ops.identity import Identity
from pennylane.ops.qubit import PauliX, PauliY, PauliZ

from pennylane.ops.op_math.sum import Sum
from pennylane.ops.op_math.prod import Prod
from pennylane.ops.op_math.sprod import SProd


PAULI_OPS = (PauliX, PauliY, PauliZ)
commutation_relations = {
    (PauliX, PauliX): (0, Identity),
    (PauliY, PauliY): (0, Identity),
    (PauliZ, PauliZ): (0, Identity),

    (PauliX, PauliY): (2j, PauliZ),
    (PauliY, PauliZ): (2j, PauliX),
    (PauliZ, PauliX): (2j, PauliY),

    (PauliY, PauliX): (-2j, PauliZ),
    (PauliZ, PauliY): (-2j, PauliX),
    (PauliX, PauliZ): (-2j, PauliY),
}

anti_commutation_relations = {
    (PauliX, PauliX): (2, Identity),
    (PauliY, PauliY): (2, Identity),
    (PauliZ, PauliZ): (2, Identity),

    (PauliX, PauliY): (0, Identity),
    (PauliY, PauliZ): (0, Identity),
    (PauliZ, PauliX): (0, Identity),

    (PauliY, PauliX): (0, Identity),
    (PauliZ, PauliY): (0, Identity),
    (PauliX, PauliZ): (0, Identity),
}


def Sigma(i: int, n: int, of: callable) -> Operator:
    """Simple utility function to generate the sum operator for
    summing over a given index for some function which depends on the index.
    """
    return Sum(*(of(index) for index in range(i, n+1)))


def Pi(i: int, n: int, of: callable) -> Operator:
    """Simple utility function to generate the product operator for
    multiplying over a given index for some function which depends on the index. """
    return Prod(*(of(index) for index in range(i, n+1)))


def commutator(op1: Operator, op2: Operator, lazy=False) -> Operator:
    r"""Computes the commutator of the given operators.
        This is given by the expression: $[A, B] = AB - BA$
    """
    if not lazy:
        types = (type(op1), type(op2))
        wires = (op1.wires, op2.wires)

        if (Identity in types) or (wires[0] != wires[1]):
            op = Prod(*(Identity(i) for i in wires[0].tolist() + wires[1].tolist()))
            return SProd(0, op)

        if all(type_op in PAULI_OPS for type_op in types):
            scalar, op = commutation_relations[(types[0], types[1])]
            return SProd(scalar, op(wires[0]))

    return Sum(Prod(op1, op2), SProd(-1, Prod(op2, op1)))  # return arithmetic commutator


def anti_commutator(op1: Operator, op2: Operator, lazy=False) -> Operator:
    r"""Computes the anti-commutator of the given operators.
        This is given by the expression: ${A, B} = AB + BA$
    """
    if not lazy:
        types = (type(op1), type(op2))
        wires = (op1.wires, op2.wires)

        if (Identity in types) or (wires[0] != wires[1]):
            return SProd(2, Prod(op1, op2))

        if all(type_op in PAULI_OPS for type_op in types):
            scalar, op = anti_commutation_relations[(types[0], types[1])]  # get the op after anti-commutation
            return SProd(scalar, op(wires[0]))

    return Sum(Prod(op1, op2), Prod(op2, op1))  # return arithmetic anti-commutator
