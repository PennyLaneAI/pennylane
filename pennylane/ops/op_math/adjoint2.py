# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the base class for the adjoint of operators."""

from typing_extensions import override

import pennylane as qp
from pennylane import math
from pennylane._class_property import classproperty
from pennylane.operation2 import Operator2

from .symbolicop2 import SymbolicOp2


class Adjoint2(SymbolicOp2):
    """The adjoint of an operator."""

    def __init__(self, base: Operator2):
        super().__init__(base)

    @property
    @override
    def pauli_rep(self):
        if not self.base.pauli_rep:
            return None
        rep = {pw: math.conjugate(c) for pw, c in self.base.pauli_rep.items()}
        return qp.pauli.PauliSentence(rep)

    def __repr__(self) -> str:
        return f"Adjoint({self.base})"

    @property
    @override
    def name(self) -> str:
        return f"Adjoint({self.base.name})"

    @override
    def label(self, decimals=None, base_label=None, cache=None) -> str:
        base_label = self.base.label(decimals, base_label, cache=cache)
        if self.base.arithmetic_depth > 0 and len(base_label) > 1:
            base_label = f"({base_label})"
        return f"{base_label}†"

    @staticmethod
    @override
    def compute_matrix(base):  # pylint: disable=arguments-differ
        base_matrix = base.matrix()
        return math.moveaxis(math.conj(base_matrix), -2, -1)

    @staticmethod
    @override
    def compute_sparse_matrix(base, format="csr"):  # pylint: disable=arguments-differ
        base_matrix = base.sparse_matrix()
        return math.transpose(math.conj(base_matrix)).asformat(format=format)

    @staticmethod
    @override
    def compute_eigvals(base):  # pylint: disable=arguments-differ
        return math.conj(base.eigvals())

    @property
    @override
    def has_diagonalizing_gates(self):  # pylint: disable=arguments-differ,invalid-overridden-method
        return self.base.has_diagonalizing_gates

    @staticmethod
    @override
    def compute_diagonalizing_gates(base):  # pylint: disable=arguments-differ
        return base.diagonalizing_gates

    @classproperty
    @classmethod
    @override
    def has_adjoint(cls):
        return True

    @override
    def adjoint(self):
        return type(self.base)(**self.base.arguments)

    @override
    def simplify(self):
        base = self.base.simplify()
        if base.has_adjoint:
            return base.adjoint().simplify()
        return Adjoint2(base)

    @property
    @override
    def has_generator(self):  # pylint: disable=arguments-differ,invalid-overridden-method
        return self.base.has_generator

    @override
    def generator(self):
        return -1 * self.base.generator()
