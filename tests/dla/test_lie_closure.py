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
"""Tests for pennylane/dla/lie_closure.py functionality"""
# pylint: disable=too-few-public-methods, protected-access
import pytest
import numpy as np

import pennylane as qml

from pennylane import X, Y, Z
from pennylane.dla import lie_closure
from pennylane.dla.lie_closure import VSpace
from pennylane.pauli import PauliWord, PauliSentence

ops1 = [
    PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0, PauliWord({0: "Y", 1: "Y"}): 1.0}),
    PauliSentence(
        {
            PauliWord({0: "X", 1: "X"}): 1.0,
        }
    ),
    PauliSentence(
        {
            PauliWord({0: "Y", 1: "Y"}): 2.0,
        }
    ),
]

ops2 = [
    PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0, PauliWord({0: "Y", 1: "Y"}): 1.0}),
    PauliSentence(
        {
            PauliWord({0: "X", 1: "X"}): 1.0,
        }
    ),
]

ops2plusY10 = ops2 + [PauliSentence({PauliWord({10: "Y"}): 1.0})]


class TestVSpace:
    """Unit and integration tests for VSpace class"""

    def test_init(self):
        """Unit tests for initialization"""
        vspace = VSpace(ops1)

        assert all(isinstance(op, PauliSentence) for op in vspace.basis)
        assert np.allclose(vspace._M, [[1.0, 1.0], [1.0, 0.0]]) or np.allclose(
            vspace._M, [[1.0, 0.0], [1.0, 1.0]]
        )  # the ordering is random as it is taken from a dictionary that has no natural ordering
        assert vspace.basis == ops1[:-1]
        assert vspace._rank == 2
        assert vspace._num_pw == 2
        assert len(vspace._pw_to_idx) == 2

    ADD_LINEAR_INDEPENDENT = (
        (ops2, PauliWord({10: "Y"}), ops2plusY10),
        (ops2, PauliSentence({PauliWord({10: "Y"}): 1.0}), ops2plusY10),
        (ops2, qml.PauliY(10), ops2plusY10),
    )

    @pytest.mark.parametrize("ops, op, true_new_basis", ADD_LINEAR_INDEPENDENT)
    def test_add_linear_independent(self, ops, op, true_new_basis):
        """Test that adding new (linearly independent) operators works as expected"""
        vspace = VSpace(ops)
        new_basis = vspace.add(op)
        assert new_basis == true_new_basis

    ADD_LINEAR_DEPENDENT = (
        (ops2, PauliWord({0: "Y", 1: "Y"}), ops2),
        (ops2, PauliSentence({PauliWord({0: "Y", 1: "Y"}): 1.0}), ops2),
        (ops2, qml.PauliY(0) @ qml.PauliY(1), ops2),
        (ops2, 0.5 * ops2[0], ops2),
        (ops2, 0.5 * ops2[1], ops2),
    )

    @pytest.mark.parametrize("ops, op, true_new_basis", ADD_LINEAR_DEPENDENT)
    def test_add_lin_dependent(self, ops, op, true_new_basis):
        """Test that adding linearly dependent operators works as expected"""
        vspace = VSpace(ops)
        new_basis = vspace.add(op)
        assert new_basis == true_new_basis

    @pytest.mark.parametrize("ops, op, true_new_basis", ADD_LINEAR_INDEPENDENT)
    def test_len(self, ops, op, true_new_basis):
        """Test the length of the VSpace instance with inplace addition to the basis"""
        vspace = VSpace(ops)
        len_before_adding = len(vspace)
        len_basis_before_adding = len(vspace.basis)

        _ = vspace.add(op)
        len_after_adding = len(vspace)
        assert len_after_adding != len_before_adding
        assert len_after_adding == len(true_new_basis)
        assert len_before_adding == len_basis_before_adding

    def test_eq_True(self):
        """Test that equivalent vspaces are correctly determined"""
        gens1 = [
            PauliSentence({PauliWord({0: "X"}): 1.0}),
            PauliSentence({PauliWord({0: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "Z"}): 1.0}),
        ]

        gens2 = [
            PauliSentence({PauliWord({0: "Z"}): 1.0, PauliWord({0: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "X"}): 1.0, PauliWord({0: "Z"}): 1.0}),
            PauliSentence({PauliWord({0: "Y"}): 1.0}),
        ]

        vspace1 = VSpace(gens1)
        vspace2 = VSpace(gens2)
        assert vspace1 == vspace2

    def test_eq_False0(self):
        """Test that non-equivalent vspaces are correctly determined"""
        # Different _num_pw
        gens1 = [
            PauliSentence({PauliWord({0: "X"}): 1.0}),
            PauliSentence({PauliWord({0: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "Z"}): 1.0}),
        ]

        gens2 = [
            PauliSentence({PauliWord({0: "Z"}): 1.0}),
            PauliSentence({PauliWord({0: "X"}): 1.0, PauliWord({0: "Z"}): 1.0}),
        ]

        vspace1 = VSpace(gens1)
        vspace2 = VSpace(gens2)
        assert vspace1 != vspace2

    def test_eq_False1(self):
        """Test that non-equivalent vspaces are correctly determined"""
        # Same num_pw but acting on different wires
        gens1 = [
            PauliSentence({PauliWord({0: "X"}): 1.0}),
            PauliSentence({PauliWord({0: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "Z"}): 1.0}),
        ]

        gens2 = [
            PauliSentence({PauliWord({1: "X"}): 1.0}),
            PauliSentence({PauliWord({0: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "Z"}): 1.0}),
        ]

        vspace1 = VSpace(gens1)
        vspace2 = VSpace(gens2)
        assert vspace1 != vspace2

    def test_eq_False2(self):
        """Test that non-equivalent vspaces are correctly determined"""
        # Same num_pw but acting on different wires
        gens1 = [
            PauliSentence({PauliWord({0: "X"}): 1.0}),
            PauliSentence({PauliWord({0: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "Z"}): 1.0}),
        ]

        gens2 = [
            PauliSentence({PauliWord({1: "Z"}): 1.0, PauliWord({0: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "X"}): 1.0, PauliWord({1: "Z"}): 1.0}),
            PauliSentence({PauliWord({0: "Y"}): 1.0}),
        ]

        vspace1 = VSpace(gens1)
        vspace2 = VSpace(gens2)
        assert vspace1 != vspace2

    def test_eq_False3(self):
        """Test that non-equivalent vspaces are correctly determined"""
        # Same num_pw, even same pws, but not spanning the same space
        # vector equivalent of ((1,1,0), (0, 0, 1)) and ((1,0,0), (0,1,0), (0,0,1))
        # I.e. different rank
        gens1 = [
            PauliSentence({PauliWord({0: "X"}): 1.0}),
            PauliSentence({PauliWord({0: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "Z"}): 1.0}),
        ]

        gens2 = [
            PauliSentence({PauliWord({1: "X"}): 1.0, PauliWord({0: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "Z"}): 1.0}),
        ]

        vspace1 = VSpace(gens1)
        vspace2 = VSpace(gens2)
        assert vspace1 != vspace2

    def test_eq_False4(self):
        """Test case where both vspaces have the same rank, same PauliWords but span different spaces"""
        v1 = VSpace(
            [
                PauliSentence({PauliWord({0: "X"}): 1.0, PauliWord({0: "Y"}): 1.0}),
                PauliSentence({PauliWord({0: "Z"}): 1.0, PauliWord({0: "Y"}): 1.0}),
            ]
        )
        v2 = VSpace(
            [
                PauliSentence({PauliWord({0: "X"}): 1.0, PauliWord({0: "Z"}): 1.0}),
                PauliSentence({PauliWord({0: "Y"}): 1.0}),
            ]
        )
        assert v1 != v2


dla11 = [
    PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0, PauliWord({0: "Y", 1: "Y"}): 1.0}),
    PauliSentence(
        {
            PauliWord({0: "Z"}): 1.0,
        }
    ),
    PauliSentence(
        {
            PauliWord({1: "Z"}): 1.0,
        }
    ),
    PauliSentence({PauliWord({0: "Y", 1: "X"}): -1.0, PauliWord({0: "X", 1: "Y"}): 1.0}),
]


class TestLieClosure:
    """Tests for qml.dla.lie_closure()"""

    def test_simple_lie_closure(self):
        """Test simple lie_closure example"""
        gen11 = dla11[:-1]
        res11 = lie_closure(gen11)
        assert res11 == dla11

        dla12 = [
            PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0, PauliWord({0: "Y", 1: "Y"}): 1.0}),
            PauliSentence(
                {
                    PauliWord({0: "Z"}): 1.0,
                }
            ),
            PauliSentence({PauliWord({0: "Y", 1: "X"}): -1.0, PauliWord({0: "X", 1: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "Z"}): -2.0, PauliWord({1: "Z"}): 2.0}),
        ]
        gen12 = dla12[:-1]
        res12 = lie_closure(gen12)
        assert VSpace(res12) == VSpace(dla12)

    def test_lie_closure_with_pl_ops(self):
        """Test that lie_closure works properly with PennyLane ops instead of PauliSentences"""
        dla = [
            qml.sum(qml.prod(X(0), X(1)), qml.prod(Y(0), Y(1))),
            Z(0),
            Z(1),
            qml.sum(qml.prod(Y(0), X(1)), qml.s_prod(-1.0, qml.prod(X(0), Y(1)))),
        ]
        gen11 = dla[:-1]
        res11 = lie_closure(gen11)
        assert VSpace(res11) == VSpace(dla11)

    def test_lie_closure_with_PauliWords(self):
        """Test that lie_closure works properly with PauliWords"""
        gen = [
            PauliWord({0: "X", 1: "X"}),
            PauliWord({0: "Z"}),
            PauliWord({1: "Z"}),
        ]
        dla = gen + [
            PauliWord({0: "Y", 1: "X"}),
            PauliWord({0: "X", 1: "Y"}),
            PauliWord({0: "Y", 1: "Y"}),
        ]
        dla = [op.pauli_rep for op in dla]

        res = lie_closure(gen)
        assert VSpace(res) == VSpace(dla)

    @pytest.mark.parametrize("n", range(2, 5))
    def test_lie_closure_transverse_field_ising_1D_open(self, n):
        """Test the lie closure works correctly for the transverse Field Ising model with open boundary conditions, a8 in theorem IV.1 in https://arxiv.org/pdf/2309.05690.pdf"""
        generators = [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "X"}): 1.0}) for i in range(n - 1)
        ]
        generators += [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "Z"}): 1.0}) for i in range(n - 1)
        ]

        res = qml.dla.lie_closure(generators)
        assert len(res) == (2 * n - 1) * (2 * n - 2) // 2

    @pytest.mark.parametrize("n", range(3, 5))
    def test_lie_closure_transverse_field_ising_1D_cyclic(self, n):
        """Test the lie closure works correctly for the transverse Field Ising model with cyclic boundary conditions, a8 in theorem IV.2 in https://arxiv.org/pdf/2309.05690.pdf"""
        generators = [PauliSentence({PauliWord({i: "X", (i + 1) % n: "X"}): 1.0}) for i in range(n)]
        generators += [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "Z"}): 1.0}) for i in range(n)
        ]

        res = qml.dla.lie_closure(generators)
        assert len(res) == 2 * n * (2 * n - 1)

    def test_lie_closure_heisenberg_generators_odd(self):
        """Test the resulting DLA from Heisenberg generators with odd n, a7 in theorem IV.1 in https://arxiv.org/pdf/2309.05690.pdf"""
        n = 3
        # dim of su(N) is N ** 2 - 1
        # Heisenberg generates su(2**(n-1)) for n odd            => dim should be (2**(n-1))**2 - 1
        generators = [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "X"}): 1.0}) for i in range(n - 1)
        ]
        generators += [
            PauliSentence({PauliWord({i: "Y", (i + 1) % n: "Y"}): 1.0}) for i in range(n - 1)
        ]
        generators += [
            PauliSentence({PauliWord({i: "Z", (i + 1) % n: "Z"}): 1.0}) for i in range(n - 1)
        ]

        res = qml.dla.lie_closure(generators)
        assert len(res) == (2 ** (n - 1)) ** 2 - 1

    def test_lie_closure_heisenberg_generators_even(self):
        """Test the resulting DLA from Heisenberg generators with odd n, a7 in theorem IV.1 in https://arxiv.org/pdf/2309.05690.pdf"""
        n = 4
        # dim of su(N) is N ** 2 - 1
        # Heisenberg generates (su(2**(n-2)))**4 for n>=4 even   => dim should be 4*((2**(n-2))**2 - 1)
        generators = [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "X"}): 1.0}) for i in range(n - 1)
        ]
        generators += [
            PauliSentence({PauliWord({i: "Y", (i + 1) % n: "Y"}): 1.0}) for i in range(n - 1)
        ]
        generators += [
            PauliSentence({PauliWord({i: "Z", (i + 1) % n: "Z"}): 1.0}) for i in range(n - 1)
        ]

        res = qml.dla.lie_closure(generators)
        assert len(res) == 4 * ((2 ** (n - 2)) ** 2 - 1)
