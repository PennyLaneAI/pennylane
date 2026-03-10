# Copyright 2025 Xanadu Quantum Technologies Inc.

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
from copy import copy

import numpy as np
import pytest

import pennylane as qml
from pennylane.pauli import PauliSentence, PauliVSpace, PauliWord

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


class TestPauliVSpace:
    """Unit and integration tests for PauliVSpace class"""

    def test_init(self):
        """Unit tests for initialization"""
        vspace = PauliVSpace(ops1)

        assert all(isinstance(op, PauliSentence) for op in vspace.basis)
        assert np.allclose(
            vspace._M, [[1 / np.sqrt(2), 1.0], [1 / np.sqrt(2), 0.0]]
        ) or np.allclose(
            vspace._M, [[1 / np.sqrt(2), 0.0], [1 / np.sqrt(2), 1.0]]
        )  # the ordering is random as it is taken from a dictionary that has no natural ordering
        assert vspace.basis == ops1[:-1]
        assert vspace._rank == 2
        assert vspace._num_pw == 2
        assert len(vspace._pw_to_idx) == 2
        assert vspace.tol == np.finfo(vspace._M.dtype).eps * 100

    def test_empty_init(self):
        """Test that a PauliVSpace can be initialized as an empty vector space"""
        vspace = PauliVSpace([])
        assert not vspace.basis
        assert vspace._rank == 0
        assert vspace._num_pw == 0
        assert len(vspace._pw_to_idx) == 0
        assert vspace.tol == np.finfo(vspace._M.dtype).eps * 100

    def test_getitem(self):
        """Test dunder getitem method"""
        vspace = PauliVSpace(ops1)
        assert vspace[0] == vspace.basis[0]

    def test_iter(self):
        """Test dunder iter method"""
        vspace = PauliVSpace(ops1)
        assert list(vspace) == vspace.basis

    @pytest.mark.parametrize("dtype", [float, complex])
    def test_dtype(self, dtype):
        vspace = PauliVSpace(ops1, dtype=dtype)
        assert vspace._M.dtype == dtype

    def test_set_tol(self):
        vspace = PauliVSpace(ops1, tol=1e-13)
        assert vspace.tol == 1e-13

    def test_init_with_ops(self):
        """Test that initialization with PennyLane operators, PauliWord and PauliSentence works"""
        ops = [qml.X(0), PauliWord({1: "X"})]
        vspace = qml.pauli.PauliVSpace(ops)
        vspace.add(qml.Y(0))

        true_res = qml.pauli.PauliVSpace([qml.X(0), qml.X(1), qml.Y(0)])
        assert vspace == true_res

    ADD_LINEAR_INDEPENDENT = (
        (ops2, PauliWord({10: "Y"}), ops2plusY10),
        (ops2, PauliSentence({PauliWord({10: "Y"}): 1.0}), ops2plusY10),
        (ops2, qml.PauliY(10), ops2plusY10),
    )

    def test_repr(self):
        """Test the string representation of the PauliVSpace instance"""
        assert (
            repr(PauliVSpace(ops1)) == "[1.0 * X(0) @ X(1)\n+ 1.0 * Y(0) @ Y(1), 1.0 * X(0) @ X(1)]"
        )

    @pytest.mark.parametrize("li_tol", [None, 1e-15, 1e-14])
    @pytest.mark.parametrize("ops, op, true_new_basis", ADD_LINEAR_INDEPENDENT)
    def test_add_linear_independent(self, ops, op, true_new_basis, li_tol):
        """Test that adding new (linearly independent) operators works as expected"""
        vspace = PauliVSpace(ops)
        new_basis = vspace.add(op, li_tol)
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
        vspace = PauliVSpace(ops)
        new_basis = vspace.add(op)
        assert new_basis == true_new_basis

    @pytest.mark.parametrize("ops, op, true_new_basis", ADD_LINEAR_INDEPENDENT)
    def test_len(self, ops, op, true_new_basis):
        """Test the length of the PauliVSpace instance with inplace addition to the basis"""
        vspace = PauliVSpace(ops)
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

        vspace1 = PauliVSpace(gens1)
        vspace2 = PauliVSpace(gens2)
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

        vspace1 = PauliVSpace(gens1)
        vspace2 = PauliVSpace(gens2)
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

        vspace1 = PauliVSpace(gens1)
        vspace2 = PauliVSpace(gens2)
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

        vspace1 = PauliVSpace(gens1)
        vspace2 = PauliVSpace(gens2)
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

        vspace1 = PauliVSpace(gens1)
        vspace2 = PauliVSpace(gens2)
        assert vspace1 != vspace2

    def test_eq_False4(self):
        """Test case where both vspaces have the same rank, same PauliWords but span different spaces"""
        v1 = PauliVSpace(
            [
                PauliSentence({PauliWord({0: "X"}): 1.0, PauliWord({0: "Y"}): 1.0}),
                PauliSentence({PauliWord({0: "Z"}): 1.0, PauliWord({0: "Y"}): 1.0}),
            ]
        )
        v2 = PauliVSpace(
            [
                PauliSentence({PauliWord({0: "X"}): 1.0, PauliWord({0: "Z"}): 1.0}),
                PauliSentence({PauliWord({0: "Y"}): 1.0}),
            ]
        )
        assert v1 != v2

    IS_INDEPENDENT_TEST = (
        (
            [
                PauliSentence({PauliWord({0: "X"}): 1.0}),
                PauliSentence({PauliWord({0: "Z"}): 1.0}),
            ],
            PauliSentence({PauliWord({0: "Y"}): 1.0}),
            True,
        ),
        (
            [
                PauliSentence({PauliWord({0: "X"}): 1.0}),
                PauliSentence({PauliWord({0: "Z"}): 1.0}),
            ],
            PauliSentence({PauliWord({0: "Z"}): 1.0}),
            False,
        ),
        (
            [
                PauliSentence({PauliWord({0: "X"}): 1.0, PauliWord({1: "Y"}): 1.0}),
                PauliSentence({PauliWord({0: "Z", 1: "Z"}): 1.0}),
            ],
            PauliSentence({PauliWord({0: "Z"}): 1.0}),
            True,
        ),
        (
            [
                PauliSentence({PauliWord({0: "X"}): 1.0, PauliWord({1: "Y"}): 1.0}),
                PauliSentence({PauliWord({0: "Z", 1: "Z"}): 1.0}),
            ],
            PauliSentence({PauliWord({0: "Z", 1: "Z"}): 1.0}),
            False,
        ),
    )

    @pytest.mark.parametrize("tol", [None, 1e-15, 1e-8])
    @pytest.mark.parametrize("ops, op, is_independent_true", IS_INDEPENDENT_TEST)
    def test_is_independent(self, ops, op, is_independent_true, tol):
        """Test the `is_independent` method returns correct results and leaves class attributes intact"""
        v1 = PauliVSpace(ops)
        vcopy = copy(v1)

        is_independent = v1.is_independent(op, tol=tol)
        assert is_independent == is_independent_true
        assert qml.math.allclose(v1._M, vcopy._M)
        assert v1._pw_to_idx == vcopy._pw_to_idx
        assert v1._rank == vcopy._rank
        assert v1._num_pw == vcopy._num_pw
