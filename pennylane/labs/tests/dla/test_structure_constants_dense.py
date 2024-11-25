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
"""Tests for pennylane/labs/dla/structure_constants.py functionality"""
# pylint: disable=no-self-use

import numpy as np
import pytest

import pennylane as qml
from pennylane import X, Y, Z
from pennylane.labs.dla import (
    batched_pauli_decompose,
    check_orthonormal,
    orthonormalize,
    structure_constants_dense,
    trace_inner_product,
)
from pennylane.pauli import PauliSentence, PauliWord

## Construct some example DLAs
# TFIM
gens = [PauliSentence({PauliWord({i: "X", i + 1: "X"}): 1.0}) for i in range(1)]
gens += [PauliSentence({PauliWord({i: "Z"}): 1.0}) for i in range(2)]
Ising2 = qml.lie_closure(gens, pauli=True)

gens = [PauliSentence({PauliWord({i: "X", i + 1: "X"}): 1.0}) for i in range(2)]
gens += [PauliSentence({PauliWord({i: "Z"}): 1.0}) for i in range(3)]
Ising3 = qml.lie_closure(gens, pauli=True)

# XXZ-type DLA, i.e. with true PauliSentences
gens2 = [
    PauliSentence(
        {
            PauliWord({i: "X", i + 1: "X"}): 1.0,
            PauliWord({i: "Y", i + 1: "Y"}): 1.0,
        }
    )
    for i in range(2)
]
gens2 += [PauliSentence({PauliWord({i: "Z"}): 1.0}) for i in range(3)]
XXZ3 = qml.lie_closure(gens2, pauli=True)

gens3 = [X(i) @ X(i + 1) + Y(i) @ Y(i + 1) + Z(i) @ Z(i + 1) for i in range(2)]
Heisenberg3_sum = qml.lie_closure(gens3)
Heisenberg3_sum = [op.pauli_rep for op in Heisenberg3_sum]

coeffs = np.random.random((len(XXZ3), len(XXZ3)))
sum_XXZ3 = [qml.sum(*(c * op for c, op in zip(_coeffs, XXZ3))).pauli_rep for _coeffs in coeffs]


class TestAdjointRepr:
    """Tests for structure_constants"""

    def test_structure_constants_dim(self):
        """Test the dimension of the adjoint repr"""
        d = len(Ising3)
        Ising3_dense = np.array([qml.matrix(op, wire_order=range(3)) for op in Ising3])
        adjoint = structure_constants_dense(Ising3_dense)
        assert adjoint.shape == (d, d, d)
        assert adjoint.dtype == float

    def test_structure_constants_with_is_orthonormal(self):
        """Test that the structure constants with is_orthonormal=True/False match for
        orthonormal inputs."""

        Ising3_dense = np.array([qml.matrix(op, wire_order=range(3)) for op in Ising3])
        assert check_orthonormal(Ising3_dense, trace_inner_product)
        adjoint_true = structure_constants_dense(Ising3_dense, is_orthonormal=True)
        adjoint_false = structure_constants_dense(Ising3_dense, is_orthonormal=False)
        assert np.allclose(adjoint_true, adjoint_false)

    @pytest.mark.parametrize("dla", [Ising2, Ising3, XXZ3, Heisenberg3_sum, sum_XXZ3])
    @pytest.mark.parametrize("use_orthonormal", [True, False])
    def test_structure_constants_elements(self, dla, use_orthonormal):
        r"""Test relation :math:`[i G_α, i G_β] = \sum_{γ=0}^{d-1} f^γ_{α,β} iG_γ_`."""

        d = len(dla)
        dla_dense = np.array([qml.matrix(op, wire_order=range(3)) for op in dla])

        if use_orthonormal:
            dla_dense = orthonormalize(dla_dense)
            assert check_orthonormal(dla_dense, trace_inner_product)
            dla = batched_pauli_decompose(dla_dense, pauli=True)
            assert check_orthonormal(dla, trace_inner_product)

        ad_rep_non_dense = qml.structure_constants(dla, is_orthogonal=False)
        ad_rep = structure_constants_dense(dla_dense, is_orthonormal=use_orthonormal)
        assert np.allclose(ad_rep, ad_rep_non_dense)
        for i in range(d):
            for j in range(d):

                comm_res = 1j * dla[i].commutator(dla[j])
                comm_res.simplify()
                res = sum((c + 0j) * dla[gamma] for gamma, c in enumerate(ad_rep[:, i, j]))
                res.simplify()
                assert set(comm_res) == set(res)  # Assert equal keys
                if len(comm_res) > 0:
                    assert np.allclose(*zip(*[(comm_res[key], res[key]) for key in res.keys()]))

    @pytest.mark.parametrize("dla", [Ising3, XXZ3])
    @pytest.mark.parametrize("use_orthonormal", [False, True])
    def test_use_operators(self, dla, use_orthonormal):
        """Test that operators can be passed and lead to the same result"""
        if use_orthonormal:
            dla = orthonormalize(dla)

        ops = np.array([qml.matrix(op.operation(), wire_order=range(3)) for op in dla])

        ad_rep_true = qml.structure_constants(dla)
        ad_rep = structure_constants_dense(ops, is_orthonormal=use_orthonormal)
        assert qml.math.allclose(ad_rep, ad_rep_true)
