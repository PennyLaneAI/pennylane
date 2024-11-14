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
import scipy as sp

import pennylane as qml
from pennylane import X, Y, Z
from pennylane.labs.dla import structure_constants_dense
from pennylane.pauli import PauliSentence, PauliWord

## Construct some example DLAs
# TFIM
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

gens3 = [X(i) @ X(i+1) + Y(i) @ Y(i+1) + Z(i) @ Z(i+1) for i in range(2)]
Heisenberg3_sum = qml.lie_closure(gens3)
Heisenberg3_sum = [op.pauli_rep for op in Heisenberg3_sum]


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
        orthogonal inputs."""

        Ising3_dense = np.array([qml.matrix(op, wire_order=range(3)) for op in Ising3]) / np.sqrt(8)
        adjoint_true = structure_constants_dense(Ising3_dense, is_orthonormal=True)
        adjoint_false = structure_constants_dense(Ising3_dense, is_orthonormal=False)
        assert np.allclose(adjoint_true, adjoint_false)

    @pytest.mark.parametrize("dla, use_orthonormal", [(Ising3, True), (Ising3, False), (XXZ3, True), (XXZ3, False), (Heisenberg3_sum, False), (Heisenberg3_sum, True)])
    def test_structure_constants_elements(self, dla, use_orthonormal):
        r"""Test relation :math:`[i G_\alpha, i G_\beta] = \sum_{\gamma = 0}^{\mathfrak{d}-1} f^\gamma_{\alpha, \beta} iG_\gamma`."""

        d = len(dla)
        dla_dense = np.array([qml.matrix(op, wire_order=range(3)) for op in dla])
        if use_orthonormal:
            gram_inv = np.linalg.pinv(
                sp.linalg.sqrtm(np.tensordot(dla_dense, dla_dense, axes=[[1, 2], [2, 1]]).real)
            )
            dla_dense = np.tensordot(gram_inv, dla_dense, axes=1)
            dla = [(scale * op).pauli_rep for scale, op in zip(np.diag(gram_inv), dla)]

        ad_rep = structure_constants_dense(dla_dense, is_orthonormal=use_orthonormal)
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
        ops = np.array([qml.matrix(op.operation(), wire_order=range(3)) for op in dla])

        if use_orthonormal:
            gram_inv = sp.linalg.sqrtm(
                np.linalg.pinv(np.tensordot(ops, ops, axes=[[1, 2], [2, 1]]).real)
            )
            ops = np.tensordot(gram_inv, ops, axes=1)
            dla = [(scale * op).pauli_rep for scale, op in zip(np.diag(gram_inv), dla)]

        ad_rep_true = qml.pauli.dla.structure_constants(dla)
        ad_rep = structure_constants_dense(ops, is_orthonormal=use_orthonormal)
        assert qml.math.allclose(ad_rep, ad_rep_true)
