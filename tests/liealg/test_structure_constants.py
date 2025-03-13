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
"""Tests for pennylane/pauli/dla/structure_constants.py functionality"""
import numpy as np
import pytest

import pennylane as qml
from pennylane import structure_constants
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


class TestAdjointRepr:
    """Tests for structure_constants"""

    def test_structure_constants_dim(self):
        """Test the dimension of the adjoint repr"""
        d = len(Ising3)
        adjoint = structure_constants(Ising3, pauli=True)
        assert adjoint.shape == (d, d, d)
        assert adjoint.dtype == float

    @pytest.mark.parametrize("dla", [Ising3, XXZ3])
    @pytest.mark.parametrize("assume_orthogonal", [True, False])
    @pytest.mark.parametrize("change_norms", [False, True])
    @pytest.mark.parametrize("matrix", [False, True])
    def test_structure_constants_elements_with_orthogonal_basis(
        self, dla, assume_orthogonal, change_norms, matrix
    ):
        r"""Test relation :math:`[i G_α, i G_β] = \sum_{γ=0}^{d-1} f^γ_{α,β} iG_γ_` with orthogonal
        bases.
        The input ``assume_orthogonal`` toggles whether ``structure_constants`` will assume the
        input basis to be orthogonal. In this test we only ever pass orthogonal bases, so both
        options should be valid and return the same result.
        The input ``change_norms`` applies some random rescaling to the DLA elements to test
        correctness of ``strucure_constants`` for non-normalized operators.
        """

        d = len(dla)
        if change_norms:
            # Sample some random new coefficients between 0.5 and 1.5
            coeffs = np.random.random(d) + 0.5
            dla = [c * op for c, op in zip(coeffs, dla)]
        ad_rep = structure_constants(
            dla, pauli=True, matrix=matrix, is_orthogonal=assume_orthogonal
        )
        for alpha in range(d):
            for beta in range(d):

                comm_res = 1j * dla[alpha].commutator(dla[beta])

                res = sum(ad_rep[gamma, alpha, beta] * dla[gamma] for gamma in range(d))
                res.simplify()
                assert set(comm_res) == set(res)  # Compare keys
                assert all(np.isclose(comm_res[k], res[k]) for k in res)

    @pytest.mark.parametrize("ortho_dla", [Ising3, XXZ3])
    @pytest.mark.parametrize("matrix", [False, True])
    def test_structure_constants_elements_with_non_orthogonal(self, ortho_dla, matrix, seed):
        r"""Test relation :math:`[i G_α, i G_β] = \sum_{γ=0}^{d-1} f^γ_{α,β} iG_γ_` with
        non-orthogonal bases.
        """
        d = len(ortho_dla)

        rng = np.random.default_rng(seed)
        coeffs = rng.uniform(0.5, 1.5, size=(d, d))
        dla = [sum(c * op for c, op in zip(_coeffs, ortho_dla)) for _coeffs in coeffs]
        ad_rep = structure_constants(dla, pauli=True, matrix=matrix, is_orthogonal=False)
        for alpha in range(d):
            for beta in range(d):

                comm_res = 1j * dla[alpha].commutator(dla[beta])
                comm_res.simplify()

                res = sum(ad_rep[gamma, alpha, beta] * dla[gamma] for gamma in range(d))
                res.simplify()
                assert set(comm_res) == set(res)  # Compare keys
                assert all(np.isclose(comm_res[k], res[k]) for k in res)

        # Manually check the transformation behaviour of the structure constants under basis change
        ortho_ad_rep = structure_constants(ortho_dla, pauli=True)
        transf_ortho_ad_rep = np.tensordot(coeffs, ortho_ad_rep, axes=[[1], [2]])
        transf_ortho_ad_rep = np.tensordot(coeffs, transf_ortho_ad_rep, axes=[[1], [2]])
        transf_ortho_ad_rep = np.tensordot(
            np.linalg.pinv(coeffs).T, transf_ortho_ad_rep, axes=[[1], [2]]
        )
        assert np.allclose(transf_ortho_ad_rep, ad_rep)

    @pytest.mark.parametrize("dla", [Ising3, XXZ3])
    @pytest.mark.parametrize("matrix", [False, True])
    def test_use_operators(self, dla, matrix):
        """Test that operators can be passed and lead to the same result"""
        ad_rep_true = structure_constants(dla, pauli=True, matrix=matrix)

        ops = [op.operation() for op in dla]
        ad_rep = structure_constants(ops, pauli=False, matrix=matrix)
        assert qml.math.allclose(ad_rep, ad_rep_true)

    @pytest.mark.parametrize("dla", [Ising3, XXZ3])
    def test_matrix_input(self, dla):
        """Test structure constants work as expected for matrix inputs"""
        dla_m = [qml.matrix(op, wire_order=range(3)) for op in dla]
        adj = qml.structure_constants(dla, matrix=True, is_orthogonal=False)
        adj_m = qml.structure_constants(dla_m, matrix=True, is_orthogonal=False)

        assert np.allclose(adj, adj_m)

    def test_raise_error_for_non_paulis(self):
        """Test that an error is raised when passing operators that do not have a pauli_rep"""
        generators = [qml.Hadamard(0), qml.X(0)]
        with pytest.raises(
            ValueError, match="Cannot compute adjoint representation of non-pauli operators"
        ):
            structure_constants(generators)


dla0 = qml.lie_closure([qml.X(0) @ qml.X(1), qml.Z(0), qml.Z(1)], matrix=True)
adj0 = qml.structure_constants(dla0, matrix=True)


class TestInterfacesStructureConstants:
    """Test interfaces jax, torch and tensorflow with structure constants"""

    @pytest.mark.jax
    def test_jax_structure_constants(self):
        """Test jax interface for structure constants"""

        import jax.numpy as jnp

        dla_jax = jnp.array(dla0)
        adj_jax = qml.structure_constants(dla_jax, matrix=True)

        assert qml.math.allclose(adj_jax, adj0)
        assert qml.math.get_interface(adj_jax) == "jax"

    @pytest.mark.torch
    def test_torch_structure_constants(self):
        """Test torch interface for structure constants"""

        import torch

        dla_torch = torch.tensor(dla0)
        adj_torch = qml.structure_constants(dla_torch, matrix=True)

        assert qml.math.allclose(adj_torch, adj0)
        assert qml.math.get_interface(adj_torch) == "torch"

    @pytest.mark.tf
    def test_tf_structure_constants(self):
        """Test tf interface for structure constants"""

        import tensorflow as tf

        dla_tf = tf.constant(dla0)
        adj_tf = qml.structure_constants(dla_tf, matrix=True)

        assert qml.math.allclose(adj_tf, adj0)
        assert qml.math.get_interface(adj_tf) == "tensorflow"
