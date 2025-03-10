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
from pennylane.pauli import PauliSentence, PauliWord, structure_constants

## Construct some example DLAs
# TFIM
gens = [PauliSentence({PauliWord({i: "X", i + 1: "X"}): 1.0}) for i in range(2)]
gens += [PauliSentence({PauliWord({i: "Z"}): 1.0}) for i in range(3)]
Ising3 = qml.pauli.lie_closure(gens, pauli=True)

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
XXZ3 = qml.pauli.lie_closure(gens2, pauli=True)


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
    def test_structure_constants_elements_with_non_orthogonal(self, ortho_dla, matrix):
        r"""Test relation :math:`[i G_α, i G_β] = \sum_{γ=0}^{d-1} f^γ_{α,β} iG_γ_` with
        non-orthogonal bases.
        """
        d = len(ortho_dla)

        coeffs = np.array(
            [
                [
                    0.87632975,
                    0.99734637,
                    1.14627469,
                    0.87077159,
                    0.72335985,
                    0.69134297,
                    1.33445484,
                    0.85575064,
                    1.02587348,
                    1.37977055,
                    1.09725003,
                    1.29153242,
                    0.74686785,
                    1.35586552,
                    0.97190644,
                ],
                [
                    1.18920102,
                    1.01351015,
                    0.5642774,
                    0.91066075,
                    1.19478573,
                    1.14819517,
                    0.93491657,
                    0.56558588,
                    1.38829663,
                    1.41286648,
                    0.53543494,
                    1.25874301,
                    1.31100716,
                    0.80541896,
                    0.51298552,
                ],
                [
                    0.79314765,
                    1.40975345,
                    1.04028023,
                    0.90036632,
                    0.63962954,
                    0.56624707,
                    1.36821595,
                    1.04101811,
                    0.68345803,
                    1.16190994,
                    0.79876645,
                    1.24691492,
                    1.29040538,
                    1.12561661,
                    0.76651242,
                ],
                [
                    1.39701302,
                    1.42955489,
                    0.7754423,
                    1.16279495,
                    1.1673055,
                    1.23315312,
                    0.83922968,
                    1.10022965,
                    1.25800816,
                    0.6125199,
                    0.78300589,
                    1.16471841,
                    1.25531344,
                    1.22672349,
                    0.88213697,
                ],
                [
                    1.21245241,
                    0.8180376,
                    1.45309174,
                    0.53451035,
                    0.66813455,
                    1.11099869,
                    0.91761782,
                    1.18144074,
                    1.10771086,
                    0.8969892,
                    0.87114053,
                    1.4978591,
                    1.18659464,
                    0.54578536,
                    1.25259782,
                ],
                [
                    0.55995753,
                    1.42934239,
                    0.84056521,
                    1.14731318,
                    1.04123299,
                    1.46146374,
                    0.55228209,
                    1.16878878,
                    0.71669839,
                    1.38043078,
                    1.06190386,
                    0.83077777,
                    1.32046924,
                    1.44597618,
                    1.44251766,
                ],
                [
                    0.89443358,
                    0.96984937,
                    0.60558213,
                    0.89236259,
                    0.77458767,
                    1.03775925,
                    1.40449435,
                    1.48825914,
                    1.47454524,
                    1.27678524,
                    0.61242633,
                    1.09777538,
                    1.49592723,
                    1.08201822,
                    0.87178585,
                ],
                [
                    0.73947568,
                    1.13662841,
                    0.83825676,
                    0.93075799,
                    0.75300268,
                    1.14279098,
                    0.6826948,
                    1.18059673,
                    1.39428721,
                    0.62093051,
                    0.55402898,
                    0.62924987,
                    1.18776586,
                    0.57120519,
                    0.90786947,
                ],
                [
                    1.04455526,
                    1.16109643,
                    0.98293779,
                    1.46916665,
                    0.9282653,
                    1.1566083,
                    0.85551882,
                    1.37524984,
                    1.1358546,
                    0.79991597,
                    1.16944656,
                    0.55279668,
                    0.98367337,
                    0.94241047,
                    0.77651419,
                ],
                [
                    1.09087171,
                    0.59280447,
                    1.23142616,
                    0.71974834,
                    0.59256391,
                    0.55785004,
                    0.97060203,
                    1.13286906,
                    0.81000058,
                    0.94322281,
                    0.75326669,
                    1.47798592,
                    0.73970418,
                    0.95382792,
                    1.33435114,
                ],
                [
                    0.62520788,
                    0.8130477,
                    1.22827088,
                    1.24441336,
                    0.81655349,
                    1.06403204,
                    1.4623155,
                    1.0188817,
                    0.61455805,
                    1.29714639,
                    0.5417235,
                    0.85622961,
                    1.10786286,
                    1.34066554,
                    0.82527299,
                ],
                [
                    1.209229,
                    1.02120134,
                    0.98580285,
                    1.43065704,
                    0.87729275,
                    1.14710053,
                    1.41854939,
                    0.68617831,
                    1.4393291,
                    0.79322775,
                    0.91163182,
                    0.61286039,
                    0.51296263,
                    0.55783837,
                    0.76557744,
                ],
                [
                    1.49904669,
                    1.39637521,
                    1.49625533,
                    0.51227843,
                    1.49352052,
                    1.46494957,
                    0.50914572,
                    1.04097673,
                    0.59483312,
                    0.70158197,
                    0.62644909,
                    1.03129645,
                    1.1492102,
                    1.41443936,
                    1.27247974,
                ],
                [
                    1.46689099,
                    1.09236398,
                    1.03995864,
                    0.66689632,
                    0.91300391,
                    0.58372147,
                    0.53625978,
                    1.39644882,
                    0.55133412,
                    0.80111682,
                    1.24112084,
                    1.37203582,
                    1.4847002,
                    0.60146886,
                    1.29733256,
                ],
                [
                    1.11239854,
                    1.24901234,
                    1.42884628,
                    0.67060191,
                    1.34071019,
                    1.34108399,
                    1.30299027,
                    1.286162,
                    0.75825189,
                    0.70681421,
                    0.69007366,
                    0.62042816,
                    0.50749492,
                    1.25818659,
                    1.3822408,
                ],
            ]
        )
        coeffs = coeffs[:d, :d]
        dla = [sum(c * op for c, op in zip(_coeffs, ortho_dla)).pauli_rep for _coeffs in coeffs]
        ad_rep = structure_constants(dla, pauli=True, is_orthogonal=False)
        for alpha in range(d):
            for beta in range(d):

                comm_res = 1j * dla[alpha].commutator(dla[beta])
                comm_res.simplify()

                res = sum(ad_rep[gamma, alpha, beta] * dla[gamma] for gamma in range(d))
                res.simplify()
                assert set(comm_res) == set(res)  # Compare keys
                assert all(np.isclose(comm_res[k], res[k]) for k in res)

        # Manually check the transformation behaviour of the structure constants under basis change
        ortho_ad_rep = structure_constants(ortho_dla, matrix=matrix, pauli=True)
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
            qml.pauli.structure_constants(generators)


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
