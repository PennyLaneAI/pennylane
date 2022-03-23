# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for generating basis set default parameters.
"""
# pylint: disable=no-self-use
import pytest

from pennylane import numpy as np
from pennylane import qchem


class TestBasis:
    """Tests for generating basis set default parameters"""

    basis_data_H = [
        (
            "sto-3g",
            "H",
            1,
            [(0, 0, 0)],
            [0.3425250914e01, 0.6239137298e00, 0.1688554040e00],
            [0.1543289673e00, 0.5353281423e00, 0.4446345422e00],
            [0.0000000000e00, 0.0000000000e00, -0.694349000e00],
        )
    ]

    @pytest.mark.parametrize("basis_data", basis_data_H)
    def test_basisfunction(self, basis_data):
        """Test that BasisFunction class creates basis function objects correctly."""
        _, _, _, l, alpha, coeff, r = basis_data

        basis_function = qchem.hf.BasisFunction(l, alpha, coeff, r)

        assert np.allclose(np.array(basis_function.alpha), np.array(alpha))
        assert np.allclose(np.array(basis_function.coeff), np.array(coeff))
        assert np.allclose(np.array(basis_function.r), np.array(r))

        assert np.allclose(basis_function.params[0], alpha)
        assert np.allclose(basis_function.params[1], coeff)
        assert np.allclose(basis_function.params[2], r)

    @pytest.mark.parametrize(
        ("basis_name", "atom_name", "params_ref"),
        [
            (
                "sto-3g",
                "H",
                (
                    (0, 0, 0),
                    [0.3425250914e01, 0.6239137298e00, 0.1688554040e00],
                    [0.1543289673e00, 0.5353281423e00, 0.4446345422e00],
                ),
            ),
        ],
    )
    def test_atom_basis_data(self, basis_name, atom_name, params_ref):
        """Test that correct basis set parameters are generated for a given atom."""
        params = qchem.hf.atom_basis_data(basis_name, atom_name)

        assert np.allclose(params, params_ref)

    @pytest.mark.parametrize(
        ("basis_name", "atom_name", "params_ref"),
        [
            (
                "6-31g",
                "H",
                (
                    (
                        (0, 0, 0),
                        [0.1873113696e02, 0.2825394365e01, 0.6401216923e00],
                        [0.3349460434e-01, 0.2347269535e00, 0.8137573261e00],
                    ),
                    (
                        (0, 0, 0),
                        [0.1612777588e00],
                        [1.0000000],
                    ),
                ),
            ),
        ],
    )
    def test_atom_basis_data_631g(self, basis_name, atom_name, params_ref):
        """Test that correct basis set parameters are generated for a given atom."""
        params = qchem.hf.atom_basis_data(basis_name, atom_name)

        assert np.allclose(params[0], params_ref[0])
        assert np.allclose(params[1][0], params_ref[1][0])
        assert np.allclose(params[1][1:], params_ref[1][1:])

    basis_data_HF = [
        (
            "sto-3g",
            ["H", "F"],
            [1, 5],
            (
                (
                    (0, 0, 0),
                    [0.3425250914e01, 0.6239137298e00, 0.1688554040e00],
                    [0.1543289673e00, 0.5353281423e00, 0.4446345422e00],
                ),
                (
                    (0, 0, 0),
                    [0.1666791340e03, 0.3036081233e02, 0.8216820672e01],
                    [0.1543289673e00, 0.5353281423e00, 0.4446345422e00],
                ),
                (
                    (0, 0, 0),
                    [0.6464803249e01, 0.1502281245e01, 0.4885884864e00],
                    [-0.9996722919e-01, 0.3995128261e00, 0.7001154689e00],
                ),
                (
                    (1, 0, 0),
                    [0.6464803249e01, 0.1502281245e01, 0.4885884864e00],
                    [0.1559162750e00, 0.6076837186e00, 0.3919573931e00],
                ),
                (
                    (0, 1, 0),
                    [0.6464803249e01, 0.1502281245e01, 0.4885884864e00],
                    [0.1559162750e00, 0.6076837186e00, 0.3919573931e00],
                ),
                (
                    (0, 0, 1),
                    [0.6464803249e01, 0.1502281245e01, 0.4885884864e00],
                    [0.1559162750e00, 0.6076837186e00, 0.3919573931e00],
                ),
            ),
        )
    ]

    @pytest.mark.parametrize("basis_data", basis_data_HF)
    def test_mol_basis_data(self, basis_data):
        """Test that correct basis set parameters are generated for a given molecule represented as
        a list of atoms."""
        basis, symbols, n_ref, params_ref = basis_data
        n_basis, params = qchem.hf.mol_basis_data(basis, symbols)

        assert n_basis == n_ref

        assert np.allclose(params, params_ref)
