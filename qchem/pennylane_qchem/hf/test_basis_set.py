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

import pytest
from pennylane import numpy as np
from basis_set import BasisFunction, atom_basis_data, mol_basis_data


class TestBasis:
    """Tests for generating basis set default parameters"""

    @pytest.mark.parametrize(
        "l, alpha, coeff, rgaus",
        [
            (
                (0, 0, 0),
                np.array([3.4252509140, 0.6239137298, 0.168855404]),
                np.array([0.1543289673, 0.5353281423, 0.4446345422]),
                np.array([0.0000000000, 0.0000000000, -0.694349000]),
            )
        ],
    )
    def test_basisfunction(self, l, alpha, coeff, rgaus):
        """Test that BasisFunction class creates basis function objects correctly."""
        basis_function = BasisFunction(l, alpha, coeff, rgaus)

        assert np.allclose(basis_function.alpha, alpha)
        assert np.allclose(basis_function.coeff, coeff)
        assert np.allclose(basis_function.rgaus, rgaus)

        assert np.allclose(basis_function.params[0], alpha)
        assert np.allclose(basis_function.params[1], coeff)
        assert np.allclose(basis_function.params[2], rgaus)

    @pytest.mark.parametrize(
        "basisset, symbol, l, coefficients, exponents",
        [
            (
                "sto-3g",
                "H",
                (0, 0, 0),
                [0.3425250914e01, 0.6239137298e00, 0.1688554040e00],
                [0.1543289673e00, 0.5353281423e00, 0.4446345422e00],
            )
        ],
    )
    def test_atom_basis_data(self, basisset, symbol, l, coefficients, exponents):
        """Test that correct basis set parameters are generated for a given atom."""
        params = atom_basis_data(basisset, symbol)[0]

        assert np.allclose(params[0], l)
        assert np.allclose(params[1], coefficients)
        assert np.allclose(params[2], exponents)

    @pytest.mark.parametrize(
        "basisset, symbols, nbasis, l, coefficients, exponents",
        [
            (
                "sto-3g",
                ["H", "H"],
                [1, 1],
                (0, 0, 0),
                [0.3425250914e01, 0.6239137298e00, 0.1688554040e00],
                [0.1543289673e00, 0.5353281423e00, 0.4446345422e00],
            )
        ],
    )
    def test_mol_basis_data(self, basisset, symbols, nbasis, l, coefficients, exponents):
        """Test that correct basis set parameters are generated for a given molecule represented as
        a list of atom."""
        n_basis, params = mol_basis_data(basisset, symbols)

        assert n_basis == nbasis

        for i in [0, 1]:
            assert np.allclose(params[i][0], l)
            assert np.allclose(params[i][1], coefficients)
            assert np.allclose(params[i][2], exponents)
