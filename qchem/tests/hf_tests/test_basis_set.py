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

    @pytest.mark.parametrize(
        ("basis_name", "atom_name", "l", "alpha", "coeff", "r"),
        [
            (
                "sto-3g",
                "H",
                (0, 0, 0),
                # data manually copied from https://www.basissetexchange.org/
                [0.3425250914e01, 0.6239137298e00, 0.1688554040e00],
                [0.1543289673e00, 0.5353281423e00, 0.4446345422e00],
                [0.0000000000e00, 0.0000000000e00, -0.694349000e00],
            ),
        ],
    )
    def test_basisfunction(self, basis_name, atom_name, l, alpha, coeff, r):
        """Test that BasisFunction class creates basis function objects correctly."""

        basis_function = qchem.BasisFunction(l, alpha, coeff, r)

        assert np.allclose(np.array(basis_function.alpha), np.array(alpha))
        assert np.allclose(np.array(basis_function.coeff), np.array(coeff))
        assert np.allclose(np.array(basis_function.r), np.array(r))

        assert np.allclose(basis_function.params[0], alpha)
        assert np.allclose(basis_function.params[1], coeff)
        assert np.allclose(basis_function.params[2], r)

    @pytest.mark.parametrize(
        ("basis_name", "atom_name", "params_ref"),
        [  # data manually copied from https://www.basissetexchange.org/
            (
                "sto-3g",
                "H",
                (
                    [
                        (
                            (0, 0, 0),  # l
                            [0.3425250914e01, 0.6239137298e00, 0.1688554040e00],  # alpha
                            [0.1543289673e00, 0.5353281423e00, 0.4446345422e00],  # coeff
                        )
                    ]
                ),
            ),
            (
                "6-31g",
                "H",
                (
                    (
                        (0, 0, 0),  # l
                        [0.1873113696e02, 0.2825394365e01, 0.6401216923e00],  # alpha
                        [0.3349460434e-01, 0.2347269535e00, 0.8137573261e00],  # coeff
                    ),
                    (
                        (0, 0, 0),  # l
                        [0.1612777588e00],  # alpha
                        [1.0000000],  # coeff
                    ),
                ),
            ),
            (
                "6-31g",
                "O",
                (
                    (
                        (0, 0, 0),  # l
                        [
                            0.5484671660e04,
                            0.8252349460e03,
                            0.1880469580e03,
                            0.5296450000e02,
                            0.1689757040e02,
                            0.5799635340e01,
                        ],  # alpha
                        [
                            0.1831074430e-02,
                            0.1395017220e-01,
                            0.6844507810e-01,
                            0.2327143360e00,
                            0.4701928980e00,
                            0.3585208530e00,
                        ],  # coeff
                    ),
                    (
                        (0, 0, 0),  # l
                        [0.1553961625e02, 0.3599933586e01, 0.1013761750e01],  # alpha
                        [-0.1107775495e00, -0.1480262627e00, 0.1130767015e01],
                        # coeff
                    ),
                    (
                        (1, 0, 0),  # l
                        [0.1553961625e02, 0.3599933586e01, 0.1013761750e01],  # alpha
                        [0.7087426823e-01, 0.3397528391e00, 0.7271585773e00],  # coeff
                    ),
                    (
                        (0, 1, 0),  # l
                        [0.1553961625e02, 0.3599933586e01, 0.1013761750e01],  # alpha
                        [0.7087426823e-01, 0.3397528391e00, 0.7271585773e00],  # coeff
                    ),
                    (
                        (0, 0, 1),  # l
                        [0.1553961625e02, 0.3599933586e01, 0.1013761750e01],  # alpha
                        [0.7087426823e-01, 0.3397528391e00, 0.7271585773e00],  # coeff
                    ),
                    (
                        (0, 0, 0),  # l
                        [0.2700058226e00],  # alpha
                        [0.1000000000e01],  # coeff
                    ),
                    (
                        (1, 0, 0),  # l
                        [0.2700058226e00],  # alpha
                        [0.1000000000e01],  # coeff
                    ),
                    (
                        (0, 1, 0),  # l
                        [0.2700058226e00],  # alpha
                        [0.1000000000e01],  # coeff
                    ),
                    (
                        (0, 0, 1),  # l
                        [0.2700058226e00],  # alpha
                        [0.1000000000e01],  # coeff
                    ),
                ),
            ),
        ],
    )
    def test_atom_basis_data(self, basis_name, atom_name, params_ref):
        """Test that correct basis set parameters are generated for a given atom."""
        params = qchem.atom_basis_data(basis_name, atom_name)

        l = [p[0] for p in params]
        l_ref = [p[0] for p in params_ref]

        alpha = [p[1] for p in params]
        alpha_ref = [p[1] for p in params_ref]

        coeff = [p[2] for p in params]
        coeff_ref = [p[2] for p in params_ref]

        assert l == l_ref
        assert alpha == alpha_ref
        assert coeff == coeff_ref

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
        n_basis, params = qchem.mol_basis_data(basis, symbols)

        assert n_basis == n_ref

        assert np.allclose(params, params_ref)
