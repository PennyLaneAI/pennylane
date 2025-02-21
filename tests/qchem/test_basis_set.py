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
# pylint: disable=too-many-arguments
import sys

import pytest

from pennylane import numpy as np
from pennylane import qchem


class TestBasis:
    """Tests for generating basis set default parameters"""

    @pytest.mark.parametrize(
        ("_", "__", "l", "alpha", "coeff", "r"),
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
    def test_basisfunction(self, _, __, l, alpha, coeff, r):
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
            (
                "6-311g",
                "O",
                (
                    (
                        (0, 0, 0),  # l
                        [8588.5, 1297.23, 299.296, 87.3771, 25.6789, 3.74004],  # alpha
                        [0.00189515, 0.0143859, 0.070732, 0.240001, 0.594797, 0.280802],
                        # coeff
                    ),
                    (
                        (0, 0, 0),  # l
                        [42.11750, 9.628370, 2.853320],  # alpha
                        [0.113889, 0.920811, -0.00327447],
                        # coeff
                    ),
                    (
                        (1, 0, 0),  # l
                        [42.11750, 9.628370, 2.853320],  # alpha
                        [0.0365114, 0.237153, 0.819702],  # coeff
                    ),
                    (
                        (0, 1, 0),  # l
                        [42.11750, 9.628370, 2.853320],  # alpha
                        [0.0365114, 0.237153, 0.819702],  # coeff
                    ),
                    (
                        (0, 0, 1),  # l
                        [42.11750, 9.628370, 2.853320],  # alpha
                        [0.0365114, 0.237153, 0.819702],  # coeff
                    ),
                    (
                        (0, 0, 0),  # l
                        [0.905661],  # alpha
                        [1.000000],  # coeff
                    ),
                    (
                        (1, 0, 0),  # l
                        [0.905661],  # alpha
                        [1.000000],  # coeff
                    ),
                    (
                        (0, 1, 0),  # l
                        [0.905661],  # alpha
                        [1.000000],  # coeff
                    ),
                    (
                        (0, 0, 1),  # l
                        [0.905661],  # alpha
                        [1.000000],  # coeff
                    ),
                    (
                        (0, 0, 0),  # l
                        [0.255611],  # alpha
                        [1.000000],  # coeff
                    ),
                    (
                        (1, 0, 0),  # l
                        [0.255611],  # alpha
                        [1.000000],  # coeff
                    ),
                    (
                        (0, 1, 0),  # l
                        [0.255611],  # alpha
                        [1.000000],  # coeff
                    ),
                    (
                        (0, 0, 1),  # l
                        [0.255611],  # alpha
                        [1.000000],  # coeff
                    ),
                ),
            ),
            (
                "cc-pvdz",
                "O",
                (
                    (
                        (0, 0, 0),  # l
                        [
                            1.172000e04,
                            1.759000e03,
                            4.008000e02,
                            1.137000e02,
                            3.703000e01,
                            1.327000e01,
                            5.025000e00,
                            1.013000e00,
                            3.023000e-01,
                        ],  # alpha
                        [
                            7.100000e-04,
                            5.470000e-03,
                            2.783700e-02,
                            1.048000e-01,
                            2.830620e-01,
                            4.487190e-01,
                            2.709520e-01,
                            1.545800e-02,
                            -2.585000e-03,
                        ],
                        # coeff
                    ),
                    (
                        (0, 0, 0),  # l
                        [
                            1.172000e04,
                            1.759000e03,
                            4.008000e02,
                            1.137000e02,
                            3.703000e01,
                            1.327000e01,
                            5.025000e00,
                            1.013000e00,
                            3.023000e-01,
                        ],  # alpha
                        [
                            -1.600000e-04,
                            -1.263000e-03,
                            -6.267000e-03,
                            -2.571600e-02,
                            -7.092400e-02,
                            -1.654110e-01,
                            -1.169550e-01,
                            5.573680e-01,
                            5.727590e-01,
                        ],
                        # coeff
                    ),
                    (
                        (0, 0, 0),  # l
                        [
                            1.172000e04,
                            1.759000e03,
                            4.008000e02,
                            1.137000e02,
                            3.703000e01,
                            1.327000e01,
                            5.025000e00,
                            1.013000e00,
                            3.023000e-01,
                        ],  # alpha
                        [
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            1.000000e00,
                        ],
                        # coeff
                    ),
                    (
                        (1, 0, 0),  # l
                        [1.770000e01, 3.854000e00, 1.046000e00, 2.753000e-01],
                        # alpha
                        [4.301800e-02, 2.289130e-01, 5.087280e-01, 4.605310e-01],
                        # coeff
                    ),
                    (
                        (0, 1, 0),  # l
                        [1.770000e01, 3.854000e00, 1.046000e00, 2.753000e-01],
                        # alpha
                        [4.301800e-02, 2.289130e-01, 5.087280e-01, 4.605310e-01],
                        # coeff
                    ),
                    (
                        (0, 0, 1),  # l
                        [1.770000e01, 3.854000e00, 1.046000e00, 2.753000e-01],
                        # alpha
                        [4.301800e-02, 2.289130e-01, 5.087280e-01, 4.605310e-01],
                        # coeff
                    ),
                    (
                        (1, 0, 0),  # l
                        [1.770000e01, 3.854000e00, 1.046000e00, 2.753000e-01],
                        # alpha
                        [0.000000e00, 0.000000e00, 0.000000e00, 1.000000e00],
                        # coeff
                    ),
                    (
                        (0, 1, 0),  # l
                        [1.770000e01, 3.854000e00, 1.046000e00, 2.753000e-01],
                        # alpha
                        [0.000000e00, 0.000000e00, 0.000000e00, 1.000000e00],
                        # coeff
                    ),
                    (
                        (0, 0, 1),  # l
                        [1.770000e01, 3.854000e00, 1.046000e00, 2.753000e-01],
                        # alpha
                        [0.000000e00, 0.000000e00, 0.000000e00, 1.000000e00],
                        # coeff
                    ),
                    (
                        (1, 1, 0),  # l
                        [1.185000e00],  # alpha
                        [1.0000000],  # coeff
                    ),
                    (
                        (1, 0, 1),  # l
                        [1.185000e00],  # alpha
                        [1.0000000],  # coeff
                    ),
                    (
                        (0, 1, 1),  # l
                        [1.185000e00],  # alpha
                        [1.0000000],  # coeff
                    ),
                    (
                        (2, 0, 0),  # l
                        [1.185000e00],  # alpha
                        [1.0000000],  # coeff
                    ),
                    (
                        (0, 2, 0),  # l
                        [1.185000e00],  # alpha
                        [1.0000000],  # coeff
                    ),
                    (
                        (0, 0, 2),  # l
                        [1.185000e00],  # alpha
                        [1.0000000],  # coeff
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

    @pytest.mark.parametrize(
        ("basis_name", "atom_name"),
        [
            (
                "sto-3g",
                "H",
            ),
            (
                "6-311g",
                "C",
            ),
            (
                "cc-pvdz",
                "O",
            ),
        ],
    )
    def test_data_basis_set_exchange(self, basis_name, atom_name):
        """Test that correct basis set parameters are loaded from basis_set_exchange library."""

        pytest.importorskip("basis_set_exchange")

        data_load = qchem.atom_basis_data(basis_name, atom_name, load_data=True)
        data_read = qchem.atom_basis_data(basis_name, atom_name, load_data=False)

        assert data_load == data_read

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

    def test_mol_basis_data_error(self):
        """Test that correct error is raised if the element is not present in the internal basis-sets"""

        with pytest.raises(ValueError, match="The requested basis set data is not available for"):
            qchem.basis_set.atom_basis_data(name="sto-3g", atom="Os")

        with pytest.raises(ValueError, match="Requested element Ox doesn't exist"):
            qchem.basis_data.load_basisset(basis="sto-3g", element="Ox")


class TestLoadBasis:
    """Tests for loading data from external libraries."""

    @pytest.mark.parametrize(
        ("basis_name", "atom_name", "params_ref"),
        [  # data manually copied from https://www.basissetexchange.org/
            (
                "sto-3g",
                "H",
                (
                    [
                        (
                            "S",  # l
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
                        "S",  # l
                        [0.1873113696e02, 0.2825394365e01, 0.6401216923e00],  # alpha
                        [0.3349460434e-01, 0.2347269535e00, 0.8137573261e00],  # coeff
                    ),
                    (
                        "S",  # l
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
                        "S",  # l
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
                        "S",  # l
                        [0.1553961625e02, 0.3599933586e01, 0.1013761750e01],  # alpha
                        [-0.1107775495e00, -0.1480262627e00, 0.1130767015e01],
                        # coeff
                    ),
                    (
                        "P",  # l
                        [0.1553961625e02, 0.3599933586e01, 0.1013761750e01],  # alpha
                        [0.7087426823e-01, 0.3397528391e00, 0.7271585773e00],  # coeff
                    ),
                    (
                        "S",  # l
                        [0.2700058226e00],  # alpha
                        [0.1000000000e01],  # coeff
                    ),
                    (
                        "P",  # l
                        [0.2700058226e00],  # alpha
                        [0.1000000000e01],  # coeff
                    ),
                ),
            ),
            (
                "6-311g",
                "O",
                (
                    (
                        "S",  # l
                        [8588.5, 1297.23, 299.296, 87.3771, 25.6789, 3.74004],  # alpha
                        [0.00189515, 0.0143859, 0.070732, 0.240001, 0.594797, 0.280802],
                        # coeff
                    ),
                    (
                        "S",  # l
                        [42.11750, 9.628370, 2.853320],  # alpha
                        [0.113889, 0.920811, -0.00327447],
                        # coeff
                    ),
                    (
                        "P",  # l
                        [42.11750, 9.628370, 2.853320],  # alpha
                        [0.0365114, 0.237153, 0.819702],  # coeff
                    ),
                    (
                        "S",  # l
                        [0.905661],  # alpha
                        [1.000000],  # coeff
                    ),
                    (
                        "P",  # l
                        [0.905661],  # alpha
                        [1.000000],  # coeff
                    ),
                    (
                        "S",  # l
                        [0.255611],  # alpha
                        [1.000000],  # coeff
                    ),
                    (
                        "P",  # l
                        [0.255611],  # alpha
                        [1.000000],  # coeff
                    ),
                ),
            ),
            (
                "cc-pvdz",
                "O",
                (
                    (
                        "S",  # l
                        [
                            1.172000e04,
                            1.759000e03,
                            4.008000e02,
                            1.137000e02,
                            3.703000e01,
                            1.327000e01,
                            5.025000e00,
                            1.013000e00,
                            3.023000e-01,
                        ],  # alpha
                        [
                            7.100000e-04,
                            5.470000e-03,
                            2.783700e-02,
                            1.048000e-01,
                            2.830620e-01,
                            4.487190e-01,
                            2.709520e-01,
                            1.545800e-02,
                            -2.585000e-03,
                        ],
                        # coeff
                    ),
                    (
                        "S",  # l
                        [
                            1.172000e04,
                            1.759000e03,
                            4.008000e02,
                            1.137000e02,
                            3.703000e01,
                            1.327000e01,
                            5.025000e00,
                            1.013000e00,
                            3.023000e-01,
                        ],  # alpha
                        [
                            -1.600000e-04,
                            -1.263000e-03,
                            -6.267000e-03,
                            -2.571600e-02,
                            -7.092400e-02,
                            -1.654110e-01,
                            -1.169550e-01,
                            5.573680e-01,
                            5.727590e-01,
                        ],
                        # coeff
                    ),
                    (
                        "S",  # l
                        [
                            1.172000e04,
                            1.759000e03,
                            4.008000e02,
                            1.137000e02,
                            3.703000e01,
                            1.327000e01,
                            5.025000e00,
                            1.013000e00,
                            3.023000e-01,
                        ],  # alpha
                        [
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            0.000000e00,
                            1.000000e00,
                        ],
                        # coeff
                    ),
                    (
                        "P",  # l
                        [1.770000e01, 3.854000e00, 1.046000e00, 2.753000e-01],
                        # alpha
                        [4.301800e-02, 2.289130e-01, 5.087280e-01, 4.605310e-01],
                        # coeff
                    ),
                    (
                        "P",  # l
                        [1.770000e01, 3.854000e00, 1.046000e00, 2.753000e-01],
                        # alpha
                        [0.000000e00, 0.000000e00, 0.000000e00, 1.000000e00],
                        # coeff
                    ),
                    (
                        "D",  # l
                        [1.185000e00],  # alpha
                        [1.0000000],  # coeff
                    ),
                ),
            ),
            (
                "6-31+G*",
                "B",
                (
                    (
                        "S",  # l
                        [
                            0.2068882250e04,
                            0.3106495700e03,
                            0.7068303300e02,
                            0.1986108030e02,
                            0.6299304840e01,
                            0.2127026970e01,
                        ],  # alpha
                        [
                            0.1866274590e-02,
                            0.1425148170e-01,
                            0.6955161850e-01,
                            0.2325729330e00,
                            0.4670787120e00,
                            0.3634314400e00,
                        ],  # coeff
                    ),
                    (
                        "S",  # l
                        [0.4727971071e01, 0.1190337736e01, 0.3594116829e00],  # alpha
                        [-0.1303937974e00, -0.1307889514e00, 0.1130944484e01],
                        # coeff
                    ),
                    (
                        "P",  # l
                        [0.4727971071e01, 0.1190337736e01, 0.3594116829e00],  # alpha
                        [0.7459757992e-01, 0.3078466771e00, 0.7434568342e00],  # coeff
                    ),
                    (
                        "S",  # l
                        [0.1267512469e00],  # alpha
                        [0.1000000000e01],  # coeff
                    ),
                    (
                        "P",  # l
                        [0.1267512469e00],  # alpha
                        [0.1000000000e01],  # coeff
                    ),
                    (
                        "D",  # l
                        [0.6000000000e00],  # alpha
                        [1.0000000],  # coeff
                    ),
                    (
                        "S",  # l
                        [0.3150000000e-01],  # alpha
                        [0.1000000000e01],  # coeff
                    ),
                    (
                        "P",  # l
                        [0.3150000000e-01],  # alpha
                        [0.1000000000e01],  # coeff
                    ),
                ),
            ),
            (
                "sto-3g",
                "Ag",
                (
                    (
                        "S",  # l
                        [4744.521634, 864.2205383, 233.8918045],  # alpha
                        [0.1543289673, 0.5353281423, 0.4446345422],  # coeff
                    ),
                    (
                        "S",  # l
                        [414.9652069, 96.42898995, 31.36170035],  # alpha
                        [-0.09996722919, 0.3995128261, 0.7001154689],  # coeff
                    ),
                    (
                        "P",  # l
                        [414.9652069, 96.42898995, 31.36170035],  # alpha
                        [0.155916275, 0.6076837186, 0.3919573931],  # coeff
                    ),
                    (
                        "S",  # l
                        [5.29023045, 2.059988316, 0.9068119281],  # alpha
                        [-0.3306100626, 0.05761095338, 1.115578745],  # coeff
                    ),
                    (
                        "P",  # l
                        [5.29023045, 2.059988316, 0.9068119281],  # alpha
                        [-0.1283927634, 0.5852047641, 0.543944204],  # coeff
                    ),
                    (
                        "S",  # l
                        [0.4370804803, 0.2353408164, 0.1039541771],  # alpha
                        [-0.3842642608, -0.1972567438, 1.375495512],  # coeff
                    ),
                    (
                        "P",  # l
                        [0.4370804803, 0.2353408164, 0.1039541771],  # alpha
                        [-0.3481691526, 0.629032369, 0.6662832743],  # coeff
                    ),
                    (
                        "SPD",  # l
                        [49.41048605, 15.07177314, 5.815158634],  # alpha
                        [-0.2277635023, 0.2175436044, 0.9166769611],  # coeff
                    ),
                    (
                        "SPD",  # l
                        [49.41048605, 15.07177314, 5.815158634],  # alpha
                        [0.004951511155, 0.5777664691, 0.4846460366],  # coeff
                    ),
                    (
                        "SPD",  # l
                        [49.41048605, 15.07177314, 5.815158634],  # alpha
                        [0.2197679508, 0.6555473627, 0.286573259],  # coeff
                    ),
                    (
                        "D",  # l
                        [3.283395668, 1.278537254, 0.5628152469],  # alpha
                        [0.1250662138, 0.6686785577, 0.3052468245],  # coeff
                    ),
                ),
            ),
        ],
    )
    def test_load_basis_data(self, basis_name, atom_name, params_ref):
        """Test that correct basis set parameters are loaded for a given atom."""

        pytest.importorskip("basis_set_exchange")

        data = qchem.load_basisset(basis_name, atom_name)

        l_ref = [p[0] for p in params_ref]
        alpha_ref = [p[1] for p in params_ref]
        coeff_ref = [p[2] for p in params_ref]

        assert data["orbitals"] == l_ref
        assert data["exponents"] == alpha_ref
        assert data["coefficients"] == coeff_ref

    def test_fail_import(self, monkeypatch):
        """Test if an ImportError is raised when basis_set_exchange is requested but
        not installed."""

        pytest.importorskip("basis_set_exchange")

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "basis_set_exchange", None)

            with pytest.raises(ImportError, match="This feature requires basis_set_exchange"):
                qchem.load_basisset("sto-3g", "H")
