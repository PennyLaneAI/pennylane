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
Unit tests for loading data from external libraries.
"""
import sys

# pylint: disable=no-self-use
import pytest

from pennylane.qchem.basis_data import load_basisset

bse = pytest.importorskip("basis_set_exchange")


class TestBasis:
    """Tests for generating basis set default parameters"""

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
        ],
    )
    def test_load_basis_data(self, basis_name, atom_name, params_ref):
        """Test that correct basis set parameters are loaded for a given atom."""

        data = load_basisset(basis_name, atom_name)

        l_ref = [p[0] for p in params_ref]
        alpha_ref = [p[1] for p in params_ref]
        coeff_ref = [p[2] for p in params_ref]

        assert data["orbitals"] == l_ref
        assert data["exponents"] == alpha_ref
        assert data["coefficients"] == coeff_ref

    def test_fail_import(self, monkeypatch):
        """Test if an ImportError is raised when basis_set_exchange is requested but
        not installed."""

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "basis_set_exchange", None)

            with pytest.raises(ImportError, match="This feature requires basis_set_exchange"):
                load_basisset("sto-3g", "H")
