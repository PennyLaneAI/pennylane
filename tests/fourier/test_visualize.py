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
Unit tests for :mod:`fourier` visualization functions.
"""

import pytest

matplotlib = pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt

from pennylane import numpy as np

from pennylane.fourier.visualize import _validate_coefficients

from pennylane.fourier.visualize import (
    violin,
    bar,
    box,
    panel,
    radial_box,
)


coeffs_1D_valid_1 = np.array([0.5, 0, 0.25j, 0.25j, 0])
coeffs_1D_valid_2 = [0.5, 0.1j, -0.25j, 0.25j, -0.1j]
coeffs_1D_invalid = np.array([0.5, 0, 0.25j, 0.25j])

coeffs_1D_valid_list = [coeffs_1D_valid_1, coeffs_1D_valid_2]

coeffs_2D_valid_1 = np.array(
    [
        [
            0.07469786 + 0.0000e00j,
            0.0 + 4.3984e-04j,
            0.00101184 - 0.0000e00j,
            0.00101184 + 0.0000e00j,
            0.0 - 4.3984e-04j,
        ],
        [
            -0.03973803 - 1.9390e-03j,
            0.0 + 0.0000e00j,
            0.01986902 + 9.6950e-04j,
            0.01986902 + 9.6950e-04j,
            -0.0 + 0.0000e00j,
        ],
        [
            0.0121718 - 3.2000e-07j,
            0.02703674 - 7.2000e-07j,
            0.22464211 - 5.9600e-06j,
            0.22464211 - 5.9600e-06j,
            -0.02703674 + 7.2000e-07j,
        ],
        [
            0.0121718 + 3.2000e-07j,
            -0.02703674 - 7.2000e-07j,
            0.22464211 + 5.9600e-06j,
            0.22464211 + 5.9600e-06j,
            0.02703674 + 7.2000e-07j,
        ],
        [
            -0.03973803 + 1.9390e-03j,
            -0.0 - 0.0000e00j,
            0.01986902 - 9.6950e-04j,
            0.01986902 - 9.6950e-04j,
            0.0 - 0.0000e00j,
        ],
    ]
)

coeffs_2D_valid_2 = np.array(
    [
        [
            0.12707831 + 0.0j,
            -0.0 + 0.00014827j,
            0.0271287 - 0.0j,
            0.0271287 + 0.0j,
            -0.0 - 0.00014827j,
        ],
        [
            0.14675568 - 0.0061323j,
            0.0 + 0.0j,
            -0.07337784 + 0.00306615j,
            -0.07337784 + 0.00306615j,
            -0.0 - 0.0j,
        ],
        [
            0.12201549 - 0.010611j,
            0.10344825 - 0.00899631j,
            0.14288853 - 0.01242621j,
            0.14288853 - 0.01242621j,
            -0.10344825 + 0.00899631j,
        ],
        [
            0.12201549 + 0.010611j,
            -0.10344825 - 0.00899631j,
            0.14288853 + 0.01242621j,
            0.14288853 + 0.01242621j,
            0.10344825 + 0.00899631j,
        ],
        [
            0.14675568 + 0.0061323j,
            -0.0 + 0.0j,
            -0.07337784 - 0.00306615j,
            -0.07337784 - 0.00306615j,
            0.0 - 0.0j,
        ],
    ]
)

coeffs_2D_valid_list = [coeffs_2D_valid_1, coeffs_2D_valid_2]

coeffs_2D_varying_degrees = np.zeros((5, 9), dtype=complex)
coeffs_2D_varying_degrees[:5, :5] = coeffs_2D_valid_2

coeffs_2D_invalid = np.array(
    [
        [
            0.12707831 + 0.0j,
            -0.0 + 0.00014827j,
            0.0271287 - 0.0j,
            0.0271287 + 0.0j,
            -0.0 - 0.00014827j,
        ],
        [
            0.14675568 - 0.0061323j,
            0.0 + 0.0j,
            -0.07337784 + 0.00306615j,
            -0.07337784 + 0.00306615j,
            -0.0 - 0.0j,
        ],
        [
            0.12201549 - 0.010611j,
            0.10344825 - 0.00899631j,
            0.14288853 - 0.01242621j,
            0.14288853 - 0.01242621j,
            -0.10344825 + 0.00899631j,
        ],
        [
            0.12201549 + 0.010611j,
            -0.10344825 - 0.00899631j,
            0.14288853 + 0.01242621j,
            0.14288853 + 0.01242621j,
            0.10344825 + 0.00899631j,
        ],
    ]
)

coeffs_3D_valid = np.zeros((5, 5, 5), dtype=complex)
data = {
    (0, 0, 1): -0.00882888 - 0.14568055j,
    (0, 0, 4): -0.00882888 + 0.14568055j,
    (0, 1, 0): 0.38262211 + 0.0j,
    (0, 2, 0): -0.0 - 0.03218167j,
    (0, 2, 1): 0.00441444 + 0.07284027j,
    (0, 2, 4): 0.00441444 - 0.07284027j,
    (0, 3, 0): -0.0 + 0.03218167j,
    (0, 3, 1): 0.00441444 + 0.07284027j,
    (0, 3, 4): 0.00441444 - 0.07284027j,
    (0, 4, 0): 0.38262211 + 0.0j,
    (2, 0, 1): 0.0019699 - 0.00293059j,
    (2, 0, 4): -0.0023094 + 0.00267124j,
    (2, 1, 0): 0.00439013 - 0.00574692j,
    (2, 2, 0): 0.00047266 - 0.00061874j,
    (2, 2, 1): -0.00098495 + 0.00146529j,
    (2, 2, 4): 0.0011547 - 0.00133562j,
    (2, 3, 0): -0.00047266 + 0.00061874j,
    (2, 3, 1): -0.00098495 + 0.00146529j,
    (2, 3, 4): 0.0011547 - 0.00133562j,
    (2, 4, 0): 0.00439013 - 0.00574692j,
    (3, 0, 1): -0.0023094 - 0.00267124j,
    (3, 0, 4): 0.0019699 + 0.00293059j,
    (3, 1, 0): 0.00439013 + 0.00574692j,
    (3, 2, 0): -0.00047266 - 0.00061874j,
    (3, 2, 1): 0.0011547 + 0.00133562j,
    (3, 2, 4): -0.00098495 - 0.00146529j,
    (3, 3, 0): 0.00047266 + 0.00061874j,
    (3, 3, 1): 0.0011547 + 0.00133562j,
    (3, 3, 4): -0.00098495 - 0.00146529j,
    (3, 4, 0): 0.00439013 + 0.00574692j,
}
coeffs_3D_varying_degrees = np.zeros((3, 7, 5), dtype=complex)

for key, val in data.items():
    coeffs_3D_valid[key] = val
    key = (key[0] - 1 if key[0] > 0 else 0, *key[1:])
    coeffs_3D_varying_degrees[key] = val


fig_valid, ax_valid = plt.subplots(2, 1, sharex=True, sharey=True)
fig_invalid, ax_invalid = plt.subplots(3, 1, sharex=True, sharey=True)

fig_radial_valid, ax_radial_valid = plt.subplots(
    2, 1, sharex=True, sharey=True, subplot_kw=dict(polar=True)
)
fig_radial_invalid, ax_radial_invalid = plt.subplots(
    3, 1, sharex=True, sharey=True, subplot_kw=dict(polar=True)
)

fig_panel_valid, ax_panel_valid = plt.subplots(5, 5, sharex=True, sharey=True)
fig_panel_1d_valid, ax_panel_1d_valid = plt.subplots(5, 1, sharex=True, sharey=True)

fig_panel_invalid, ax_panel_invalid = plt.subplots(3, 2, sharex=True, sharey=True)


class TestValidateCoefficients:
    """Test Fourier coefficients are properly validated/invalidated."""

    @pytest.mark.parametrize(
        "coeffs,n_inputs,can_be_list,expected_coeffs",
        [
            (coeffs_1D_valid_1, 1, True, np.array([coeffs_1D_valid_1])),
            (coeffs_1D_valid_1, 1, False, np.array(coeffs_1D_valid_1)),
            (coeffs_1D_valid_2, 1, True, np.array([coeffs_1D_valid_2])),
            (coeffs_1D_valid_2, 1, False, coeffs_1D_valid_2),
            (coeffs_2D_valid_1, 2, True, np.array([coeffs_2D_valid_1])),
            (coeffs_2D_valid_list, 2, True, np.array(coeffs_2D_valid_list)),
            (coeffs_3D_valid, 3, True, np.array([coeffs_3D_valid])),
            (coeffs_3D_valid, 3, False, coeffs_3D_valid),
            (coeffs_3D_varying_degrees, 3, True, np.array([coeffs_3D_varying_degrees])),
            (coeffs_3D_varying_degrees, 3, False, coeffs_3D_varying_degrees),
        ],
    )
    def test_valid_fourier_coeffs(self, coeffs, n_inputs, can_be_list, expected_coeffs):
        """Check that valid parameters are properly processed."""
        obtained_coeffs = _validate_coefficients(coeffs, n_inputs, can_be_list)
        assert np.allclose(obtained_coeffs, expected_coeffs)

    def test_incorrect_type_fourier_coeffs(self):
        """Check that invalid type of parameters is caught"""
        with pytest.raises(TypeError, match="must be a list of numerical"):
            _validate_coefficients("A", True)

    @pytest.mark.parametrize(
        "coeffs,n_inputs,can_be_list,expected_error_message",
        [
            (coeffs_1D_invalid, 1, True, "Shape of input coefficients must be 2d"),
            (coeffs_1D_valid_1, 2, True, "Plotting function expected a list of"),
            (coeffs_2D_invalid, 2, False, "Shape of input coefficients must be 2d_i"),
        ],
    )
    def test_invalid_fourier_coeffs(self, coeffs, n_inputs, can_be_list, expected_error_message):
        """Check invalid Fourier coefficient inputs are caught."""
        with pytest.raises(ValueError, match=expected_error_message):
            _validate_coefficients(coeffs, n_inputs, can_be_list)


class TestInvalidAxesPassing:
    """Test that axes of the incorrect type are not plotted on."""

    @pytest.mark.parametrize(
        "func,coeffs,n_inputs,ax,expected_error_message",
        [
            (
                violin,
                coeffs_1D_valid_1,
                1,
                ax_invalid,
                "Matplotlib axis should consist of two subplots.",
            ),
            (
                box,
                coeffs_1D_valid_2,
                1,
                ax_invalid,
                "Matplotlib axis should consist of two subplots.",
            ),
            (
                bar,
                coeffs_1D_valid_1,
                1,
                ax_invalid,
                "Matplotlib axis should consist of two subplots.",
            ),
            (
                radial_box,
                coeffs_2D_valid_list,
                2,
                ax_radial_invalid,
                "Matplotlib axis should consist of two subplots.",
            ),
            (
                radial_box,
                coeffs_2D_valid_list,
                2,
                ax_valid,
                "Matplotlib axes for radial_box must be polar.",
            ),
            (
                panel,
                coeffs_2D_valid_list,
                2,
                ax_panel_invalid,
                "Shape of subplot axes must match the shape of the coefficient data.",
            ),
        ],
    )
    def test_invalid_axes(self, func, coeffs, n_inputs, ax, expected_error_message):
        """Test that invalid axes are not plotted on."""
        with pytest.raises(ValueError, match=expected_error_message):
            func(coeffs, n_inputs, ax)


class TestReturnType:
    """Test that the functions return an axis date type."""

    @pytest.mark.parametrize(
        "func,coeffs,n_inputs,ax,show_freqs",
        [
            (violin, coeffs_1D_valid_1, 1, ax_valid, True),
            (violin, coeffs_1D_valid_1, 1, ax_valid, False),
            (violin, coeffs_2D_varying_degrees, 2, ax_valid, True),
            (box, coeffs_1D_valid_1, 1, ax_valid, True),
            (box, coeffs_1D_valid_1, 1, ax_valid, False),
            (box, coeffs_3D_valid, 3, ax_valid, True),
            (bar, coeffs_1D_valid_1, 1, ax_valid, True),
            (bar, coeffs_1D_valid_1, 1, ax_valid, False),
            (bar, coeffs_3D_varying_degrees, 3, ax_valid, False),
            (radial_box, coeffs_2D_valid_list, 2, ax_radial_valid, True),
            (radial_box, coeffs_2D_valid_list, 2, ax_radial_valid, False),
            (panel, coeffs_2D_valid_list, 2, ax_panel_valid, None),
            (panel, coeffs_1D_valid_list, 1, ax_panel_1d_valid, None),
        ],
    )
    def test_correct_return_type(self, func, coeffs, n_inputs, ax, show_freqs):
        """Test that invalid axes are not plotted on."""
        if show_freqs is None:
            res = func(coeffs, n_inputs, ax)
        else:
            res = func(coeffs, n_inputs, ax, show_freqs=show_freqs)

        assert isinstance(res, type(ax))


def test_panel_n_inputs():
    """Tests that error is raised if n_inputs not 1 or 2."""
    with pytest.raises(ValueError, match="Panel plot function accepts input"):
        panel(coeffs_1D_valid_list, 3, ax_panel_1d_valid)
