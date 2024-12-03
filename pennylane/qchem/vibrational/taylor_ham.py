# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The functions related to the construction of the taylor form Hamiltonian."""
import itertools

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from pennylane.bose import BoseSentence, BoseWord


def _obtain_r2(ytrue, yfit):
    """Calculates coefficient of determination of accuracy of fit of a model."""
    ymean = np.sum(ytrue) / len(ytrue)
    ssres = np.sum((ytrue - yfit) ** 2)
    sstot = np.sum((ytrue - ymean) ** 2)

    return 1 - ssres / sstot


def _remove_harmonic(freqs, pes_onemode):
    """Removes the harmonic part from the PES

    Args:
        freqs (list(float)): normal mode frequencies
        pes_onemode (TensorLike[float]): one mode PES

    Returns:
        harmonic part of the PES
    """
    nmodes, quad_order = np.shape(pes_onemode)
    gauss_grid, _ = np.polynomial.hermite.hermgauss(quad_order)

    harmonic_pes = np.zeros((nmodes, quad_order))
    anh_pes = np.zeros((nmodes, quad_order))

    for ii in range(nmodes):
        ho_const = freqs[ii] / 2
        harmonic_pes[ii, :] = ho_const * (gauss_grid**2)
        anh_pes[ii, :] = pes_onemode[ii, :] - harmonic_pes[ii, :]

    return nmodes, quad_order, anh_pes, harmonic_pes


def _fit_onebody(anh_pes, deg, min_deg=3):
    r"""Fits the one-body PES to get one-body coefficients.

    Args:
        anh_pes (list(list(float))): anharmonic part of the PES object
        deg (int): maximum degree of taylor form polynomial
        min_deg (int): minimum degree of taylor form polynomial

    Returns:
        tuple (list(list(float)), list(list(float))):
            - the one-body coefficients
            - the predicted one-body PES using fitted coefficients
    """
    if deg < min_deg:
        raise Exception(
            f"Taylor expansion degree is {deg}<{min_deg}, minimal degree is set by min_deg keyword!"
        )

    nmodes, quad_order = np.shape(anh_pes)
    grid, _ = np.polynomial.hermite.hermgauss(quad_order)
    coeffs = np.zeros((nmodes, deg - min_deg + 1))

    predicted_1D = np.zeros_like(anh_pes)

    for i1 in range(nmodes):
        poly1D = PolynomialFeatures(degree=(min_deg, deg), include_bias=False)
        poly1D_features = poly1D.fit_transform(grid.reshape(-1, 1))
        poly1D_reg_model = LinearRegression()
        poly1D_reg_model.fit(poly1D_features, anh_pes[i1, :])
        coeffs[i1, :] = poly1D_reg_model.coef_
        predicted_1D[i1, :] = poly1D_reg_model.predict(poly1D_features)

    return coeffs, predicted_1D


def _twobody_degs(deg, min_deg=3):
    """Finds the degree of fit for two-body coefficients.

    Args:
        deg (int): the maximum total degree of the polynomial expansion
        min_deg (int): The minimum degree to include in the expansion.
            Defaults to 3.

    Returns:
        list(tuple): A list of tuples `(q1deg, q2deg)` where:
            - `q1deg` (int): The degree of the polynomial in the first variable (`q1`).
            - `q2deg` (int): The degree of the polynomial in the second variable (`q2`).
            - `q1deg + q2deg = feat_deg` for each combination, where `min_deg <= feat_deg <= deg`.
    """
    fit_degs = []
    for feat_deg in range(min_deg, deg + 1):
        max_deg = feat_deg - 1
        for deg_dist in range(1, max_deg + 1):
            q1deg = max_deg - deg_dist + 1
            q2deg = deg_dist
            fit_degs.append((q1deg, q2deg))

    return fit_degs


def _fit_twobody(pes_twomode, deg, min_deg=3):
    r"""Fits the two-body PES to get two-body coefficients.

    Args:
        two-body PES (TensorLike[float]): two-mode PES
        deg (int): maximum degree of taylor form polynomial
        min_deg (int): minimum degree of taylor form polynomial

    Returns:
        tuple (list(list(float)), list(list(float))):
            - the two-body coefficients
            - the predicted two-body PES using fitted coefficients
    """
    nmodes, _, quad_order, _ = np.shape(pes_twomode)
    gauss_grid, _ = np.polynomial.hermite.hermgauss(quad_order)

    if deg < min_deg:
        raise Exception(
            f"Taylor expansion degree is {deg}<{min_deg}, minimal degree is set by min_deg keyword!"
        )

    fit_degs = _twobody_degs(deg, min_deg)
    num_coeffs = len(fit_degs)
    coeffs = np.zeros((nmodes, nmodes, num_coeffs))

    predicted_2D = np.zeros_like(pes_twomode)

    grid_2D = np.array(np.meshgrid(gauss_grid, gauss_grid))
    q1 = grid_2D[0, ::].flatten()
    q2 = grid_2D[1, ::].flatten()
    idx_2D = np.array(np.meshgrid(range(quad_order), range(quad_order)))
    idx1 = idx_2D[0, ::].flatten()
    idx2 = idx_2D[1, ::].flatten()
    num_2D = len(q1)

    features = np.zeros((num_2D, num_coeffs))
    for deg_idx, (q1deg, q2deg) in enumerate(fit_degs):
          features[:, deg_idx] = (q1 ** q1deg) * (q2 ** q2deg)

    for i1 in range(nmodes):
        for i2 in range(i1):
            Y = []
            for idx in range(num_2D):
                idx_q1 = idx1[idx]
                idx_q2 = idx2[idx]
                Y.append(pes_twomode[i1, i2, idx_q1, idx_q2])
            poly2D_reg_model = LinearRegression()
            poly2D_reg_model.fit(features, Y)
            coeffs[i1, i2, :] = poly2D_reg_model.coef_
            predicted = poly2D_reg_model.predict(features)
            for idx in range(num_2D):
                predicted_2D[i1, i2, idx1[idx], idx2[idx]] = predicted[idx]

    return coeffs, predicted_2D


def _generate_bin_occupations(max_occ, nbins):
    combinations = list(itertools.product(range(max_occ + 1), repeat=nbins))

    # Filter valid combinations
    valid_combinations = [combo for combo in combinations if sum(combo) == max_occ]

    return valid_combinations


def _threebody_degs(deg, min_deg=3):
    """Finds the degree of fit for three-body coefficients.

    Args:
        deg (int): the maximum total degree of the polynomial expansion
        min_deg (int): The minimum degree to include in the expansion.
            Defaults to 3.

    Returns:
        list(tuple): A list of tuples `(q1deg, q2deg, q3deg)` where:
            - `q1deg` (int): The degree of the polynomial in the first variable (`q1`).
            - `q2deg` (int): The degree of the polynomial in the second variable (`q2`).
            - `q3deg` (int): The degree of the polynomial in the second variable (`q3`).
            - `q1deg + q2deg + q3deg = feat_deg` for each combination, where `min_deg <= feat_deg <= deg`.
    """
    fit_degs = []
    for feat_deg in range(min_deg, deg + 1):
        max_deg = feat_deg - 3
        if max_deg < 0:
            continue
        possible_occupations = _generate_bin_occupations(max_deg, 3)
        for occ in possible_occupations:
            q1deg = 1 + occ[0]
            q2deg = 1 + occ[1]
            q3deg = 1 + occ[2]
            fit_degs.append((q1deg, q2deg, q3deg))

    return fit_degs


def _fit_threebody(pes_threemode, deg, min_deg=3):
    r"""Fits the three-body PES to get three-body coefficients.

    Args:
        three-body PES (TensorLike[float]): three-mode PES
        deg (int): maximum degree of taylor form polynomial
        min_deg (int): minimum degree of taylor form polynomial

    Returns:
        tuple (list(list(float)), list(list(float))):
            - the three-body coefficients
            - the predicted three-body PES using fitted coefficients
    """
    nmodes, _, _, quad_order, _, _ = np.shape(pes_threemode)
    gauss_grid, _ = np.polynomial.hermite.hermgauss(quad_order)

    if deg < min_deg:
        raise Exception(
            f"Taylor expansion degree is {deg}<{min_deg}, minimal degree is set by min_deg keyword!"
        )

    predicted_3D = np.zeros_like(pes_threemode)
    fit_degs = _threebody_degs(deg)
    num_coeffs = len(fit_degs)
    coeffs = np.zeros((nmodes, nmodes, nmodes, num_coeffs))

    grid_3D = np.array(np.meshgrid(gauss_grid, gauss_grid, gauss_grid))
    q1 = grid_3D[0, ::].flatten()
    q2 = grid_3D[1, ::].flatten()
    q3 = grid_3D[2, ::].flatten()
    idx_3D = np.array(np.meshgrid(range(quad_order), range(quad_order), range(quad_order)))
    idx1 = idx_3D[0, ::].flatten()
    idx2 = idx_3D[1, ::].flatten()
    idx3 = idx_3D[2, ::].flatten()
    num_3D = len(q1)

    features = np.zeros((num_3D, num_coeffs))
    for deg_idx, Qs in enumerate(fit_degs):
        q1deg, q2deg, q3deg = Qs
        features[:, deg_idx] = q1 ** (q1deg) * q2 ** (q2deg) * q3 ** (q3deg)

    for i1 in range(nmodes):
        for i2 in range(i1):
            for i3 in range(i2):
                Y = []
                for idx in range(num_3D):
                    idx_q1 = idx1[idx]
                    idx_q2 = idx2[idx]
                    idx_q3 = idx3[idx]
                    Y.append(pes_threemode[i1, i2, i3, idx_q1, idx_q2, idx_q3])

                poly3D_reg_model = LinearRegression()
                poly3D_reg_model.fit(features, Y)
                coeffs[i1, i2, i3, :] = poly3D_reg_model.coef_
                predicted = poly3D_reg_model.predict(features)
                for idx in range(num_3D):
                    idx_q1 = idx1[idx]
                    idx_q2 = idx2[idx]
                    idx_q3 = idx3[idx]
                    predicted_3D[i1, i2, i3, idx_q1, idx_q2, idx_q3] = predicted[idx]

    return coeffs, predicted_3D


def taylor_coeffs(pes, deg=4, min_deg=3):
    r"""Computes the Taylor form fitted coefficients for Hamiltonian construction

    Args:
        pes (VibrationalPES): the PES object
        deg (int): maximum degree of taylor form polynomial
        min_deg (int): minimum degree of taylor form polynomial

    Returns:
        coeff_arr (list(list(floats))): the coeffs of the one-body, two-body, three-body terms
    """

    _, _, anh_pes, harmonic_pes = _remove_harmonic(pes.freqs, pes.pes_onemode)
    coeff_1D, predicted_1D = _fit_onebody(anh_pes, deg, min_deg=min_deg)
    predicted_1D += harmonic_pes
    coeff_arr = [coeff_1D]
    predicted_arr = [predicted_1D]

    if pes.pes_twomode is not None:
        coeff_2D, predicted_2D = _fit_twobody(pes.pes_twomode, deg, min_deg=min_deg)
        coeff_arr.append(coeff_2D)
        predicted_arr.append(predicted_2D)

    if pes.pes_threemode is not None:
        coeff_3D, predicted_3D = _fit_threebody(pes.pes_threemode, deg, min_deg=min_deg)
        coeff_arr.append(coeff_3D)
        predicted_arr.append(predicted_3D)

    return coeff_arr


def taylor_dipole_coeffs(pes, deg=4, min_deg=1):
    r"""Calculates Taylor form fitted coefficients for dipole construction.

    Args:
        pes (VibrationalPES): the PES object
        deg (int): the maximum degree of the taylor polynomial
        min_deg (int): the minimum degree of the taylor polynomial

    Returns:
        tuple: a tuple containing:
            - coeffs_x_arr (list(floats)): coefficients for x-displacements
            - coeffs_y_arr (list(floats)): coefficients for y-displacements
            - coeffs_z_arr (list(floats)): coefficients for z-displacements
    """
    coeffs_x_1D, predicted_x_1D = _fit_onebody(pes.dipole_onemode[:, :, 0], deg, min_deg=min_deg)
    coeffs_x_arr = [coeffs_x_1D]
    predicted_x_arr = [predicted_x_1D]

    coeffs_y_1D, predicted_y_1D = _fit_onebody(pes.dipole_onemode[:, :, 1], deg, min_deg=min_deg)
    coeffs_y_arr = [coeffs_y_1D]
    predicted_y_arr = [predicted_y_1D]

    coeffs_z_1D, predicted_z_1D = _fit_onebody(pes.dipole_onemode[:, :, 2], deg, min_deg=min_deg)
    coeffs_z_arr = [coeffs_z_1D]
    predicted_z_arr = [predicted_z_1D]

    if pes.dipole_twomode is not None:
        coeffs_x_2D, predicted_x_2D = _fit_twobody(
            pes.dipole_twomode[:, :, :, :, 0], deg, min_deg=min_deg
        )
        coeffs_x_arr.append(coeffs_x_2D)
        predicted_x_arr.append(predicted_x_2D)

        coeffs_y_2D, predicted_y_2D = _fit_twobody(
            pes.dipole_twomode[:, :, :, :, 1], deg, min_deg=min_deg
        )
        coeffs_y_arr.append(coeffs_y_2D)
        predicted_y_arr.append(predicted_y_2D)

        coeffs_z_2D, predicted_z_2D = _fit_twobody(
            pes.dipole_twomode[:, :, :, :, 2], deg, min_deg=min_deg
        )
        coeffs_z_arr.append(coeffs_z_2D)
        predicted_z_arr.append(predicted_z_2D)

    if pes.dipole_threemode is not None:
        coeffs_x_3D, predicted_x_3D = _fit_threebody(
            pes.dipole_threemode[:, :, :, :, :, :, 0], deg, min_deg=min_deg
        )
        coeffs_x_arr.append(coeffs_x_3D)
        predicted_x_arr.append(predicted_x_3D)

        coeffs_y_3D, predicted_y_3D = _fit_threebody(
            pes.dipole_threemode[:, :, :, :, :, :, 1], deg, min_deg=min_deg
        )
        coeffs_y_arr.append(coeffs_y_3D)
        predicted_y_arr.append(predicted_y_3D)

        coeffs_z_3D, predicted_z_3D = _fit_threebody(
            pes.dipole_threemode[:, :, :, :, :, :, 2], deg, min_deg=min_deg
        )
        coeffs_z_arr.append(coeffs_z_3D)
        predicted_z_arr.append(predicted_z_3D)

    return coeffs_x_arr, coeffs_y_arr, coeffs_z_arr


def _position_to_boson(index, op):
    """Convert position operator `p` or `q` into respective bosonic operator. The conversion is
    described in `Eq. 6 and 7 <https://arxiv.org/pdf/1703.09313>`_.

    Args:
        index (int): the index of the operator
        op (str): the position operator, either ``"p"`` or ``"q"``

    Returns:
        BoseSentence: bosonic form of the position operator given
    """
    factor = 1j / np.sqrt(2) if op == "p" else 1 / np.sqrt(2)
    bop = factor * BoseWord({(0, index): "-"})
    bdag = factor * BoseWord({(0, index): "+"})
    return bdag - bop if op == "p" else bdag + bop


def taylor_anharmonic(taylor_coeffs, start_deg=2):
    """Build anharmonic term of taylor form bosonic observable from provided coefficients described
    in `Eq. 10 <https://arxiv.org/pdf/1703.09313>`_.

    Args:
        taylor_coeffs (list(float)): the coeffs of the taylor expansion
        start_deg (int): the starting degree

    Returns:
        BoseSentence: anharmonic term of the taylor hamiltonian for given coeffs
    """
    num_coups = len(taylor_coeffs)

    taylor_1D = taylor_coeffs[0]
    num_modes, num_1D_coeffs = np.shape(taylor_1D)

    taylor_deg = num_1D_coeffs + start_deg - 1

    ordered_dict = BoseSentence({})

    if num_coups > 3:
        raise ValueError("Found 4-mode expansion coefficients, not defined!")

    # One-mode expansion
    for mode in range(num_modes):
        bosonized_qm = _position_to_boson(mode, "q")
        for deg_i in range(start_deg, taylor_deg + 1):
            coeff = taylor_1D[mode, deg_i - start_deg]
            qpow = bosonized_qm**deg_i
            ordered_dict += (coeff * qpow).normal_order()
    # Two-mode expansion
    if num_coups > 1:
        taylor_2D = taylor_coeffs[1]
        degs_2d = _twobody_degs(taylor_deg, min_deg=start_deg)
        for m1 in range(num_modes):
            bosonized_qm1 = _position_to_boson(m1, "q")
            for m2 in range(m1):
                bosonized_qm2 = _position_to_boson(m2, "q")
                for deg_idx, Qs in enumerate(degs_2d):
                    q1deg, q2deg = Qs[:2]
                    coeff = taylor_2D[m1, m2, deg_idx]
                    bosonized_qm1_pow = bosonized_qm1**q1deg
                    bosonized_qm2_pow = bosonized_qm2**q2deg
                    ordered_dict += (coeff * bosonized_qm1_pow * bosonized_qm2_pow).normal_order()

    # Three-mode expansion
    if num_coups > 2:
        degs_3d = _threebody_degs(taylor_deg, min_deg=start_deg)
        taylor_3D = taylor_coeffs[2]
        for m1 in range(num_modes):
            bosonized_qm1 = _position_to_boson(m1, "q")
            for m2 in range(m1):
                bosonized_qm2 = _position_to_boson(m2, "q")
                for m3 in range(m2):
                    bosonized_qm3 = _position_to_boson(m3, "q")
                    for deg_idx, Qs in enumerate(degs_3d):
                        q1deg, q2deg, q3deg = Qs[:3]
                        coeff = taylor_3D[m1, m2, m3, deg_idx]
                        bosonized_qm1_pow = bosonized_qm1**q1deg
                        bosonized_qm2_pow = bosonized_qm2**q2deg
                        bosonized_qm3_pow = bosonized_qm3**q3deg
                        ordered_dict += (
                            coeff * bosonized_qm1_pow * bosonized_qm2_pow * bosonized_qm3_pow
                        ).normal_order()

    return BoseSentence(ordered_dict).normal_order()


def taylor_kinetic(taylor_coeffs, freqs, is_loc=True, Uloc=None):
    """Build kinetic term of taylor form bosonic observable from provided coefficients

    Args:
        taylor_coeffs (list(float)): the coeffs of the taylor expansion
        freqs (list(float)): the frequencies
        is_loc (bool): whether or not if localized
        Uloc (list(float)): localization matrix indicating the relationship between original and
            localized modes

    Returns:
        BoseSentence: anharmonic term of the taylor hamiltonian for given coeffs
    """
    taylor_1D = taylor_coeffs[0]
    num_modes, _ = np.shape(taylor_1D)

    if is_loc:
        alphas_arr = np.einsum("ij,ik,j,k->jk", Uloc, Uloc, np.sqrt(freqs), np.sqrt(freqs))
    else:
        alphas_arr = np.zeros((num_modes, num_modes))
        for m in range(num_modes):
            alphas_arr[m, m] = freqs[m]

    kin_ham = BoseSentence({})
    for m1 in range(num_modes):
        pm1 = _position_to_boson(m1, "p")
        for m2 in range(num_modes):
            pm2 = _position_to_boson(m2, "p")
            kin_ham += (0.5 * alphas_arr[m1, m2]) * (pm1 * pm2).normal_order()

    return kin_ham.normal_order()


def taylor_harmonic(taylor_coeffs, freqs):
    """Build harmonic term of taylor form bosonic observable from provided coefficients, see first
    term of `Eq. 4 and Eq. 7 <https://arxiv.org/pdf/1703.09313>`_.

    Args:
        taylor_coeffs (list(float)): the coeffs of the taylor expansion
        freqs (list(float)): the harmonic frequencies

    Returns:
        BoseSentence: harmonic term of the taylor hamiltonian for given coeffs
    """
    taylor_1D = taylor_coeffs[0]
    num_modes, _ = np.shape(taylor_1D)
    harm_pot = BoseSentence({})
    for mode in range(num_modes):
        bosonized_qm2 = (
            _position_to_boson(mode, "q") * _position_to_boson(mode, "q")
        ).normal_order()
        harm_pot += bosonized_qm2 * freqs[mode] * 0.5

    return harm_pot.normal_order()


def taylor_bosonic(taylor_coeffs, freqs, is_loc=True, Uloc=None):
    """Build taylor form bosonic observable from provided coefficients, following `Eq. 4 and Eq. 7
    <https://arxiv.org/pdf/1703.09313>`_.

    Args:
        taylor_coeffs (list(float)): the coeffs of the taylor expansion
        freqs (list(float)): the harmonic frequencies
        is_loc (bool): whether or not if localized
        Uloc (list(float)): localization matrix indicating the relationship between original and
            localized modes

    Returns:
        BoseSentence: taylor hamiltonian for given coeffs
    """
    if is_loc:
        start_deg = 2
    else:
        start_deg = 3

    harm_pot = taylor_harmonic(taylor_coeffs, freqs)
    ham = taylor_anharmonic(taylor_coeffs, start_deg) + harm_pot
    kin_ham = taylor_kinetic(taylor_coeffs, freqs, is_loc, Uloc)
    ham += kin_ham
    return ham.normal_order()


def taylor_hamiltonian(pes_object, deg=4, min_deg=3):
    """Compute taylor hamiltonian from PES object

    Args:
        pes_object(VibrationalPES): the PES object
        deg (int): the maximum degree of the taylor polynomial
        min_deg (int): the minimum degree of the taylor polynomial

    Returns:
        BoseSentence: taylor hamiltonian for given PES and degree
    """
    coeffs_arr = taylor_coeffs(pes_object, deg, min_deg)
    ham = taylor_bosonic(
        coeffs_arr, pes_object.freqs, is_loc=pes_object.localized, Uloc=pes_object.uloc
    )
    return ham
