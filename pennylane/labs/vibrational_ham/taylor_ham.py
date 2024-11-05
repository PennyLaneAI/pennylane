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

from pennylane.labs.vibrational_ham.bosonic import BoseSentence, BoseWord


def _obtain_r2(ytrue, yfit):
    ymean = np.sum(ytrue) / len(ytrue)
    ssres = np.sum((ytrue - yfit) ** 2)
    sstot = np.sum((ytrue - ymean) ** 2)

    return 1 - ssres / sstot


def _remove_harmonic(freqs, pes_onebody):
    nmodes, quad_order = np.shape(pes_onebody)
    gauss_grid, gauss_weights = np.polynomial.hermite.hermgauss(quad_order)

    harmonic_pes = np.zeros((nmodes, quad_order))
    anh_pes = np.zeros((nmodes, quad_order))

    for ii in range(nmodes):
        ho_const = freqs[ii] / 2
        harmonic_pes[ii, :] = ho_const * (gauss_grid**2)
        anh_pes[ii, :] = pes_onebody[ii, :] - harmonic_pes[ii, :]

    return nmodes, quad_order, anh_pes, harmonic_pes


def _fit_onebody(anh_pes, deg, min_deg=3):
    if deg < min_deg:
        raise Exception(
            f"Taylor expansion degree is {deg}<{min_deg}, minimal degree is set by min_deg keyword!"
        )

    nmodes, quad_order = np.shape(anh_pes)
    gauss_grid, _ = np.polynomial.hermite.hermgauss(quad_order)
    fs = np.zeros((nmodes, deg - min_deg + 1))

    predicted_1D = np.zeros_like(anh_pes)

    for i1 in range(nmodes):
        poly1D = PolynomialFeatures(degree=(min_deg, deg), include_bias=False)
        poly1D_features = poly1D.fit_transform(gauss_grid.reshape(-1, 1))
        poly1D_reg_model = LinearRegression()
        poly1D_reg_model.fit(poly1D_features, anh_pes[i1, :])
        fs[i1, :] = poly1D_reg_model.coef_
        predicted_1D[i1, :] = poly1D_reg_model.predict(poly1D_features)

    return fs, predicted_1D


def _twobody_degs(deg, min_deg=3):
    fit_degs = []
    deg_idx = 0
    for feat_deg in range(min_deg, deg + 1):
        max_deg = feat_deg - 1
        for deg_dist in range(1, max_deg + 1):
            q1deg = max_deg - deg_dist + 1
            q2deg = deg_dist
            fit_degs.append((q1deg, q2deg))

    return fit_degs


def _fit_twobody(pes_twobody, deg, min_deg=3):
    nmodes, _, quad_order, _ = np.shape(pes_twobody)
    gauss_grid, _ = np.polynomial.hermite.hermgauss(quad_order)

    if deg < min_deg:
        raise Exception(
            f"Taylor expansion degree is {deg}<{min_deg}, minimal degree is set by min_deg keyword!"
        )

    fit_degs = _twobody_degs(deg, min_deg)
    num_fs = len(fit_degs)
    fs = np.zeros((nmodes, nmodes, num_fs))

    predicted_2D = np.zeros_like(pes_twobody)

    grid_2D = np.array(np.meshgrid(gauss_grid, gauss_grid))
    q1 = grid_2D[0, ::].flatten()
    q2 = grid_2D[1, ::].flatten()
    idx_2D = np.array(np.meshgrid(range(quad_order), range(quad_order)))
    idx1 = idx_2D[0, ::].flatten()
    idx2 = idx_2D[1, ::].flatten()
    num_2D = len(q1)

    features = np.zeros((num_2D, num_fs))
    for deg_idx, Qs in enumerate(fit_degs):
        q1deg, q2deg = Qs
        features[:, deg_idx] = q1 ** (q1deg) * q2 ** (q2deg)

    for i1 in range(nmodes):
        for i2 in range(i1):
            poly2D = PolynomialFeatures(
                degree=(min_deg, deg), include_bias=False, interaction_only=True
            )
            Y = []
            for idx in range(num_2D):
                idx_q1 = idx1[idx]
                idx_q2 = idx2[idx]
                Y.append(pes_twobody[i1, i2, idx_q1, idx_q2])
            poly2D_reg_model = LinearRegression()
            poly2D_reg_model.fit(features, Y)
            fs[i1, i2, :] = poly2D_reg_model.coef_
            predicted = poly2D_reg_model.predict(features)
            for idx in range(num_2D):
                idx_q1 = idx1[idx]
                idx_q2 = idx2[idx]
                predicted_2D[i1, i2, idx_q1, idx_q2] = predicted[idx]

    return fs, predicted_2D


def _generate_bin_occupations(max_occ, nbins):
    # Generate all combinations placing max_occ balls in nbins
    combinations = list(itertools.product(range(max_occ + 1), repeat=nbins))

    # Filter valid combinations
    valid_combinations = [combo for combo in combinations if sum(combo) == max_occ]

    return valid_combinations


def _threebody_degs(deg, min_deg=3):
    fit_degs = []
    deg_idx = 0
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


def _fit_threebody(pes_threebody, deg, min_deg=3):
    nmodes, _, _, quad_order, _, _ = np.shape(pes_threebody)
    gauss_grid, _ = np.polynomial.hermite.hermgauss(quad_order)

    if deg < min_deg:
        raise Exception(
            f"Taylor expansion degree is {deg}<{min_deg}, minimal degree is set by min_deg keyword!"
        )

    predicted_3D = np.zeros_like(pes_threebody)
    fit_degs = _threebody_degs(deg)
    num_fs = len(fit_degs)
    fs = np.zeros((nmodes, nmodes, nmodes, num_fs))

    grid_3D = np.array(np.meshgrid(gauss_grid, gauss_grid, gauss_grid))
    q1 = grid_3D[0, ::].flatten()
    q2 = grid_3D[1, ::].flatten()
    q3 = grid_3D[2, ::].flatten()
    idx_3D = np.array(np.meshgrid(range(quad_order), range(quad_order), range(quad_order)))
    idx1 = idx_3D[0, ::].flatten()
    idx2 = idx_3D[1, ::].flatten()
    idx3 = idx_3D[2, ::].flatten()
    num_3D = len(q1)

    features = np.zeros((num_3D, num_fs))
    for deg_idx, Qs in enumerate(fit_degs):
        q1deg, q2deg, q3deg = Qs
        features[:, deg_idx] = q1 ** (q1deg) * q2 ** (q2deg) * q3 ** (q3deg)

    for i1 in range(nmodes):
        for i2 in range(i1):
            for i3 in range(i2):
                poly3D = PolynomialFeatures(
                    degree=(min_deg, deg), include_bias=False, interaction_only=True
                )
                Y = []
                for idx in range(num_3D):
                    idx_q1 = idx1[idx]
                    idx_q2 = idx2[idx]
                    idx_q3 = idx3[idx]
                    Y.append(pes_threebody[i1, i2, i3, idx_q1, idx_q2, idx_q3])

                poly3D_reg_model = LinearRegression()
                poly3D_reg_model.fit(features, Y)
                fs[i1, i2, i3, :] = poly3D_reg_model.coef_
                predicted = poly3D_reg_model.predict(features)
                for idx in range(num_3D):
                    idx_q1 = idx1[idx]
                    idx_q2 = idx2[idx]
                    idx_q3 = idx3[idx]
                    predicted_3D[i1, i2, i3, idx_q1, idx_q2, idx_q3] = predicted[idx]

    return fs, predicted_3D


def taylor_integrals(pes, deg=4, min_deg=3):
    r"""Returns the coefficients for real-space Hamiltonian.
    Args:
            pes: PES object.
            deg:
            min_deg:
    """

    nmodes, quad_order, anh_pes, harmonic_pes = _remove_harmonic(pes.freqs, pes.pes_onebody)
    coeff_1D, predicted_1D = _fit_onebody(anh_pes, deg, min_deg=min_deg)
    predicted_1D += harmonic_pes
    coeff_arr = [coeff_1D]
    predicted_arr = [predicted_1D]

    if pes.pes_twobody is not None:
        coeff_2D, predicted_2D = _fit_twobody(pes.pes_twobody, deg, min_deg=min_deg)
        coeff_arr.append(coeff_2D)
        predicted_arr.append(predicted_2D)

    if pes.pes_threebody is not None:
        coeff_3D, predicted_3D = _fit_threebody(pes.pes_threebody, deg, min_deg=min_deg)
        coeff_arr.append(coeff_3D)
        predicted_arr.append(predicted_3D)

    return coeff_arr


def taylor_integrals_dipole(pes, deg=4, min_deg=1):

    nmodes, quad_order, _ = pes.dipole_onebody.shape

    f_x_1D, predicted_x_1D = _fit_onebody(pes.dipole_onebody[:, :, 0], deg, min_deg=min_deg)
    f_x_arr = [f_x_1D]
    predicted_x_arr = [predicted_x_1D]

    f_y_1D, predicted_y_1D = _fit_onebody(pes.dipole_onebody[:, :, 1], deg, min_deg=min_deg)
    f_y_arr = [f_y_1D]
    predicted_y_arr = [predicted_y_1D]

    f_z_1D, predicted_z_1D = _fit_onebody(pes.dipole_onebody[:, :, 2], deg, min_deg=min_deg)
    f_z_arr = [f_z_1D]
    predicted_z_arr = [predicted_z_1D]

    if pes.dipole_twobody is not None:
        f_x_2D, predicted_x_2D = _fit_twobody(
            pes.dipole_twobody[:, :, :, :, 0], deg, min_deg=min_deg
        )
        f_x_arr.append(f_x_2D)
        predicted_x_arr.append(predicted_x_2D)

        f_y_2D, predicted_y_2D = _fit_twobody(
            pes.dipole_twobody[:, :, :, :, 1], deg, min_deg=min_deg
        )
        f_y_arr.append(f_y_2D)
        predicted_y_arr.append(predicted_y_2D)

        f_z_2D, predicted_z_2D = _fit_twobody(
            pes.dipole_twobody[:, :, :, :, 2], deg, min_deg=min_deg
        )
        f_z_arr.append(f_z_2D)
        predicted_z_arr.append(predicted_z_2D)

    if pes.dipole_threebody is not None:
        f_x_3D, predicted_x_3D = _fit_threebody(
            pes.dipole_threebody[:, :, :, :, :, :, 0], deg, min_deg=min_deg
        )
        f_x_arr.append(f_x_3D)
        predicted_x_arr.append(predicted_x_3D)

        f_y_3D, predicted_y_3D = _fit_threebody(
            pes.dipole_threebody[:, :, :, :, :, :, 1], deg, min_deg=min_deg
        )
        f_y_arr.append(f_y_3D)
        predicted_y_arr.append(predicted_y_3D)

        f_z_3D, predicted_z_3D = _fit_threebody(
            pes.dipole_threebody[:, :, :, :, :, :, 2], deg, min_deg=min_deg
        )
        f_z_arr.append(f_z_3D)
        predicted_z_arr.append(predicted_z_3D)

    return f_x_arr, f_y_arr, f_z_arr


def _position_to_boson(index, op):
    """Convert position operator `p` or `q` into respective bosonic operator
    
    Args:
        index (int): the index of the operator
        op (str): the position operator, either `"p"` or `"q"`

    Returns:
        BoseSentence: bosonic form of the position operator given
    """
    factor = 1j / np.sqrt(2) if op == "p" else 1 / np.sqrt(2)
    bop = factor * BoseWord({(0, index): "-"})
    bdag = factor * BoseWord({(0, index): "+"})
    return bdag - bop if op == "p" else bdag + bop


def taylor_anharmonic(taylor_coeffs, start_deg=2):
    """Build anharmonic term of taylor form bosonic observable from provided integrals
    
    Args:
        taylor_coeffs (list(float)): the coeffs of the taylor integrals
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
    """Build kinetic term of taylor form bosonic observable from provided integrals
    
    Args:
        taylor_coeffs (list(float)): the coeffs of the taylor integrals
        freqs (list(float)): the frequencies
        is_loc (bool): whether or not if localized
        Uloc (list(float)): not sure

    Returns:
        BoseSentence: anharmonic term of the taylor hamiltonian for given coeffs
    """
    taylor_1D = taylor_coeffs[0]
    num_modes, num_1D_coeffs = np.shape(taylor_1D)

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
    """Build harmonic term of taylor form bosonic observable from provided integrals
    
    Args:
        taylor_coeffs (list(float)): the coeffs of the taylor integrals
        freqs (list(float)): the harmonic frequencies

    Returns:
        BoseSentence: harmonic term of the taylor hamiltonian for given coeffs
    """
    taylor_1D = taylor_coeffs[0]
    num_modes, num_1D_coeffs = np.shape(taylor_1D)
    harm_pot = BoseSentence({})
    # Add Harmonic component
    for mode in range(num_modes):
        bosonized_qm2 = (
            _position_to_boson(mode, "q") * _position_to_boson(mode, "q")
        ).normal_order()
        harm_pot += bosonized_qm2 * freqs[mode] * 0.5

    return harm_pot.normal_order()


def taylor_bosonic(taylor_coeffs, freqs, is_loc=True, Uloc=None):
    """Build taylor form bosonic observable from provided integrals
    
    Args:
        taylor_coeffs (list(float)): the coeffs of the taylor integrals
        freqs (list(float)): the harmonic frequencies
        is_loc (bool): whether or not if localized
        Uloc (): not sure

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
    """Compute taylor hamiltonian from PES object"""
    coeffs_arr = taylor_integrals(pes_object, deg, min_deg)
    freqs = taylor_integrals_dipole(pes_object, deg, min_deg)
    ham = taylor_bosonic(coeffs_arr, freqs)
    return ham
