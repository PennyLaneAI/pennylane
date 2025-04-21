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
"""The functions related to the construction of the Taylor form Hamiltonian."""
import itertools

import numpy as np

from pennylane.bose import BoseSentence, BoseWord, binary_mapping, unary_mapping

# pylint: disable=import-outside-toplevel


def _import_sklearn():
    """Import sklearn."""
    try:
        import sklearn
    except ImportError as Error:
        raise ImportError(
            "This feature requires sklearn. It can be installed with: pip install scikit-learn."
        ) from Error

    return sklearn


def _remove_harmonic(freqs, onemode_pes):
    """Removes the harmonic part from the PES.

    Args:
        freqs (list(float)): the harmonic frequencies in atomic units
        onemode_pes (TensorLike[float]): one mode PES

    Returns:
        tuple: A tuple containing the following:
            - TensorLike[float] : anharmonic part of the PES
            - TensorLike[float] : harmonic part of the PES
    """
    nmodes, quad_order = np.shape(onemode_pes)
    grid, _ = np.polynomial.hermite.hermgauss(quad_order)
    harmonic_pes = np.zeros((nmodes, quad_order))
    anh_pes = np.zeros((nmodes, quad_order))

    for ii in range(nmodes):
        ho_const = freqs[ii] / 2
        harmonic_pes[ii, :] = ho_const * (grid**2)
        anh_pes[ii, :] = onemode_pes[ii, :] - harmonic_pes[ii, :]

    return anh_pes, harmonic_pes


def _fit_onebody(onemode_op, max_deg, min_deg=3):
    r"""Fits the one-mode operator to get one-body coefficients.

    Args:
        onemode_op (TensorLike[float]): one-mode operator
        max_deg (int): maximum degree of Taylor form polynomial
        min_deg (int): minimum degree of Taylor form polynomial

    Returns:
        tuple (TensorLike[float], TensorLike[float]):
            - the one-body coefficients
            - the predicted one-body PES using fitted coefficients
    """
    _import_sklearn()

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    if max_deg < min_deg:
        raise ValueError(
            f"Taylor expansion degree is {max_deg}<{min_deg}, please set max_deg greater than min_deg."
        )

    nmodes, quad_order = np.shape(onemode_op)
    grid, _ = np.polynomial.hermite.hermgauss(quad_order)
    coeffs = np.zeros((nmodes, max_deg - min_deg + 1))

    predicted_1D = np.zeros_like(onemode_op)

    for i1 in range(nmodes):
        poly1D = PolynomialFeatures(degree=(min_deg, max_deg), include_bias=False)
        poly1D_features = poly1D.fit_transform(grid.reshape(-1, 1))
        poly1D_reg_model = LinearRegression()
        poly1D_reg_model.fit(poly1D_features, onemode_op[i1, :])
        coeffs[i1, :] = poly1D_reg_model.coef_
        predicted_1D[i1, :] = poly1D_reg_model.predict(poly1D_features)

    return coeffs, predicted_1D


def _twobody_degs(max_deg, min_deg=3):
    """Finds the degree of fit for two-body coefficients.

    Args:
        max_deg (int): maximum degree of Taylor form polynomial
        min_deg (int): minimum degree of Taylor form polynomial

    Returns:
        list(tuple): A list of tuples `(q1deg, q2deg)` where the sum of the two values is
            guaranteed to be between the maximum total degree and minimum degree.
    """
    fit_degs = []
    for feat_deg in range(min_deg, max_deg + 1):
        max_deg = feat_deg - 1
        for deg_dist in range(1, max_deg + 1):
            q1deg = max_deg - deg_dist + 1
            q2deg = deg_dist
            fit_degs.append((q1deg, q2deg))

    return fit_degs


def _fit_twobody(twomode_op, max_deg, min_deg=3):
    r"""Fits the two-mode operator to get two-body coefficients.

    Args:
        twomode_op (TensorLike[float]): two-mode operator
        max_deg (int): maximum degree of Taylor form polynomial
        min_deg (int): minimum degree of Taylor form polynomial

    Returns:
        tuple (TensorLike[float], TensorLike[float]):
            - the two-body coefficients
            - the predicted two-body PES using fitted coefficients
    """
    _import_sklearn()
    from sklearn.linear_model import LinearRegression

    nmodes, _, quad_order, _ = np.shape(twomode_op)
    gauss_grid, _ = np.polynomial.hermite.hermgauss(quad_order)

    if max_deg < min_deg:
        raise ValueError(
            f"Taylor expansion degree is {max_deg}<{min_deg}, please set max_deg greater than min_deg."
        )

    fit_degs = _twobody_degs(max_deg, min_deg)
    num_coeffs = len(fit_degs)
    coeffs = np.zeros((nmodes, nmodes, num_coeffs))

    predicted_2D = np.zeros_like(twomode_op)

    grid_2D = np.array(np.meshgrid(gauss_grid, gauss_grid))
    q1, q2 = (grid.flatten() for grid in grid_2D)
    idx_2D = np.array(np.meshgrid(range(quad_order), range(quad_order)))
    idx1, idx2 = (idx.flatten() for idx in idx_2D)
    num_2D = len(q1)

    features = np.zeros((num_2D, num_coeffs))
    for deg_idx, (q1deg, q2deg) in enumerate(fit_degs):
        features[:, deg_idx] = (q1**q1deg) * (q2**q2deg)

    for i1 in range(nmodes):
        for i2 in range(i1):
            Y = twomode_op[i1, i2, idx1, idx2]
            poly2D_reg_model = LinearRegression()
            poly2D_reg_model.fit(features, Y)
            coeffs[i1, i2, :] = poly2D_reg_model.coef_
            predicted = poly2D_reg_model.predict(features)
            for idx in range(num_2D):
                predicted_2D[i1, i2, idx1[idx], idx2[idx]] = predicted[idx]

    return coeffs, predicted_2D


def _generate_bin_occupations(max_occ, nbins):
    """
    Generate all valid combinations of bin occupations for a given number of bins
    and a total maximum occupancy.

    Args:
        max_occ(int): the maximum total number of items to be distributed across bins
        nbins(int): the number of bins to distribute the items into

    Returns
        list(tuple): A list of tuples, where each tuple represents a valid combination of item counts for the bins.
    """
    combinations = list(itertools.product(range(max_occ + 1), repeat=nbins))

    valid_combinations = [combo for combo in combinations if sum(combo) == max_occ]

    return valid_combinations


def _threebody_degs(max_deg, min_deg=3):
    """Finds the degree of fit for three-body coefficients.

    Args:
        max_deg (int): maximum degree of Taylor form polynomial
        min_deg (int): minimum degree of Taylor form polynomial

    Returns:
        list(tuple): A list of tuples `(q1deg, q2deg, q3deg)` where the sum of the three values is
            guaranteed to be between the maximum total degree and minimum degree.
    """
    fit_degs = []
    for feat_deg in range(min_deg, max_deg + 1):
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


def _fit_threebody(threemode_op, max_deg, min_deg=3):
    r"""Fits the three-mode operator to get three-body coefficients.

    Args:
        threemode_op (TensorLike[float]): threemode operator
        max_deg (int): maximum degree of Taylor form polynomial
        min_deg (int): minimum degree of Taylor form polynomial

    Returns:
        tuple (TensorLike[float], TensorLike[float]):
            - the three-body coefficients
            - the predicted three-body PES using fitted coefficients
    """
    _import_sklearn()
    from sklearn.linear_model import LinearRegression

    nmodes, _, _, quad_order, _, _ = np.shape(threemode_op)
    gauss_grid, _ = np.polynomial.hermite.hermgauss(quad_order)

    if max_deg < min_deg:
        raise ValueError(
            f"Taylor expansion degree is {max_deg}<{min_deg}, please set max_deg greater than min_deg."
        )

    predicted_3D = np.zeros_like(threemode_op)
    fit_degs = _threebody_degs(max_deg)
    num_coeffs = len(fit_degs)
    coeffs = np.zeros((nmodes, nmodes, nmodes, num_coeffs))

    grid_3D = np.array(np.meshgrid(gauss_grid, gauss_grid, gauss_grid))
    q1, q2, q3 = (grid.flatten() for grid in grid_3D)
    idx_3D = np.array(np.meshgrid(range(quad_order), range(quad_order), range(quad_order)))
    idx1, idx2, idx3 = (idx.flatten() for idx in idx_3D)
    num_3D = len(q1)

    features = np.zeros((num_3D, num_coeffs))
    for deg_idx, Qs in enumerate(fit_degs):
        q1deg, q2deg, q3deg = Qs
        features[:, deg_idx] = q1 ** (q1deg) * q2 ** (q2deg) * q3 ** (q3deg)

    for i1 in range(nmodes):
        for i2 in range(i1):
            for i3 in range(i2):
                Y = threemode_op[i1, i2, i3, idx1, idx2, idx3]

                poly3D_reg_model = LinearRegression()
                poly3D_reg_model.fit(features, Y)
                coeffs[i1, i2, i3, :] = poly3D_reg_model.coef_
                predicted = poly3D_reg_model.predict(features)
                for idx in range(num_3D):
                    predicted_3D[i1, i2, i3, idx1[idx], idx2[idx], idx3[idx]] = predicted[idx]

    return coeffs, predicted_3D


def taylor_coeffs(pes, max_deg=4, min_deg=3):
    r"""Compute fitted coefficients for Taylor vibrational Hamiltonian.

    The coefficients are defined following Eq. 5 of `arXiv:1703.09313
    <https://arxiv.org/abs/1703.09313>`_ as:

    .. math::

        \Phi_{ijk} = \frac{k_{ijk}}{\sqrt{\omega_i \omega_j \omega_k}}
        \quad \text{and} \quad
        \Phi_{ijkl} = \frac{k_{ijkl}}{\sqrt{\omega_i \omega_j \omega_k \omega_l}},

    where :math:`\Phi_{ijk}` and :math:`\Phi_{ijkl}` are the third- and fourth-order reduced force
    constants, respectively, defined in terms of the third- and fourth-order partial derivatives
    of the potential energy surface data.

    Args:
        pes (VibrationalPES): object containing the vibrational potential energy surface data
        max_deg (int): maximum degree of Taylor form polynomial
        min_deg (int): minimum degree of Taylor form polynomial

    Returns:
        tuple(TensorLike[float]): the coefficients of the one-body, two-body and three-body terms

    **Example**

    >>> pes_onemode = np.array([[0.309, 0.115, 0.038, 0.008, 0.000, 0.006, 0.020, 0.041, 0.070]])
    >>> pes_twomode = np.zeros((1, 1, 9, 9))
    >>> dipole_onemode = np.zeros((1, 9, 3))
    >>> gauss_weights = np.array([3.96e-05, 4.94e-03, 8.85e-02,
    ...                           4.33e-01, 7.20e-01, 4.33e-01,
    ...                           8.85e-02, 4.94e-03, 3.96e-05])
    >>> grid = np.array([-3.19, -2.27, -1.47, -0.72,  0.0,  0.72,  1.47,  2.27,  3.19])
    >>> pes_object = qml.qchem.VibrationalPES(
    ...     freqs=np.array([0.025]),
    ...     grid=grid,
    ...     uloc=np.array([[1.0]]),
    ...     gauss_weights=gauss_weights,
    ...     pes_data=[pes_onemode, pes_twomode],
    ...     dipole_data=[dipole_onemode],
    ...     localized=True,
    ...     dipole_level=1,
    ... )
    >>> one, two = qml.qchem.taylor_coeffs(pes_object, 4, 2)
    >>> print(one)
    [[-0.00088528 -0.00361425  0.00068143]]
    """

    anh_pes, harmonic_pes = _remove_harmonic(pes.freqs, pes.pes_onemode)
    coeff_1D, predicted_1D = _fit_onebody(anh_pes, max_deg, min_deg=min_deg)
    predicted_1D += harmonic_pes
    coeff_arr = [coeff_1D]
    predicted_arr = [predicted_1D]

    if pes.pes_twomode is not None:
        coeff_2D, predicted_2D = _fit_twobody(pes.pes_twomode, max_deg, min_deg=min_deg)
        coeff_arr.append(coeff_2D)
        predicted_arr.append(predicted_2D)

    if pes.pes_threemode is not None:
        coeff_3D, predicted_3D = _fit_threebody(pes.pes_threemode, max_deg, min_deg=min_deg)
        coeff_arr.append(coeff_3D)
        predicted_arr.append(predicted_3D)

    return coeff_arr


def taylor_dipole_coeffs(pes, max_deg=4, min_deg=1):
    r"""Compute fitted coefficients for the Taylor dipole operator.

    Args:
        pes (VibrationalPES): object containing the vibrational potential energy surface data
        max_deg (int): maximum degree of Taylor form polynomial
        min_deg (int): minimum degree of Taylor form polynomial

    Returns:
        tuple: a tuple containing:
            - list(array(floats)): coefficients for x-displacements
            - list(array(floats)): coefficients for y-displacements
            - list(array(floats)): coefficients for z-displacements

    **Example**

    >>> freqs = np.array([0.01885397])
    >>> grid, weights = np.polynomial.hermite.hermgauss(9)
    >>> pes_onebody = np.array([[0.05235573, 0.03093067, 0.01501878, 0.00420778, 0.0,
    ...                          0.00584504, 0.02881817, 0.08483433, 0.22025702]])
    >>> pes_twobody = None
    >>> dipole_onebody = np.array([[[-1.92201700e-16,  1.45397041e-16, -1.40451549e-01],
    ...                             [-1.51005108e-16,  9.53185441e-17, -1.03377032e-01],
    ...                             [-1.22793018e-16,  7.22781963e-17, -6.92825934e-02],
    ...                             [-1.96537436e-16, -5.86686504e-19, -3.52245369e-02],
    ...                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    ...                             [ 5.24758835e-17, -1.40650833e-16,  3.69955543e-02],
    ...                             [-4.52407941e-17,  1.38406311e-16,  7.60888733e-02],
    ...                             [-4.63820104e-16,  5.42928787e-17,  1.17726042e-01],
    ...                             [ 1.19224372e-16,  9.12491386e-17,  1.64013197e-01]]])
    >>> vib_obj = qml.qchem.VibrationalPES(
    ...     freqs=freqs,
    ...     grid=grid,
    ...     gauss_weights=weights,
    ...     uloc=None,
    ...     pes_data=[pes_onebody, pes_twobody],
    ...     dipole_data=[dipole_onebody],
    ...     localized=False
    ... )
    >>> x, y, z = qml.qchem.taylor_dipole_coeffs(vib_obj, 4, 2)
    >>> print(z)
    [array([[ 1.64124324e-03,  5.39120159e-03, -4.80053702e-05]])]
    """
    coeffs_x_1D, predicted_x_1D = _fit_onebody(
        pes.dipole_onemode[:, :, 0], max_deg, min_deg=min_deg
    )
    coeffs_x_arr = [coeffs_x_1D]
    predicted_x_arr = [predicted_x_1D]

    coeffs_y_1D, predicted_y_1D = _fit_onebody(
        pes.dipole_onemode[:, :, 1], max_deg, min_deg=min_deg
    )
    coeffs_y_arr = [coeffs_y_1D]
    predicted_y_arr = [predicted_y_1D]

    coeffs_z_1D, predicted_z_1D = _fit_onebody(
        pes.dipole_onemode[:, :, 2], max_deg, min_deg=min_deg
    )
    coeffs_z_arr = [coeffs_z_1D]
    predicted_z_arr = [predicted_z_1D]

    if pes.dipole_twomode is not None:
        coeffs_x_2D, predicted_x_2D = _fit_twobody(
            pes.dipole_twomode[:, :, :, :, 0], max_deg, min_deg=min_deg
        )
        coeffs_x_arr.append(coeffs_x_2D)
        predicted_x_arr.append(predicted_x_2D)

        coeffs_y_2D, predicted_y_2D = _fit_twobody(
            pes.dipole_twomode[:, :, :, :, 1], max_deg, min_deg=min_deg
        )
        coeffs_y_arr.append(coeffs_y_2D)
        predicted_y_arr.append(predicted_y_2D)

        coeffs_z_2D, predicted_z_2D = _fit_twobody(
            pes.dipole_twomode[:, :, :, :, 2], max_deg, min_deg=min_deg
        )
        coeffs_z_arr.append(coeffs_z_2D)
        predicted_z_arr.append(predicted_z_2D)

    if pes.dipole_threemode is not None:
        coeffs_x_3D, predicted_x_3D = _fit_threebody(
            pes.dipole_threemode[:, :, :, :, :, :, 0], max_deg, min_deg=min_deg
        )
        coeffs_x_arr.append(coeffs_x_3D)
        predicted_x_arr.append(predicted_x_3D)

        coeffs_y_3D, predicted_y_3D = _fit_threebody(
            pes.dipole_threemode[:, :, :, :, :, :, 1], max_deg, min_deg=min_deg
        )
        coeffs_y_arr.append(coeffs_y_3D)
        predicted_y_arr.append(predicted_y_3D)

        coeffs_z_3D, predicted_z_3D = _fit_threebody(
            pes.dipole_threemode[:, :, :, :, :, :, 2], max_deg, min_deg=min_deg
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
        BoseSentence: bosonic form of the given position operator
    """
    factor = 1j / np.sqrt(2) if op == "p" else 1 / np.sqrt(2)
    bop = factor * BoseWord({(0, index): "-"})
    bdag = factor * BoseWord({(0, index): "+"})
    return bdag - bop if op == "p" else bdag + bop


def _taylor_anharmonic(taylor_coeffs_array, start_deg=2):
    """Build anharmonic term of Taylor form bosonic observable from provided coefficients described
    in `Eq. 10 <https://arxiv.org/pdf/1703.09313>`_.

    Args:
        taylor_coeffs_array (list(float)): the coeffs of the Taylor expansion
        start_deg (int): the starting degree

    Returns:
        pennylane.bose.BoseSentence: anharmonic part of the Taylor hamiltonian for given coeffs
    """
    num_coups = len(taylor_coeffs_array)

    taylor_1D = taylor_coeffs_array[0]
    num_modes, num_1D_coeffs = np.shape(taylor_1D)

    taylor_deg = num_1D_coeffs + start_deg - 1

    ordered_dict = BoseSentence({})

    # One-mode expansion
    for mode in range(num_modes):
        bosonized_qm = _position_to_boson(mode, "q")
        for deg_i in range(start_deg, taylor_deg + 1):
            coeff = taylor_1D[mode, deg_i - start_deg]
            qpow = bosonized_qm**deg_i
            ordered_dict += (coeff * qpow).normal_order()
    # Two-mode expansion
    if num_coups > 1:
        taylor_2D = taylor_coeffs_array[1]
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
        taylor_3D = taylor_coeffs_array[2]
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


def _taylor_kinetic(taylor_coeffs_array, freqs, is_local=True, uloc=None):
    """Build kinetic term of Taylor form bosonic observable from provided coefficients

    Args:
        taylor_coeffs_array (list(float)): the coeffs of the Taylor expansion
        freqs (list(float)): the harmonic frequencies in atomic units
        is_local (bool): Flag whether the vibrational modes are localized. Default is ``True``.
        uloc (list(list(float))): localization matrix indicating the relationship between original
            and localized modes

    Returns:
        pennylane.bose.BoseSentence: kinetic term of the Taylor hamiltonian for given coeffs
    """
    taylor_1D = taylor_coeffs_array[0]
    num_modes, _ = np.shape(taylor_1D)

    if is_local:
        alphas_arr = np.einsum("ij,ik,j,k->jk", uloc, uloc, np.sqrt(freqs), np.sqrt(freqs))
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


def _taylor_harmonic(taylor_coeffs_array, freqs):
    """Build harmonic term of Taylor form bosonic observable from provided coefficients, see first
    term of `Eq. 4 and Eq. 7 <https://arxiv.org/pdf/1703.09313>`_.

    Args:
        taylor_coeffs_array (list(float)): the coeffs of the Taylor expansion
        freqs (list(float)): the harmonic frequencies in atomic units

    Returns:
        pennylane.bose.BoseSentence: harmonic term of the Taylor hamiltonian for given coeffs
    """
    taylor_1D = taylor_coeffs_array[0]
    num_modes, _ = np.shape(taylor_1D)
    harm_pot = BoseSentence({})
    for mode in range(num_modes):
        bosonized_qm2 = (
            _position_to_boson(mode, "q") * _position_to_boson(mode, "q")
        ).normal_order()
        harm_pot += bosonized_qm2 * freqs[mode] * 0.5

    return harm_pot.normal_order()


def taylor_bosonic(coeffs, freqs, is_local=True, uloc=None):
    """Return Taylor bosonic vibrational Hamiltonian.

    The construction of the Hamiltonian is based on Eqs. 4-7 of `arXiv:1703.09313 <https://arxiv.org/abs/1703.09313>`_.

    Args:
        coeffs (list(float)): the coefficients of the Hamiltonian
        freqs (list(float)): the harmonic frequencies in atomic units
        is_local (bool): Flag whether the vibrational modes are localized. Default is ``True``.
        uloc (list(list(float))): localization matrix indicating the relationship between original
            and localized modes

    Returns:
        pennylane.bose.BoseSentence: Taylor bosonic hamiltonian

    **Example**

    >>> one_mode = np.array([[-0.00088528, -0.00361425,  0.00068143]])
    >>> two_mode = np.array([[[0., 0., 0., 0., 0., 0.]]])
    >>> freqs = np.array([0.025])
    >>> uloc = np.array([[1.0]])
    >>> ham = qml.qchem.taylor_bosonic(coeffs=[one_mode, two_mode], freqs=freqs, uloc=uloc)
    >>> print(ham)
    -0.0012778303419517393 * b⁺(0) b⁺(0) b⁺(0)
    + -0.0038334910258552178 * b⁺(0) b⁺(0) b(0)
    + -0.0038334910258552178 * b⁺(0)
    + -0.0038334910258552178 * b⁺(0) b(0) b(0)
    + -0.0038334910258552178 * b(0)
    + -0.0012778303419517393 * b(0) b(0) b(0)
    + (0.0005795050000000001+0j) * b⁺(0) b⁺(0)
    + (0.026159009999999996+0j) * b⁺(0) b(0)
    + (0.012568432499999997+0j) * I
    + (0.0005795050000000001+0j) * b(0) b(0)
    + 0.00017035749999999995 * b⁺(0) b⁺(0) b⁺(0) b⁺(0)
    + 0.0006814299999999998 * b⁺(0) b⁺(0) b⁺(0) b(0)
    + 0.0010221449999999997 * b⁺(0) b⁺(0) b(0) b(0)
    + 0.0006814299999999998 * b⁺(0) b(0) b(0) b(0)
    + 0.00017035749999999995 * b(0) b(0) b(0) b(0)
    """
    if is_local:
        start_deg = 2
    else:
        start_deg = 3

    harm_pot = _taylor_harmonic(coeffs, freqs)
    ham = _taylor_anharmonic(coeffs, start_deg) + harm_pot
    kin_ham = _taylor_kinetic(coeffs, freqs, is_local, uloc)
    ham += kin_ham
    return ham.normal_order()


# pylint: disable=too-many-positional-arguments, too-many-arguments
def taylor_hamiltonian(
    pes, max_deg=4, min_deg=3, mapping="binary", n_states=2, wire_map=None, tol=1e-12
):
    """Return Taylor vibrational Hamiltonian.

    The construction of the Hamiltonian is based on Eqs. 4-7 of `arXiv:1703.09313 <https://arxiv.org/abs/1703.09313>`_.
    The Hamiltonian is then converted to a qubit operator with a selected ``mapping`` method.

    Args:
        pes (VibrationalPES): object containing the vibrational potential energy surface data
        max_deg (int): maximum degree of Taylor form polynomial
        min_deg (int): minimum degree of Taylor form polynomial
        mapping (str): Method used to map to qubit basis. Input values can be ``"binary"``
            or ``"unary"``. Default is ``"binary"``.
        n_states(int): maximum number of allowed bosonic states
        wire_map (dict): A dictionary defining how to map the states of the Bose operator to qubit
            wires. If ``None``, integers used to label the bosonic states will be used as wire labels.
            Defaults to ``None``.
        tol (float): tolerance for discarding the imaginary part of the coefficients

    Returns:
        Operator: the Taylor Hamiltonian

    **Example**

    >>> pes_onemode = np.array([[0.309, 0.115, 0.038, 0.008, 0.000, 0.006, 0.020, 0.041, 0.070]])
    >>> pes_twomode = np.zeros((1, 1, 9, 9))
    >>> dipole_onemode = np.zeros((1, 9, 3))
    >>> gauss_weights = np.array([3.96e-05, 4.94e-03, 8.85e-02,
    ...                           4.33e-01, 7.20e-01, 4.33e-01,
    ...                           8.85e-02, 4.94e-03, 3.96e-05])
    >>> grid = np.array([-3.19, -2.27, -1.47, -0.72,  0.0,  0.72,  1.47,  2.27,  3.19])
    >>> pes_object = qml.qchem.VibrationalPES(
    ...     freqs=np.array([0.025]),
    ...     grid=grid,
    ...     uloc=np.array([[1.0]]),
    ...     gauss_weights=gauss_weights,
    ...     pes_data=[pes_onemode, pes_twomode],
    ...     dipole_data=[dipole_onemode],
    ...     localized=True,
    ...     dipole_level=1,
    ... )
    >>> qml.qchem.taylor_hamiltonian(pes_object, 4, 2)
    (
        -0.003833496032473659 * X(0)
        + (0.0256479442871582+0j) * I(0)
        + (-0.013079509779221888+0j) * Z(0)
    )
    """
    coeffs_arr = taylor_coeffs(pes, max_deg, min_deg)
    bose_op = taylor_bosonic(coeffs_arr, pes.freqs, is_local=pes.localized, uloc=pes.uloc)
    mapping = mapping.lower().strip()
    if mapping == "binary":
        ham = binary_mapping(bose_operator=bose_op, n_states=n_states, wire_map=wire_map, tol=tol)
    elif mapping == "unary":
        ham = unary_mapping(bose_operator=bose_op, n_states=n_states, wire_map=wire_map, tol=tol)
    else:
        raise ValueError(
            f"Specified mapping {mapping}, is not found. Please use either 'binary' or 'unary' mapping."
        )

    return ham
