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
    r"""Computes the coefficients of a Taylor vibrational Hamiltonian.

    The coefficients are computed from a multi-dimensional polynomial fit over potential energy data
    computed along normal coordinates, with a polynomial specified by ``min_deg`` and ``max_deg``.

    Args:
        pes (VibrationalPES): the vibrational potential energy surface object
        max_deg (int): maximum degree of the polynomial used to compute the coefficients
        min_deg (int): minimum degree of the polynomial used to compute the coefficients

    Returns:
        List(TensorLike[float]): the coefficients of the Taylor vibrational Hamiltonian

    **Example**

    >>> freqs = np.array([0.0249722])
    >>> pes_onemode = np.array([[0.08477, 0.01437, 0.00000, 0.00937, 0.03414]])
    >>> pes_object = qml.qchem.VibrationalPES(freqs=freqs, pes_data=[pes_onemode])
    >>> coeffs = qml.qchem.taylor_coeffs(pes_object, 4, 2)
    >>> print(coeffs)
    [array([[-4.73959071e-05, -3.06785775e-03,  5.21798831e-04]])]

    .. details::
        :title: Theory

        A molecular potential energy surface can be defined as [Eq. 7 of
        `J. Chem. Phys. 135, 134108 (2011) <https://pubs.aip.org/aip/jcp/article-abstract/135/13/134108/191108/Size-extensive-vibrational-self-consistent-field?redirectedFrom=PDF>`_]:

        .. math::

            V = V_0 + \sum_{i} F_i q_i + \sum_{i,j} F_{ij} q_i q_j +
                       \sum_{i,j,k} F_{ijk} q_i q_j q_k + \cdots,

        where :math:`q` is a normal coordinate and :math:`F` represents the derivatives of the
        potential energy surface.

        This function computes these derivatives via Taylor expansion of the potential energy data
        by performing a multi-dimensional polynomial fit.

        The potential energy surface along the normal coordinate can be defined as

        .. math::

            V(q_1,\cdots,q_M) = V_0 + \sum_{i=1}^M V_1^{(i)}(q_i) + \sum_{i>j}
            V_2^{(i,j)}(q_i,q_j) + \sum_{i<j<k} V_3^{(i,j,k)}(q_i,q_j,q_k) + \cdots,

        where :math:`V_n` represents the :math:`n`-mode component of the potential energy surface
        computed along the normal coordinate. The :math:`V_n` terms are defined as:

        .. math::

            V_0 &\equiv  V(q_1=0,\cdots,q_M=0) \\
            V_1^{(i)}(q_i) &\equiv  V(0,\cdots,0,q_i,0,\cdots,0) -  V_0 \\
            V_2^{(i,j)}(q_i,q_j) &\equiv  V(0,\cdots,q_i,\cdots,q_j,\cdots,0) -
            V_1^{(i)}(q_i) -  V_1^{(j)}(q_j) -  V_0  \\
            \nonumber \vdots

        Note that the terms :math:`V_n` are represented here by an array of energy points computed
        along the normal coordinates. These energy data are then used in a multi-dimensional
        polynomial fit where each term :math:`V_n` is expanded in terms of products of :math:`q`
        with exponents specified by ``min_deg`` and ``max_deg``.

        The one-mode Taylor coefficients, :math:`\Phi`, computed here are related to the potential
        energy surface as:

        .. math::

            V_1^{(j)}(q_j) \approx \Phi^{(2)}_j q_j^2 + \Phi^{(3)}_j q_j^3 + ... + \Phi^{(n)}_j q_j^n,

        where the largest power :math:`n` is determined by ``max_deg``. Similarly, the two-mode and
        three-mode Taylor coefficients are computed if the two-mode and three-mode potential energy
        surface data, :math:`V_2^{(j, k)}(q_j, q_k)` and :math:`V_3^{(j, k, l)}(q_j, q_k, q_l)`, are
        provided.
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
    r"""Computes the coefficients of a Taylor dipole operator.

    The coefficients are computed from a multi-dimensional polynomial fit over dipole moment data
    computed along normal coordinates, with a polynomial specified by ``min_deg`` and ``max_deg``.

    Args:
        pes (VibrationalPES): the vibrational potential energy surface object
        max_deg (int): maximum degree of the polynomial used to compute the coefficients
        min_deg (int): minimum degree of the polynomial used to compute the coefficients

    Returns:
        tuple: a tuple containing:
            - List(TensorLike[float]): coefficients for x-displacements
            - List(TensorLike[float]): coefficients for y-displacements
            - List(TensorLike[float]): coefficients for z-displacements

    **Example**

    >>> freqs = np.array([0.0249722])
    >>> dipole_onemode = np.array([[[-1.24222060e-16, -6.29170686e-17, -7.04678188e-02],
    ...                             [ 3.83941489e-16, -2.31579327e-18, -3.24444991e-02],
    ...                             [ 1.67813138e-17, -5.63904474e-17, -5.60662627e-15],
    ...                             [-7.37584781e-17, -5.51948189e-17,  2.96786374e-02],
    ...                             [ 1.40526000e-16, -3.67126324e-17,  5.92006212e-02]]])
    >>> pes_object = qml.qchem.VibrationalPES(freqs=freqs, dipole_data=[dipole_onemode])
    >>> coeffs_x, coeffs_y, coeffs_z = qml.qchem.taylor_dipole_coeffs(pes_object, 4, 2)
    >>> print(coeffs_z)
    [array([[-1.54126823e-03,  8.17300533e-03,  3.94178001e-05]])]

    .. details::
        :title: Theory

        The dipole :math:`D` along each of the :math:`x, y,` and :math:`z` directions is defined as:

        .. math::

            D(q_1,\cdots,q_M) = D_0 + \sum_{i=1}^M D_1^{(i)}(q_i) + \sum_{i>j}
            D_2^{(i,j)}(q_i,q_j) + \sum_{i<j<k} D_3^{(i,j,k)}(q_i,q_j,q_k) + \cdots,

        where :math:`q` is a normal coordinate and :math:`D_n` represents the :math:`n`-mode
        component of the dipole computed along the normal coordinate. The :math:`D_n` terms are
        defined as:

        .. math::

            D_0 &\equiv D(q_1=0,\cdots,q_M=0) \\
            D_1^{(i)}(q_i) &\equiv D(0,\cdots,0,q_i,0,\cdots,0) - D_0 \\
            D_2^{(i,j)}(q_i,q_j) &\equiv D(0,\cdots,q_i,\cdots,q_j,\cdots,0) -
            D_1^{(i)}(q_i) - D_1^{(j)}(q_j) - D_0  \\
            \nonumber \vdots

        The one-mode Taylor dipole coefficients, :math:`\Phi`, computed here are related to the
        dipole data as:

        .. math::

            D_1^{(j)}(q_j) \approx \Phi^{(2)}_j q_j^2 + \Phi^{(3)}_j q_j^3 + ....

        Similarly, the two-mode and three-mode Taylor dipole coefficients are computed if the
        two-mode and three-mode dipole data, :math:`D_2^{(j, k)}(q_j, q_k)` and
        :math:`D_3^{(j, k, l)}(q_j, q_k, q_l)`, are provided.
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
    r"""Returns a Taylor bosonic vibrational Hamiltonian.

    The Taylor vibrational Hamiltonian is defined in terms of kinetic :math:`T` and potential
    :math:`V` components  as:

    .. math::

        H = T + V.

    The kinetic term is defined in terms of momentum :math:`p` operators as

    .. math::

        T = \sum_{i\geq j} K_{ij} p_i  p_j,

    where the :math:`K` matrix is defined in terms of vibrational frequencies, :math:`\omega`, and
    mode localization unitary matrix, :math:`U`, as:

    .. math::

        K_{ij} = \sum_{k=1}^M \frac{\omega_k}{2} U_{ki} U_{kj}.

    The potential term is defined in terms of normal coordinate operator :math:`q` as:

    .. math::

        V(q_1,\cdots,q_M) = V_0 + \sum_{i=1}^M V_1^{(i)}(q_i) + \sum_{i>j}
        V_2^{(i,j)}(q_i,q_j) + \sum_{i<j<k} V_3^{(i,j,k)}(q_i,q_j,q_k) + \cdots,

    where :math:`V_n` represents the :math:`n`-mode component of the potential energy surface
    computed along the normal coordinate. The :math:`V_n` terms are defined as:

    .. math::

		V_0 &\equiv  V(q_1=0,\cdots,q_M=0) \\
		V_1^{(i)}(q_i) &\equiv  V(0,\cdots,0,q_i,0,\cdots,0) -  V_0 \\
		V_2^{(i,j)}(q_i,q_j) &\equiv  V(0,\cdots,q_i,\cdots,q_j,\cdots,0) -
		V_1^{(i)}(q_i) -  V_1^{(j)}(q_j) -  V_0  \\
		\nonumber \vdots

    These terms are then used in a multi-dimensional polynomial fit to get :math:`n`-mode Taylor
    coefficients. For instance, the one-mode Taylor coefficient :math:`\Phi` is related to the
    one-mode potential energy surface data as:

    .. math::

        V_1^{(j)}(q_j) \approx \Phi^{(2)}_j q_j^2 + \Phi^{(3)}_j q_j^3 + ...

    Similarly, the two-mode and three-mode Taylor coefficients are computed if the two-mode and
    three-mode potential energy surface data, :math:`V_2^{(j, k)}(q_j, q_k)` and
    :math:`V_3^{(j, k, l)}(q_j, q_k, q_l)`, are provided.

    This real-space form of the vibrational Hamiltonian can be represented in the bosonic basis by
    using equations defined in Eqs. 6, 7 of `arXiv:1703.09313 <https://arxiv.org/abs/1703.09313>`_:

    .. math::

        \hat q_i = \frac{1}{\sqrt{2}}(b_i^\dagger + b_i), \quad
        \hat p_i = \frac{1}{\sqrt{2}}(b_i^\dagger - b_i),

    where :math:`b^\dagger` and :math:`b` are bosonic creation and annihilation operators,
    respectively.

    Args:
        coeffs (list(tensorlike(float))): the coefficients of a Taylor vibrational Hamiltonian
        freqs (array(float)): the harmonic vibrational frequencies in atomic units
        is_local (bool): Whether the vibrational modes are localized. Default is ``True``.
        uloc (tensorlike(float)): normal mode localization matrix with shape ``(m, m)`` where
            ``m = len(freqs)``

    Returns:
        pennylane.bose.BoseSentence: Taylor bosonic Hamiltonian

    **Example**

    >>> freqs = np.array([0.025])
    >>> one_mode = np.array([[-0.00088528, -0.00361425,  0.00068143]])
    >>> uloc = np.array([[1.0]])
    >>> ham = qml.qchem.taylor_bosonic(coeffs=[one_mode], freqs=freqs, uloc=uloc)
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
    r"""Returns Taylor vibrational Hamiltonian.

    The Taylor vibrational Hamiltonian is defined in terms of kinetic :math:`T` and potential
    :math:`V` components  as:

    .. math::

        H = T + V.

    The kinetic term is defined in terms of momentum :math:`p` operator as

    .. math::

        T = \sum_{i\geq j} K_{ij} p_i  p_j,

    where the :math:`K` matrix is defined in terms of vibrational frequencies, :math:`\omega`, and
    mode localization unitary matrix, :math:`U`, as:

    .. math::

        K_{ij} = \sum_{k=1}^M \frac{\omega_k}{2} U_{ki} U_{kj}.

    The potential term is defined in terms of the normal coordinate operator :math:`q` as:

    .. math::

        V(q_1,\cdots,q_M) = V_0 + \sum_{i=1}^M V_1^{(i)}(q_i) + \sum_{i>j}
        V_2^{(i,j)}(q_i,q_j) + \sum_{i<j<k} V_3^{(i,j,k)}(q_i,q_j,q_k) + \cdots,

    where :math:`V_n` represents the :math:`n`-mode component of the potential energy surface
    computed along the normal coordinate. The :math:`V_n` terms are defined as:

    .. math::

		V_0 &\equiv  V(q_1=0,\cdots,q_M=0) \\
		V_1^{(i)}(q_i) &\equiv  V(0,\cdots,0,q_i,0,\cdots,0) -  V_0 \\
		V_2^{(i,j)}(q_i,q_j) &\equiv  V(0,\cdots,q_i,\cdots,q_j,\cdots,0) -
		V_1^{(i)}(q_i) -  V_1^{(j)}(q_j) -  V_0  \\
		\nonumber \vdots

    These terms are then used in a multi-dimensional polynomial fit with a polynomial specified by
    ``min_deg`` and ``max_deg`` to get :math:`n`-mode Taylor coefficients. For instance, the
    one-mode Taylor coefficient :math:`\Phi` is related to the one-mode potential energy surface
    data as:

    .. math::

        V_1^{(j)}(q_j) \approx \Phi^{(2)}_j q_j^2 + \Phi^{(3)}_j q_j^3 + ...

    Similarly, the two-mode and three-mode Taylor coefficients are computed if the two-mode and
    three-mode potential energy surface data, :math:`V_2^{(j, k)}(q_j, q_k)` and
    :math:`V_3^{(j, k, l)}(q_j, q_k, q_l)`, are provided.

    This real space form of the vibrational Hamiltonian can be represented in the bosonic basis by
    using equations defined in Eqs. 6, 7 of `arXiv:1703.09313 <https://arxiv.org/abs/1703.09313>`_:

    .. math::

        \hat q_i = \frac{1}{\sqrt{2}}(b_i^\dagger + b_i), \quad
        \hat p_i = \frac{1}{\sqrt{2}}(b_i^\dagger - b_i),

    where :math:`b^\dagger` and :math:`b` are bosonic creation and annihilation operators,
    respectively.

    The bosonic Hamiltonian is then converted to a qubit operator with a selected ``mapping``
    method to obtain a linear combination as:

    .. math::

        H = \sum_{i} c_i P_i,

    where :math:`P` is a tensor product of Pauli operators and :math:`c` is a constant.

    Args:
        pes (VibrationalPES): object containing the vibrational potential energy surface data
        max_deg (int): maximum degree of the polynomial used to compute the coefficients
        min_deg (int): minimum degree of the polynomial used to compute the coefficients
        mapping (str): Method used to map to qubit basis. Input values can be ``"binary"``
            or ``"unary"``. Default is ``"binary"``.
        n_states(int): maximum number of allowed bosonic states
        wire_map (dict): A dictionary defining how to map the states of the Bose operator to qubit
            wires. If ``None``, integers used to label the bosonic states will be used as wire labels.
            Defaults to ``None``.
        tol (float): tolerance for discarding the imaginary part of the coefficients during mapping

    Returns:
        Operator: the Taylor Hamiltonian

    **Example**

    >>> freqs = np.array([0.0249722])
    >>> pes_onemode = np.array([[0.08477, 0.01437, 0.00000, 0.00937, 0.03414]])
    >>> pes_object = qml.qchem.VibrationalPES(freqs=freqs, pes_data=[pes_onemode], localized=False)
    >>> ham = qml.qchem.taylor_hamiltonian(pes_object)
    >>> print(ham)
    0.026123120450329353 * I(0) + -0.01325338030021957 * Z(0) + -0.0032539545260859464 * X(0)
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
