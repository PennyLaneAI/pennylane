# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains methods for computing Fourier coefficients and frequency spectra of quantum functions ."""
from itertools import product
from .custom_decompositions import *

custom_decomps_required = {"CRot": custom_CRot_decomposition}


def fourier_coefficients(f, n_inputs, degree, lowpass_filter=True, filter_threshold=None):
    r"""Computes the first :math:`2d+1` Fourier coefficients of a :math:`2\pi`
    periodic function, where :math:`d` is the highest desired frequency in the
    Fourier spectrum.

    By default, a low-pass filter is applied prior to computing the coefficients
    in order to mitigate the effects of aliasing. Coefficients up to a threshold
    value are computed, and then frequencies higher than the degree are simply removed. This
    ensures that the coefficients returned will have the correct values, though they
    may not be the full set of coefficients. If no threshold value is provided, the
    threshold will be set to ``2 * degree``.

    Args:
        f (callable): function that takes an array of :math:`N` scalar inputs
        n_inputs (int): number of function inputs
        degree (int): max frequency of Fourier coeffs to be computed. For degree :math:`d`,
            the coefficients from frequencies :math:`-d, -d+1,...0,..., d-1, d ` will be computed.
        lowpass_filter (bool): If True (default), a simple low-pass filter is applied prior to
            computing the set of coefficients in order to filter out frequencies above the
            given degree.
        filter_threshold (None or int): The integer frequency at which to filter. If no value is
            specified, ``2 * degree`` is used.

    Returns:
        array[complex]: The Fourier coefficients of the function f up to the specified degree.

    **Example**

    .. code-block:: python

        import pennylane as qml
        from pennylane import numpy as anp

        # Expected Fourier series over 2 parameters with frequencies 0 and 1
        num_inputs = 2
        degree = 1

        weights = anp.array([0.5, 0.2], requires_grad=True, is_input=False)

        dev = qml.device('default.qubit', wires=['a'])

        @qml.qnode(dev)
        def circuit(weights, inpt):
            qml.RX(inpt[0], wires='a')
            qml.Rot(0.1, 0.2, 0.3, wires='a')

            qml.RY(inpt[1], wires='a')
            qml.Rot(-4.1, 3.2, 1.3, wires='a')

            return qml.expval(qml.PauliZ(wires='a'))

        # Coefficients of the "inpt" variable will be computed
        coeffs = fourier_coefficients(partial(circuit, weights), num_inputs, degree)

    """
    if not lowpass_filter:
        return _fourier_coefficients_no_filter(f, n_inputs, degree)

    if filter_threshold is None:
        filter_threshold = 2 * degree

    # Compute the fft of the function at 2x the specified degree
    unfiltered_coeffs = _fourier_coefficients_no_filter(f, n_inputs, filter_threshold)

    # Shift the frequencies so that the 0s are at the centre
    shifted_unfiltered_coeffs = np.fft.fftshift(unfiltered_coeffs)

    # Next, slice up the array so that we get only the coefficients we care about,
    # those between -degree and degree
    range_slices = list(
        range(
            filter_threshold - degree,
            shifted_unfiltered_coeffs.shape[0] - (filter_threshold - degree),
        )
    )

    shifted_filtered_coeffs = shifted_unfiltered_coeffs.copy()

    # Go axis by axis and take only the central components
    for axis in range(n_inputs - 1, -1, -1):
        shifted_filtered_coeffs = np.take(shifted_filtered_coeffs, range_slices, axis=axis)

    # Shift everything back into "normal" fft ordering
    filtered_coeffs = np.fft.ifftshift(shifted_filtered_coeffs)

    # Compute the inverse FFT
    f_discrete_filtered = np.fft.ifftn(filtered_coeffs)

    # Now compute the FFT again on the filtered data
    coeffs = np.fft.fftn(f_discrete_filtered)

    return coeffs


def _fourier_coefficients_no_filter(f, n_inputs, degree):
    r"""Computes the first :math:`2d+1` Fourier coefficients of a :math:`2\pi` periodic
    function, where :math:`d` is the highest desired frequency in the Fourier spectrum.

    This function computes the coefficients blindly without any filtering applied, and
    is thus used as a helper function for the true ``fourier_coefficients`` function.

    Args:
        f (callable): function that takes an array of :math:`N` scalar inputs
        n_inputs (int): number of function inputs
        degree (int): max frequency of Fourier coeffs to be computed. For degree :math:`d`,
            the coefficients from frequencies :math:`-d, -d+1,...0,..., d-1, d ` will be computed.

    Returns:
        array[complex]: The Fourier coefficients of the function f up to the specified degree.
    """

    # number of integer values for the indices n_i = -degree,...,0,...,degree
    k = 2 * degree + 1

    # create generator for indices nvec = (n1, ..., nN), ranging from (-d,...,-d) to (d,...,d).
    n_range = np.array(range(-degree, degree + 1))
    n_ranges = [n_range] * n_inputs
    nvecs = product(*n_ranges)

    # here we will collect the discretized values of function f
    shp = tuple([k] * n_inputs)
    f_discrete = np.zeros(shape=shp)

    for nvec in nvecs:
        nvec = np.array(nvec)

        # compute the evaluation points for frequencies nvec
        sample_points = 2 * np.pi / k * nvec

        # fill discretized function array with value of f at inpts
        f_discrete[tuple(nvec)] = f(sample_points)

    coeffs = np.fft.fftn(f_discrete) / f_discrete.size

    return coeffs
