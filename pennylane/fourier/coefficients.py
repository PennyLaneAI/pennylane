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
"""Contains methods for computing Fourier coefficients and frequency spectra of quantum functions."""
from itertools import product
import numpy as np


def coefficients(f, n_inputs, degree, lowpass_filter=False, filter_threshold=None):
    r"""Computes the first :math:`2d+1` Fourier coefficients of a :math:`2\pi`
    periodic function, where :math:`d` is the highest desired frequency (the
    degree) of the Fourier spectrum.

    While this function can be used to compute Fourier coefficients in general,
    the specific use case in PennyLane is to compute coefficients of the
    functions that result from measuring expectation values of parametrized
    quantum circuits, as described in `Schuld, Sweke and Meyer (2020)
    <https://arxiv.org/abs/2008.08605>`__ and `Vidal and Theis, 2019
    <https://arxiv.org/abs/1901.11434>`__.

    **Details**

    Consider a quantum circuit that depends on a
    parameter vector :math:`x` with
    length :math:`N`. The circuit involves application of some unitary
    operations :math:`U(x)`, and then measurement of an observable
    :math:`\langle \hat{O} \rangle`. Analytically, the expectation value is

    .. math::

       \langle \hat{O} \rangle = \langle 0 \vert U^\dagger (x) \hat{O} U(x) \vert 0\rangle = \langle
       \psi(x) \vert \hat{O} \vert \psi (x)\rangle.

    This output is simply a function :math:`f(x) = \langle \psi(x) \vert \hat{O} \vert \psi
    (x)\rangle`. Notably, it is a periodic function of the parameters, and
    it can thus be expressed as a multidimensional Fourier series:

    .. math::

        f(x) = \sum \limits_{n_1\in \Omega_1} \dots \sum \limits_{n_N \in \Omega_N}
        c_{n_1,\dots, n_N} e^{-i x_1 n_1} \dots e^{-i x_N n_N},

    where :math:`n_i` are integer-valued frequencies, :math:`\Omega_i` are the set
    of available values for the integer frequencies, and the
    :math:`c_{n_1,\ldots,n_N}` are Fourier coefficients.

    Args:
        f (callable): Function that takes a 1D tensor of ``n_inputs`` scalar inputs. The function can be a QNode, but
            has to return a real scalar value (such as an expectation).
        n_inputs (int): number of function inputs
        degree (int): max frequency of Fourier coeffs to be computed. For degree :math:`d`,
            the coefficients from frequencies :math:`-d, -d+1,...0,..., d-1, d` will be computed.
        lowpass_filter (bool): If ``True``, a simple low-pass filter is applied prior to
            computing the set of coefficients in order to filter out frequencies above the
            given degree. See examples below.
        filter_threshold (None or int): The integer frequency at which to filter. If
            ``lowpass_filter`` is set to ``True,`` but no value is specified, ``2 * degree`` is used.

    Returns:
        array[complex]: The Fourier coefficients of the function ``f`` up to the specified degree.

    **Example**

    Suppose we have the following quantum function and wish to compute its Fourier
    coefficients with respect to the variable ``inpt``, which is an array with 2 values:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=['a'])

        @qml.qnode(dev)
        def circuit(weights, inpt):
            qml.RX(inpt[0], wires='a')
            qml.Rot(*weights[0], wires='a')

            qml.RY(inpt[1], wires='a')
            qml.Rot(*weights[1], wires='a')

            return qml.expval(qml.PauliZ(wires='a'))

    .. note::

        The QNode has to return a scalar value (such as a single expectation).

    Unless otherwise specified, the coefficients will be computed for all input
    values. To compute coefficients with respect to only a subset of the input
    values, it is necessary to use a wrapper function (e.g.,
    ``functools.partial``). We do this below, while fixing a value for
    ``weights``:

    >>> from functools import partial
    >>> weights = np.array([[0.1, 0.2, 0.3], [-4.1, 3.2, 1.3]])
    >>> partial_circuit = partial(circuit, weights)

    Now we must specify the number of inputs, and the maximum desired
    degree. Based on the underlying theory, we expect the degree to be 1
    (frequencies -1, 0, and 1).

    >>> num_inputs = 2
    >>> degree = 1

    Then we can obtain the coefficients:

    >>> coeffs = coefficients(partial_circuit, num_inputs, degree)
    >>> print(coeffs)
    [[ 0.    +0.j     -0.    +0.j     -0.    +0.j    ]
    [-0.0014-0.022j  -0.3431-0.0408j -0.1493+0.0374j]
    [-0.0014+0.022j  -0.1493-0.0374j -0.3431+0.0408j]]

    If the specified degree is lower than the highest frequency of the function,
    aliasing may occur, and the resultant coefficients will be incorrect as they
    will include components of the series expansion from higher frequencies. In
    order to mitigate aliasing, setting ``lowpass_filter=True`` will apply a
    simple low-pass filter prior to computing the coefficients. Coefficients up
    to a specified value are computed, and then frequencies higher than the
    degree are simply removed. This ensures that the coefficients returned will
    have the correct values, though they may not be the full set of
    coefficients. If no threshold value is provided, the threshold will be set
    to ``2 * degree``.

    Consider the circuit below:

    .. code-block:: python

        @qml.qnode(dev)
        def circuit(inpt):
            qml.RX(inpt[0], wires=0)
            qml.RY(inpt[0], wires=1)
            qml.CNOT(wires=[1, 0])
            return qml.expval(qml.PauliZ(0))

    One can work out by hand that the Fourier coefficients are :math:`c_0 = 0.5, c_1 = c_{-1} = 0,`
    and :math:`c_2 = c_{-2} = 0.25`. Suppose we would like only to obtain the coefficients
    :math:`c_0` and :math:`c_1, c_{-1}`. If we simply ask for the coefficients of degree 1,
    we will obtain incorrect values due to aliasing:

    >>> coefficients(circuit, 1, 1)
    array([0.5 +0.j, 0.25+0.j, 0.25+0.j])

    However if we enable the low-pass filter, we can still obtain the correct coefficients:

    >>> coefficients(circuit, 1, 1, lowpass_filter=True)
    array([0.5+0.j, 0. +0.j, 0. +0.j])

    Note that in this case, ``2 * degree`` gives us exactly the maximum coefficient;
    in other situations it may be desirable to set the threshold value explicitly.

    The `coefficients` function can handle qnodes from all PennyLane interfaces.
    """
    if not lowpass_filter:
        return _coefficients_no_filter(f, n_inputs, degree)

    if filter_threshold is None:
        filter_threshold = 2 * degree

    # Compute the fft of the function at 2x the specified degree
    unfiltered_coeffs = _coefficients_no_filter(f, n_inputs, filter_threshold)

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


def _coefficients_no_filter(f, n_inputs, degree):
    r"""Computes the first :math:`2d+1` Fourier coefficients of a :math:`2\pi` periodic
    function, where :math:`d` is the highest desired frequency in the Fourier spectrum.

    This function computes the coefficients blindly without any filtering applied, and
    is thus used as a helper function for the true ``coefficients`` function.

    Args:
        f (callable): function that takes a 1D array of ``n_inputs`` scalar inputs
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
