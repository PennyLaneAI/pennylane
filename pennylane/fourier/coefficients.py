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

custom_decomps_required = {
    "CRot" : custom_CRot_decomposition
}


def fourier_coefficients(f, n_inputs, degree, apply_rfftn=False):
    """Computes the first :math:`2d+1` Fourier coefficients of a :math:`2\pi` periodic
    function, where :math:`d` is the highest desired frequency in the Fourier spectrum.

    The Fourier coefficients are computed for the parameters in the *last*
    argument of a quantum function.

    Args:
        f (callable): function that takes an array of :math:`N` scalar inputs. Function should
            have structure ``f(weights, params)`` where params are the paramters that the
            Fourier coefficients are taken with respect to.
        n_inputs (int): dimension of the input
        degree (int): max frequency of Fourier coeffs to be computed. For degree :math:`d`,
            the coefficients from frequencies :math:`-d, -d+1,...0,..., d-1, d ` will be computed.
        apply_rfftn (bool): If True, call rfftn instead of fftn.

    Returns:
        (np.ndarray): The Fourier coefficients of the function f.

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

    # Now we have a discretized verison of f we can use
    # the discrete fourier transform.
    # The normalization factor is the number of discrete points (??)
    if apply_rfftn:
        coeffs = np.fft.rfftn(f_discrete) / f_discrete.size
    else:
        coeffs = np.fft.fftn(f_discrete) / f_discrete.size

    return coeffs
