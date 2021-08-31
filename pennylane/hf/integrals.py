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
TThis module contains functions for computing integrals over basis functions.
"""
import numpy as np
import autograd.numpy as anp
from scipy.special import factorial2 as fac2


def primitive_norm(l, alpha):
    r"""Compute the normalization constant for a primitive Gaussian function.

    The normalization constant is computed as

    .. math::

        N(l, \alpha) = (\frac{2\alpha}{\pi})^{3/4} \frac{(4 \alpha)^{(l_x + l_y + l_z)/2}}
        {(2\l_x-1)!! (2\l_y-1)!! (2\l_z-1)!!)^{1/2}}

    where :math:`l` and :math:`\alpha` are the Cartesian angular momentum vector and the exponent of
    the Gaussian function, respectively.

    Args:
        l (tuple): angular momentum numbers
        alpha (array(float)): exponent of the Gaussian function

    Returns:
        n (float): normalization coefficient

    **Example**
    >>> l = (0, 0, 0)
    >>> alpha = np.array([3.425250914])
    >>> n = gaussian_norm(l, alpha)
    >>> print(n)
    array([1.79444183])
    """
    lx, ly, lz = l
    n = (2 * alpha / np.pi)**0.75 * (4 * alpha)**(sum(l) / 2) / \
    anp.sqrt(fac2(2 * lx - 1) * fac2(2 * ly - 1) * fac2(2 * lz - 1))
    return n

def contracted_norm(l, alpha, c):
    r"""Compute the normalization constant for a contracted Gaussian function.

    The normalization constant is computed as

    .. math::

        N(l, \alpha, c) = [\frac{\pi^{3/2}(2\l_x-1)!! (2\l_y-1)!! (2\l_z-1)!!}{2^{l_x + l_y + l_z}}
        \sum_{i,j} \frac{c_i c_j}{(\alpha_i + \alpha_j)^{2^{l_x + l_y + l_z+3/2}}}]^{-1/2}

    where :math:`l`, :math:`\alpha` and :math:`c` are the Cartesian angular momentum vector,the
    exponent of the Gaussian function and the contraction coefficients, respectively.

    Args:
        l (tuple): angular momentum numbers
        alpha (array(float)): exponent of the Gaussian function
        c (array(float)): contraction coefficients of the Gaussian function

    Returns:
        n (float): normalization coefficient

    **Example**
    >>> l = (0, 0, 0)
    >>> alpha = np.array([3.425250914, 0.6239137298, 0.168855404])
    >>> c = np.array([1.79444183, 0.50032649, 0.18773546])
    >>> n = contracted_norm(l, alpha, c)
    >>> print(n)
    0.39969026908800853
    """
    lx, ly, lz = l
    coeff = np.pi ** 1.5 / 2 ** sum(l) * fac2(2 * lx - 1) * fac2(2 * ly - 1) * fac2(2 * lz - 1)
    s = ((c[:,anp.newaxis] * c) / ((alpha[:,anp.newaxis] + alpha) ** (sum(l) + 1.5))).sum()
    n = 1 / anp.sqrt(coeff * s)
    return n
