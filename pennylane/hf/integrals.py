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


def gaussian_norm(l, alpha):
    r"""Compute the normalization constant for a Gaussian function.
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
    r"""Compute the normalization constant for contracted Gaussian function.
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

def generate_params(params, args):
    """
    Generate basis set parameters. The default values are used for the non-differentiable parameters
    and the user-defined values are used for the differentiable ones.
    """
    basis_params = []
    c = 0

    for i, p in enumerate(params):
        if p is None:
            basis_params.append(args[c])
            c += 1
        else:
            basis_params.append(p)
    return tuple(basis_params)


def expansion(la, lb, ra, rb, alpha, beta, t):
    r"""Compute Hermite Gaussian expansion coefficients recursively for a set of Gaussian functions
    centered on the same position.
    Args:
        la (integer): angular momentum component for the first Gaussian function
        lb (integer): angular momentum component for the second Gaussian function
        ra (float): position component of the the first Gaussian function
        rb (float): position component of the the second Gaussian function
        alpha (array(float)): exponent of the first Gaussian function
        beta (array(float)): exponent of the second Gaussian function
        t(integer): number of nodes in the Hermite Gaussian
    Returns:
        array(float): expansion coefficients for each Gaussian combination
    **Example**
    >>> la, lb = 0, 0
    >>> ra, rb = 0.0, 0.0
    >>> alpha = [[3.42525091], [0.62391373], [0.1688554 ]]
    >>> beta = [3.42525091, 0.62391373, 0.1688554]
    >>> alpha = np.array([3.425250914, 0.6239137298, 0.168855404])
    >>> t = 0
    >>> c = expansion(la, lb, ra, rb, alpha, beta, t)
    >>> print(c)
    [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]
    """
    p = alpha + beta
    q = alpha * beta / p
    r = ra - rb

    if la == lb == t == 0:
        return anp.exp(-q * r * r)

    elif t < 0 or t > (la + lb):
        return 0.0

    elif lb == 0:
        return (1 / (2 * p)) * expansion(la - 1, lb, ra, rb, alpha, beta, t - 1) - \
               (q * r / alpha) * expansion(la - 1, lb, ra, rb, alpha, beta, t) + \
               (t + 1) * expansion(la - 1, lb, ra, rb, alpha, beta, t + 1)
    else:
        return (1 / (2 * p)) * expansion(la, lb - 1, ra, rb, alpha, beta, t - 1) + \
               (q * r / beta) * expansion(la, lb - 1, ra, rb, alpha, beta, t) + \
               (t + 1) * expansion(la, lb - 1, ra, rb, alpha, beta, t + 1)

def gaussian_overlap(la, lb, ra, rb, alpha, beta):
    r"""Compute overlap integrals for two sets of Gaussian functions.
    Args:
        la (integer): angular momentum for the first Gaussian function
        lb (integer): angular momentum for the second Gaussian function
        ra (float): position vector of the the first Gaussian function
        rb (float): position vector of the the second Gaussian function
        alpha (array(float)): exponent of the first Gaussian function
        beta (array(float)): exponent of the second Gaussian function
    Returns:
        array(float): overlap integrals for each Gaussian combination
    """
    p = alpha + beta
    s = 1.0
    for i in range(3):
        s = s * anp.sqrt(anp.pi / p) * expansion(la[i], lb[i], ra[i], rb[i], alpha, beta, 0)
    return s


def generate_overlap(basis_a, basis_b):
    """Return a function that normalizes and computes the overlap integral for two contracted
    Gaussian orbitals.
    """
    def overlap_integral(*args):

        ra, ca, alpha = generate_params(basis_a.params, args[0])
        rb, cb, beta = generate_params(basis_b.params, args[1])

        ca = ca * gaussian_norm(basis_a.L, alpha)
        cb = cb * gaussian_norm(basis_b.L, beta)

        na = contracted_norm(basis_a.L, alpha, ca)
        nb = contracted_norm(basis_b.L, beta, cb)

        return na * nb * ((ca[:,anp.newaxis] * cb) * gaussian_overlap(basis_a.L, basis_b.L, ra, rb, alpha[:,anp.newaxis], beta)).sum()
    return overlap_integral
