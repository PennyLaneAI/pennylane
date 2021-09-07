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
This module contains the functions needed for computing integrals over basis functions.
"""
import autograd.numpy as anp
import numpy as np
from scipy.special import factorial2 as fac2


def primitive_norm(l, alpha):
    r"""Compute the normalization constant for a primitive Gaussian function.

    A Gaussian function centred at the position :math:`r = (x, y, z)` is defined as

    .. math::

        G = x^{l_x} y^{l_y} z^{l_z} e^{-\alpha r^2},

    where :math:`l = (l_x, l_y, l_z)` defines the angular momentum quantum number and :math:`\alpha`
    is the Gaussian function exponent. The normalization constant for this function is computed as

    .. math::

        N(l, \alpha) = (\frac{2\alpha}{\pi})^{3/4} \frac{(4 \alpha)^{(l_x + l_y + l_z)/2}}
        {(2l_x-1)!! (2l_y-1)!! (2l_z-1)!!)^{1/2}}.

    Args:
        l (tuple[int]): angular momentum quantum number of the basis function
        alpha (array[float]): exponent of the primitive Gaussian function

    Returns:
        array[float]: normalization coefficient

    **Example**

    >>> l = (0, 0, 0)
    >>> alpha = np.array([3.425250914])
    >>> n = gaussian_norm(l, alpha)
    >>> print(n)
    array([1.79444183])
    """
    lx, ly, lz = l
    n = (
        (2 * alpha / anp.pi) ** 0.75
        * (4 * alpha) ** (sum(l) / 2)
        / anp.sqrt(fac2(2 * lx - 1) * fac2(2 * ly - 1) * fac2(2 * lz - 1))
    )
    return n


def contracted_norm(l, alpha, a):
    r"""Compute the normalization constant for a contracted Gaussian function.

    A contracted Gaussian function is defined as

    .. math::

        \psi = a_1 G_1 + a_2 G_2 + a_3 G_3,

    where :math:`a` denotes the contraction coefficients and :math:`G` is a primitive Gaussian function. The
    normalization constant for this function is computed as

    .. math::

        N(l, \alpha, a) = [\frac{\pi^{3/2}(2l_x-1)!! (2l_y-1)!! (2l_z-1)!!}{2^{l_x + l_y + l_z}}
        \sum_{i,j} \frac{a_i a_j}{(\alpha_i + \alpha_j)^{{l_x + l_y + l_z+3/2}}}]^{-1/2}

    where :math:`l` and :math:`\alpha` denote the angular momentum quantum number and the exponent
    of the Gaussian function, respectively.

    Args:
        l (tuple[int]): angular momentum quantum number of the primitive Gaussian functions
        alpha (array[float]): exponents of the primitive Gaussian functions
        a (array[float]): coefficients of the contracted Gaussian functions

    Returns:
        array[float]: normalization coefficient

    **Example**

    >>> l = (0, 0, 0)
    >>> alpha = np.array([3.425250914, 0.6239137298, 0.168855404])
    >>> a = np.array([1.79444183, 0.50032649, 0.18773546])
    >>> n = contracted_norm(l, alpha, a)
    >>> print(n)
    0.39969026908800853
    """
    lx, ly, lz = l
    c = anp.pi ** 1.5 / 2 ** sum(l) * fac2(2 * lx - 1) * fac2(2 * ly - 1) * fac2(2 * lz - 1)
    s = (
        (a.reshape(len(a), 1) * a) / ((alpha.reshape(len(alpha), 1) + alpha) ** (sum(l) + 1.5))
    ).sum()
    n = 1 / anp.sqrt(c * s)
    return n


def gaussian_kinetic(la, lb, ra, rb, alpha, beta):
    r"""
    """
    l1, m1, n1 = la
    l2, m2, n2 = lb

    k1 = beta * (2 * (l2 + m2 + n2) + 3) * gaussian_overlap((l1, m1, n1), (l2, m2, n2), ra, rb, alpha, beta)

    k2 = -2 * (beta ** 2) * \
            (gaussian_overlap((l1, m1, n1), (l2 + 2, m2, n2), ra, rb, alpha, beta) +
             gaussian_overlap((l1, m1, n1), (l2, m2 + 2, n2), ra, rb, alpha, beta) +
             gaussian_overlap((l1, m1, n1), (l2, m2, n2 + 2), ra, rb, alpha, beta))

    k3 = -0.5 * (l2 * (l2 - 1) * gaussian_overlap((l1, m1, n1), (l2 - 2, m2, n2), ra, rb, alpha, beta) +
                 m2 * (m2 - 1) * gaussian_overlap((l1, m1, n1), (l2, m2 - 2, n2), ra, rb, alpha, beta) +
                 n2 * (n2 - 1) * gaussian_overlap((l1, m1, n1), (l2, m2, n2 - 2), ra, rb, alpha, beta))

    return k1 + k2 + k3

def generate_kinetic(basis_a, basis_b):
    r"""
    """
    def kinetic_integral(*args):

        ra, ca, alpha = generate_params(basis_a.params, args[0])
        rb, cb, beta = generate_params(basis_b.params, args[1])

        ca = ca * gaussian_norm(basis_a.L, alpha)
        cb = cb * gaussian_norm(basis_b.L, beta)

        na = contracted_norm(basis_a.L, alpha, ca)
        nb = contracted_norm(basis_b.L, beta, cb)

        return na * nb * ((ca[:, anp.newaxis] * cb) * gaussian_kinetic(basis_a.L, basis_b.L, ra, rb,
                                                                       alpha[:, anp.newaxis],
                                                                       beta)).sum()
    return kinetic_integral

def nuclear_attraction(la, lb, ra, rb, alpha, beta, r):
    """
    Computes nuclear attraction between Gaussian primitives
    Note that C is the coordinates of the nuclear centre
    """
    l1, m1, n1 = la
    l2, m2, n2 = lb
    p = alpha + beta
    gp = gaussian_prod(alpha, beta, ra[:,anp.newaxis,anp.newaxis], rb[:,anp.newaxis,anp.newaxis])
    dr = gp - anp.array(r)[:,anp.newaxis,anp.newaxis]

    val = 0.0
    for t in range(l1 + l2 + 1):
        for u in range(m1 + m2 + 1):
            for v in range(n1 + n2 + 1):
                val = val + expansion(l1, l2, ra[0], rb[0], alpha, beta, t) * \
                            expansion(m1, m2, ra[1], rb[1], alpha, beta, u) * \
                            expansion(n1, n2, ra[2], rb[2], alpha, beta, v) * \
                            hermite_coulomb(t, u, v, 0, p, dr)
    val = val * 2 * anp.pi / p
    return val

def generate_attraction(basis_a, basis_b):
    """
    Computes the nuclear attraction integral
    """
    def attraction_integral(*args):

        print(*args)
        print()

        r = args[0]
        ra, ca, alpha = generate_params(basis_a.params, args[1])
        rb, cb, beta = generate_params(basis_b.params, args[2])

        ca = ca * gaussian_norm(basis_a.L, alpha)
        cb = cb * gaussian_norm(basis_b.L, beta)


        na = contracted_norm(basis_a.L, alpha, ca)
        nb = contracted_norm(basis_b.L, beta, cb)

        v = na * nb * ((ca * cb[:,anp.newaxis]) * nuclear_attraction(basis_a.L, basis_b.L, ra, rb, alpha, beta[:,anp.newaxis], r)).sum()
        return v
    return attraction_integral


def electron_repulsion(la, lb, lc, ld, ra, rb, rc, rd, alpha, beta, gamma, delta):
    """Electron repulsion between Gaussians"""
    l1, m1, n1 = la
    l2, m2, n2 = lb
    l3, m3, n3 = lc
    l4, m4, n4 = ld

    p = alpha + beta
    q = gamma + delta
    quotient = (p * q)/(p + q)

    p_ab = gaussian_prod(alpha, beta, ra[:,anp.newaxis,anp.newaxis,anp.newaxis,anp.newaxis],
                      rb[:,anp.newaxis,anp.newaxis,anp.newaxis,anp.newaxis]) # A and B composite center
    p_cd = gaussian_prod(gamma, delta, rc[:,anp.newaxis,anp.newaxis,anp.newaxis,anp.newaxis],
                      rd[:,anp.newaxis,anp.newaxis,anp.newaxis,anp.newaxis]) # C and D composite center

    e = 0.0
    for t in range(l1+l2+1):
        for u in range(m1+m2+1):
            for v in range(n1+n2+1):
                for tau in range(l3+l4+1):
                    for nu in range(m3+m4+1):
                        for phi in range(n3+n4+1):
                            e = e + expansion(l1, l2, ra[0], rb[0], alpha, beta, t) * \
                                    expansion(m1, m2, ra[1], rb[1], alpha, beta, u) * \
                                    expansion(n1, n2, ra[2], rb[2], alpha, beta, v) * \
                                    expansion(l3, l4, rc[0], rd[0], gamma, delta, tau) * \
                                    expansion(m3, m4, rc[1], rd[1], gamma, delta, nu) * \
                                    expansion(n3, n4, rc[2], rd[2], gamma, delta, phi) * \
                                   ((-1) ** (tau + nu + phi)) * \
                                    hermite_coulomb(t + tau, u + nu, v + phi, 0, quotient, p_ab - p_cd)

    e = e * 2 * (anp.pi ** 2.5) / (p * q * anp.sqrt(p+q))
    return e


def generate_repulsion(basis_a, basis_b, basis_c, basis_d):
    """
    Computes the two electron repulsion integral
    """
    def repulsion_integral(*args):

        ra, ca, alpha = generate_params(basis_a.params, args[0])
        rb, cb, beta = generate_params(basis_b.params, args[1])
        rc, cc, gamma = generate_params(basis_c.params, args[2])
        rd, cd, delta = generate_params(basis_d.params, args[3])

        ca = ca * gaussian_norm(basis_a.L, alpha)
        cb = cb * gaussian_norm(basis_b.L, beta)
        cc = cc * gaussian_norm(basis_c.L, gamma)
        cd = cd * gaussian_norm(basis_d.L, delta)

        n1 = contracted_norm(basis_a.L, alpha, ca)
        n2 = contracted_norm(basis_b.L, beta, cb)
        n3 = contracted_norm(basis_c.L, gamma, cc)
        n4 = contracted_norm(basis_d.L, delta, cd)

        e = n1 * n2 * n3 * n4 * (
                (ca * cb[:,anp.newaxis] * cc[:,anp.newaxis,anp.newaxis] * cd[:,anp.newaxis,anp.newaxis,anp.newaxis]) *
                electron_repulsion(basis_a.L, basis_b.L, basis_c.L, basis_d.L, ra, rb, rc, rd,
                alpha, beta[:,anp.newaxis], gamma[:,anp.newaxis,anp.newaxis], delta[:,anp.newaxis,anp.newaxis,anp.newaxis])
        ).sum()
        return e
    return repulsion_integral


def boys(a, b):
    r"""
    """
    f = anp.piecewise(b, [b == 0, b != 0], [lambda b : 1 / (2 * a + 1),
    lambda b : sc.special.gamma(0.5 + a) * sc.special.gammainc(0.5 + a, b) / (2 * (b ** (0.5 + a)))])
    return f


def gaussian_prod(alpha, beta, ra, rb):
    """Returns the Gaussian product center"""
    return (alpha * anp.array(ra) + beta * anp.array(rb)) / (alpha + beta)