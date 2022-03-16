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
# pylint: disable= unbalanced-tuple-unpacking, too-many-arguments
import itertools as it
import autograd.numpy as anp
import autograd.scipy as asp
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
    >>> n = primitive_norm(l, alpha)
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
    c = anp.pi**1.5 / 2 ** sum(l) * fac2(2 * lx - 1) * fac2(2 * ly - 1) * fac2(2 * lz - 1)
    s = (
        (a.reshape(len(a), 1) * a) / ((alpha.reshape(len(alpha), 1) + alpha) ** (sum(l) + 1.5))
    ).sum()
    n = 1 / anp.sqrt(c * s)
    return n


def _generate_params(params, args):
    """Generate basis set parameters. The default values are used for the non-differentiable
    parameters and the user-defined values are used for the differentiable ones.

    Args:
        params (list(array[float])): default values of the basis set parameters
        args (list(array[float])): initial values of the differentiable basis set parameters

    Returns:
        list(array[float]): basis set parameters
    """
    basis_params = []
    c = 0
    for p in params:
        if p.requires_grad:
            basis_params.append(args[c])
            c += 1
        else:
            basis_params.append(p)
    return basis_params


def expansion(la, lb, ra, rb, alpha, beta, t):
    r"""Compute Hermite Gaussian expansion coefficients recursively for two Gaussian functions.

    An overlap distribution, which defines the product of two Gaussians, can be written as a Hermite
    expansion as [`Helgaker (1995) p798 <https://www.worldscientific.com/doi/abs/10.1142/9789812832115_0001>`_]

    .. math::

        \Omega_{ij} = \sum_{t=0}^{i+j} E_t^{ij} \Lambda_t,

    where :math:`\Lambda` is a Hermite polynomial of degree :math:`t`, :math:`E` denotes the expansion
    coefficients, :math:`\Omega_{ij} = G_i G_j`, and :math:`G` is a Gaussian function. The overlap
    integral between two Gaussian functions can be simply computed by integrating over the overlap
    distribution which requires obtaining the expansion coefficients. This can be done recursively
    as [`Helgaker (1995) p799 <https://www.worldscientific.com/doi/abs/10.1142/9789812832115_0001>`_]

    .. math::

        E_t^{i+1,j} = \frac{1}{2p} E_{t-1}^{ij} - \frac{qr}{\alpha} E_{t}^{ij} + (t+1) E_{t+1}^{ij},

    and

    .. math::

        E_t^{i,j+1} = \frac{1}{2p} E_{t-1}^{ij} + \frac{qr}{\beta} E_{t}^{ij} + (t+1) E_{t+1}^{ij},

    where :math:`p = \alpha + \beta` and :math:`q = \alpha \beta / (\alpha + \beta)` are computed
    from the Gaussian exponents :math:`\alpha, \beta` and the position :math:`r` is computed as
    :math:`r = r_\alpha - r_\beta`. The starting coefficient is

    .. math::

        E_0^{00} = e^{-qr^2},

    and :math:`E_t^{ij} = 0` is :math:`t < 0` or :math:`t > (i+j)`.

    Args:
        la (integer): angular momentum component for the first Gaussian function
        lb (integer): angular momentum component for the second Gaussian function
        ra (float): position component of the the first Gaussian function
        rb (float): position component of the the second Gaussian function
        alpha (array[float]): exponent of the first Gaussian function
        beta (array[float]): exponent of the second Gaussian function
        t (integer): number of nodes in the Hermite Gaussian

    Returns:
        array[float]: expansion coefficients for each Gaussian combination

    **Example**

    >>> la, lb = 0, 0
    >>> ra, rb = 0.0, 0.0
    >>> alpha = np.array([3.42525091])
    >>> beta =  np.array([3.42525091])
    >>> t = 0
    >>> c = expansion(la, lb, ra, rb, alpha, beta, t)
    >>> c
    array([1.])
    """
    p = anp.array(alpha + beta)
    q = anp.array(alpha * beta / p)
    r = anp.array(ra - rb)

    if la == lb == t == 0:
        return anp.exp(-q * r**2)

    if t < 0 or t > (la + lb):
        return 0.0

    if lb == 0:
        return (
            (1 / (2 * p)) * expansion(la - 1, lb, ra, rb, alpha, beta, t - 1)
            - (q * r / alpha) * expansion(la - 1, lb, ra, rb, alpha, beta, t)
            + (t + 1) * expansion(la - 1, lb, ra, rb, alpha, beta, t + 1)
        )

    return (
        (1 / (2 * p)) * expansion(la, lb - 1, ra, rb, alpha, beta, t - 1)
        + (q * r / beta) * expansion(la, lb - 1, ra, rb, alpha, beta, t)
        + (t + 1) * expansion(la, lb - 1, ra, rb, alpha, beta, t + 1)
    )


def gaussian_overlap(la, lb, ra, rb, alpha, beta):
    r"""Compute overlap integral for two primitive Gaussian functions.

    The overlap integral between two Gaussian functions denoted by :math:`a` and :math:`b` can be
    computed as [`Helgaker (1995) p803 <https://www.worldscientific.com/doi/abs/10.1142/9789812832115_0001>`_]:

    .. math::

        S_{ab} = E^{ij} E^{kl} E^{mn} \left (\frac{\pi}{p}  \right )^{3/2},

    where :math:`E` is a coefficient that can be computed recursively, :math:`i-n` are the angular
    momentum quantum numbers corresponding to different Cartesian components and :math:`p` is
    computed from the exponents of the two Gaussian functions as :math:`p = \alpha + \beta`.

    Args:
        la (integer): angular momentum for the first Gaussian function
        lb (integer): angular momentum for the second Gaussian function
        ra (float): position vector of the the first Gaussian function
        rb (float): position vector of the the second Gaussian function
        alpha (array[float]): exponent of the first Gaussian function
        beta (array[float]): exponent of the second Gaussian function

    Returns:
        array[float]: overlap integral between primitive Gaussian functions

    **Example**

    >>> la, lb = (0, 0, 0), (0, 0, 0)
    >>> ra, rb = np.array([0., 0., 0.]), np.array([0., 0., 0.])
    >>> alpha = np.array([np.pi/2])
    >>> beta = np.array([np.pi/2])
    >>> o = gaussian_overlap(la, lb, ra, rb, alpha, beta)
    >>> o
    array([1.])
    """
    p = alpha + beta
    s = 1.0
    for i in range(3):
        s = s * anp.sqrt(anp.pi / p) * expansion(la[i], lb[i], ra[i], rb[i], alpha, beta, 0)
    return s


def generate_overlap(basis_a, basis_b):
    r"""Return a function that computes the overlap integral for two contracted Gaussian functions.

    Args:
        basis_a (BasisFunction): first basis function
        basis_b (BasisFunction): second basis function

    Returns:
        function: function that computes the overlap integral

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> mol = qml.hf.Molecule(symbols, geometry)
    >>> args = []
    >>> generate_overlap(mol.basis_set[0], mol.basis_set[0])(*args)
    1.0
    """

    def overlap_integral(*args):
        r"""Normalize and compute the overlap integral for two contracted Gaussian functions.

        Args:
            args (array[float]): initial values of the differentiable parameters

        Returns:
            array[float]: the overlap integral between two contracted Gaussian orbitals
        """

        args_a = [arg[0] for arg in args]
        args_b = [arg[1] for arg in args]
        alpha, ca, ra = _generate_params(basis_a.params, args_a)
        beta, cb, rb = _generate_params(basis_b.params, args_b)

        ca = ca * primitive_norm(basis_a.l, alpha)
        cb = cb * primitive_norm(basis_b.l, beta)

        na = contracted_norm(basis_a.l, alpha, ca)
        nb = contracted_norm(basis_b.l, beta, cb)

        return (
            na
            * nb
            * (
                (ca[:, anp.newaxis] * cb)
                * gaussian_overlap(basis_a.l, basis_b.l, ra, rb, alpha[:, anp.newaxis], beta)
            ).sum()
        )

    return overlap_integral


def _diff2(i, j, ri, rj, alpha, beta):
    r"""Compute the second order differentiated integral needed for evaluating a kinetic integral.

    The second-order integral :math:`D_{ij}^2`, where :math:`i` and :math:`j` denote angular
    momentum components of Gaussian functions, is computed from overlap integrals :math:`S` and the
    Gaussian exponent :math:`\beta` as
    [`Helgaker (1995) p804 <https://www.worldscientific.com/doi/abs/10.1142/9789812832115_0001>`_]:

    .. math::

        D_{ij}^2 = j(j-1)S_{i,j-2}^0 - 2\beta(2j+1)S_{i,j}^0 + 4\beta^2 S_{i,j+2}^0.

    Args:
        i (integer): angular momentum component for the first Gaussian function
        j (integer): angular momentum component for the second Gaussian function
        ri (array[float]): position component of the the first Gaussian function
        rj (array[float]): position component of the the second Gaussian function
        alpha (array[float]): exponent of the first Gaussian function
        beta (array[float]): exponent of the second Gaussian function

    Returns:
        array[float]: second-order differentiated integral between two Gaussian functions
    """
    p = alpha + beta

    d1 = j * (j - 1) * anp.sqrt(anp.pi / p) * expansion(i, j - 2, ri, rj, alpha, beta, 0)
    d2 = -2 * beta * (2 * j + 1) * anp.sqrt(anp.pi / p) * expansion(i, j, ri, rj, alpha, beta, 0)
    d3 = 4 * beta**2 * anp.sqrt(anp.pi / p) * expansion(i, j + 2, ri, rj, alpha, beta, 0)

    return d1 + d2 + d3


def gaussian_kinetic(la, lb, ra, rb, alpha, beta):
    r"""Compute the kinetic integral for two primitive Gaussian functions.

    The kinetic integral between two Gaussian functions denoted by :math:`a` and :math:`b` is
    computed as
    [`Helgaker (1995) p805 <https://www.worldscientific.com/doi/abs/10.1142/9789812832115_0001>`_]:

    .. math::

        T_{ab} = -\frac{1}{2} \left ( D_{ij}^2 D_{kl}^0 D_{mn}^0 + D_{ij}^0 D_{kl}^2 D_{mn}^0 + D_{ij}^0 D_{kl}^0 D_{mn}^2\right ),

    where :math:`D_{ij}^0 = S_{ij}^0` is an overlap integral and :math:`D_{ij}^2` is computed from
    overlap integrals :math:`S` and the Gaussian exponent :math:`\beta` as

    .. math::

        D_{ij}^2 = j(j-1)S_{i,j-2}^0 - 2\beta(2j+1)S_{i,j}^0 + 4\beta^2 S_{i,j+2}^0.

    Args:
        la (tuple[int]): angular momentum for the first Gaussian function
        lb (tuple[int]): angular momentum for the second Gaussian function
        ra (array[float]): position vector of the the first Gaussian function
        rb (array[float]): position vector of the the second Gaussian function
        alpha (array[float]): exponent of the first Gaussian function
        beta (array[float]): exponent of the second Gaussian function

    Returns:
        array[float]: kinetic integral between two Gaussian functions

    **Example**

    >>> la, lb = (0, 0, 0), (0, 0, 0)
    >>> ra = np.array([0., 0., 0.])
    >>> rb = rb = np.array([0., 0., 0.])
    >>> alpha = np.array([np.pi/2])
    >>> beta = np.array([np.pi/2])
    >>> t = gaussian_kinetic(la, lb, ra, rb, alpha, beta)
    >>> t
    array([2.35619449])
    """

    p = alpha + beta

    t1 = (
        _diff2(la[0], lb[0], ra[0], rb[0], alpha, beta)
        * anp.sqrt(anp.pi / p)
        * expansion(la[1], lb[1], ra[1], rb[1], alpha, beta, 0)
        * anp.sqrt(anp.pi / p)
        * expansion(la[2], lb[2], ra[2], rb[2], alpha, beta, 0)
    )

    t2 = (
        anp.sqrt(anp.pi / p)
        * expansion(la[0], lb[0], ra[0], rb[0], alpha, beta, 0)
        * _diff2(la[1], lb[1], ra[1], rb[1], alpha, beta)
        * anp.sqrt(anp.pi / p)
        * expansion(la[2], lb[2], ra[2], rb[2], alpha, beta, 0)
    )

    t3 = (
        anp.sqrt(anp.pi / p)
        * expansion(la[0], lb[0], ra[0], rb[0], alpha, beta, 0)
        * anp.sqrt(anp.pi / p)
        * expansion(la[1], lb[1], ra[1], rb[1], alpha, beta, 0)
        * _diff2(la[2], lb[2], ra[2], rb[2], alpha, beta)
    )

    return -0.5 * (t1 + t2 + t3)


def generate_kinetic(basis_a, basis_b):
    r"""Return a function that computes the kinetic integral for two contracted Gaussian functions.

    Args:
        basis_a (BasisFunction): first basis function
        basis_b (BasisFunction): second basis function

    Returns:
        function: function that computes the kinetic integral

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.425250914, 0.6239137298, 0.168855404],
    >>>                   [3.425250914, 0.6239137298, 0.168855404]], requires_grad = True)
    >>> mol = qml.hf.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [mol.alpha]
    >>> generate_kinetic(mol.basis_set[0], mol.basis_set[1])(*args)
    0.38325367405312843
    """

    def kinetic_integral(*args):
        r"""Compute the kinetic integral for two contracted Gaussian functions.

        Args:
            args (array[float]): initial values of the differentiable parameters

        Returns:
            array[float]: the kinetic integral between two contracted Gaussian orbitals
        """
        args_a = [arg[0] for arg in args]
        args_b = [arg[1] for arg in args]
        alpha, ca, ra = _generate_params(basis_a.params, args_a)
        beta, cb, rb = _generate_params(basis_b.params, args_b)

        ca = ca * primitive_norm(basis_a.l, alpha)
        cb = cb * primitive_norm(basis_b.l, beta)

        na = contracted_norm(basis_a.l, alpha, ca)
        nb = contracted_norm(basis_b.l, beta, cb)

        return (
            na
            * nb
            * (
                (ca[:, anp.newaxis] * cb)
                * gaussian_kinetic(basis_a.l, basis_b.l, ra, rb, alpha[:, anp.newaxis], beta)
            ).sum()
        )

    return kinetic_integral


def _boys(n, t):
    r"""Evaluate the Boys function.

    The :math:`n`-th order `Boys function <https://arxiv.org/abs/2107.01488>`_ is defined as

    .. math::

        F_n(t) = \int_{0}^{1}x^{2n} e^{-tx^2}dx.

    The Boys function is related to the lower incomplete Gamma
    `function <https://en.wikipedia.org/wiki/Incomplete_gamma_function>`_, :math:`\gamma`, as

    .. math::

        F_n(t) = \frac{1}{2t^{n + 0.5}} \gamma(n + 0.5, t),

    where

    .. math::

        \gamma(m, t) = \int_{0}^{t} x^{m-1} e^{-x} dx.

    Args:
        n (float): order of the Boys function
        t (float): exponent of the Boys function

    Returns:
        float: value of the Boys function
    """
    if t == 0.0:
        return 1 / (2 * n + 1)

    return asp.special.gammainc(n + 0.5, t) * asp.special.gamma(n + 0.5) / (2 * t ** (n + 0.5))


def _hermite_coulomb(t, u, v, n, p, dr):
    """Evaluate the Hermite integral needed to compute the nuclear attraction and electron repulsion
    integrals.

    These integrals are computed recursively starting from the Boys function
    [`Helgaker (1995) p817 <https://www.worldscientific.com/doi/abs/10.1142/9789812832115_0001>`_]:


    .. math::

        R_{000}^n = (-2p)^n F_n(pR_{CP}^2),

    where :math:`F_n` is the Boys function, :math:`p` is computed from the exponents of the two
    Gaussian functions as :math:`p = \alpha + \beta`, and :math:`R_{CP}` is the distance between the
    center of the composite Gaussian centered at :math:`P` and the electrostatic potential generated
    by a nucleus at :math:`C`. The following recursive equations are used to compute the
    higher-order Hermite integrals

    .. math::

        R_{t+1, u, v}^n = t R_{t-1, u, v}^{n+1} + x R_{t, u, v}^{n+1},

        R_{t, u+1, v}^n = u R_{t, u-1, v}^{n+1} + y R_{t, u, v}^{n+1},

        R_{t, u, v+1}^n = v R_{t, u, v-1}^{n+1} + z R_{t, u, v}^{n+1},

    where :math:`x`, :math:`y` and :math:`z` are the Cartesian components of :math:`R_{CP}`.

    Args:
        t (integer): order of Hermite derivative in x
        u (integer): order of Hermite derivative in y
        v (float): order of Hermite derivative in z
        n (integer): order of the Boys function
        p (float): sum of the Gaussian exponents
        dr (array[float]): distance between the center of the composite Gaussian and the nucleus

    Returns:
        array[float]: value of the Hermite integral
    """
    x, y, z = dr[0], dr[1], dr[2]
    T = p * (dr**2).sum(axis=0)
    r = 0

    if t == u == v == 0:
        f = []
        for term in T.flatten():
            f.append(_boys(n, term))
        return ((-2 * p) ** n) * anp.array(f).reshape(T.shape)

    if t == u == 0:
        if v > 1:
            r = r + (v - 1) * _hermite_coulomb(t, u, v - 2, n + 1, p, dr)
        r = r + z * _hermite_coulomb(t, u, v - 1, n + 1, p, dr)
        return r

    if t == 0:
        if u > 1:
            r = r + (u - 1) * _hermite_coulomb(t, u - 2, v, n + 1, p, dr)
        r = r + y * _hermite_coulomb(t, u - 1, v, n + 1, p, dr)
        return r

    if t > 1:
        r = r + (t - 1) * _hermite_coulomb(t - 2, u, v, n + 1, p, dr)
    r = r + x * _hermite_coulomb(t - 1, u, v, n + 1, p, dr)
    return r


def nuclear_attraction(la, lb, ra, rb, alpha, beta, r):
    r"""Compute nuclear attraction integral between primitive Gaussian functions.

    The nuclear attraction integral between two Gaussian functions denoted by :math:`a` and
    :math:`b` can be computed as
    [`Helgaker (1995) p820 <https://www.worldscientific.com/doi/abs/10.1142/9789812832115_0001>`_]

    .. math::

        V_{ab} = \frac{2\pi}{p} \sum_{tuv} E_t^{ij} E_u^{kl} E_v^{mn} R_{tuv},

    where :math:`E` and :math:`R` represent the Hermite Gaussian expansion coefficients and the
    Hermite Coulomb integral, respectively. The sum goes over :math:`i + j + 1`, :math:`k + l + 1`
    and :math:`m + n + 1` for :math:`t`, :math:`u` and :math:`v`, respectively, and :math:`p` is
    computed from the exponents of the two Gaussian functions as :math:`p = \alpha + \beta`.

    Args:
        la (tuple[int]): angular momentum for the first Gaussian function
        lb (tuple[int]): angular momentum for the second Gaussian function
        ra (array[float]): position vector of the the first Gaussian function
        rb (array[float]): position vector of the the second Gaussian function
        alpha (array[float]): exponent of the first Gaussian function
        beta (array[float]): exponent of the second Gaussian function
        r (array[float]): position vector of nucleus

    Returns:
        array[float]: nuclear attraction integral between two Gaussian functions
    """
    l1, m1, n1 = la
    l2, m2, n2 = lb
    p = alpha + beta
    rgp = (alpha * ra[:, anp.newaxis, anp.newaxis] + beta * rb[:, anp.newaxis, anp.newaxis]) / (
        alpha + beta
    )
    dr = rgp - anp.array(r)[:, anp.newaxis, anp.newaxis]

    a = 0.0
    for t, u, v in it.product(*[range(l) for l in [l1 + l2 + 1, m1 + m2 + 1, n1 + n2 + 1]]):
        a = a + expansion(l1, l2, ra[0], rb[0], alpha, beta, t) * expansion(
            m1, m2, ra[1], rb[1], alpha, beta, u
        ) * expansion(n1, n2, ra[2], rb[2], alpha, beta, v) * _hermite_coulomb(t, u, v, 0, p, dr)
    a = a * 2 * anp.pi / p
    return a


def generate_attraction(r, basis_a, basis_b):
    r"""Return a function that computes the nuclear attraction integral for two contracted Gaussian
    functions.

    Args:
        r (array[float]): position vector of nucleus
        basis_a (BasisFunction): first basis function
        basis_b (BasisFunction): second basis function

    Returns:
        function: function that computes the electron-nuclear attraction integral

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.425250914, 0.6239137298, 0.168855404],
    >>>                   [3.425250914, 0.6239137298, 0.168855404]], requires_grad = True)
    >>> mol = qml.hf.Molecule(symbols, geometry, alpha=alpha)
    >>> basis_a = mol.basis_set[0]
    >>> basis_b = mol.basis_set[1]
    >>> args = [mol.alpha]
    >>> generate_attraction(geometry[0], basis_a, basis_b)(*args)
    0.801208332328965
    """

    def attraction_integral(*args):
        r"""Compute the electron-nuclear attraction integral for two contracted Gaussian functions.

        Args:
            args (array[float]): initial values of the differentiable parameters

        Returns:
            array[float]: the electron-nuclear attraction integral
        """
        if r.requires_grad:
            coor = args[0]
            args_a = [arg[0] for arg in args[1:]]
            args_b = [arg[1] for arg in args[1:]]
        else:
            coor = r
            args_a = [arg[0] for arg in args]
            args_b = [arg[1] for arg in args]

        alpha, ca, ra = _generate_params(basis_a.params, args_a)
        beta, cb, rb = _generate_params(basis_b.params, args_b)

        ca = ca * primitive_norm(basis_a.l, alpha)
        cb = cb * primitive_norm(basis_b.l, beta)

        na = contracted_norm(basis_a.l, alpha, ca)
        nb = contracted_norm(basis_b.l, beta, cb)

        v = (
            na
            * nb
            * (
                (ca * cb[:, anp.newaxis])
                * nuclear_attraction(
                    basis_a.l, basis_b.l, ra, rb, alpha, beta[:, anp.newaxis], coor
                )
            ).sum()
        )
        return v

    return attraction_integral


def electron_repulsion(la, lb, lc, ld, ra, rb, rc, rd, alpha, beta, gamma, delta):
    r"""Compute the electron-electron repulsion integral between four primitive Gaussian functions.

    The electron repulsion integral between four Gaussian functions denoted by :math:`a`, :math:`b`
    , :math:`c` and :math:`d` can be computed as
    [`Helgaker (1995) p820 <https://www.worldscientific.com/doi/abs/10.1142/9789812832115_0001>`_]

    .. math::

        g_{abcd} = \frac{2\pi^{5/2}}{pq\sqrt{p+q}} \sum_{tuv} E_t^{o_a o_b} E_u^{m_a m_b}
        E_v^{n_a n_b} \sum_{rsw} (-1)^{r+s+w} E_r^{o_c o_d} E_s^{m_c m_d} E_w^{n_c n_d}
        R_{t+r, u+s, v+w},

    where :math:`E` and :math:`R` are the Hermite Gaussian expansion coefficients and the
    Hermite Coulomb integral, respectively. The sums go over the angular momentum quantum numbers
    :math:`o_i + o_j + 1`, :math:`m_i + m_j + 1` and :math:`n_i + n_j + 1` respectively for
    :math:`t, u, v` and :math:`r, s, w`. The exponents of the Gaussian functions are used to compute
    :math:`p` and :math:`q` as :math:`p = \alpha + \beta` and :math:`q = \gamma + \delta`.

    Args:
        la (tuple[int]): angular momentum for the first Gaussian function
        lb (tuple[int]): angular momentum for the second Gaussian function
        lc (tuple[int]): angular momentum for the third Gaussian function
        ld (tuple[int]): angular momentum for the forth Gaussian function
        ra (array[float]): position vector of the the first Gaussian function
        rb (array[float]): position vector of the the second Gaussian function
        rc (array[float]): position vector of the the third Gaussian function
        rd (array[float]): position vector of the the forth Gaussian function
        alpha (array[float]): exponent of the first Gaussian function
        beta (array[float]): exponent of the second Gaussian function
        gamma (array[float]): exponent of the third Gaussian function
        delta (array[float]): exponent of the forth Gaussian function

    Returns:
        array[float]: electron-electron repulsion integral between four Gaussian functions
    """
    l1, m1, n1 = la
    l2, m2, n2 = lb
    l3, m3, n3 = lc
    l4, m4, n4 = ld

    p = alpha + beta
    q = gamma + delta

    p_ab = (
        alpha * ra[:, anp.newaxis, anp.newaxis, anp.newaxis, anp.newaxis]
        + beta * rb[:, anp.newaxis, anp.newaxis, anp.newaxis, anp.newaxis]
    ) / (alpha + beta)

    p_cd = (
        gamma * rc[:, anp.newaxis, anp.newaxis, anp.newaxis, anp.newaxis]
        + delta * rd[:, anp.newaxis, anp.newaxis, anp.newaxis, anp.newaxis]
    ) / (gamma + delta)

    g = 0.0
    lengths = [l1 + l2 + 1, m1 + m2 + 1, n1 + n2 + 1, l3 + l4 + 1, m3 + m4 + 1, n3 + n4 + 1]
    for t, u, v, r, s, w in it.product(*[range(length) for length in lengths]):
        g = g + expansion(l1, l2, ra[0], rb[0], alpha, beta, t) * expansion(
            m1, m2, ra[1], rb[1], alpha, beta, u
        ) * expansion(n1, n2, ra[2], rb[2], alpha, beta, v) * expansion(
            l3, l4, rc[0], rd[0], gamma, delta, r
        ) * expansion(
            m3, m4, rc[1], rd[1], gamma, delta, s
        ) * expansion(
            n3, n4, rc[2], rd[2], gamma, delta, w
        ) * (
            (-1) ** (r + s + w)
        ) * _hermite_coulomb(
            t + r, u + s, v + w, 0, (p * q) / (p + q), p_ab - p_cd
        )

    g = g * 2 * (anp.pi**2.5) / (p * q * anp.sqrt(p + q))

    return g


def generate_repulsion(basis_a, basis_b, basis_c, basis_d):
    r"""Return a function that computes the electron-electron repulsion integral for four contracted
    Gaussian functions.

    Args:
        basis_a (BasisFunction): first basis function
        basis_b (BasisFunction): second basis function
        basis_c (BasisFunction): third basis function
        basis_d (BasisFunction): fourth basis function
    Returns:
        function: function that computes the electron repulsion integral

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.425250914, 0.6239137298, 0.168855404],
    >>>                   [3.425250914, 0.6239137298, 0.168855404],
    >>>                   [3.425250914, 0.6239137298, 0.168855404],
    >>>                   [3.425250914, 0.6239137298, 0.168855404]], requires_grad = True)
    >>> mol = qml.hf.Molecule(symbols, geometry, alpha=alpha)
    >>> basis_a = mol.basis_set[0]
    >>> basis_b = mol.basis_set[1]
    >>> args = [mol.alpha]
    >>> generate_repulsion(basis_a, basis_b, basis_a, basis_b)(*args)
    0.45590152106593573
    """

    def repulsion_integral(*args):
        r"""Compute the electron-electron repulsion integral for four contracted Gaussian functions.

        Args:
            args (array[float]): initial values of the differentiable parameters

        Returns:
            array[float]: the electron repulsion integral between four contracted Gaussian functions
        """
        args_a = [arg[0] for arg in args]
        args_b = [arg[1] for arg in args]
        args_c = [arg[2] for arg in args]
        args_d = [arg[3] for arg in args]

        alpha, ca, ra = _generate_params(basis_a.params, args_a)
        beta, cb, rb = _generate_params(basis_b.params, args_b)
        gamma, cc, rc = _generate_params(basis_c.params, args_c)
        delta, cd, rd = _generate_params(basis_d.params, args_d)

        ca = ca * primitive_norm(basis_a.l, alpha)
        cb = cb * primitive_norm(basis_b.l, beta)
        cc = cc * primitive_norm(basis_c.l, gamma)
        cd = cd * primitive_norm(basis_d.l, delta)

        n1 = contracted_norm(basis_a.l, alpha, ca)
        n2 = contracted_norm(basis_b.l, beta, cb)
        n3 = contracted_norm(basis_c.l, gamma, cc)
        n4 = contracted_norm(basis_d.l, delta, cd)

        e = (
            n1
            * n2
            * n3
            * n4
            * (
                (
                    ca
                    * cb[:, anp.newaxis]
                    * cc[:, anp.newaxis, anp.newaxis]
                    * cd[:, anp.newaxis, anp.newaxis, anp.newaxis]
                )
                * electron_repulsion(
                    basis_a.l,
                    basis_b.l,
                    basis_c.l,
                    basis_d.l,
                    ra,
                    rb,
                    rc,
                    rd,
                    alpha,
                    beta[:, anp.newaxis],
                    gamma[:, anp.newaxis, anp.newaxis],
                    delta[:, anp.newaxis, anp.newaxis, anp.newaxis],
                )
            ).sum()
        )
        return e

    return repulsion_integral
