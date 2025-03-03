"""
This module contains the functions needed for computing matrices.
"""
import itertools as it

from jax import numpy as jnp
from jax import scipy as jsp
from jax import config

config.update("jax_enable_x64", True)

import jax
import itertools as it

import numpy as np


from .integrals import (
    repulsion_integral,
)


def repulsion_tensor(basis_functions):
    r"""Return a function that computes the electron repulsion tensor for a given set of basis
    functions.

    Args:
        basis_functions (list[~qchem.basis_set.BasisFunction]): basis functions

    Returns:
        function: function that computes the electron repulsion tensor

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]])
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> repulsion_tensor(mol.basis_set)
    """
    n = len(basis_functions)
    tensor = np.zeros((n, n, n, n))
    e_calc = np.full((n, n, n, n), np.nan)

    alphas =  np.array([basis.alpha for basis in basis_functions])
    coeffs = np.array([basis.coeff for basis in basis_functions])
    rs = np.array([basis.r for basis in basis_functions])
    ls = np.array([basis.l for basis in basis_functions])

    jitted_repulsion_integral = jax.jit(repulsion_integral, static_argnums=(3))
    i, j, k, l = 0, 0, 0, 0
    alphas_ = alphas[np.array([i, j, k, l])]
    coeffs_ = coeffs[np.array([i, j, k, l])]
    rs_ = rs[np.array([i, j, k, l])]
    ls_ = tuple([tuple(x) for x in ls[jnp.array([i, j, k, l])]])

    jitted_repulsion_integral(alphas_, coeffs_, rs_, ls_)

    for (i, j, k, l) in it.product(np.arange(n), repeat=4):
        if np.isnan(e_calc[(i, j, k, l)]):
            alphas_ = alphas[np.array([i, j, k, l])]
            coeffs_ = coeffs[np.array([i, j, k, l])]
            rs_ = rs[np.array([i, j, k, l])]
            ls_ = tuple([tuple(x) for x in ls[jnp.array([i, j, k, l])]])

            integral = jitted_repulsion_integral(alphas_, coeffs_, rs_, ls_)

            permutations = [
                (i, j, k, l),
                (k, l, i, j),
                (j, i, l, k),
                (l, k, j, i),
                (j, i, k, l),
                (l, k, i, j),
                (i, j, l, k),
                (k, l, j, i),
            ]

            o = np.zeros((n, n, n, n))
            for perm in permutations:
                o[perm] = 1.0
                e_calc[perm] = 1.0
            tensor = tensor + integral * o
    return tensor
