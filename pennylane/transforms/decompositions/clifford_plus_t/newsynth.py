# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The gridsynth method, adapted from the ``newsynth`` Haskell package."""

from typing import Tuple, Iterator
import numpy as np

import pennylane as qml

from .conversion import (
    denomexp,
    from_d_root_two,
    point_from_d_root_two,
)
from .diophantine import diophantine_dyadic
from .gridpoints import gridpoints2_increasing
from .shapes import ConvexSet, Ellipse, Point
from .rings import Matrix, DOmega, omega

OMEGA = omega(0, 0, 1, 0)


def gridsynth(prec, theta, effort):
    """
    The gridsynth algorithm. Converts RZ(theta) into a sequence of Clifford+T gates.

    Args:
        g (TensorLike): source of randomness
        prec (Number): used to define the epsilon (error) value, 2 ** -prec
        theta (float): the angle we are trying to approximate
        effort (int): the number of candidates to try before giving up

    Returns:
        (U2 DOmega, Maybe Double, [(DOmega, Integer, DStatus)])
    """
    if effort < 1:
        raise ValueError("`effort` must be a positive integer.")

    epsilon = 2**-prec
    region = epsilon_region(epsilon, theta)
    candidates = gridpoints2_increasing(region)
    candidate_info = first_solvable(candidates, effort)
    u, _, t = candidate_info[0]
    if not t:
        raise ValueError("could not find valid candidate.")
    uU, log_err = with_succesful_candidate(u, t, theta)
    return uU, log_err, candidate_info


def epsilon_region(epsilon, theta):
    """The epsilon region."""
    zx = np.cos(-theta / 2)
    zy = np.sin(-theta / 2)
    z = (zx, zy)
    d = 1 - (epsilon**2 / 2)

    ev1 = 4 * (epsilon**-4)
    ev2 = epsilon**-2
    ctr = Point(d * zx, d * zy)
    mmat = np.array([[ev1, 0], [0, ev2]])
    bmat = np.array([[zx, -zy], [zy, zx]])
    mat = bmat @ mmat @ np.linalg.inv(bmat)
    ell = Ellipse(mat, ctr)

    def characteristic_fn(x, y):
        return x**2 + y**2 <= 1 and from_d_root_two(x) + zy * from_d_root_two(y) >= d

    def intersector(p, v):
        p, v = (list(p), list(v))
        a = np.inner(v, v)
        b = 2 * np.inner(v, p)
        c = np.inner(p, p) - 1
        q = [r for r in np.roots((a, b, c)) if np.isclose(np.imag(r), 0)]
        if len(q) != 2:
            return None
        t0, t1 = sorted(q)
        vz = np.inner(point_from_d_root_two(v), z)
        rhs = d - np.inner(point_from_d_root_two(p), z)
        t2 = rhs / vz
        if vz > 0:
            return max(t0, t2), t1
        if vz == 0:
            return (t0, t1) if rhs <= 0 else None
        return t0, min(t1, t2)

    return ConvexSet(ell, characteristic_fn, intersector)


def tcount_for(k):
    """Returns the T-count for a given k-value"""
    return 2 * k - 2 if k > 0 else 0


def first_solvable(candidates: Iterator[Tuple[DOmega, int]], effort: int):
    """Get the first solvable ``u`` from a list of candidates."""
    # TODO: do we need to track ``infos``?
    infos = []
    for u, tcount in candidates:
        u_dagger_u = u.conjugate() * u
        xi = 1 - u_dagger_u.real
        t = diophantine_dyadic(xi, effort)
        infos.append((u, tcount, t))
        if t is not None:
            return infos

    raise ValueError("no valid candidates")


def with_succesful_candidate(u, t, theta):
    """Validate a successful candidate and return the matrix."""
    if denomexp(u + t) < denomexp(u + OMEGA * t):
        t *= OMEGA
    uU = Matrix.array([[u, -np.conj(t)], [t, np.conj(u)]])
    uU_fixed = uU.astype(complex)
    zrot_fixed = qml.RZ.compute_matrix(theta).view(Matrix)
    err = np.sqrt(np.real(hs_sqnorm(uU_fixed - zrot_fixed)) / 2)
    log_err = -np.log2(err)
    return uU, log_err


def hs_sqnorm(m: Matrix):
    """Returns the Hilbert-Schmidt operator norm."""
    return np.trace(m @ m.adjoint())
