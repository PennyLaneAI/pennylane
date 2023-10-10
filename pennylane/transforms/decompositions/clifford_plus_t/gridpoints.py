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

from typing import Tuple, Generator
import numpy as np

from .conversion import (
    action,
    adj2,
    denomexp,
    fatten_interval,
    operator_to_bl2z,
)
from .rings import Matrix, DRootTwo, ZRootTwo, DOmega, SQRT2
from .shapes import ConvexSet, Ellipse, Point

LAMBDA = ZRootTwo(1, 1)
LAMBDA_INV = ZRootTwo(-1, 1)
ROOT_NEG1 = DOmega(0, 1, 0, 0)
log_lambda = lambda x: np.emath.logn(1 + SQRT2, float(x))


def gridpoints2_increasing(region: ConvexSet) -> Generator[Tuple[DOmega, int], None, None]:
    """Returns a list of candidates."""
    unitdisk = create_unitdisk()
    matA = region.ellipse.operator
    matA = matA / np.sqrt(np.linalg.det(matA))
    opG = reduction(matA, unitdisk.ellipse.operator)
    opG_inv = opG.inverse()
    setA = region.transform(opG_inv)
    setB = unitdisk.transform(adj2(opG_inv))
    (x0A, x1A), (y0A, y1A) = setA.ellipse.bounding_box()
    (x0B, x1B), (y0B, y1B) = setB.ellipse.bounding_box()

    def solutions_fn(k):
        """Given a k value, produce results."""
        xs = gridpoints_scaled((x0A, x1A + LAMBDA), (x0B, x1B + LAMBDA), k + 1)
        x0 = next(xs, None)
        if x0 is None:
            return
        dx_inv = ZRootTwo(0, 1) ** k
        dx = 1 / dx_inv
        x0_bul = x0.adj2()
        dx_bul = dx.adj2()
        for beta_ in gridpoints_scaled(fatten_interval(y0A, y1A), fatten_interval(y0B, y1B), k + 1):
            beta_bul = beta_.adj2()
            iA = setA.line_intersector(Point(x0, beta_), Point(dx, 0))
            iB = setB.line_intersector(Point(x0_bul, beta_bul), Point(dx_bul, 0))
            if iA is None or iB is None:
                continue
            t0A, t1A = iA
            t0B, t1B = iB
            dtA = 10 / max(10, 2**k * (t1B - t0B))
            dtB = 10 / max(10, 2**k * (t1A - t0A))
            alpha_offs = gridpoints_scaled_parity(
                (beta_ - x0) * dx_inv, (t0A - dtA, t1A + dtA), (t0B - dtB, t1B + dtB), 1
            )
            alphas = [dx * a + x0 for a in alpha_offs]
            for alpha_ in alphas:
                alpha, beta = Point(alpha_, beta_).transform(opG)
                if region.characteristic_fn(alpha, beta) and unitdisk.characteristic_fn(
                    alpha.adj2(), beta.adj2()
                ):
                    yield DOmega.from_root_two(alpha) + ROOT_NEG1 * DOmega.from_root_two(beta)

    for u in solutions_fn(0):
        yield (u, 0)

    k = 1
    while True:
        for u in solutions_fn(k):
            if denomexp(u) == k:
                yield (u, k)
        k += 1


def create_unitdisk():
    """Function to generate a new unitdisk instance."""
    ell = Ellipse(np.eye(2), Point(0, 0))

    def intersector(p, v):
        """The line intersector of the unitdisk."""
        p, v = (list(p), list(v))
        a = np.inner(v, v)
        b = 2 * np.inner(v, p)
        c = np.inner(p, p) - 1
        q = tuple(r for r in np.roots((a, b, c)) if np.isclose(np.imag(r), 0))
        return sorted(q) if len(q) == 2 else None

    return ConvexSet(ell, lambda x, y: x**2 + y**2 <= 1, intersector)


def gridpoints_scaled(x, y, k) -> Generator[DRootTwo, None, None]:
    """Gridpoints satisfying the scaling criteria so they are in D[omega]"""
    scale_inv = ZRootTwo(0, 1) ** k
    scale = 1 / scale_inv
    x = (scale_inv * x[0], scale_inv * x[1])
    y = (-scale_inv * y[1], -scale_inv * y[0]) if k % 2 else (scale_inv * y[0], scale_inv * y[1])
    a = int(x[0] + y[0]) // 2
    b = int(SQRT2 * (x[0] - y[0])) // 4
    alpha = ZRootTwo(a, b)
    xoff = a + SQRT2 * b
    yoff = a - SQRT2 * b
    x_ = (x[0] - xoff, x[1] - xoff)
    y_ = (y[0] - yoff, y[1] - yoff)
    test = lambda v: x[0] <= v <= x[1] and y[0] <= v.adj2() <= y[1]
    beta = (b + alpha for b in gridpoints_internal(x_, y_))
    return (scale * b for b in beta if test(b))


def gridpoints_scaled_parity(beta, x, y, k):
    """Like gridpoints_scaled, but with k >= 1"""
    if denomexp(beta) <= k - 1:
        return gridpoints_scaled(x, y, k - 1)
    offs = 1 / ZRootTwo(0, 1) ** k
    offs_bul = offs.adj2()
    return (
        z - offs
        for z in gridpoints_scaled(
            (x[0] + offs, x[1] + offs), (y[0] + offs_bul, y[1] + offs_bul), k - 1
        )
    )


def gridpoints_internal(x, y):
    """The actual implementation for gridpoints computation."""
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    if dy <= 0 < dx:
        return (gp.adj2() for gp in gridpoints_internal(x=y, y=x))

    if dy <= 0:
        return iter(())

    n, _ = floorlog(LAMBDA, dy)
    y1, y0 = y if n % 2 else (y[1], y[0])  # swap order if odd

    if dy >= float(LAMBDA):
        lambda_n = LAMBDA**n
        lambda_inv_n = LAMBDA_INV**n
        lambda_bul_n = (-LAMBDA_INV) ** n
        new_x = (lambda_n * x[0], lambda_n * x[1])
        new_y = (lambda_bul_n * y0, lambda_bul_n * y1)
        return (lambda_inv_n * g for g in gridpoints_internal(new_x, new_y))

    if 0 < dy < 1:
        m = -n
        lambda_inv_m = LAMBDA_INV**m
        lambda_m = LAMBDA**m
        lambda_bul_inv_m = (-LAMBDA) ** m
        new_x = (lambda_inv_m * x[0], lambda_inv_m * x[1])
        new_y = (lambda_bul_inv_m * y0, lambda_bul_inv_m * y1)
        return (lambda_m * g for g in gridpoints_internal(new_x, new_y))

    amin = int(np.ceil((x[0] + y[0]) / 2))
    amax = int(x[1] + y[1]) // 2
    bmin = lambda a: int(np.ceil((a - y[1]) / SQRT2))
    bmax = lambda a: int((a - y[0]) // SQRT2)
    return (ZRootTwo(a, b) for a in range(amin, amax + 1) for b in range(bmin(a), bmax(a) + 1))


def floorlog(b, x):
    """The floor of log x (with base b)."""
    if x <= 0:
        raise ValueError("x value must be greater than 0; got:", x)
    if 1 <= x < b:
        return 0, x
    if x < 1 <= (x * b):
        return -1, b * x
    n, r = floorlog(b**2, x)
    if r < b:
        return 2 * n, r
    return 2 * n + 1, r / b


def step_lemma(matA: np.ndarray, matB: np.ndarray) -> Matrix:
    """The Step Lemma from the paper.

    Returns:
        Matrix[DRootTwo]: the result of the Step Lemma
    """
    # pylint:disable=too-many-return-statements
    b, l2z = operator_to_bl2z(matA)
    beta, l2zeta = operator_to_bl2z(matB)

    if beta < -1e-16:
        return wlog_using(matA, matB, Matrix.array([[1, 0], [0, -1]]))

    if l2z * l2zeta < 1:
        return wlog_using(matA, matB, Matrix.array([[0, 1], [1, 0]]))

    l2z_minus_zeta = l2z / l2zeta
    if l2z_minus_zeta > 33.971 or l2z_minus_zeta < 0.029437:
        s_power = int(np.round(log_lambda(l2z_minus_zeta) / 8))
        s_mat = Matrix.array([[LAMBDA**s_power, 0], [0, LAMBDA_INV**s_power]])
        return wlog_using(matA, matB, s_mat)

    if matA[0][1] * matA[1][0] + matB[0][1] * matB[1][0] <= 15:
        return None

    if l2z_minus_zeta > 5.8285 or l2z_minus_zeta < 0.17157:
        return with_shift(matA, matB, int(np.round(log_lambda(l2z_minus_zeta) / 4)))

    if 0.24410 <= l2z <= 4.0968 and 0.24410 <= l2zeta <= 4.0968:
        return Matrix.array([[1, -1], [1, 1]]) / ZRootTwo(0, 1)

    if b >= 0:
        if l2z <= 1.6969:
            return Matrix.array([[-LAMBDA_INV, -1], [LAMBDA, 1]]) / ZRootTwo(0, 1)
        if l2zeta <= 1.6969:
            return adj2(Matrix.array([[-LAMBDA_INV, -1], [LAMBDA, 1]]) / ZRootTwo(0, 1))
        l2c = min(l2z, l2zeta)
        power = max(1, int(np.sqrt(l2c // 4)))
        return Matrix.array([[1, -2 * power], [0, 1]])

    l2c = min(l2z, l2zeta)
    power = max(1, int(np.sqrt(l2c // 2)))
    return Matrix.array([[1, ZRootTwo(0, power)], [0, 1]])


def wlog_using(matA, matB, op):
    """Helper function for Step Lemma."""
    matA, matB = action(matA, matB, op)
    op2 = step_lemma(matA, matB)
    return op if op2 is None else op @ op2


def with_shift(matA, matB, k):
    """Shift helper for Step Lemma."""
    matA = shift_sigma(k, matA)
    matB = shift_tau(k, matB)
    op2 = step_lemma(matA, matB)
    return None if op2 is None else shift_sigma(k, op2)


def shift_sigma(k, op):
    """Shift helper for Step Lemma."""
    (a, b), (c, d) = op
    lambda_k = LAMBDA**k
    return Matrix.array([[a * lambda_k, b], [c, d / lambda_k]])


def shift_tau(k, op):
    """Shift helper for Step Lemma."""
    (a, b), (c, d) = op
    lambda_k = LAMBDA**k
    if k % 2:
        b = -b
        c = -c
    return Matrix.array([[a / lambda_k, b], [c, d * lambda_k]])


def reduction(matA, matB) -> Matrix:
    """Applies the Step Lemma until the skew is 15 or less"""
    if (opG := step_lemma(matA, matB)) is None:
        return Matrix.array([[1, 0], [0, 1]])
    return opG @ reduction(*action(matA, matB, opG))
