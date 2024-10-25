# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Contains the implementation of the angle solver for GQSP, QSP and QSVT.
"""

import pennylane as qml
import numpy as np
from numpy.polynomial import Polynomial, chebyshev


def get_complementary_poly(P, precision = 1e-7):
    # TODO :  Update (QLTR)
    r"""Computes the Q polynomial given P

        Computes polynomial $Q$ of degree at-most that of $P$, satisfying

            $$ \abs{P(e^{i\theta})}^2 + \abs{Q(e^{i\theta})}^2 = 1 $$

        for every $\theta \in \mathbb{R}$.

        The exact method for computing $Q$ is described in the proof of Theorem 4.
        The method computes an auxillary polynomial R, whose roots are computed
        and re-interpolated to obtain the required polynomial Q.

        Args:
            P: Co-efficients of a complex polynomial.
            verify: sanity check the computed polynomial roots (defaults to False).
            verify_precision: precision to compare values while verifying

        References:
            [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
                Motlagh and Wiebe. (2023). Theorem 4.
        """
    d = len(P) - 1  # degree

    # R(z) = z^d (1 - P^*(z) P(z))
    # obtained on simplifying Eq. 34, Eq. 35 by substituting H, T.
    # The definition of $R$ is given in the text from Eq.34 - Eq. 35,
    # following the chain of definitions below:
    #
    #     $$
    #     T(\theta) = \abs{P(e^{i\theta}),
    #     H = I - T,
    #     H = e^{-id\theta} R(e^{i\theta})
    #     $$
    #
    # Here H and T are defined on reals, so the initial definition of R is only on the unit circle.
    # We analytically continue this definition to the entire complex plane by replacing $e^{i\theta}$ by $z$.
    R = Polynomial.basis(d) - Polynomial(P) * Polynomial(np.conj(P[::-1]))
    roots = R.roots()

    # R is self-inversive, so larger_roots and smaller_roots occur in conjugate pairs.
    units: list[complex] = []  # roots r s.t. \abs{r} = 1
    larger_roots: list[complex] = []  # roots r s.t. \abs{r} > 1
    smaller_roots: list[complex] = []  # roots r s.t. \abs{r} < 1

    for r in roots:
        if np.allclose(np.abs(r), 1):
            units.append(r)
        elif np.abs(r) > 1:
            larger_roots.append(r)
        else:
            smaller_roots.append(r)

    # pair up roots in `units`, claimed in Eq. 40 and the explanation preceding it.
    # all unit roots must have even multiplicity.
    paired_units: list[complex] = []
    unpaired_units: list[complex] = []
    for z in units:
        matched_z = None
        for w in unpaired_units:
            if np.allclose(z, w, rtol=precision):
                matched_z = w
                break

        if matched_z is not None:
            paired_units.append(z)
            unpaired_units.remove(matched_z)
        else:
            unpaired_units.append(z)


    # Q = G \hat{G}, where
    # - \hat{G}^2 is the monomials which are unit roots of R, which occur in pairs.
    # - G*(z) G(z) is the interpolation of the conjugate paired non-unit roots of R,
    #   described in Eq. 37 - Eq. 38

    # Leading co-efficient of R described in Eq. 37.
    # Note: In the paper, the polynomial is interpolated from larger_roots,
    #       but this is swapped in our implementation to reduce the error in Q.
    c = R.coef[-1]
    scaling_factor = np.sqrt(np.abs(c * np.prod(larger_roots)))

    Q = scaling_factor * Polynomial.fromroots(paired_units + smaller_roots)

    return Q.coef

def get_QSP_angles(P):

    parity = (len(P) - 1) % 2

    aux_poly = np.concatenate([P[1 - parity::2], chebyshev.poly2cheb(P)[parity::2]])
    aux_complementary = np.array(get_complementary_poly(aux_poly))

    S = np.array([aux_poly, aux_complementary])
    n = S.shape[1]

    theta = np.zeros(n)

    def RY_matrix(theta):
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ])

    for d in reversed(range(n)):
        a, b = S[:, d]
        theta[d] = np.arctan(b.real / a.real)

        S = RY_matrix(-theta[d]) @ S
        S = np.array([S[0][1: d + 1], S[1][0:d]])

    if P[-1] < 0:
        theta[0] += np.pi
    return theta


def get_GQSP_angles(P):

    Q = get_complementary_poly(P)
    if len(P) != len(Q):
        raise ValueError("Polynomials P and Q must have the same degree.")

    S = np.array([P, Q])
    n = S.shape[1]

    theta = np.zeros(n)
    phi = np.zeros(n)
    lambd = 0

    def U3_paper(theta, phi, lambd):

        exp_phi = np.exp(1j * phi)
        exp_lambda = np.exp(1j * lambd)
        exp_lambda_phi = np.exp(1j * (lambd + phi))

        R = np.array([
            [exp_lambda_phi * np.cos(theta), exp_phi * np.sin(theta)],
            [exp_lambda * np.sin(theta), -np.cos(theta)]
        ], dtype=complex)

        return R

    def safe_angle(x):
        return 0 if np.isclose(x, 0, atol=1e-10) else np.angle(x)

    for d in reversed(range(n)):
        assert S.shape == (2, d + 1)

        a, b = S[:, d]
        theta[d] = np.arctan2(np.abs(b), np.abs(a))
        # \phi_d = arg(a / b)
        phi[d] = 0 if np.isclose(np.abs(b), 0, atol=1e-10) else safe_angle(a * np.conj(b))

        if d == 0:
            lambd = safe_angle(b)
        else:
            S = U3_paper(theta[d], phi[d], 0).conj().T @ S
            S = np.array([S[0][1: d + 1], S[1][0:d]])

    return theta, phi, lambd

def transform_angles(angles, routine1, routine2):

    if routine1 == "QSP" and routine2 == "QSVT":
        num_angles = len(angles)
        update_vals = np.zeros(num_angles)

        update_vals[0] = 3 * np.pi / 4 - (3 + len(angles) % 4) * np.pi / 2
        update_vals[1:-1] = np.pi / 2
        update_vals[-1] = -np.pi / 4

        return angles + update_vals

    if routine1 == "QSVT" and routine2 == "QSP":
        num_angles = len(angles)
        update_vals = np.zeros(num_angles)

        update_vals[0] = 3 * np.pi / 4 - (3 + len(angles) % 4) * np.pi / 2
        update_vals[1:-1] = np.pi / 2
        update_vals[-1] = -np.pi / 4

        return angles - update_vals

    if routine1 == "QSP" and routine2 == "GQSP":
        thetas = angles
        thetas[0] += np.pi
        return [thetas, np.pi * np.ones(len(angles)), 0]


def poly_to_angles(P, routine):

    if routine == "QSP":
        return get_QSP_angles(P)

    if routine == "QSVT":
        return transform_angles(get_QSP_angles(P), "QSP", "QSVT")

    if routine == "GQSP":
        return get_GQSP_angles(P)


def test_QSVT(poly):

    angles = get_QSVT_angles(poly)
    x = 0.5

    block_encoding = qml.RX(-2 * np.arccos(x), wires=0)
    projectors = [qml.PCPhase(angle, dim=1, wires=0) for angle in angles]

    @qml.qnode(qml.device("default.qubit"))
    def circuit():
        """
        qml.RX(2*angles[0], wires = 0)

        for ind, angle in enumerate(angles[1:]):

            qml.RZ(-2* np.arccos(x), wires = 0)

            qml.RX(2*angle, wires = 0)

        """
        qml.QSVT(block_encoding, projectors)

        return qml.state()

    output = qml.matrix(circuit, wire_order=[0])()[0, 0]
    #print("con x ", sum(coef * (x ** i) for i, coef in enumerate(poly)))

    z = qml.matrix(qml.RZ(-2 * np.arccos(x), wires=0))[0, 0]
    # print("con z ", np.round(sum(coef * (z ** i) for i, coef in enumerate(poly)),5))

    #print("output", output)
    if not np.isclose(sum(coef * (x ** i) for i, coef in enumerate(poly)), output.real):
        print(sum(coef * (x ** i) for i, coef in enumerate(poly)), output.real)

"""
for seed in range(100):
    d = np.random.randint(500) + 1
    print(f"test {seed + 1} with degree {2*d}")
    np.random.seed(seed)
    coeffs = np.random.randn(d)/d
    poly = [0.001]

    for c in coeffs:
        poly.append(0)
        poly.append(c)

    #print(np.round(poly,2))
    test_QSVT(poly)


poly = [0, 0.2, 0, 0.3]

print(get_GQSP_angles(poly))

"""




