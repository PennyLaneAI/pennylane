# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for pennylane/labs/dla/lie_closure_dense.py functionality"""
from functools import partial

import numpy as np

# pylint: disable=too-few-public-methods, protected-access, no-self-use
import pytest

import pennylane as qml
from pennylane import X, Y, Z
from pennylane.labs.dla import AI, AII, AIII, BDI, CI, CII, DIII, ClassB, khaneja_glaser_involution
from pennylane.labs.dla.involutions import Ipq, J, Kpq


class TestMatrixConstructors:
    """Tests for the matrix constructing methods used in Cartan involutions."""

    @pytest.mark.parametrize(
        "num_wires, wire",
        [(N, w) for N in range(5) for w in range(1, N + 1)] + [(N, None) for N in range(5)],
    )
    def test_J_qubit_cases(self, num_wires, wire):
        """Test J matrix constructor for qubit cases."""
        out = J(2**num_wires, wire)
        op = qml.Y(0 if wire is None else wire)
        expected = op.matrix(wire_order=range(num_wires + 1))
        assert np.allclose(out, expected)

    @pytest.mark.parametrize("n", [3, 5, 6, 7])
    def test_J_non_qubit_cases(self, n):
        """Test J matrix constructor for non-qubit cases."""
        out = J(n)
        expected = np.zeros((2 * n, 2 * n), dtype=complex)
        for i in range(n):
            expected[i, n + i] = -1j
            expected[n + i, i] = 1j
        assert np.allclose(out, expected)

    @pytest.mark.parametrize(
        "num_wires, wire",
        [(N, w) for N in range(5) for w in range(1, N + 1)] + [(N, None) for N in range(5)],
    )
    def test_I_qubit_cases(self, num_wires, wire):
        """Test I_pq matrix constructor for qubit cases."""
        p = 2**num_wires
        out = Ipq(p, p, wire)
        op = qml.Z(0 if wire is None else wire)
        expected = op.matrix(wire_order=range(num_wires + 1))
        assert np.allclose(out, expected)

    @pytest.mark.parametrize("p, q", [(1, 3), (2, 5), (3, 2), (6, 6)])
    def test_Ipq_non_qubit_cases(self, p, q):
        """Test I_pq matrix constructor for non-qubit cases."""
        out = Ipq(p, q)
        expected = np.diag(np.concatenate([np.ones(p), -np.ones(q)], axis=0))
        assert np.allclose(out, expected)

    @pytest.mark.parametrize(
        "num_wires, wire",
        [(N, w) for N in range(5) for w in range(1, N + 1)] + [(N, None) for N in range(1, 5)],
    )
    def test_Kpq_qubit_cases(self, num_wires, wire):
        """Test K_pq matrix constructor for qubit cases."""
        p = 2**num_wires
        out = Kpq(p, p, wire)
        op = qml.Z(1 if wire is None else wire)
        expected = op.matrix(wire_order=range(num_wires + 1))
        assert np.allclose(out, expected)

    @pytest.mark.parametrize("p, q", [(1, 3), (2, 5), (3, 2), (6, 6)])
    def test_Kpq_non_qubit_cases(self, p, q):
        """Test K_pq matrix constructor for non-qubit cases."""
        out = Kpq(p, q)
        expected = np.diag(
            np.concatenate([np.ones(p), -np.ones(q), np.ones(p), -np.ones(q)], axis=0)
        )
        assert np.allclose(out, expected)


class TestInvolutionExceptions:
    """Test exceptions being raised by involutions."""

    def test_non_qubit_wire_given(self):
        """Test that an error is raised if the wire is specified but it is a non-qubit case."""
        with pytest.raises(ValueError, match="wire argument is only supported"):
            J(3, wire=0)

        with pytest.raises(ValueError, match="wire argument is only supported"):
            Ipq(3, 5, wire=0)

        with pytest.raises(ValueError, match="wire argument is only supported"):
            Kpq(3, 5, wire=0)

    def test_p_or_q_missing(self):
        """Test that an error is raised if p or q (or both) are not given for the involutions
        AIII, BDI, CII."""
        for p, q in [(None, 2), (5, None), (None, None)]:
            for invol in [AIII, BDI, CII]:
                with pytest.raises(ValueError, match="please specify p and q"):
                    invol(X(0) @ Y(1), p=p, q=q, wire=0)

    def test_Khaneja_Glaser_exceptions(self):
        """Test that the Khaneja-Glaser involution raises custom exceptions related
        to wire and infering p and q."""
        op = qml.X(0) @ qml.Y(1)
        with pytest.raises(ValueError, match="Please specify the wire for the Khaneja"):
            khaneja_glaser_involution(op)

        [op] = op.pauli_rep
        with pytest.raises(ValueError, match="Can't infer p and q from operator of type <class"):
            khaneja_glaser_involution(op, wire=0)


AI_cases = [
    (X(0) @ Y(1) @ Z(2), True),
    (X(0) @ Y(1) @ Z(2) - Z(0) @ X(1) @ Y(2), True),
    (X(0) @ Y(1) @ Z(2) - Y(0) @ Y(1) @ Y(2), True),
    (Y(0) @ Y(1) @ Z(2), False),
    (X(0) @ X(1) @ Z(2) + X(0) @ Y(1) @ Y(2), False),
]

AII_cases = [  # (#_Y is odd, I/Y on wire 0)
    (X(0) @ Y(1) @ Z(2), False),  # (True, False) -> -1
    (X(0) @ Y(1) @ Z(2) - Z(0) @ X(1) @ Y(2), False),  # (True, False) -> -1
    (X(1) @ Y(0) @ Z(2) - Y(0) @ Y(1) @ Y(2), True),  # (True, True) -> 1
    (Y(0) @ Z(2) @ Y(1), False),  # (False, True) -> -1
    (X(1) @ Z(2) @ X(0) + X(0) @ Y(1) @ Y(2), True),  # (False, False) -> 1
]

AIII_cases = [  # I/Z on wire 0?
    (X(0) @ Y(1) @ Z(2), False),
    (X(0) @ Y(1) @ Z(2) - Y(0) @ X(1) @ Y(2), False),
    (Z(0) @ X(1) @ Z(2) - Z(0) @ Y(1) @ Y(2), True),
    (Y(0) @ Y(1) @ Z(2), False),
    (Y(1) @ Z(2), True),
    (Z(0) + 0.2 * Y(1) @ Y(2), True),
]

BDI_cases = AIII_cases  # BDI = AIII

CI_cases = AI_cases  # CI = AI

CII_cases = [  # I/Z on wire 1?
    (X(0) @ Z(1) @ Z(2), True),
    (X(0) @ Z(2) - Z(0) @ Z(1) @ Y(2), True),
    (Z(0) @ X(1) @ Z(2) - Y(0) @ Y(1) @ Y(2), False),
    (Y(0) @ X(1) @ Z(2), False),
    (Y(1) @ Z(2), False),
    (Z(0) + 0.2 * Y(0) @ Y(2), True),
]

DIII_cases = [  # I/Y on wire 0?
    (X(0) @ Y(1) @ Z(2), False),
    (X(0) @ Y(1) @ Z(2) - Z(0) @ X(1) @ Y(2), False),
    (X(1) @ Y(0) @ Z(2) - Y(0) @ Y(1) @ Y(2), True),
    (Y(0) @ Y(1) @ Z(2), True),
    (X(1) @ Z(2), True),
    (Z(0) + 0.2 * X(2) @ X(0), False),
]

ClassB_cases = DIII_cases  # ClassB = DIII


class TestInvolutions:
    """Test the involutions themselves."""

    def run_test_case(self, op, expected, invol):
        """Run a generic test case for a given operator and involution"""
        inputs = [op, op.pauli_rep, qml.matrix(op, wire_order=[0, 1, 2])]
        outputs = [invol(_input) for _input in inputs]
        if expected:
            assert all(outputs)
        else:
            assert not any(outputs)

    @pytest.mark.parametrize("op, expected", AI_cases)
    def test_AI(self, op, expected):
        """Test singledispatch for AI involution"""
        self.run_test_case(op, expected, AI)

    @pytest.mark.parametrize("op, expected", AII_cases)
    def test_AII(self, op, expected):
        """Test singledispatch for AII involution"""
        self.run_test_case(op, expected, AII)

    @pytest.mark.parametrize("op, expected", AIII_cases)
    def test_AIII(self, op, expected):
        """Test singledispatch for AIII involution"""
        self.run_test_case(op, expected, partial(AIII, p=4, q=4))
        # Khaneja-Glaser is just AIII with automatically inferred p and q.
        self.run_test_case(op, expected, partial(khaneja_glaser_involution, wire=0))

    @pytest.mark.parametrize("op, expected", BDI_cases)
    def test_BDI(self, op, expected):
        """Test singledispatch for BDI involution"""
        self.run_test_case(op, expected, partial(BDI, p=4, q=4))

    @pytest.mark.parametrize("op, expected", CI_cases)
    def test_CI(self, op, expected):
        """Test singledispatch for CI involution"""
        self.run_test_case(op, expected, CI)

    @pytest.mark.parametrize("op, expected", CII_cases)
    def test_CII(self, op, expected):
        """Test singledispatch for CII involution"""
        self.run_test_case(op, expected, partial(CII, p=4, q=4))

    @pytest.mark.parametrize("op, expected", DIII_cases)
    def test_DIII(self, op, expected):
        """Test singledispatch for DIII involution"""
        self.run_test_case(op, expected, DIII)

    @pytest.mark.parametrize("op, expected", ClassB_cases)
    def test_ClassB(self, op, expected):
        """Test singledispatch for ClassB involution"""
        self.run_test_case(op, expected, ClassB)
