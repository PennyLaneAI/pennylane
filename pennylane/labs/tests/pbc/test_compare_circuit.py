# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Tests for PBC functionality
"""

import numpy as np
import pytest

import pennylane as qp
from pennylane import X, Y, Z
from pennylane.labs.pbc import compare_circuits, controlled, measure, ppr


class TestCompareCircuits:
    """Tests for compare_circuits"""

    def test_CNOT(self):
        """Basic test that CNOT works as expected"""

        def circuit1():
            qp.CNOT((0, 1))

        def circuit2():
            controlled(qp.Z(0), qp.X(1))

        assert compare_circuits(
            circuit1,
            circuit2,
            wires=range(2),
        ) == (True, "up to global phase -0.2500π")

    @pytest.mark.parametrize(
        "P1, P2, Pdash",
        [
            (X(0), Z(1), Y(0)),  # note that the P2 does not really matter here
            (X(0), Z(1) @ Y(2), Y(0)),
            (X(0), Z(2) @ Y(3), Y(0) @ X(1)),
        ],
    )
    def test_ppr_control_commutation_non_commuting(self, P1, P2, Pdash):
        r"""
        Test case P1 and P' dont commute
           +----+   +----+ +---)           +------+         +----+
        ---|    |---|    |-|   |--      ---|      |---------|    |---
        ---| P1 |---| P' |-| φ |--      ---|  P'  |---------| P1 |---
        ---|    |---|    |-|   |--      ---|      |--+---)--|    |---
           +----+   +----+ +---)           |      |  |   |  +----+
             |                      =      |      |  | φ |    |
           +----+                          |      |  |   |  +----+
        ---|    |-----------------      ---|      |--+---)--|    |---
        ---| P2 |-----------------      ---|  P2  |---------| P2 |---
        ---|    |-----------------      ---|      |---------|    |---
           +----+                          +------+         +----+
        """

        def circuit1(P1, P2, Pdash):
            controlled(P1, P2)
            ppr(0.5, Pdash)

        def circuit2(P1, P2, Pdash):
            ppr(0.5, Pdash @ P2)
            controlled(P1, P2)

        wires = qp.wires.Wires.all_wires([P1.wires, P2.wires])

        assert compare_circuits(circuit1, circuit2, wires=wires, P1=P1, P2=P2, Pdash=Pdash) == (
            True,
            "exact",
        )

    @pytest.mark.parametrize(
        "P1, P2, Pdash",
        [
            (X(0), Z(1), X(0)),  # note that the P2 does not really matter here
            (Y(0), Z(1) @ Y(2), Y(0)),
            (X(0) @ X(1), Z(2) @ Y(3), Y(0) @ Y(1)),
        ],
    )
    def test_ppr_control_commutation_commuting(self, P1, P2, Pdash):
        r"""
        Test case P1 and P' do commute
        """

        def circuit1(P1, P2, Pdash):
            controlled(P1, P2)
            ppr(0.5, Pdash)

        def circuit2(P1, P2, Pdash):
            ppr(0.5, Pdash)
            controlled(P1, P2)

        wires = qp.wires.Wires.all_wires([P1.wires, P2.wires])

        assert compare_circuits(circuit1, circuit2, wires=wires, P1=P1, P2=P2, Pdash=Pdash) == (
            True,
            "exact",
        )

    @pytest.mark.parametrize("P, Pdash", ((Z(0) @ Z(1), X(0) @ X(1)),))
    def test_ppr_pauli_measure_commutation(self, P, Pdash):
        """Test that ppr and pauli_measure commute"""

        def circuit1(P, Pdash):
            ppr(np.pi / 2, P)
            measure(Pdash)

        def circuit2(P, Pdash):
            measure(Pdash)  # when they commute
            ppr(np.pi / 2, P)

        wires = qp.wires.Wires.all_wires([P.wires, Pdash.wires])

        assert compare_circuits(circuit1, circuit2, wires=wires, P=P, Pdash=Pdash) == (
            True,
            "exact",
        )

    @pytest.mark.parametrize("P, Pdash", ((Z(0), X(0)),))
    def test_ppr_pauli_measure_non_commuting(self, P, Pdash):
        """Test that ppr and pauli_measure yield new operation iPP' when they dont commute"""

        def circuit1(P, Pdash):
            ppr(np.pi / 2, P)
            measure(Pdash)

        def circuit2(P, Pdash):
            measure(1j * P @ Pdash)  # when they commute
            ppr(np.pi / 2, P)

        wires = qp.wires.Wires.all_wires([P.wires, Pdash.wires])

        assert compare_circuits(circuit1, circuit2, wires=wires, P=P, Pdash=Pdash) == (
            True,
            "exact",
        )

    @pytest.mark.parametrize(
        "P1, P3, P2, P4",
        (
            (X(0), Z(0), X(1), X(1)),
            (X(0), Z(0), X(1) @ X(2), Z(1) @ Z(2)),
            (X(0), Z(0), X(1) @ X(2), Y(1) @ Y(2)),
            (X(0), Z(0), X(1) @ X(2), X(1) @ X(2)),
            (X(0), Y(0), X(1) @ X(2), X(1) @ X(2)),
            (X("a") @ X(0), Y(0), X(1) @ X(2), X(1) @ X(2)),
            (X("a") @ X(0), Z("a") @ X(0), X(1) @ X(2), X(1) @ X(2)),
        ),
    )
    def test_controlled_measure_commutation(self, P1, P2, P3, P4):
        """Check commutation relation between a control and measurement when the following commutation relations hold

        0: ─╭P1●─╭P3↗─┤  = ─╭───P3↗───╭P1●─┤
        1: ─╰P2○─╰P4↗─┤  = ─╰P2@P4↗───╰P2○─┤

        holds when
        * [P1, P3] != 0
        * [P2, P4] = 0
        """

        def circuit1(P1, P2, P3, P4):
            controlled(P1, P2)
            measure(P3 @ P4)

        def circuit2(P1, P2, P3, P4):
            measure(P3 @ P2 @ P4)
            controlled(P1, P2)

        assert qp.commutator(P1, P3) != qp.simplify(0 * P1 @ P3), "P1 and P3 need to anti-commute"
        assert qp.commutator(P2, P4) == qp.simplify(0 * P2 @ P4), "P2 and P4 need to commute"
        wires = qp.wires.Wires.all_wires([P1.wires, P2.wires, P3.wires, P4.wires])

        assert compare_circuits(
            circuit1, circuit2, wires=wires, P1=P1, P2=P2, P3=P3, P4=P4, verbose=True
        ) == (True, "exact")
