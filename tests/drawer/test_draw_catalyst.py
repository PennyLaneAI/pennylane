# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test the pennylane drawer with Catalyst."""

# pylint: disable=import-outside-toplevel,protected-access
import pytest

import pennylane as qml

catalyst = pytest.importorskip("catalyst")
mpl = pytest.importorskip("matplotlib")

pytestmark = pytest.mark.external


class TestCatalystDraw:
    """Drawer integration test with Catalyst jitted QNodes."""

    def test_simple_circuit(self):
        """Test a simple circuit that does not use Catalyst features."""

        @qml.qjit
        @qml.qnode(qml.device("lightning.qubit", wires=(0, "a", 1.234)))
        def circuit(x, y, z):
            """A quantum circuit on three wires."""
            qml.RX(x, wires=0)
            qml.RY(y, wires="a")
            qml.RZ(z, wires=1.234)
            return qml.expval(qml.PauliZ(0))

        expected = "    0: ──RX─┤  <Z>\n    a: ──RY─┤     \n1.234: ──RZ─┤     "
        assert qml.draw(circuit, decimals=None)(1.234, 2.345, 3.456) == expected

    @pytest.mark.parametrize("c", [0, 1])
    def test_cond_circuit(self, c):
        """Test a circuit with a Catalyst conditional."""

        import catalyst  # pylint: disable=redefined-outer-name

        @qml.qjit
        @qml.qnode(qml.device("lightning.qubit", wires=(0, "a", 1.234)))
        def circuit(x, y, z, c):
            """A quantum circuit on three wires."""

            @catalyst.cond(c)
            def conditional_flip():
                qml.PauliX(wires=0)

            qml.RX(x, wires=0)
            conditional_flip()
            qml.RY(y, wires="a")
            qml.RZ(z, wires=1.234)
            return qml.expval(qml.PauliZ(0))

        expected = [
            "    0: ──RX─┤  <Z>\n    a: ──RY─┤     \n1.234: ──RZ─┤     ",
            "    0: ──RX──X─┤  <Z>\n    a: ──RY────┤     \n1.234: ──RZ────┤     ",
        ]
        assert qml.draw(circuit, decimals=None)(1.234, 2.345, 3.456, c) == expected[c]

    @pytest.mark.parametrize("c", [1, 2])
    def test_for_loop_circuit(self, c):
        """Test a circuit with a Catalyst for_loop"""

        import catalyst  # pylint: disable=redefined-outer-name

        @qml.qjit
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit(x, y, z, c):
            """A quantum circuit on three wires."""

            @catalyst.for_loop(0, c, 1)
            def loop(i):
                qml.Hadamard(wires=i)

            qml.RX(x, wires=0)
            loop()  # pylint: disable=no-value-for-parameter
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            return qml.expval(qml.PauliZ(0))

        expected = [
            "0: ──RX──H─┤  <Z>\n1: ──RY────┤     \n2: ──RZ────┤     ",
            "0: ──RX──H──┤  <Z>\n1: ──H───RY─┤     \n2: ──RZ─────┤     ",
        ]
        assert qml.draw(circuit, decimals=None)(1.234, 2.345, 3.456, c) == expected[c - 1]

    @pytest.mark.parametrize("c", [0, 1])
    def test_while_loop_circuit(self, c):
        """Test a circuit with a Catalyst while_loop"""

        import catalyst  # pylint: disable=redefined-outer-name

        @qml.qjit
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit(x, y, z, c):
            """A quantum circuit on three wires."""

            @catalyst.while_loop(lambda x: x < 2.0)
            def loop_rx(x):
                # perform some work and update (some of) the arguments
                qml.RX(x, wires=0)
                return x + 1

            # apply the while loop
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            loop_rx(c)
            qml.RZ(z, wires=2)
            return qml.expval(qml.PauliZ(0))

        expected = [
            "0: ──RX──RX──RX─┤  <Z>\n1: ──RY─────────┤     \n2: ──RZ─────────┤     ",
            "0: ──RX──RX─┤  <Z>\n1: ──RY─────┤     \n2: ──RZ─────┤     ",
        ]
        assert qml.draw(circuit, decimals=None)(1.234, 2.345, 3.456, c) == expected[c]


class TestCatalystDrawMpl:
    """MPL Drawer integration test with Catalyst jitted QNodes."""

    def test_simple_circuit(self):
        """Test a simple circuit that does not use Catalyst features."""

        @qml.qjit
        @qml.qnode(qml.device("lightning.qubit", wires=(0, "a", 1.234)))
        def circuit(x, y, z):
            """A quantum circuit on three wires."""
            qml.RX(x, wires=0)
            qml.RY(y, wires="a")
            qml.RZ(z, wires=1.234)
            return qml.expval(qml.PauliZ(0))

        fig, ax = qml.draw_mpl(circuit, decimals=None)(1.234, 2.345, 3.456)
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, mpl.axes._axes.Axes)

    @pytest.mark.parametrize("c", [0, 1])
    def test_cond_circuit(self, c):
        """Test a circuit with a Catalyst conditional."""

        import catalyst  # pylint: disable=redefined-outer-name

        @qml.qjit
        @qml.qnode(qml.device("lightning.qubit", wires=(0, "a", 1.234)))
        def circuit(x, y, z, c):
            """A quantum circuit on three wires."""

            @catalyst.cond(c)
            def conditional_flip():
                qml.PauliX(wires=0)

            qml.RX(x, wires=0)
            conditional_flip()
            qml.RY(y, wires="a")
            qml.RZ(z, wires=1.234)
            return qml.expval(qml.PauliZ(0))

        fig, ax = qml.draw_mpl(circuit, decimals=None)(1.234, 2.345, 3.456, c)
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, mpl.axes._axes.Axes)

    @pytest.mark.parametrize("c", [1, 2])
    def test_for_loop_circuit(self, c):
        """Test a circuit with a Catalyst for_loop"""

        import catalyst  # pylint: disable=redefined-outer-name

        @qml.qjit
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit(x, y, z, c):
            """A quantum circuit on three wires."""

            @catalyst.for_loop(0, c, 1)
            def loop(i):
                qml.Hadamard(wires=i)

            qml.RX(x, wires=0)
            loop()  # pylint: disable=no-value-for-parameter
            qml.RY(y, wires=1)
            qml.RZ(z, wires=2)
            return qml.expval(qml.PauliZ(0))

        fig, ax = qml.draw_mpl(circuit, decimals=None)(1.234, 2.345, 3.456, c)
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, mpl.axes._axes.Axes)

    @pytest.mark.parametrize("c", [0, 1])
    def test_while_loop_circuit(self, c):
        """Test a circuit with a Catalyst while_loop"""

        import catalyst  # pylint: disable=redefined-outer-name

        @qml.qjit
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit(x, y, z, c):
            """A quantum circuit on three wires."""

            @catalyst.while_loop(lambda x: x < 2.0)
            def loop_rx(x):
                # perform some work and update (some of) the arguments
                qml.RX(x, wires=0)
                return x + 1

            # apply the while loop
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            loop_rx(c)
            qml.RZ(z, wires=2)
            return qml.expval(qml.PauliZ(0))

        fig, ax = qml.draw_mpl(circuit, decimals=None)(1.234, 2.345, 3.456, c)
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, mpl.axes._axes.Axes)
