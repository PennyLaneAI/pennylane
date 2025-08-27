# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit test module for the draw function in the Python Compiler visualization module."""


import pytest

pytestmark = pytest.mark.external

pytest.importorskip("xdsl")
pytest.importorskip("catalyst")


# pylint: disable=wrong-import-position
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.visualization import draw

# pylint: disable=implicit-str-concat


@pytest.mark.usefixtures("enable_disable_plxpr")
class Testdraw:
    """Unit tests for the draw function in the Python Compiler visualization module."""

    @pytest.fixture
    def transforms_circuit(self):
        """Fixture for a circuit."""

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circ():
            qml.RX(1, 0)
            qml.RX(2.0, 0)
            qml.RY(3.0, 1)
            qml.RY(4.0, 1)
            qml.RZ(5.0, 2)
            qml.RZ(6.0, 2)
            qml.Hadamard(0)
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            qml.CNOT([0, 1])
            qml.Hadamard(1)
            qml.Hadamard(1)
            qml.RZ(7.0, 0)
            qml.RZ(8.0, 0)
            qml.CNOT([0, 2])
            qml.CNOT([0, 2])
            return qml.state()

        return circ

    @pytest.mark.parametrize("qjit", [True, False])
    @pytest.mark.parametrize(
        "level, expected",
        [
            (
                0,
                "0: ──RX──RX──H──H─╭●─╭●──RZ──RZ─╭●─╭●─┤  State\n"
                "1: ──RY──RY───────╰X─╰X──H───H──│──│──┤  State\n"
                "2: ──RZ──RZ─────────────────────╰X─╰X─┤  State",
            ),
            (
                1,
                "0: ──RX──H──H─╭●─╭●──RZ────╭●─╭●─┤  State\n"
                "1: ──RY───────╰X─╰X──H───H─│──│──┤  State\n"
                "2: ──RZ────────────────────╰X─╰X─┤  State",
            ),
            (2, "0: ──RX──RZ─┤  State\n" "1: ──RY─────┤  State\n" "2: ──RZ─────┤  State"),
            (None, "0: ──RX──RZ─┤  State\n" "1: ──RY─────┤  State\n" "2: ──RZ─────┤  State"),
            (50, "0: ──RX──RZ─┤  State\n" "1: ──RY─────┤  State\n" "2: ──RZ─────┤  State"),
        ],
    )
    def test_multiple_levels_xdsl(self, transforms_circuit, level, qjit, expected):
        """Test that multiple levels of transformation are applied correctly with xDSL compilation passes."""

        transforms_circuit = qml.compiler.python_compiler.transforms.merge_rotations_pass(
            qml.compiler.python_compiler.transforms.iterative_cancel_inverses_pass(
                transforms_circuit
            )
        )

        if qjit:
            transforms_circuit = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(
                transforms_circuit
            )

        assert draw(transforms_circuit, level=level)() == expected

    @pytest.mark.parametrize("qjit", [True, False])
    @pytest.mark.parametrize(
        "level, expected",
        [
            (
                0,
                "0: ──RX──RX──H──H─╭●─╭●──RZ──RZ─╭●─╭●─┤  State\n"
                "1: ──RY──RY───────╰X─╰X──H───H──│──│──┤  State\n"
                "2: ──RZ──RZ─────────────────────╰X─╰X─┤  State",
            ),
            (
                1,
                "0: ──RX──H──H─╭●─╭●──RZ────╭●─╭●─┤  State\n"
                "1: ──RY───────╰X─╰X──H───H─│──│──┤  State\n"
                "2: ──RZ────────────────────╰X─╰X─┤  State",
            ),
            (2, "0: ──RX──RZ─┤  State\n" "1: ──RY─────┤  State\n" "2: ──RZ─────┤  State"),
            (None, "0: ──RX──RZ─┤  State\n" "1: ──RY─────┤  State\n" "2: ──RZ─────┤  State"),
            (50, "0: ──RX──RZ─┤  State\n" "1: ──RY─────┤  State\n" "2: ──RZ─────┤  State"),
        ],
    )
    def test_multiple_levels_catalyst(self, transforms_circuit, level, qjit, expected):
        """Test that multiple levels of transformation are applied correctly with Catalyst compilation passes."""

        transforms_circuit = qml.transforms.merge_rotations(
            qml.transforms.cancel_inverses(transforms_circuit)
        )

        if qjit:
            transforms_circuit = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(
                transforms_circuit
            )

        assert draw(transforms_circuit, level=level)() == expected

    @pytest.mark.parametrize("qjit", [True, False])
    @pytest.mark.parametrize(
        "level, expected",
        [
            (
                0,
                "0: ──RX──RX──H──H─╭●─╭●──RZ──RZ─╭●─╭●─┤  State\n"
                "1: ──RY──RY───────╰X─╰X──H───H──│──│──┤  State\n"
                "2: ──RZ──RZ─────────────────────╰X─╰X─┤  State",
            ),
            (
                1,
                "0: ──RX──H──H─╭●─╭●──RZ────╭●─╭●─┤  State\n"
                "1: ──RY───────╰X─╰X──H───H─│──│──┤  State\n"
                "2: ──RZ────────────────────╰X─╰X─┤  State",
            ),
            (2, "0: ──RX──RZ─┤  State\n" "1: ──RY─────┤  State\n" "2: ──RZ─────┤  State"),
            (None, "0: ──RX──RZ─┤  State\n" "1: ──RY─────┤  State\n" "2: ──RZ─────┤  State"),
            (50, "0: ──RX──RZ─┤  State\n" "1: ──RY─────┤  State\n" "2: ──RZ─────┤  State"),
        ],
    )
    def test_multiple_levels_xdsl_catalyst(self, transforms_circuit, level, qjit, expected):
        """Test that multiple levels of transformation are applied correctly with xDSL and Catalyst compilation passes."""

        transforms_circuit = qml.transforms.merge_rotations(
            qml.compiler.python_compiler.transforms.iterative_cancel_inverses_pass(
                transforms_circuit
            )
        )
        if qjit:
            transforms_circuit = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(
                transforms_circuit
            )

        assert draw(transforms_circuit, level=level)() == expected

    @pytest.mark.parametrize("qjit", [True, False])
    @pytest.mark.parametrize(
        "level, expected",
        [
            (
                0,
                "0: ──RX──RX──H──H─╭●─╭●──RZ──RZ─╭●─╭●─┤  State\n"
                "1: ──RY──RY───────╰X─╰X──H───H──│──│──┤  State\n"
                "2: ──RZ──RZ─────────────────────╰X─╰X─┤  State",
            ),
            (
                1,
                "0: ──RX──RX──H──H─╭●─╭●──RZ──RZ─╭●─╭●─┤  State\n"
                "1: ──RY──RY───────╰X─╰X──H───H──│──│──┤  State\n"
                "2: ──RZ──RZ─────────────────────╰X─╰X─┤  State",
            ),
            (
                2,
                "0: ──RX──RX──H──H─╭●─╭●──RZ──RZ─╭●─╭●─┤  State\n"
                "1: ──RY──RY───────╰X─╰X──H───H──│──│──┤  State\n"
                "2: ──RZ──RZ─────────────────────╰X─╰X─┤  State",
            ),
            (
                None,
                "0: ──RX──RX──H──H─╭●─╭●──RZ──RZ─╭●─╭●─┤  State\n"
                "1: ──RY──RY───────╰X─╰X──H───H──│──│──┤  State\n"
                "2: ──RZ──RZ─────────────────────╰X─╰X─┤  State",
            ),
            (
                50,
                "0: ──RX──RX──H──H─╭●─╭●──RZ──RZ─╭●─╭●─┤  State\n"
                "1: ──RY──RY───────╰X─╰X──H───H──│──│──┤  State\n"
                "2: ──RZ──RZ─────────────────────╰X─╰X─┤  State",
            ),
        ],
    )
    def test_no_passes(self, transforms_circuit, level, qjit, expected):
        """Test that if no passes are applied, the circuit is still visualized."""

        if qjit:
            transforms_circuit = qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])(
                transforms_circuit
            )

        assert draw(transforms_circuit, level=level)() == expected

    def test_adjoint(self):
        """Test that the adjoint operation is visualized correctly."""

        @qml.compiler.python_compiler.transforms.merge_rotations_pass
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circ():
            qml.adjoint(qml.RX(0.1, wires=0))
            qml.adjoint(qml.Hadamard(wires=0))
            return qml.state()

        assert draw(circ)() == "0: ──RX†──H†─┤  State"

    def test_ctrl(self):
        """Test that the controlled operation is visualized correctly."""

        @qml.compiler.python_compiler.transforms.merge_rotations_pass
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circ():
            qml.ctrl(qml.RX(0.1, wires=0), control=(1, 2, 3))
            return qml.state()

        assert (
            draw(circ)() == "1: ─╭●──┤  State\n2: ─├●──┤  State\n3: ─├●──┤  State\n0: ─╰RX─┤  State"
        )

    def test_ctrl_with_ctrl_values(self):
        """Test that the controlled operation with control values is visualized correctly."""

        @qml.compiler.python_compiler.transforms.merge_rotations_pass
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circ():
            qml.ctrl(qml.RX(0.1, wires=0), control=(1, 2, 3), control_values=(0, 1, 0))
            return qml.state()

        assert (
            draw(circ)() == "1: ─╭○──┤  State\n2: ─├●──┤  State\n3: ─├○──┤  State\n0: ─╰RX─┤  State"
        )

    def test_adjoint_ctrl(self):
        """Test that the adjoint controlled operation is visualized correctly."""

        @qml.compiler.python_compiler.transforms.merge_rotations_pass
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circ():
            qml.adjoint(qml.ctrl(qml.RX(0.1, wires=0), (1, 2, 3), control_values=(0, 1, 0)))
            return qml.state()

        assert (
            draw(circ)()
            == "1: ─╭○───┤  State\n2: ─├●───┤  State\n3: ─├○───┤  State\n0: ─╰RX†─┤  State"
        )

    def test_ctrl_adjoint(self):
        """Test that the controlled adjoint operation is visualized correctly."""

        @qml.compiler.python_compiler.transforms.merge_rotations_pass
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circ():
            qml.ctrl(qml.adjoint(qml.RX(0.1, wires=0)), (1, 2, 3), control_values=(0, 1, 0))
            return qml.state()

        assert (
            draw(circ)()
            == "1: ─╭○───┤  State\n2: ─├●───┤  State\n3: ─├○───┤  State\n0: ─╰RX†─┤  State"
        )

    def test_probs_meas(self):
        """Test that the probability measurement is visualized correctly."""

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circ():
            qml.RX(0.1, 0)
            qml.RY(0.2, 1)
            qml.RZ(0.3, 2)
            return qml.probs(0), qml.probs(1), qml.probs(2)

        assert draw(circ)() == "0: ──RX─┤  Probs\n1: ──RY─┤  Probs\n2: ──RZ─┤  Probs"

    def test_probs_meas_2(self):
        """Test that the probability measurement is visualized correctly."""

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circ():
            qml.RX(0.1, 0)
            qml.RY(0.2, 1)
            qml.RZ(0.3, 2)
            return qml.probs()

        assert draw(circ)() == "0: ──RX─┤  Probs\n1: ──RY─┤  Probs\n2: ──RZ─┤  Probs"

    def test_expval_meas(self):
        """Test that the expectation value measurement is visualized correctly."""

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circ():
            qml.RX(0.1, 0)
            qml.RY(0.2, 1)
            qml.RZ(0.3, 2)
            return qml.expval(qml.X(0)), qml.expval(qml.Y(1)), qml.expval(qml.Z(2))

        assert draw(circ)() == "0: ──RX─┤  <X>\n1: ──RY─┤  <Y>\n2: ──RZ─┤  <Z>"


if __name__ == "__main__":
    pytest.main(["-x", __file__])
