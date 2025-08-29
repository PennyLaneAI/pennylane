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

# pylint: disable=implicit-str-concat, unnecessary-lambda


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

    @pytest.mark.parametrize(
        "op, kwargs, expected",
        [
            (
                lambda: qml.ctrl(qml.RX(0.1, 0), control=(1, 2, 3)),
                {},
                "1: ─╭●──┤  State\n2: ─├●──┤  State\n3: ─├●──┤  State\n0: ─╰RX─┤  State",
            ),
            (
                lambda: qml.ctrl(qml.RX(0.1, 0), control=(1, 2, 3), control_values=(0, 1, 0)),
                {},
                "1: ─╭○──┤  State\n2: ─├●──┤  State\n3: ─├○──┤  State\n0: ─╰RX─┤  State",
            ),
            (
                lambda: qml.adjoint(qml.ctrl(qml.RX(0.1, 0), (1, 2, 3), control_values=(0, 1, 0))),
                {},
                "1: ─╭○───┤  State\n2: ─├●───┤  State\n3: ─├○───┤  State\n0: ─╰RX†─┤  State",
            ),
            (
                lambda: qml.ctrl(qml.adjoint(qml.RX(0.1, 0)), (1, 2, 3), control_values=(0, 1, 0)),
                {},
                "1: ─╭○───┤  State\n2: ─├●───┤  State\n3: ─├○───┤  State\n0: ─╰RX†─┤  State",
            ),
        ],
    )
    def test_ctrl_adjoint_variants(self, op, kwargs, expected):
        """
        Test the visualization of control and adjoint variants.
        """

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def _():
            op()
            return qml.state()

        assert draw(_)(**kwargs) == expected

    @pytest.mark.parametrize(
        "measurement, expected",
        [
            (
                lambda: (qml.probs(0), qml.probs(1), qml.probs(2)),
                "0: ──RX─┤  Probs\n1: ──RY─┤  Probs\n2: ──RZ─┤  Probs",
            ),
            (
                lambda: qml.probs(),
                "0: ──RX─┤  Probs\n1: ──RY─┤  Probs\n2: ──RZ─┤  Probs",
            ),
            (
                lambda: qml.sample(),
                "0: ──RX─┤  Sample\n1: ──RY─┤  Sample\n2: ──RZ─┤  Sample",
            ),
            (
                lambda: (qml.expval(qml.X(0)), qml.expval(qml.Y(1)), qml.expval(qml.Z(2))),
                "0: ──RX─┤  <X>\n1: ──RY─┤  <Y>\n2: ──RZ─┤  <Z>",
            ),
            (
                lambda: (
                    qml.expval(qml.X(0) @ qml.Y(1)),
                    qml.expval(qml.Y(1) @ qml.Z(2) @ qml.X(0)),
                    qml.expval(qml.Z(2) @ qml.X(0) @ qml.Y(1)),
                ),
                "0: ──RX─┤ ╭<X@Y> ╭<Y@Z@X> ╭<Z@X@Y>\n1: ──RY─┤ ╰<X@Y> ├<Y@Z@X> ├<Z@X@Y>\n2: ──RZ─┤        ╰<Y@Z@X> ╰<Z@X@Y>",
            ),
            (
                lambda: (
                    qml.expval(
                        qml.Hamiltonian([0.2, 0.2], [qml.PauliX(0), qml.Y(1)])
                        @ qml.Hamiltonian([0.1, 0.1], [qml.PauliZ(2), qml.PauliZ(3)])
                    )
                ),
                "0: ──RX─┤ ╭<(𝓗)@(𝓗)>\n1: ──RY─┤ ├<(𝓗)@(𝓗)>\n2: ──RZ─┤ ├<(𝓗)@(𝓗)>\n3: ─────┤ ╰<(𝓗)@(𝓗)>",
            ),
            (
                lambda: (qml.var(qml.X(0)), qml.var(qml.Y(1)), qml.var(qml.Z(2))),
                "0: ──RX─┤  Var[X]\n1: ──RY─┤  Var[Y]\n2: ──RZ─┤  Var[Z]",
            ),
            (
                lambda: (
                    qml.var(qml.X(0) @ qml.Y(1)),
                    qml.var(qml.Y(1) @ qml.Z(2) @ qml.X(0)),
                    qml.var(qml.Z(2) @ qml.X(0) @ qml.Y(1)),
                ),
                "0: ──RX─┤ ╭Var[X@Y] ╭Var[Y@Z@X] ╭Var[Z@X@Y]\n1: ──RY─┤ ╰Var[X@Y] ├Var[Y@Z@X] ├Var[Z@X@Y]\n2: ──RZ─┤           ╰Var[Y@Z@X] ╰Var[Z@X@Y]",
            ),
        ],
    )
    def test_measurements(self, measurement, expected):
        """
        Test the visualization of measurements.
        """

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def _():
            qml.RX(0.1, 0)
            qml.RY(0.2, 1)
            qml.RZ(0.3, 2)
            return measurement()

        if isinstance(measurement(), qml.measurements.SampleMP):
            _ = qml.set_shots(10)(_)

        assert draw(_)() == expected

    def test_global_phase(self):
        """Test the visualization of global phase shifts."""

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def _():
            qml.H(0)
            qml.H(1)
            qml.H(2)
            qml.GlobalPhase(0.5)
            return qml.state()

        assert (
            draw(_)()
            == "0: ──H─╭GlobalPhase─┤  State\n1: ──H─├GlobalPhase─┤  State\n2: ──H─╰GlobalPhase─┤  State"
        )


if __name__ == "__main__":
    pytest.main(["-x", __file__])
