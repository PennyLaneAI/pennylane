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
import jax

# pylint: disable=wrong-import-position
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import iterative_cancel_inverses_pass
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

        transforms_circuit = iterative_cancel_inverses_pass(
            qml.compiler.python_compiler.transforms.merge_rotations_pass(transforms_circuit)
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

        transforms_circuit = qml.transforms.cancel_inverses(
            qml.transforms.merge_rotations(transforms_circuit)
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

        transforms_circuit = iterative_cancel_inverses_pass(
            qml.transforms.merge_rotations(transforms_circuit)
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
        "op, expected",
        [
            (
                lambda: qml.ctrl(qml.RX(0.1, 0), control=(1, 2, 3)),
                "1: ─╭●──┤  State\n2: ─├●──┤  State\n3: ─├●──┤  State\n0: ─╰RX─┤  State",
            ),
            (
                lambda: qml.ctrl(qml.RX(0.1, 0), control=(1, 2, 3), control_values=(0, 1, 0)),
                "1: ─╭○──┤  State\n2: ─├●──┤  State\n3: ─├○──┤  State\n0: ─╰RX─┤  State",
            ),
            (
                lambda: qml.adjoint(qml.ctrl(qml.RX(0.1, 0), (1, 2, 3), control_values=(0, 1, 0))),
                "1: ─╭○───┤  State\n2: ─├●───┤  State\n3: ─├○───┤  State\n0: ─╰RX†─┤  State",
            ),
            (
                lambda: qml.ctrl(qml.adjoint(qml.RX(0.1, 0)), (1, 2, 3), control_values=(0, 1, 0)),
                "1: ─╭○───┤  State\n2: ─├●───┤  State\n3: ─├○───┤  State\n0: ─╰RX†─┤  State",
            ),
        ],
    )
    def test_ctrl_adjoint_variants(self, op, expected):
        """
        Test the visualization of control and adjoint variants.
        """

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            op()
            return qml.state()

        assert draw(circuit)() == expected

    def test_ctrl_before_custom_op(self):
        """
        Test the visualization of control operations before custom ops.
        """

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            qml.ctrl(qml.X(3), control=[0, 1, 2], control_values=[1, 0, 1])
            qml.RX(0.1, 2)
            return qml.state()

        assert (
            draw(circuit)()
            == "0: ─╭●─────┤  State\n1: ─├○─────┤  State\n2: ─├●──RX─┤  State\n3: ─╰X─────┤  State"
        )

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
        def circuit():
            qml.RX(0.1, 0)
            qml.RY(0.2, 1)
            qml.RZ(0.3, 2)
            return measurement()

        if isinstance(measurement(), qml.measurements.SampleMP):
            circuit = qml.set_shots(10)(circuit)

        assert draw(circuit)() == expected

    def test_global_phase(self):
        """Test the visualization of global phase shifts."""

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            qml.H(0)
            qml.H(1)
            qml.H(2)
            qml.GlobalPhase(0.5)
            return qml.state()

        assert (
            draw(circuit)()
            == "0: ──H─╭GlobalPhase─┤  State\n1: ──H─├GlobalPhase─┤  State\n2: ──H─╰GlobalPhase─┤  State"
        )

    @pytest.mark.parametrize(
        "postselect, mid_measure_label",
        [
            (None, "┤↗├"),
            (0, "┤↗₀├"),
            (1, "┤↗₁├"),
        ],
    )
    def test_draw_mid_circuit_measurement_postselect(self, postselect, mid_measure_label):
        """Test that mid-circuit measurements are drawn correctly."""

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit():
            qml.Hadamard(0)
            qml.measure(0, postselect=postselect)
            qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        drawing = draw(circuit)()
        expected_drawing = "0: ──H──" + mid_measure_label + "──X─┤  <Z>"

        assert drawing == expected_drawing

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "ops, expected",
        [
            (
                [
                    (qml.QubitUnitary, jax.numpy.array([[0, 1], [1, 0]]), [0]),
                    (
                        qml.QubitUnitary,
                        jax.numpy.array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]]),
                        [0, 1],
                    ),
                    (qml.QubitUnitary, jax.numpy.zeros((8, 8)), [0, 1, 2]),
                    (
                        qml.QubitUnitary,
                        jax.numpy.array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]]),
                        [0, 1],
                    ),
                    (qml.QubitUnitary, jax.numpy.array([[0, 1], [1, 0]]), [0]),
                ],
                "0: ──U(M0)─╭U(M1)─╭U(M2)─╭U(M1)──U(M0)─┤  State\n"
                "1: ────────╰U(M1)─├U(M2)─╰U(M1)────────┤  State\n"
                "2: ───────────────╰U(M2)───────────────┤  State",
            ),
            (
                [
                    (qml.StatePrep, jax.numpy.array([1, 0]), [0]),
                    (qml.StatePrep, jax.numpy.array([1, 0, 0, 0]), [0, 1]),
                    (qml.StatePrep, jax.numpy.array([1, 0, 0, 0, 1, 0, 0, 0]), [0, 1, 2]),
                    (qml.StatePrep, jax.numpy.array([1, 0, 0, 0]), [0, 1]),
                    (qml.StatePrep, jax.numpy.array([1, 0]), [0]),
                ],
                "0: ──|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩──|Ψ⟩─┤  State\n"
                "1: ──────╰|Ψ⟩─├|Ψ⟩─╰|Ψ⟩──────┤  State\n"
                "2: ───────────╰|Ψ⟩───────────┤  State",
            ),
            (
                [
                    (qml.MultiRZ, 0.1, [0]),
                    (qml.MultiRZ, 0.1, [0, 1]),
                    (qml.MultiRZ, 0.1, [0, 1, 2]),
                    (qml.MultiRZ, 0.1, [0, 1]),
                    (qml.MultiRZ, 0.1, [0]),
                ],
                "0: ──MultiRZ─╭MultiRZ─╭MultiRZ─╭MultiRZ──MultiRZ─┤  State\n"
                "1: ──────────╰MultiRZ─├MultiRZ─╰MultiRZ──────────┤  State\n"
                "2: ───────────────────╰MultiRZ───────────────────┤  State",
            ),
            (
                [
                    (qml.BasisState, jax.numpy.array([1]), [0]),
                    (qml.BasisState, jax.numpy.array([1, 0]), [0, 1]),
                    (qml.BasisState, jax.numpy.array([1, 0, 0]), [0, 1, 2]),
                    (qml.BasisState, jax.numpy.array([1, 0]), [0, 1]),
                    (qml.BasisState, jax.numpy.array([1]), [0]),
                ],
                "0: ──|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩─╭|Ψ⟩──|Ψ⟩─┤  State\n"
                "1: ──────╰|Ψ⟩─├|Ψ⟩─╰|Ψ⟩──────┤  State\n"
                "2: ───────────╰|Ψ⟩───────────┤  State",
            ),
        ],
    )
    def test_visualization_cases(self, ops, expected):
        """
        Test the visualization of the quantum operations defined in the unified compiler dialect.
        """

        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            for op, param, wires in ops:
                op(param, wires=wires)
            return qml.state()

        assert draw(circuit)() == expected

    def test_reshape(self):
        """Test that the visualization works when the parameters are reshaped."""

        one_dim = jax.numpy.array([1, 0])
        two_dim = jax.numpy.array([[0, 1], [1, 0]])
        eight_dim = jax.numpy.zeros((8, 8))

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit():
            qml.RX(one_dim[0], wires=0)
            qml.RZ(two_dim[0, 0], wires=0)
            qml.QubitUnitary(eight_dim[:2, :2], wires=0)
            qml.QubitUnitary(eight_dim[0:4, 0:4], wires=[0, 1])
            return qml.state()

        expected = (
            "0: ──RX(M0)──RZ(M0)──U(M1)─╭U(M2)─┤  State\n"
            "1: ────────────────────────╰U(M2)─┤  State"
        )
        assert draw(circuit)() == expected

    def test_args_warning(self):
        """Test that a warning is raised when dynamic arguments are used."""

        # pylint: disable=unused-argument
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circ(arg):
            qml.RX(0.1, wires=0)
            return qml.state()

        with pytest.warns(UserWarning):
            draw(circ)(0.1)

    def adjoint_op_not_implemented(self):
        """Test that NotImplementedError is raised when AdjointOp is used."""

        @qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit():
            qml.adjoint(qml.QubitUnitary)(jax.numpy.array([[0, 1], [1, 0]]), wires=[0])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(NotImplementedError, match="not yet supported"):
            print(draw(circuit)())

    def test_cond_not_implemented(self):
        """Test that NotImplementedError is raised when cond is used."""

        @qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit():
            m0 = qml.measure(0, reset=False, postselect=0)
            qml.cond(m0, qml.RX, qml.RY)(1.23, 1)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(NotImplementedError, match="not yet supported"):
            print(draw(circuit)())

    def test_for_loop_not_implemented(self):
        """Test that NotImplementedError is raised when for loop is used."""

        @qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()], autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit():
            for _ in range(3):
                qml.RX(0.1, 0)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(NotImplementedError, match="not yet supported"):
            print(draw(circuit)())

    def test_while_loop_not_implemented(self):
        """Test that NotImplementedError is raised when while loop is used."""

        @qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()], autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit():
            i = 0
            while i < 3:
                qml.RX(0.1, 0)
                i += 1
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(NotImplementedError, match="not yet supported"):
            print(draw(circuit)())


if __name__ == "__main__":
    pytest.main(["-x", __file__])
