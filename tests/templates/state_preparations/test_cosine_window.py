# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for the CosineWindow template.
"""
import numpy as np

# pylint: disable=too-few-public-methods
import pytest

import pennylane as qml
from pennylane.exceptions import WireError
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.transforms.decompose import DecomposeInterpreter
from pennylane.wires import Wires


@pytest.mark.jax
def test_standard_validity():
    """Check the operation using the assert_valid function."""

    op = qml.CosineWindow(wires=[0, 1])

    qml.ops.functions.assert_valid(op)


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize(
        "wires",
        [
            [0, 1],
            [0, 1, 2, 3, 4],
            ["a", "b", "c", "d", "e", "f"],
        ],
    )
    def test_decomposition_new(self, wires):
        """Tests the decomposition rule implemented with the new system."""
        op = qml.CosineWindow(wires=wires)

        for rule in qml.list_decomps(qml.CosineWindow):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize(
        "wires",
        [
            [0, 1],
            [0, 1, 2],
            [0, 1, 2, 3],
            [0, 1, 2, 3, 4],
        ],
    )
    @pytest.mark.capture
    def test_decomposition_new_capture(self, wires):
        """Tests the decomposition rule implemented with the new system."""
        op = qml.CosineWindow(wires=wires)

        for rule in qml.list_decomps(qml.CosineWindow):
            _test_decomposition_rule(op, rule)

    @pytest.mark.integration
    @pytest.mark.capture
    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_integration_decompose_interpreter(self):
        """Tests that a simple circuit is correctly decomposed into different gate sets."""
        import jax
        from jax import numpy as jnp

        from pennylane.tape.plxpr_conversion import CollectOpsandMeas

        def f():
            qml.CosineWindow(wires=[0, 1])

        decomposed_f = DecomposeInterpreter(
            gate_set={"Hadamard", "RZ", "PhaseShift", "ControlledPhaseShift", "SWAP"}
        )(f)
        jaxpr = jax.make_jaxpr(decomposed_f)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        assert collector.state["ops"] == [
            qml.Hadamard(1),
            qml.RZ(3.141592653589793, wires=[1]),
            qml.adjoint(qml.QFT(wires=[0, 1])),
            qml.PhaseShift(jnp.array(-2.89760778e19), wires=[0]),
            qml.PhaseShift(jnp.array(1.44880389e19), wires=[1]),
        ]

    def test_correct_gates_single_wire(self):
        """Test that the correct gates are applied."""

        op = qml.CosineWindow(wires=[0])
        queue = op.decomposition()

        assert queue[0].name == "Hadamard"
        assert queue[1].name == "RZ"
        assert queue[2].name == "Adjoint(QFT)"
        assert queue[3].name == "PhaseShift"

        assert np.isclose(queue[3].data[0], np.pi / 2)

    def test_correct_gates_many_wires(self):
        """Test that the correct gates are applied on two wires."""

        op = qml.CosineWindow(wires=[0, 1, 2, 3, 4])
        queue = op.decomposition()

        assert queue[0].name == "Hadamard"
        assert queue[1].name == "RZ"
        assert queue[2].name == "Adjoint(QFT)"

        for ind, q in enumerate(queue[3:]):
            assert q.name == "PhaseShift"
            assert np.isclose(q.data[0], np.pi / 2 ** (ind + 1))

    def test_custom_wire_labels(self):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.CosineWindow(wires=range(3))
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.CosineWindow(wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z")), qml.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2)
        assert np.allclose(state1, state2)


class TestRepresentation:
    """Test id and label."""

    def test_id(self):
        """Tests that the id attribute can be set."""
        wires = [0, 1, 2]
        template = qml.CosineWindow(wires=wires, id="a")
        assert template.id == "a"
        assert template.wires == Wires(wires)

    def test_label(self):
        """Test label method returns CosineWindow"""
        op = qml.CosineWindow(wires=[0, 1])
        assert op.label() == "CosineWindow"


class TestStateVector:
    """Test the state_vector() method of various CosineWindow operations."""

    def test_CosineWindow_state_vector(self):
        """Tests that the state vector is correct for a single wire."""
        op = qml.CosineWindow(wires=[0])
        res = op.state_vector()
        expected = np.array([0.0, 1.0])
        assert np.allclose(res, expected)

        op = qml.CosineWindow(wires=[0, 1])
        res = np.reshape(op.state_vector() ** 2, (-1,))
        expected = np.array([0.0, 0.25, 0.5, 0.25])
        assert np.allclose(res, expected)

    def test_CosineWindow_state_vector_bad_wire_order(self):
        """Tests that the provided wire_order must contain the wires in the operation."""
        qsv_op = qml.CosineWindow(wires=[0, 1])
        with pytest.raises(WireError, match="wire_order must contain all CosineWindow wires"):
            qsv_op.state_vector(wire_order=[1, 2])

    def test_CosineWindow_state_vector_wire_order(self):
        """Tests that the state vector works with a different order of wires."""
        op = qml.CosineWindow(wires=[0, 1])
        res = np.reshape(op.state_vector(wire_order=[1, 0]) ** 2, (-1,))
        expected = np.array([0.0, 0.5, 0.25, 0.25])
        assert np.allclose(res, expected)

    def test_CosineWindow_state_vector_subset_of_wires(self):
        """Tests that the state vector works with not all state wires."""
        op = qml.CosineWindow([2, 1])
        res = op.state_vector(wire_order=[0, 1, 2])
        assert res.shape == (2, 2, 2)

        expected_10 = qml.CosineWindow([0, 1]).state_vector(wire_order=[1, 0])
        expected = np.stack([expected_10, np.zeros_like(expected_10)])
        assert np.allclose(res, expected)


class TestInterfaces:
    """Test that the template works with different interfaces"""

    @pytest.mark.jax
    def test_jax_jit(self):
        """Test that the template correctly compiles with JAX JIT   ."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.CosineWindow(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        circuit2 = jax.jit(circuit)

        res = circuit()
        res2 = circuit2()
        assert qml.math.allclose(res, res2, atol=1e-6, rtol=0)
