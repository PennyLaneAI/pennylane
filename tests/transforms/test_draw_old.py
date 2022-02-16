# Copyright 2018-2022 Xanadu Quantum Technologies Inc.


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
Unit tests for the old draw transform.
"""
from functools import partial
import pytest

import pennylane as qml
from pennylane import numpy as np

from pennylane.transforms import draw_old


def test_drawing():
    """Test circuit drawing"""

    x = np.array(0.1, requires_grad=True)
    y = np.array([0.2, 0.3], requires_grad=True)
    z = np.array(0.4, requires_grad=True)

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="autograd")
    def circuit(p1, p2=y, **kwargs):
        qml.RX(p1, wires=0)
        qml.RY(p2[0] * p2[1], wires=1)
        qml.RX(kwargs["p3"], wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

    result = draw_old(circuit)(p1=x, p3=z)
    expected = """\
 0: ──RX(0.1)───RX(0.4)──╭C──╭┤ ⟨Z ⊗ X⟩ 
 1: ──RY(0.06)───────────╰X──╰┤ ⟨Z ⊗ X⟩ 
"""

    assert result == expected


def test_drawing_tf():
    """Test circuit drawing when using TensorFlow"""
    tf = pytest.importorskip("tensorflow")

    x = tf.constant(0.1)
    y = tf.constant([0.2, 0.3])
    z = tf.Variable(0.4)

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="tf")
    def circuit(p1, p2=y, **kwargs):
        qml.RX(p1, wires=0)
        qml.RY(p2[0] * p2[1], wires=1)
        qml.RX(kwargs["p3"], wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

    result = draw_old(circuit)(p1=x, p3=z)
    expected = """\
 0: ──RX(0.1)───RX(0.4)──╭C──╭┤ ⟨Z ⊗ X⟩ 
 1: ──RY(0.06)───────────╰X──╰┤ ⟨Z ⊗ X⟩ 
"""

    assert result == expected


def test_drawing_torch():
    """Test circuit drawing when using Torch"""
    torch = pytest.importorskip("torch")

    x = torch.tensor(0.1, requires_grad=True)
    y = torch.tensor([0.2, 0.3], requires_grad=True)
    z = torch.tensor(0.4, requires_grad=True)

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def circuit(p1, p2=y, **kwargs):
        qml.RX(p1, wires=0)
        qml.RY(p2[0] * p2[1], wires=1)
        qml.RX(kwargs["p3"], wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

    result = draw_old(circuit)(p1=x, p3=z)
    expected = """\
 0: ──RX(0.1)───RX(0.4)──╭C──╭┤ ⟨Z ⊗ X⟩ 
 1: ──RY(0.06)───────────╰X──╰┤ ⟨Z ⊗ X⟩ 
"""

    assert result == expected


def test_drawing_jax():
    """Test circuit drawing when using JAX"""
    jax = pytest.importorskip("jax")
    jnp = jax.numpy

    x = jnp.array(0.1)
    y = jnp.array([0.2, 0.3])
    z = jnp.array(0.4)

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="jax")
    def circuit(p1, p2=y, **kwargs):
        qml.RX(p1, wires=0)
        qml.RY(p2[0] * p2[1], wires=1)
        qml.RX(kwargs["p3"], wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

    result = draw_old(circuit)(p1=x, p3=z)
    expected = """\
 0: ──RX(0.1)───RX(0.4)──╭C──╭┤ ⟨Z ⊗ X⟩ 
 1: ──RY(0.06)───────────╰X──╰┤ ⟨Z ⊗ X⟩ 
"""

    assert result == expected


def test_drawing_ascii():
    """Test circuit drawing when using ASCII characters"""
    from pennylane import numpy as np

    x = np.array(0.1, requires_grad=True)
    y = np.array([0.2, 0.3], requires_grad=True)
    z = np.array(0.4, requires_grad=True)

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="autograd")
    def circuit(p1, p2=y, **kwargs):
        qml.RX(p1, wires=0)
        qml.RY(p2[0] * p2[1], wires=1)
        qml.RX(kwargs["p3"], wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

    result = draw_old(circuit, charset="ascii")(p1=x, p3=z)
    expected = """\
 0: --RX(0.1)---RX(0.4)--+C--+| <Z @ X> 
 1: --RY(0.06)-----------+X--+| <Z @ X> 
"""

    assert result == expected


def test_show_all_wires_error():
    """Test that show_all_wires will raise an error if the provided wire
    order does not contain all wires on the device"""

    dev = qml.device("default.qubit", wires=[-1, "a", "q2", 0])

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=-1)
        qml.CNOT(wires=[-1, "q2"])
        return qml.expval(qml.PauliX(wires="q2"))

    with pytest.raises(ValueError, match="must contain all wires"):
        draw_old(circuit, show_all_wires=True, wire_order=[-1, "a"])()


def test_missing_wire():
    """Test that wires not specifically mentioned in the wire
    reordering are appended at the bottom of the circuit drawing"""

    dev = qml.device("default.qubit", wires=["a", -1, "q2"])

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=-1)
        qml.CNOT(wires=["a", "q2"])
        qml.RX(0.2, wires="a")
        return qml.expval(qml.PauliX(wires="q2"))

    # test one missing wire
    res = draw_old(circuit, wire_order=["q2", "a"])()
    expected = [
        " q2: ──╭X───────────┤ ⟨X⟩ ",
        "  a: ──╰C──RX(0.2)──┤     ",
        " -1: ───H───────────┤     \n",
    ]

    assert res == "\n".join(expected)

    # test one missing wire
    res = draw_old(circuit, wire_order=["q2", -1])()
    expected = [
        " q2: ─────╭X───────────┤ ⟨X⟩ ",
        " -1: ──H──│────────────┤     ",
        "  a: ─────╰C──RX(0.2)──┤     \n",
    ]

    assert res == "\n".join(expected)

    # test multiple missing wires
    res = draw_old(circuit, wire_order=["q2"])()
    expected = [
        " q2: ─────╭X───────────┤ ⟨X⟩ ",
        " -1: ──H──│────────────┤     ",
        "  a: ─────╰C──RX(0.2)──┤     \n",
    ]

    assert res == "\n".join(expected)


def test_invalid_wires():
    """Test that an exception is raised if a wire in the wire
    ordering does not exist on the device"""
    dev = qml.device("default.qubit", wires=["a", -1, "q2"])

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=-1)
        qml.CNOT(wires=["a", "q2"])
        qml.RX(0.2, wires="a")
        return qml.expval(qml.PauliX(wires="q2"))

    with pytest.raises(ValueError, match="contains wires not contained on the device"):
        draw_old(circuit, wire_order=["q2", 5])()


@pytest.mark.parametrize(
    "transform",
    [
        qml.gradients.param_shift(shifts=[(0.2,)]),
        partial(qml.gradients.param_shift, shifts=[(0.2,)]),
    ],
)
def test_draw_batch_transform(transform):
    """Test that drawing a batch transform works correctly"""
    dev = qml.device("default.qubit", wires=1)

    @transform
    @qml.qnode(dev)
    def circuit(x):
        qml.Hadamard(wires=0)
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    # the parameter-shift transform will create two circuits; one with x+0.2
    # and one with x-0.2.
    res = draw_old(circuit)(np.array(0.6, requires_grad=True))
    expected = [" 0: ──H──RX(0.8)──┤ ⟨Z⟩ ", "", " 0: ──H──RX(0.4)──┤ ⟨Z⟩ ", ""]
    assert res == "\n".join(expected)


def test_direct_qnode_integration():
    """Test that a QNode renders correctly."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def qfunc(a, w):
        qml.Hadamard(0)
        qml.CRX(a, wires=[0, 1])
        qml.Rot(w[0], w[1], w[2], wires=[1])
        qml.CRX(-a, wires=[0, 1])

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    a, w = 2.3, [1.2, 3.2, 0.7]

    assert draw_old(qfunc)(a, w) == (
        " 0: ──H──╭C────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩ \n"
        + " 1: ─────╰RX(2.3)──Rot(1.2, 3.2, 0.7)──╰RX(-2.3)──╰┤ ⟨Z ⊗ Z⟩ \n"
    )

    assert draw_old(qfunc, charset="ascii")(a, w) == (
        " 0: --H--+C----------------------------+C---------+| <Z @ Z> \n"
        + " 1: -----+RX(2.3)--Rot(1.2, 3.2, 0.7)--+RX(-2.3)--+| <Z @ Z> \n"
    )


def test_same_wire_multiple_measurements():
    """Test that drawing a QNode with multiple measurements on certain wires works correctly."""
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def qnode(x, y):
        qml.RY(x, wires=0)
        qml.Hadamard(0)
        qml.RZ(y, wires=0)
        return [
            qml.expval(qml.PauliX(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliX(wires=[2])),
            qml.expval(qml.PauliX(wires=[0]) @ qml.PauliX(wires=[3])),
        ]

    expected = (
        " 0: ──RY(1)──H──RZ(2)──╭┤ ⟨X ⊗ X ⊗ X⟩ ╭┤ ⟨X ⊗ X⟩ \n"
        + " 1: ───────────────────├┤ ⟨X ⊗ X ⊗ X⟩ │┤         \n"
        + " 2: ───────────────────╰┤ ⟨X ⊗ X ⊗ X⟩ │┤         \n"
        + " 3: ────────────────────┤             ╰┤ ⟨X ⊗ X⟩ \n"
    )
    assert draw_old(qnode)(1.0, 2.0) == expected


def test_same_wire_multiple_measurements_many_obs():
    """Test that drawing a QNode with multiple measurements on certain
    wires works correctly when there are more observables than the number of
    observables for any wire.
    """
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def qnode(x, y):
        qml.RY(x, wires=0)
        qml.Hadamard(0)
        qml.RZ(y, wires=0)
        return [
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1)),
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
        ]

    expected = (
        " 0: ──RY(0.3)──H──RZ(0.2)──┤ ⟨Z⟩ ┤     ╭┤ ⟨Z ⊗ Z⟩ \n"
        + " 1: ───────────────────────┤     ┤ ⟨Z⟩ ╰┤ ⟨Z ⊗ Z⟩ \n"
    )
    assert draw_old(qnode)(0.3, 0.2) == expected


def test_qubit_circuit_length_under_max_length_kwdarg():
    """Test that a qubit circuit with a circuit length less than the max_length renders correctly."""
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def qnode():
        for i in range(3):
            qml.Hadamard(wires=i)
            qml.RX(i * 0.1, wires=i)
            qml.RY(i * 0.1, wires=i)
            qml.RZ(i * 0.1, wires=i)
        return qml.expval(qml.PauliZ(0))

    expected = (
        " 0: ──H──RX(0)────RY(0)────RZ(0)────┤ ⟨Z⟩\n"
        + " 1: ──H──RX(0.1)──RY(0.1)──RZ(0.1)──┤    \n"
        + " 2: ──H──RX(0.2)──RY(0.2)──RZ(0.2)──┤    \n"
    )
    assert draw_old(qnode, max_length=60)() == expected


def test_matrix_parameter_template():
    """Assert draw method handles templates with matrix valued parameters."""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():
        qml.AmplitudeEmbedding(np.array([0, 1]), wires=0)
        return qml.state()

    expected = " 0: ──AmplitudeEmbedding(M0)──┤ State \nM0 =\n[0.+0.j 1.+0.j]\n"
    assert draw_old(circuit)() == expected


class TestWireOrdering:
    """Tests for wire ordering functionality"""

    def test_default_ordering(self):
        """Test that the default wire ordering matches the device"""

        dev = qml.device("default.qubit", wires=["a", -1, "q2"])

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=-1)
            qml.CNOT(wires=["a", "q2"])
            qml.RX(0.2, wires="a")
            return qml.expval(qml.PauliX(wires="q2"))

        res = draw_old(circuit)()
        expected = [
            "  a: ─────╭C──RX(0.2)──┤     ",
            " -1: ──H──│────────────┤     ",
            " q2: ─────╰X───────────┤ ⟨X⟩ \n",
        ]

        assert res == "\n".join(expected)

    def test_wire_reordering(self):
        """Test that wires are correctly reordered"""

        dev = qml.device("default.qubit", wires=["a", -1, "q2"])

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=-1)
            qml.CNOT(wires=["a", "q2"])
            qml.RX(0.2, wires="a")
            return qml.expval(qml.PauliX(wires="q2"))

        res = draw_old(circuit, wire_order=["q2", "a", -1])()
        expected = [
            " q2: ──╭X───────────┤ ⟨X⟩ ",
            "  a: ──╰C──RX(0.2)──┤     ",
            " -1: ───H───────────┤     \n",
        ]

        assert res == "\n".join(expected)

    def test_include_empty_wires(self):
        """Test that empty wires are correctly included"""

        dev = qml.device("default.qubit", wires=[-1, "a", "q2", 0])

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=-1)
            qml.CNOT(wires=[-1, "q2"])
            return qml.expval(qml.PauliX(wires="q2"))

        res = draw_old(circuit, show_all_wires=True)()
        expected = [
            " -1: ──H──╭C──┤     ",
            "  a: ─────│───┤     ",
            " q2: ─────╰X──┤ ⟨X⟩ ",
            "  0: ─────────┤     \n",
        ]

        assert res == "\n".join(expected)

    def test_show_all_wires_error(self):
        """Test that show_all_wires will raise an error if the provided wire
        order does not contain all wires on the device"""

        dev = qml.device("default.qubit", wires=[-1, "a", "q2", 0])

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=-1)
            qml.CNOT(wires=[-1, "q2"])
            return qml.expval(qml.PauliX(wires="q2"))

        with pytest.raises(ValueError, match="must contain all wires"):
            draw_old(circuit, show_all_wires=True, wire_order=[-1, "a"])()

    def test_missing_wire(self):
        """Test that wires not specifically mentioned in the wire
        reordering are appended at the bottom of the circuit drawing"""

        dev = qml.device("default.qubit", wires=["a", -1, "q2"])

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=-1)
            qml.CNOT(wires=["a", "q2"])
            qml.RX(0.2, wires="a")
            return qml.expval(qml.PauliX(wires="q2"))

        # test one missing wire
        res = draw_old(circuit, wire_order=["q2", "a"])()
        expected = [
            " q2: ──╭X───────────┤ ⟨X⟩ ",
            "  a: ──╰C──RX(0.2)──┤     ",
            " -1: ───H───────────┤     \n",
        ]

        assert res == "\n".join(expected)

        # test one missing wire
        res = draw_old(circuit, wire_order=["q2", -1])()
        expected = [
            " q2: ─────╭X───────────┤ ⟨X⟩ ",
            " -1: ──H──│────────────┤     ",
            "  a: ─────╰C──RX(0.2)──┤     \n",
        ]

        assert res == "\n".join(expected)

        # test multiple missing wires
        res = draw_old(circuit, wire_order=["q2"])()
        expected = [
            " q2: ─────╭X───────────┤ ⟨X⟩ ",
            " -1: ──H──│────────────┤     ",
            "  a: ─────╰C──RX(0.2)──┤     \n",
        ]

        assert res == "\n".join(expected)

    def test_invalid_wires(self):
        """Test that an exception is raised if a wire in the wire
        ordering does not exist on the device"""
        dev = qml.device("default.qubit", wires=["a", -1, "q2"])

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=-1)
            qml.CNOT(wires=["a", "q2"])
            qml.RX(0.2, wires="a")
            return qml.expval(qml.PauliX(wires="q2"))

        with pytest.raises(ValueError, match="contains wires not contained on the device"):
            res = draw_old(circuit, wire_order=["q2", 5])()

    def test_no_ops_draws(self):
        """Test that a QNode with no operations still draws correctly"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def qnode():
            return qml.expval(qml.PauliX(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliX(wires=[2]))

        res = draw_old(qnode)()
        expected = [
            " 0: ──╭┤ ⟨X ⊗ X ⊗ X⟩ \n",
            " 1: ──├┤ ⟨X ⊗ X ⊗ X⟩ \n",
            " 2: ──╰┤ ⟨X ⊗ X ⊗ X⟩ \n",
        ]

        assert res == "".join(expected)


class TestOpsIntegration:
    """Integration tests for drawing specific operations and templates"""

    def test_approx_time_evolution(self):
        """Test that a QNode with the ApproxTimeEvolution template draws
        correctly when having the expansion strategy set."""
        H = qml.PauliX(0) + qml.PauliZ(1) + 0.5 * qml.PauliX(0) @ qml.PauliX(1)

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit(t):
            qml.ApproxTimeEvolution(H, t, 2)
            return qml.probs(wires=0)

        res = draw_old(circuit, expansion_strategy="device")(0.5)
        expected = [
            " 0: ──H────────RZ(0.5)──H──H──╭RZ(0.25)──H──H────────RZ(0.5)──H──H──╭RZ(0.25)──H──┤ Probs \n",
            " 1: ──RZ(0.5)──H──────────────╰RZ(0.25)──H──RZ(0.5)──H──────────────╰RZ(0.25)──H──┤       \n",
        ]

        assert res == "".join(expected)
