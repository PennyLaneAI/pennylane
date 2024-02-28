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
Tests for the AmplitudeAmplification template.
"""

import pytest
import numpy as np
import pennylane as qml


@qml.prod
def generator(wires):
    for wire in wires:
        qml.Hadamard(wire)


@qml.prod
def oracle(items, wires):
    for item in items:
        qml.FlipSign(item, wires=wires)


class TestInitialization:
    """Test the AmplitudeAmplification class initializes correctly."""

    @pytest.mark.parametrize(
        "fixed_point, work_wire, raise_error",
        (
            (True, 3, False),
            (True, "a", False),
            (False, 4, False),
            (True, None, True),
        ),
    )
    def test_error_none_wire(self, fixed_point, work_wire, raise_error):
        """Test an error is raised if work_wire is None and fixed_point is True."""

        U = generator(wires=range(3))
        O = oracle([0, 2], wires=range(3))

        if raise_error:
            with pytest.raises(
                qml.wires.WireError, match="work_wire must be specified if fixed_point == True."
            ):
                qml.AmplitudeAmplification(
                    U, O, iters=3, fixed_point=fixed_point, work_wire=work_wire
                )

        else:
            try:
                qml.AmplitudeAmplification(
                    U, O, iters=3, fixed_point=fixed_point, work_wire=work_wire
                )
            except TypeError:
                assert False  # test should fail if an error was raised when we expect it not to

    @pytest.mark.parametrize(
        "wires, fixed_point, work_wire, raise_error",
        (
            ([0, 1, 2], True, 2, True),
            ([0, 1, 2], True, "a", False),
            (["a", "b"], True, "a", True),
            ([0, 1], False, 0, False),
        ),
    )
    def test_error_wrong_work_wire(self, wires, fixed_point, work_wire, raise_error):
        """Test an error is raised if work_wire is part of the U wires."""

        U = generator(wires=wires)
        O = oracle([0], wires=wires)

        if raise_error:
            with pytest.raises(
                ValueError, match="work_wire must be different from the wires of U."
            ):
                qml.AmplitudeAmplification(
                    U, O, iters=3, fixed_point=fixed_point, work_wire=work_wire
                )

        else:
            try:
                qml.AmplitudeAmplification(
                    U, O, iters=3, fixed_point=fixed_point, work_wire=work_wire
                )
            except TypeError:
                assert False  # test should fail if an error was raised when we expect it not to


@pytest.mark.parametrize(
    "n_wires, items, iters",
    (
        (3, [0, 2], 1),
        (3, [1, 2], 2),
        (5, [4, 5, 7, 12], 3),
        (5, [0, 1, 2, 3, 4], 4),
    ),
)
def test_compare_grover(n_wires, items, iters):
    U = generator(wires=range(n_wires))
    O = oracle(items, wires=range(n_wires))

    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def circuit_amplitude_amplification():
        generator(wires=range(n_wires))
        qml.AmplitudeAmplification(U, O, iters)
        return qml.probs(wires=range(n_wires))

    @qml.qnode(dev)
    def circuit_grover():
        generator(wires=range(n_wires))

        for _ in range(iters):
            oracle(items, wires=range(n_wires))
            qml.GroverOperator(wires=range(n_wires))

        return qml.probs(wires=range(n_wires))

    assert np.allclose(circuit_amplitude_amplification(), circuit_grover(), atol=1e-5)


class TestIntegration:
    """Tests that the AmplitudeAmplification is executable in a QNode context"""

    @staticmethod
    def circuit():
        """Test circuit"""
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)

        qml.AmplitudeAmplification(
            generator(range(3)), oracle([0], range(3)), fixed_point=True, iters=3, work_wire=3
        )
        return qml.probs(wires=range(3))

    # not calculated analytically, we are only ensuring that the results are consistent accross interfaces

    exp_result = np.array(
        [0.52864728, 0.0673361, 0.0673361, 0.0673361, 0.0673361, 0.0673361, 0.0673361, 0.0673361]
    )

    def test_qnode_numpy(self):
        """Test that the QNode executes with Numpy."""
        dev = qml.device("default.qubit")
        qnode = qml.QNode(self.circuit, dev, interface=None)

        res = qnode()
        assert res.shape == (8,)
        assert np.allclose(res, self.exp_result, atol=0.002)

    def test_lightning_qubit(self):
        """Test that the QNode executes with the Lightning Qubit simulator."""
        dev = qml.device("lightning.qubit", wires=4)
        qnode = qml.QNode(self.circuit, dev)

        res = qnode()
        assert res.shape == (8,)
        assert np.allclose(res, self.exp_result, atol=0.002)


def test_correct_queueing():
    """Test that the AmplitudeAmplification operator is correctly queued in the circuit"""
    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def circuit1():
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)

        qml.AmplitudeAmplification(generator(range(3)), oracle([0], range(3)))
        return qml.state()

    @qml.qnode(dev)
    def circuit2():
        generator(wires=[0, 1, 2])

        qml.AmplitudeAmplification(generator(range(3)), oracle([0], range(3)))
        return qml.state()

    U = generator(wires=[0, 1, 2])
    O = oracle([0], wires=[0, 1, 2])

    @qml.qnode(dev)
    def circuit3():
        generator(wires=[0, 1, 2])

        qml.AmplitudeAmplification(U=U, O=O)
        return qml.state()

    assert np.allclose(circuit1(), circuit2())
    assert np.allclose(circuit1(), circuit3())


# pylint: disable=protected-access
def test_flatten_and_unflatten():
    """Test the _flatten and _unflatten methods for AmplitudeAmplification."""

    op = qml.AmplitudeAmplification(qml.RX(0.25, wires=0), qml.PauliZ(0))
    data, metadata = op._flatten()

    assert len(data) == 2
    assert len(metadata) == 5

    new_op = type(op)._unflatten(*op._flatten())
    assert qml.equal(op, new_op)
    assert op is not new_op

    assert hash(metadata)


def test_amplification():
    """Test that AmplitudeAmplification amplifies a marked element."""

    U = generator(wires=range(3))
    O = oracle([2], wires=range(3))

    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def circuit():
        generator(wires=range(3))
        qml.AmplitudeAmplification(U, O, iters=5, fixed_point=True, work_wire=3)

        return qml.probs(wires=range(3))

    res = np.round(circuit(), 3)
    expected = np.array([0.013, 0.013, 0.91, 0.013, 0.013, 0.013, 0.013, 0.013])

    assert np.allclose(res, expected)


@pytest.mark.parametrize(("p_min"), [0.7, 0.8, 0.9])
def test_p_min(p_min):
    """Test that the p_min parameter works correctly."""

    dev = qml.device("default.qubit")

    U = generator(wires=range(4))
    O = oracle([0], wires=range(4))

    @qml.qnode(dev)
    def circuit():
        generator(wires=range(4))

        qml.AmplitudeAmplification(U, O, fixed_point=True, work_wire=4, p_min=p_min, iters=11)

        return qml.probs(wires=range(4))

    assert circuit()[0] >= p_min
