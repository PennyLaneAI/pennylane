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
                TypeError, match="work_wire must be specified if fixed_point == True."
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
        "U, O, raise_error",
        (
            (generator(wires=[0, 1, 2]), oracle([0, 1], wires=[0, 1, 2]), False),
            (generator(wires=[0, 2, 1]), oracle([0, 1], wires=[0, 1, 2]), False),
            (generator(wires=[0, 1, 3]), oracle([0, 1], wires=[0, 1, 2]), True),
            (generator(wires=[0, 1, 2, 3]), oracle([0, 1], wires=[0, 1, 2]), True),
        ),
    )
    def test_error_wrong_wires(self, U, O, raise_error):
        """Test an error is raised if the wires of U and O are not the same."""

        if raise_error:
            with pytest.raises(TypeError, match="U and O must act on the same wires."):
                qml.AmplitudeAmplification(U, O)

        else:
            try:
                qml.AmplitudeAmplification(U, O)
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
            with pytest.raises(TypeError, match="work_wire must be different from the wires of U."):
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
        "U, O, iters, fixed_point, work_wire",
        (
            (generator(wires=range(3)), oracle([0, 1], wires=range(3)), 1, True, 3),
            (generator(wires=range(2)), oracle([0, 1], wires=range(2)), 2, False, 2),
            (generator(wires=range(4)), oracle([0, 1], wires=range(4)), 3, True, 5),
        ),
    )
    def test_init_correctly(self, U, O, iters, fixed_point, work_wire):
        """Test that all of the attributes are initalized correctly."""

        op = qml.AmplitudeAmplification(U, O, iters, fixed_point, work_wire)

        assert op.wires == U.wires + qml.wires.Wires(work_wire)
        assert op.U == U
        assert op.O == O
        assert op.iters == iters
        assert op.fixed_point == fixed_point
        assert op.work_wire == work_wire


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
        [
            0.67103818,
            0.04699455,
            0.04699455,
            0.04699455,
            0.04699455,
            0.04699455,
            0.04699455,
            0.04699455,
        ]
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
