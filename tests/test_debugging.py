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
Unit tests for the debugging module.
"""
import numpy as np
import pytest

import pennylane as qml


class TestSnapshot:
    """Test the Snapshot instruction for simulators."""

    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "backprop", "parameter-shift", "adjoint"])
    def test_default_qubit_legacy_opmath(self, method):
        """Test that multiple snapshots are returned correctly on the state-vector simulator."""
        dev = qml.device("default.qubit.legacy", wires=2)

        @qml.qnode(dev, diff_method=method)
        def circuit():
            qml.Snapshot()
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

        circuit()
        assert dev._debugger is None
        if method is not None:
            assert circuit.interface == "auto"

        result = qml.snapshots(circuit)()
        expected = {
            0: np.array([1, 0, 0, 0]),
            "very_important_state": np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0]),
            2: np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),
            "execution_results": np.array(0),
        }

        assert all(k1 == k2 for k1, k2 in zip(result.keys(), expected.keys()))
        assert all(np.allclose(v1, v2) for v1, v2 in zip(result.values(), expected.values()))

    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "backprop", "parameter-shift", "adjoint"])
    def test_default_qubit2(self, method):
        """Test that multiple snapshots are returned correctly on the new
        state-vector simulator."""
        dev = qml.device("default.qubit")

        # TODO: add additional QNode test once the new device supports it

        @qml.qnode(dev, diff_method=method)
        def circuit():
            qml.Snapshot()
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

        circuit()
        assert dev._debugger is None
        if method is not None:
            assert circuit.interface == "auto"

        result = qml.snapshots(circuit)()
        expected = {
            0: np.array([1, 0, 0, 0]),
            "very_important_state": np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0]),
            2: np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),
            "execution_results": np.array(0),
        }

        assert all(k1 == k2 for k1, k2 in zip(result.keys(), expected.keys()))
        assert all(np.allclose(v1, v2) for v1, v2 in zip(result.values(), expected.values()))

    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "parameter-shift"])
    def test_default_mixed(self, method):
        """Test that multiple snapshots are returned correctly on the density-matrix simulator."""
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev, diff_method=method)
        def circuit():
            qml.Snapshot()
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

        circuit()
        assert dev._debugger is None

        result = qml.snapshots(circuit)()
        expected = {
            0: np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            "very_important_state": np.array(
                [[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]]
            ),
            2: np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]),
            "execution_results": np.array(0),
        }

        assert all(k1 == k2 for k1, k2 in zip(result.keys(), expected.keys()))
        assert all(np.allclose(v1, v2) for v1, v2 in zip(result.values(), expected.values()))

    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "parameter-shift"])
    def test_default_gaussian(self, method):
        """Test that multiple snapshots are returned correctly on the CV simulator."""
        dev = qml.device("default.gaussian", wires=2)

        @qml.qnode(dev, diff_method=method)
        def circuit():
            qml.Snapshot()
            qml.Displacement(0.5, 0, wires=0)
            qml.Snapshot("very_important_state")
            qml.Beamsplitter(0.5, 0.7, wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.QuadX(0))

        circuit()
        assert dev._debugger is None

        result = qml.snapshots(circuit)()
        expected = {
            0: {
                "cov_matrix": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                "means": np.array([0, 0, 0, 0]),
            },
            1: {
                "cov_matrix": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                "means": np.array([1, 0, 0, 0]),
            },
            2: {
                "cov_matrix": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                "means": np.array([0.87758256, 0.36668488, 0, 0.30885441]),
            },
            "execution_results": np.array(0.87758256),
        }

        assert all(k1 == k2 for k1, k2 in zip(result.keys(), expected.keys()))
        assert np.allclose(result["execution_results"], expected["execution_results"])
        del result["execution_results"]
        del expected["execution_results"]
        assert all(
            np.allclose(v1["cov_matrix"], v2["cov_matrix"])
            for v1, v2 in zip(result.values(), expected.values())
        )
        assert all(
            np.allclose(v1["means"], v2["means"])
            for v1, v2 in zip(result.values(), expected.values())
        )

    @pytest.mark.parametrize("method", [None, "parameter-shift", "adjoint"])
    def test_lightning_qubit(self, method):
        """Test that an error is (currently) raised on the lightning simulator."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, diff_method=method)
        def circuit():
            qml.Snapshot()
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

        # can run the circuit
        result = circuit()
        assert result == 0

        with pytest.raises(qml.DeviceError, match="Device does not support snapshots."):
            qml.snapshots(circuit)()

    def test_unsupported_device(self):
        """Test that an error is raised on unsupported devices."""
        dev = qml.device("default.qubit.legacy", wires=2)
        # remove attributes to simulate unsupported device
        delattr(dev, "_debugger")
        dev.operations.remove("Snapshot")

        @qml.qnode(dev, interface=None)  # iterface=None prevents new device creation internally
        def circuit():
            qml.Snapshot()
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

        # can run the circuit
        result = circuit()
        assert result == 0

        with pytest.raises(qml.DeviceError, match="Device does not support snapshots."):
            qml.snapshots(circuit)()

        # need to revert change to not affect other tests (since operations a static attribute)
        dev.operations.add("Snapshot")

    def test_unsupported_device_new(self):
        """Test that an error is raised on unsupported devices."""

        class DummyDevice(qml.devices.Device):  # pylint: disable=too-few-public-methods
            def execute(self, *args, **kwargs):
                return args, kwargs

        dev = DummyDevice()

        with pytest.raises(qml.DeviceError, match="Device does not support snapshots."):
            with qml.debugging._Debugger(dev):
                dev.execute([])

    def test_empty_snapshots(self):
        """Test that snapshots function in the absence of any Snapshot operations."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        result = qml.snapshots(circuit)()
        expected = {"execution_results": np.array(0)}

        assert result == expected

    @pytest.mark.parametrize("shots", [None, 1, 1000000])
    def test_different_shots(self, shots):
        """Test that snapshots are returned correctly with different QNode shot values."""
        dev = qml.device("default.qubit", wires=2, shots=shots)

        @qml.qnode(dev)
        def circuit():
            qml.Snapshot()
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

        result = qml.snapshots(circuit)()
        expected = {
            0: np.array([1, 0, 0, 0]),
            "very_important_state": np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0]),
            2: np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),
            "execution_results": np.array(0),
        }

        assert all(k1 == k2 for k1, k2 in zip(result.keys(), expected.keys()))
        for v1, v2 in zip(result.values(), expected.values()):
            if v1.shape == ():
                # Expectation value comparison
                assert v2.shape == ()
                if shots != 1:
                    # If many shots or analytic mode, we can compare the result
                    assert np.allclose(v1, v2, atol=0.006)
                else:
                    # If a single shot, it must be in the eigvals of qml.Z
                    assert v1.item() in {-1.0, 1.0}
            else:
                assert np.allclose(v1, v2)

    @pytest.mark.parametrize(
        "m,expected_result",
        [
            ("expval", np.array(0)),
            ("var", np.array(1)),
            ("probs", np.array([0.5, 0, 0, 0.5])),
            ("state", np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])),
        ],
    )
    def test_different_measurements(self, m, expected_result):
        """Test that snapshots are returned correctly with different QNode measurements."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=None)
        def circuit():
            qml.Snapshot()
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            if m == "expval":
                return qml.expval(qml.PauliZ(0))
            if m == "var":
                return qml.var(qml.PauliY(1))
            if m == "probs":
                return qml.probs([0, 1])
            return qml.state()

        result = qml.snapshots(circuit)()
        expected = {
            0: np.array([1, 0, 0, 0]),
            "very_important_state": np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0]),
            2: np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),
            "execution_results": expected_result,
        }

        assert all(k1 == k2 for k1, k2 in zip(result.keys(), expected.keys()))
        assert all(np.allclose(v1, v2) for v1, v2 in zip(result.values(), expected.values()))

        if m == "state":
            assert np.allclose(result[2], result["execution_results"])

    def test_controlled_circuit(self):
        """Test that snapshots are returned correctly even with a controlled circuit."""
        dev = qml.device("default.qubit", wires=2)

        def circuit(params, wire):
            qml.Hadamard(wire)
            qml.Snapshot()
            qml.Rot(*params, wire)

        @qml.qnode(dev)
        def qnode(params):
            qml.Hadamard(0)
            qml.ctrl(circuit, 0)(params, wire=1)
            return qml.expval(qml.PauliZ(1))

        params = np.array([1.3, 1.4, 0.2])
        result = qml.snapshots(qnode)(params)
        expected = {
            0: np.array([1 / np.sqrt(2), 0, 0.5, 0.5]),
            "execution_results": np.array(0.36819668),
        }

        assert all(k1 == k2 for k1, k2 in zip(result.keys(), expected.keys()))
        assert all(np.allclose(v1, v2) for v1, v2 in zip(result.values(), expected.values()))

    def test_all_measurement_snapshot(self):
        """Test that the correct measurement snapshots are returned for different measurement types."""
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(x):
            qml.Snapshot(measurement=qml.expval(qml.PauliZ(0)))
            qml.RX(x, wires=0)
            qml.Snapshot(measurement=qml.var(qml.PauliZ(0)))
            qml.Snapshot(measurement=qml.probs(0))
            qml.Snapshot(measurement=qml.state())
            return qml.expval(qml.PauliZ(0))

        phi = 0.1
        result = qml.snapshots(circuit)(phi)
        expected = {
            0: np.array(1),
            1: np.array(1 - np.cos(phi) ** 2),
            2: np.array([np.cos(phi / 2) ** 2, np.sin(phi / 2) ** 2]),
            3: np.array([np.cos(phi / 2), -1j * np.sin(phi / 2)]),
            "execution_results": np.array(np.cos(phi)),
        }

        assert all(k1 == k2 for k1, k2 in zip(result.keys(), expected.keys()))
        assert all(np.allclose(v1, v2) for v1, v2 in zip(result.values(), expected.values()))

    def test_unsupported_snapshot_measurement(self):
        """Test that an exception is raised when an unsupported measurement is provided to the snapshot."""
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            with pytest.raises(
                ValueError,
                match="The measurement PauliZ is not supported as it is not an instance "
                "of <class 'pennylane.measurements.measurements.StateMeasurement'>",
            ):
                qml.Snapshot(measurement=qml.PauliZ(0))
            return qml.expval(qml.PauliZ(0))

        qml.snapshots(circuit)()
