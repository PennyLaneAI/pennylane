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


def _compare_numpy_dicts(dict1, dict2):
    assert all(k1 == k2 for k1, k2 in zip(dict1.keys(), dict2.keys()))
    assert all(np.allclose(v1, v2) for v1, v2 in zip(dict1.values(), dict2.values()))


class TestSnapshotTape:
    def test_snapshot_output_tapes(self):
        ops = [
            qml.Snapshot(),
            qml.Hadamard(wires=0),
            qml.Snapshot("very_important_state"),
            qml.CNOT(wires=[0, 1]),
            qml.Snapshot(),
        ]

        measurements = [qml.expval(qml.PauliX(0))]

        num_snapshots = len(tuple(filter(lambda x: isinstance(x, qml.Snapshot), ops)))

        tape = qml.tape.QuantumTape(ops, measurements)

        tapes, _ = qml.snapshots(tape)

        assert len(tapes) == num_snapshots + 1

        tape_no_meas = qml.tape.QuantumTape(ops)

        tapes_no_meas, _ = qml.snapshots(tape_no_meas)

        assert len(tapes_no_meas) == num_snapshots

    def test_snapshot_postprocessing_fn(self):
        ops = [
            qml.Snapshot(),
            qml.Hadamard(wires=0),
            qml.Snapshot("very_important_state"),
            qml.CNOT(wires=[0, 1]),
            qml.Snapshot(),
        ]

        measurements = [qml.expval(qml.PauliX(0))]

        num_snapshots = len(tuple(filter(lambda x: isinstance(x, qml.Snapshot), ops)))

        tape = qml.tape.QuantumTape(ops, measurements)

        _, fn = qml.snapshots(tape)

        expected_keys = [0, 2, "very_important_state", "execution_results"]
        assert "snapshot_tags" in fn.keywords
        assert len(fn.keywords["snapshot_tags"]) == num_snapshots + 1
        assert all(key in fn.keywords["snapshot_tags"] for key in expected_keys)

        tape_no_meas = qml.tape.QuantumTape(ops)

        _, fn_no_meas = qml.snapshots(tape_no_meas)

        expected_keys.remove("execution_results")
        assert "snapshot_tags" in fn.keywords
        assert len(fn_no_meas.keywords["snapshot_tags"]) == num_snapshots
        assert all(key in fn_no_meas.keywords["snapshot_tags"] for key in expected_keys)


class TestSnapshotSupportedQNode:
    """Test the Snapshot instruction for simulators."""

    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "backprop", "parameter-shift", "adjoint"])
    def test_default_qubit_legacy_opmath(self, method):
        """Test that multiple snapshots are returned correctly on the state-vector simulator."""
        dev = qml.device("default.qubit.legacy", wires=2)

        assert qml.debugging._is_snapshot_compatible(dev)

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

        _compare_numpy_dicts(result, expected)

    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "backprop", "parameter-shift", "adjoint"])
    def test_default_qubit2(self, method):
        """Test that multiple snapshots are returned correctly on the new
        state-vector simulator."""
        dev = qml.device("default.qubit")

        assert qml.debugging._is_snapshot_compatible(dev)

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

        _compare_numpy_dicts(result, expected)

    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "parameter-shift"])
    def test_default_mixed(self, method):
        """Test that multiple snapshots are returned correctly on the density-matrix simulator."""
        dev = qml.device("default.mixed", wires=2)

        assert qml.debugging._is_snapshot_compatible(dev)

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

        _compare_numpy_dicts(result, expected)

    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "parameter-shift"])
    def test_default_gaussian(self, method):
        """Test that multiple snapshots are returned correctly on the CV simulator."""
        dev = qml.device("default.gaussian", wires=2)

        assert qml.debugging._is_snapshot_compatible(dev)

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

    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "parameter-shift"])
    def test_default_qutrit_mixed(self, method):
        """Test that multiple snapshots are returned correctly on the density-matrix simulator."""
        np.random.seed(1)

        dev = qml.device("default.qutrit.mixed", wires=2, shots=100)

        assert qml.debugging._is_snapshot_compatible(dev)

        @qml.qnode(dev, diff_method=method)
        def circuit():
            qml.Snapshot(shots=None)
            qml.THadamard(wires=0)
            qml.Snapshot(measurement=qml.counts())
            qml.TSWAP(wires=[0, 1])
            qml.Snapshot(measurement=qml.probs(), shots=None)
            return qml.counts()

        circuit()
        assert dev._debugger is None

        result = qml.snapshots(circuit)()
        expected_first_state = np.zeros((9, 9))
        expected_first_state[0, 0] = 1.0
        expected = {
            0: expected_first_state,
            1: {"00": 31, "10": 34, "20": 35},
            2: np.array([0.33333333, 0.33333333, 0.33333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "execution_results": {"00": 24, "01": 39, "02": 37},
        }

        assert result["execution_results"] == expected["execution_results"]

        del result["execution_results"]
        del expected["execution_results"]
        del result[1]
        del expected[1]

        _compare_numpy_dicts(result, expected)

    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "parameter-shift"])
    def test_default_qutrit(self, method):
        """Test that multiple snapshots are returned correctly on the density-matrix simulator."""
        np.random.seed(7)

        dev = qml.device("default.qutrit", wires=2, shots=100)

        assert qml.debugging._is_snapshot_compatible(dev)

        @qml.qnode(dev, diff_method=method)
        def circuit():
            qml.Snapshot(shots=None)
            qml.THadamard(wires=0)
            qml.Snapshot("very_important_state", shots=None)
            qml.TSWAP(wires=[0, 1])
            qml.Snapshot(measurement=qml.probs(), shots=None)
            return qml.counts()

        circuit()
        assert dev._debugger is None

        result = qml.snapshots(circuit)()
        expected = {
            0: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "very_important_state": np.array(
                [
                    0.0 - 0.57735027j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 - 0.57735027j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 - 0.57735027j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ]
            ),
            2: np.array([0.33, 0.33, 0.34, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "execution_results": {"00": 39, "01": 31, "02": 30},
        }

        assert result["execution_results"] == expected["execution_results"]

        del result["execution_results"]
        del expected["execution_results"]

        _compare_numpy_dicts(result, expected)

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

    @pytest.mark.parametrize("shots,expected_expval", [(None, 0), (1, -1), (100, -0.22)])
    def test_different_shots(self, shots, expected_expval):
        """Test that snapshots are returned correctly with different QNode shot values."""
        dev = qml.device("default.qubit", wires=2, shots=shots)

        @qml.qnode(dev)
        def circuit():
            qml.Snapshot(shots=None)
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state", shots=None)
            qml.CNOT(wires=[0, 1])
            qml.Snapshot(shots=None)
            return qml.expval(qml.PauliX(0))

        result = qml.snapshots(circuit)()

        expected = {
            0: np.array([1, 0, 0, 0]),
            "very_important_state": np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0]),
            2: np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),
            "execution_results": np.array(expected_expval),
        }

        _compare_numpy_dicts(result, expected)

    @pytest.mark.parametrize(
        "m,expected_result",
        [
            ("expval", np.array(0)),
            ("var", np.array(1)),
            ("probs", np.array([0.5, 0, 0, 0.5])),
            ("state", np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])),
        ],
    )
    @pytest.mark.parametrize("force_qnode_transform", [False, True])
    def test_different_measurements(self, m, expected_result, force_qnode_transform, mocker):
        """Test that snapshots are returned correctly with different QNode measurements."""
        mocker.patch(
            "pennylane.debugging._is_snapshot_compatible", return_value=not force_qnode_transform
        )
        # Since we can't spy on `get_snapshots`, we do it the other way around
        # and verify that the default transform function is(not) called
        spy = mocker.spy(qml.debugging.snapshots, "default_qnode_transform")

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

        assert spy.call_count == (1 if force_qnode_transform else 0)

        _compare_numpy_dicts(result, expected)

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

        _compare_numpy_dicts(result, expected)

    def test_all_state_measurement_snapshot(self):
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

        _compare_numpy_dicts(result, expected)

    def test_all_sample_measurement_snapshot(self):
        """Test that the correct measurement snapshots are returned for different measurement types."""
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.RX(phi=0.1, wires=0)
            # Shot-based measurements
            qml.Snapshot(measurement=qml.expval(qml.PauliZ(0)), shots=500)
            qml.Snapshot(measurement=qml.var(qml.PauliZ(0)), shots=500)
            qml.Snapshot(measurement=qml.probs(0), shots=500)
            qml.Snapshot(measurement=qml.counts(wires=0), shots=500)
            qml.Snapshot(measurement=qml.sample(wires=0), shots=5)

            return qml.expval(qml.PauliZ(0))

        result = qml.snapshots(circuit)()

        expected = {
            0: 1.0,
            1: 0.03174399999999999,
            2: np.array([1.0, 0.0]),
            3: {"0": 500},
            4: np.array([0, 0, 0, 0, 0]),
            "execution_results": np.array(0.99500417),
        }

        assert result[3]["0"] == 500
        del result[3]
        del expected[3]

        _compare_numpy_dicts(result, expected)

    def test_sample_measurement_with_missing_shots(self):
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.Snapshot(measurement=qml.sample())
            return qml.expval(qml.PauliZ(0))

        # Expect a NotImplemented to be raised here since no shots has
        # been provided to the snapshot due to the analytical device
        with pytest.raises(NotImplementedError):
            qml.snapshots(circuit)()

    def test_state_measurement_with_unexpected_shots(self):
        dev = qml.device("default.qubit", shots=200)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.Snapshot(measurement=qml.state())
            return qml.expval(qml.PauliZ(0))

        # Expect a NotImplemented to be raised here since no shots has
        # been provided to the snapshot due to the analytical device
        with pytest.raises(
            AttributeError, match="'StateMP' object has no attribute 'process_samples'"
        ):
            qml.snapshots(circuit)()

    @pytest.mark.parametrize("force_qnode_transform", (False, True))
    def test_override_snapshot_shots(self, force_qnode_transform, mocker):
        mocker.patch(
            "pennylane.debugging._is_snapshot_compatible", return_value=not force_qnode_transform
        )

        dev = qml.device("default.qubit", shots=200)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.Snapshot(measurement=qml.counts())
            return qml.expval(qml.PauliZ(0))

        results = qml.snapshots(circuit)()
        assert sum(results[0].values()) == 200

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

    @pytest.mark.xfail
    def test_default_clifford_with_arbitrary_measurement(self):
        dev = qml.device("default.clifford")

        assert qml.debugging._is_snapshot_compatible(dev)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.Snapshot(measurement=qml.expval(qml.X(0)))
            qml.S(wires=0)
            return qml.expval(qml.X(0))

        qml.snapshots(circuit)()


class TestSnapshotUnsupportedQNode:
    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "parameter-shift"])
    def test_lightning_qubit(self, method):
        dev = qml.device("lightning.qubit", wires=2)

        assert not qml.debugging._is_snapshot_compatible(dev)

        @qml.qnode(dev, diff_method=method)
        def circuit():
            qml.Snapshot()
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

        # can run the circuit
        assert circuit() == 0

        with pytest.warns(UserWarning, match="resulting in a total of 4 executions."):
            result = qml.snapshots(circuit)()

        expected = {
            0: np.array([1.0, 0.0, 0.0, 0.0]),
            "very_important_state": np.array([0.70710678, 0.0, 0.70710678, 0.0]),
            2: np.array([0.70710678, 0.0, 0.0, 0.70710678]),
            "execution_results": 0.0,
        }

        _compare_numpy_dicts(result, expected)

    def test_lightning_qubit_adjoint_fails_for_empty_snapshots(self):
        """Test lightning with adjoing differentiation fails with default snapshot as it
        falls to qml.state() which is not supported"""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, diff_method="adjoint")
        def circuit():
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

        with pytest.raises(
            qml.DeviceError, match="not accepted for analytic simulation on adjoint"
        ):
            qml.snapshots(circuit)()

    def test_lightning_counts(self):
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev, diff_method=None)
        def circuit():
            qml.Hadamard(0)
            qml.Snapshot(measurement=qml.counts(wires=1), shots=200)
            return qml.expval(qml.PauliZ(0))

        result = qml.snapshots(circuit)()
        assert result[0]["0"] == 200
