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
from contextlib import nullcontext

import numpy as np
import pytest
from flaky import flaky

import pennylane as qml


def _compare_numpy_dicts(dict1, dict2):
    assert dict1.keys() == dict2.keys()
    assert all(np.allclose(dict1[key], dict_2_val) for key, dict_2_val in dict2.items())


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

    def test_snapshot_fails_with_mcm(self):
        with pytest.raises(
            ValueError, match="Mid-circuit measurements can not be used in snapshots"
        ):
            qml.Snapshot(measurement=qml.measurements.MidMeasureMP(1))

    def test_snapshot_fails_with_non_str_tags(self):
        with pytest.raises(ValueError, match="tags can only be of type 'str'"):
            qml.Snapshot(123, qml.state())

        with pytest.raises(ValueError, match="tags can only be of type 'str'"):
            qml.Snapshot(qml.state())


@pytest.mark.parametrize(
    "dev",
    [
        # Two supported devices
        qml.device("default.qubit"),
        qml.device("default.mixed", wires=2),
        # Two non-supported devices
        qml.device("default.qutrit", wires=2),
        qml.device("lightning.qubit", wires=2),
    ],
)
class TestSnapshotGeneral:
    def test_sample_measurement_with_analytical_device_fails(self, dev):
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.Snapshot(measurement=qml.sample())
            return qml.expval(qml.PauliZ(0))

        # Expect a DeviceError to be raised here since no shots has
        # been provided to the snapshot due to the analytical device
        with pytest.raises(qml.DeviceError):
            qml.snapshots(circuit)()

    def test_state_measurement_with_finite_shot_device_fails(self, dev):
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.Snapshot(measurement=qml.state())
            return qml.expval(qml.PauliZ(0))

        # Expect a DeviceError to be raised here since no shots has
        # been provided to the snapshot due to the finite-shot device
        with pytest.raises(qml.DeviceError):
            qml.snapshots(circuit)(shots=200)

    @pytest.mark.parametrize("diff_method", [None, "parameter-shift"])
    def test_all_state_measurement_snapshot_pure_dev(self, dev, diff_method):
        """Test that the correct measurement snapshots are returned for different measurement types."""
        if isinstance(
            dev, (qml.devices.default_mixed.DefaultMixed, qml.devices.default_qutrit.DefaultQutrit)
        ):
            pytest.skip()

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qml.Snapshot(measurement=qml.expval(qml.PauliZ(0)))
            qml.RX(x, wires=0)
            qml.Snapshot(measurement=qml.var(qml.PauliZ(0)))
            qml.Snapshot(measurement=qml.probs(0))
            qml.Snapshot(measurement=qml.state())

            return qml.expval(qml.PauliZ(0))

        phi = 0.1
        with (
            pytest.warns(UserWarning, match="resulting in a total of 5 executions.")
            if "lightning" in dev.name
            else nullcontext()
        ):
            result = qml.snapshots(circuit)(phi)

        expected = {
            0: np.array(1),
            1: np.array(1 - np.cos(phi) ** 2),
            2: np.array([np.cos(phi / 2) ** 2, np.sin(phi / 2) ** 2]),
            3: np.array([np.cos(phi / 2), -1j * np.sin(phi / 2)]),
            "execution_results": np.array(np.cos(phi)),
        }

        if "lightning" in dev.name:
            expected[3] = np.kron(expected[3], np.array([1.0, 0.0]))

        _compare_numpy_dicts(result, expected)

    def test_empty_snapshots(self, dev):
        """Test that snapshots function in the absence of any Snapshot operations."""

        @qml.qnode(dev)
        def circuit():
            if isinstance(dev, qml.QutritDevice):
                qml.THadamard(wires=0)
                return qml.expval(qml.GellMann(0, index=6))

            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliX(0))

        result = qml.snapshots(circuit)()
        if isinstance(dev, qml.QutritDevice):
            expected = {"execution_results": np.array(0.66666667)}
        else:
            expected = {"execution_results": np.array(1.0)}

        _compare_numpy_dicts(result, expected)


class TestSnapshotSupportedQNode:
    """Test the Snapshot instruction for simulators."""

    # pylint: disable=protected-access
    @pytest.mark.parametrize("diff_method", ["backprop", "adjoint"])
    def test_default_qubit_with_backprob_and_adjoint(self, diff_method):
        dev = qml.device("default.qubit")

        assert qml.debugging._is_snapshot_compatible(dev)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit():
            qml.Hadamard(wires=0)
            qml.Snapshot("important_expval", measurement=qml.expval(qml.PauliX(0)))
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

        circuit()
        assert dev._debugger is None
        if diff_method is not None:
            assert circuit.interface == "auto"

        result = qml.snapshots(circuit)()
        expected = {
            "important_expval": np.array(1.0),
            1: np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),
            "execution_results": np.array(0),
        }

        _compare_numpy_dicts(result, expected)

    @pytest.mark.parametrize("diff_method", [None, "backprop", "parameter-shift", "adjoint"])
    def test_default_qubit_legacy_only_supports_state(self, diff_method):
        dev = qml.device("default.qubit.legacy", wires=2)

        assert qml.debugging._is_snapshot_compatible(dev)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit_faulty():
            qml.Hadamard(wires=0)
            qml.Snapshot("important_expval", measurement=qml.expval(qml.PauliX(0)))
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

        circuit_faulty()
        assert dev._debugger is None
        if diff_method is not None:
            assert circuit_faulty.interface == "auto"

        with pytest.raises(NotImplementedError, match="only supports `qml.state` measurements"):
            qml.snapshots(circuit_faulty)()

        @qml.qnode(dev, diff_method=diff_method)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

        with pytest.raises(qml.DeviceError, match="not accepted with finite shots"):
            qml.snapshots(circuit)(shots=200)

        result = qml.snapshots(circuit)()
        expected = {
            0: np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]),
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
            qml.Snapshot(measurement=qml.expval(qml.PauliX(0)))
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot(measurement=qml.probs())
            return qml.expval(qml.PauliX(0))

        circuit()
        assert dev._debugger is None

        result = qml.snapshots(circuit)()
        expected = {
            0: np.array(0),
            2: np.array([0.5, 0.0, 0.0, 0.5]),
            "very_important_state": np.array(
                [[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]]
            ),
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
    @pytest.mark.parametrize("diff_method", [None, "parameter-shift"])
    def test_default_qutrit_mixed_finite_shot(self, diff_method):
        """Test that multiple snapshots are returned correctly on the qutrit density-matrix simulator."""
        dev = qml.device("default.qutrit.mixed", wires=2, shots=100)

        assert qml.debugging._is_snapshot_compatible(dev)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(add_bad_snapshot: bool):
            qml.THadamard(wires=0)
            qml.Snapshot(measurement=qml.counts())
            qml.TSWAP(wires=[0, 1])
            if add_bad_snapshot:
                qml.Snapshot(measurement=qml.probs())
            return qml.counts()

        circuit(False)
        assert dev._debugger is None

        # This should fail since finite-shot probs() isn't supported
        with pytest.raises(NotImplementedError):
            qml.snapshots(circuit)(add_bad_snapshot=True)

        result = qml.snapshots(circuit)(add_bad_snapshot=False)

        expected = {
            0: {"00": 34, "10": 37, "20": 29},
            "execution_results": {"00": 37, "01": 33, "02": 30},
        }

        assert result[0] == expected[0]
        assert result["execution_results"] == expected["execution_results"]

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
            qml.Snapshot(measurement=qml.probs())
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
            1: np.array([0.5, 0.0, 0.25, 0.25]),
            "execution_results": np.array(0.36819668),
        }

        _compare_numpy_dicts(result, expected)

    def test_adjoint_circuit(self):
        """Test that snapshots are returned correctly when adjointed."""
        dev = qml.device("default.qubit", wires=2)

        def circuit(params, wire):
            qml.Rot(*params, wire)
            qml.Snapshot()
            qml.Snapshot(measurement=qml.probs())
            qml.Hadamard(wire)

        @qml.qnode(dev)
        def qnode(params):
            qml.Hadamard(0)
            qml.adjoint(circuit, 0)(params, wire=1)
            return qml.expval(qml.PauliZ(1))

        params = np.array([1.3, 1.4, 0.2])
        result = qml.snapshots(qnode)(params)
        expected = {
            0: np.array([0.25, 0.25, 0.25, 0.25]),
            1: np.array([0.5, 0.5, 0.5, 0.5]),
            "execution_results": np.array(0.96580634),
        }

        _compare_numpy_dicts(result, expected)

    def test_all_sample_measurement_snapshot(self):
        """Test that the correct measurement snapshots are returned for different measurement types."""
        dev = qml.device("default.qubit", wires=1, shots=10)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            # Shot-based measurements
            qml.Snapshot(measurement=qml.expval(qml.PauliZ(0)))
            qml.Snapshot(measurement=qml.var(qml.PauliZ(0)))
            qml.Snapshot(measurement=qml.probs(0))
            qml.Snapshot(measurement=qml.counts(wires=0))
            qml.Snapshot(measurement=qml.sample(wires=0))

            return qml.expval(qml.PauliZ(0))

        result = qml.snapshots(circuit)()

        expected = {
            0: -0.6,
            1: 0.64,
            2: np.array([0.6, 0.4]),
            3: {"0": 2, "1": 8},
            4: np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 0]),
            "execution_results": np.array(0.2),
        }

        assert result[3]["0"] == 2
        assert result[3]["1"] == 8

        del result[3]
        del expected[3]

        _compare_numpy_dicts(result, expected)

    def test_unsupported_snapshot_measurement(self):
        """Test that an exception is raised when an unsupported measurement is provided to the snapshot."""
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            with pytest.raises(
                ValueError,
                match="The measurement PauliZ is not supported as it is not an instance "
                "of <class 'pennylane.measurements.measurements.MeasurementProcess'>",
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
    @flaky(max_runs=3)
    def test_lightning_qubit_finite_shots(self):
        dev = qml.device("lightning.qubit", wires=2, shots=200)

        @qml.qnode(dev, diff_method=None)
        def circuit():
            qml.Hadamard(0)
            qml.Snapshot(measurement=qml.counts(wires=0))
            return qml.expval(qml.PauliX(1))

        result = qml.snapshots(circuit)()
        assert 90 <= result[0]["0"] <= 110
        assert np.allclose(result["execution_results"], np.array(0.0))

    @pytest.mark.parametrize("diff_method", ["backprop", "adjoint"])
    def test_lightning_qubit_fails_for_state_snapshots_with_adjoint_and_backprop(self, diff_method):
        """Test lightning with backprop and adjoint differentiation fails with default snapshot as it
        falls to qml.state() which is not supported"""

        dev = qml.device("lightning.qubit", wires=2)

        with pytest.raises(
            qml.QuantumFunctionError, match=f"does not support {diff_method} with requested circuit"
        ):

            @qml.qnode(dev, diff_method=diff_method)
            def circuit():
                qml.Snapshot()
                return qml.expval(qml.PauliX(0))

            qml.snapshots(circuit)()

    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "parameter-shift"])
    def test_default_qutrit(self, method):
        """Test that multiple snapshots are returned correctly on the pure qutrit simulator."""
        np.random.seed(7)

        dev = qml.device("default.qutrit", wires=2, shots=100)

        assert not qml.debugging._is_snapshot_compatible(dev)

        @qml.snapshots
        @qml.qnode(dev, diff_method=method)
        def circuit():
            qml.THadamard(wires=0)
            qml.Snapshot(measurement=qml.probs())
            qml.TSWAP(wires=[0, 1])
            return qml.counts()

        with pytest.warns(UserWarning, match="total of 2 executions."):
            result = circuit()

        expected = {
            0: np.array([0.27, 0.0, 0.0, 0.38, 0.0, 0.0, 0.35, 0.0, 0.0]),
            "execution_results": {"00": 39, "01": 31, "02": 30},
        }

        assert result["execution_results"] == expected["execution_results"]

        del result["execution_results"]
        del expected["execution_results"]

        _compare_numpy_dicts(result, expected)
