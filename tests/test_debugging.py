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
from unittest.mock import patch

import numpy as np
import pytest
from flaky import flaky
from scipy.stats import ttest_ind

import pennylane as qml
from pennylane import numpy as qnp
from pennylane.debugging import PLDB, pldb_device_manager
from pennylane.ops.functions.equal import assert_equal


def _compare_numpy_dicts(dict1, dict2):
    assert dict1.keys() == dict2.keys()
    assert all(np.allclose(dict1[key], dict_2_val) for key, dict_2_val in dict2.items())


class TestSnapshotTape:
    @pytest.mark.parametrize("shots", [None, 100])
    def test_snapshot_output_tapes(self, shots):
        ops = [
            qml.Snapshot(),
            qml.Hadamard(wires=0),
            qml.Snapshot("very_important_state"),
            qml.CNOT(wires=[0, 1]),
            qml.Snapshot(),
        ]

        measurements = [qml.expval(qml.PauliX(0))]

        num_snapshots = len(tuple(filter(lambda x: isinstance(x, qml.Snapshot), ops)))

        expected_tapes = [
            qml.tape.QuantumTape([], [qml.state()], shots=shots),
            qml.tape.QuantumTape([qml.Hadamard(0)], [qml.state()], shots=shots),
            qml.tape.QuantumTape([qml.Hadamard(0), qml.CNOT((0, 1))], [qml.state()], shots=shots),
            qml.tape.QuantumTape(
                [qml.Hadamard(0), qml.CNOT((0, 1))], [qml.expval(qml.X(0))], shots=shots
            ),
        ]

        tape = qml.tape.QuantumTape(ops, measurements, shots=shots)
        tapes, _ = qml.snapshots(tape)

        assert len(tapes) == num_snapshots + 1

        for out, expected in zip(tapes, expected_tapes):
            assert_equal(out, expected)

        tape_no_meas = qml.tape.QuantumTape(ops, shots=shots)
        tapes_no_meas, _ = qml.snapshots(tape_no_meas)

        assert len(tapes_no_meas) == num_snapshots

        for out, expected in zip(tapes_no_meas, expected_tapes[:-1]):
            assert_equal(out, expected)

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
        assert fn(["a", 1, 2, []]) == {
            0: "a",
            "very_important_state": 1,
            2: 2,
            "execution_results": [],
        }

        tape_no_meas = qml.tape.QuantumTape(ops)

        _, fn_no_meas = qml.snapshots(tape_no_meas)

        expected_keys.remove("execution_results")
        assert "snapshot_tags" in fn.keywords
        assert len(fn_no_meas.keywords["snapshot_tags"]) == num_snapshots
        assert all(key in fn_no_meas.keywords["snapshot_tags"] for key in expected_keys)
        assert fn(["a", 1, 2]) == {
            0: "a",
            "very_important_state": 1,
            2: 2,
        }

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

    def test_non_StateMP_state_measurements_with_finite_shot_device_fails(self, dev):
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.Snapshot(measurement=qml.mutual_info(0, 1))
            return qml.expval(qml.PauliZ(0))

        # Expect a DeviceError to be raised here since no shots has
        # been provided to the snapshot due to the finite-shot device
        with pytest.raises(qml.DeviceError):
            qml.snapshots(circuit)(shots=200)

    def test_StateMP_with_finite_shot_device_passes(self, dev):
        if "lightning" in dev.name or "mixed" in dev.name:
            pytest.skip()

        @qml.qnode(dev)
        def circuit():
            qml.Snapshot(measurement=qml.state())
            qml.Snapshot()

            if isinstance(dev, qml.devices.QutritDevice):
                return qml.expval(qml.GellMann(0, 1))

            return qml.expval(qml.PauliZ(0))

        with (
            pytest.warns(UserWarning, match="Requested state or density matrix with finite shots")
            if isinstance(dev, qml.devices.default_qutrit.DefaultQutrit)
            else nullcontext()
        ):
            qml.snapshots(circuit)(shots=200)

    @pytest.mark.parametrize("diff_method", [None, "parameter-shift"])
    def test_all_state_measurement_snapshot_pure_qubit_dev(self, dev, diff_method):
        """Test that the correct measurement snapshots are returned for different measurement types."""
        if isinstance(dev, (qml.devices.default_mixed.DefaultMixed, qml.devices.QutritDevice)):
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
            pytest.warns(UserWarning, match="Snapshots are not supported for the given device")
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
            if isinstance(dev, qml.devices.QutritDevice):
                qml.THadamard(wires=0)
                return qml.expval(qml.GellMann(0, index=6))

            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliX(0))

        result = qml.snapshots(circuit)()
        if isinstance(dev, qml.devices.QutritDevice):
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

        assert qml.debugging.snapshot._is_snapshot_compatible(dev)

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

    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "parameter-shift"])
    def test_default_mixed(self, method):
        """Test that multiple snapshots are returned correctly on the density-matrix simulator."""
        dev = qml.device("default.mixed", wires=2)

        assert qml.debugging.snapshot._is_snapshot_compatible(dev)

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

        assert qml.debugging.snapshot._is_snapshot_compatible(dev)

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

        # TODO: not sure what to do with this test so leaving this here for now.
        np.random.seed(9872653)

        dev = qml.device("default.qutrit.mixed", wires=2, shots=100)

        assert qml.debugging.snapshot._is_snapshot_compatible(dev)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(add_bad_snapshot: bool):
            qml.THadamard(wires=0)
            qml.Snapshot(measurement=qml.counts())
            qml.TSWAP(wires=[0, 1])
            if add_bad_snapshot:
                qml.Snapshot(measurement=qml.probs())
            qml.Snapshot()
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
        assert np.allclose(
            result[1][:3],
            np.array([0.33333333, 0.33333333, 0.33333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        assert result["execution_results"] == expected["execution_results"]

        # Make sure shots are overridden correctly
        result = qml.snapshots(circuit)(add_bad_snapshot=False, shots=200)
        assert result[0] == {"00": 74, "10": 58, "20": 68}

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
            "pennylane.debugging.snapshot._is_snapshot_compatible",
            return_value=not force_qnode_transform,
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

        # TODO: not sure what to do with this test so leaving this here for now.
        np.random.seed(9872653)

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

        # TODO: The fact that this entire test depends on a global seed is not good
        np.random.seed(9872653)

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
            qml.Snapshot()

            return qml.expval(qml.PauliZ(0))

        result = qml.snapshots(circuit)()

        expected = {
            0: -0.6,
            1: 0.64,
            2: np.array([0.6, 0.4]),
            3: {"0": 2, "1": 8},
            4: np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 0]),
            5: np.array([0.70710678, 0.70710678]),
            "execution_results": np.array(0.2),
        }

        assert result[3]["0"] == 2
        assert result[3]["1"] == 8

        del result[3]
        del expected[3]

        _compare_numpy_dicts(result, expected)

        # Make sure shots are overridden correctly
        result = qml.snapshots(circuit)(shots=200)
        assert result[3] == {"0": 98, "1": 102}
        assert np.allclose(result[5], expected[5])

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


class TestSnapshotUnsupportedQNode:
    """Unit tests for qml.snapshots when using with qnodes with unsupported devices"""

    def test_unsupported_device_warning(self):
        """Test that a warning is raised when the device being used by a qnode does not natively support
        qml.Snapshot"""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.Snapshot()
            qml.CNOT([0, 1])
            return qml.expval(qml.Z(1))

        with pytest.warns(UserWarning, match="Snapshots are not supported"):
            _ = qml.snapshots(circuit)

    @flaky(max_runs=3)
    def test_lightning_qubit_finite_shots(self):
        dev = qml.device("lightning.qubit", wires=2, shots=500)

        @qml.qnode(dev, diff_method=None)
        def circuit():
            qml.Hadamard(0)
            qml.Snapshot(measurement=qml.counts(wires=0))
            return qml.expval(qml.PauliX(1))

        # TODO: fallback to simple `np.allclose` tests once `setRandomSeed` is exposed from the lightning C++ code
        counts, expvals = tuple(zip(*(qml.snapshots(circuit)().values() for _ in range(50))))
        assert ttest_ind([count["0"] for count in counts], 250).pvalue >= 0.75
        assert ttest_ind(expvals, 0.0).pvalue >= 0.75

        # Make sure shots are overridden correctly
        counts, _ = tuple(zip(*(qml.snapshots(circuit)(shots=1000).values() for _ in range(50))))
        assert ttest_ind([count["0"] for count in counts], 500).pvalue >= 0.75

    @pytest.mark.parametrize("diff_method", ["backprop", "adjoint"])
    def test_lightning_qubit_fails_for_state_snapshots_with_adjoint_and_backprop(self, diff_method):
        """Test lightning with backprop and adjoint differentiation fails with default snapshot as it
        falls to qml.state() which is not supported"""

        dev = qml.device("lightning.qubit", wires=2)

        with (
            pytest.raises(
                qml.DeviceError,
                match=r"not accepted for analytic simulation on adjoint \+ lightning.qubit",
            )
            if diff_method == "adjoint"
            else pytest.raises(
                qml.QuantumFunctionError,
                match=f"does not support {diff_method} with requested circuit",
            )
        ):

            @qml.qnode(dev, diff_method=diff_method)
            def circuit():
                qml.Snapshot()
                return qml.expval(qml.PauliX(0))

            qml.snapshots(circuit)()

    def test_state_wire_order_preservation(self):
        """Test that the snapshots wire order reflects the wire order on the device."""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit():
            qml.X(1)
            qml.Snapshot()
            return qml.state()

        out = qml.snapshots(circuit)()

        assert qml.math.allclose(out[0], out["execution_results"])

    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "parameter-shift"])
    def test_default_qutrit(self, method):
        """Test that multiple snapshots are returned correctly on the pure qutrit simulator."""

        dev = qml.device("default.qutrit", wires=2)

        assert not qml.debugging.snapshot._is_snapshot_compatible(dev)

        @qml.qnode(dev, diff_method=method)
        def circuit():
            qml.THadamard(wires=0)
            qml.Snapshot(measurement=qml.probs())
            qml.TSWAP(wires=[0, 1])
            return qml.probs()

        with pytest.warns(UserWarning, match="Snapshots are not supported for the given device"):
            circuit = qml.snapshots(circuit)

        result = circuit()
        expected = {
            0: np.array([1 / 3, 0.0, 0.0, 1 / 3, 0.0, 0.0, 1 / 3, 0.0, 0.0]),
            "execution_results": np.array([1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }

        assert np.allclose(result["execution_results"], expected["execution_results"])

        del result["execution_results"]  # pylint: disable=unsupported-delete-operation
        del expected["execution_results"]

        _compare_numpy_dicts(result, expected)

        # Make sure shots are overridden correctly
        result = circuit(shots=200)
        assert np.allclose(
            result[0],
            np.array([1 / 3, 0.0, 0.0, 1 / 3, 0.0, 0.0, 1 / 3, 0.0, 0.0]),
            atol=0.1,
            rtol=0,
        )


# pylint: disable=protected-access
class TestPLDB:
    """Test the interactive debugging integration"""

    def test_pldb_init(self):
        """Test that PLDB initializes correctly"""
        debugger = PLDB()
        assert debugger.prompt == "[pldb] "
        assert getattr(debugger, "_PLDB__active_dev") is None

    def test_valid_context_outside_qnode(self):
        """Test that valid_context raises an error when breakpoint
        is called outside of a qnode execution."""

        with pytest.raises(
            RuntimeError, match="Can't call breakpoint outside of a qnode execution"
        ):
            with qml.queuing.AnnotatedQueue() as _:
                qml.X(0)
                qml.breakpoint()
                qml.Hadamard(0)

        def my_qfunc():
            qml.X(0)
            qml.breakpoint()
            qml.Hadamard(0)
            return qml.expval(qml.Z(0))

        with pytest.raises(
            RuntimeError, match="Can't call breakpoint outside of a qnode execution"
        ):
            _ = my_qfunc()

    def test_valid_context_not_compatible_device(self):
        """Test that valid_context raises an error when breakpoint
        is called with an incompatible device."""
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev)
        def my_circ():
            qml.X(0)
            qml.breakpoint()
            qml.Hadamard(0)
            return qml.expva(qml.Z(0))

        with pytest.raises(TypeError, match="Breakpoints not supported on this device"):
            _ = my_circ()

        PLDB.reset_active_dev()

    def test_add_device(self):
        """Test that we can add a device to the global active device list."""
        assert not PLDB.has_active_dev()

        dev1, dev2, dev3 = (
            qml.device("default.qubit", wires=3),
            qml.device("default.qubit"),
            qml.device("lightning.qubit", wires=1),
        )

        PLDB.add_device(dev1)
        assert PLDB.get_active_device() == dev1

        PLDB.add_device(dev2)  # overwrites dev1
        PLDB.add_device(dev3)  # overwrites dev2

        assert PLDB.get_active_device() == dev3

        PLDB.reset_active_dev()  # clean up the debugger active devices

    dev_names = (
        "default.qubit",
        "lightning.qubit",
    )

    @pytest.mark.parametrize("device_name", dev_names)
    def test_get_active_device(self, device_name):
        """Test that we can access the active device."""
        dev = qml.device(device_name, wires=2)
        with pldb_device_manager(dev) as _:
            assert PLDB.get_active_device() is dev

    def test_get_active_device_error_when_no_active_device(self):
        """Test that an error is raised if we try to get
        the active device when there are no active devices."""
        assert not PLDB.has_active_dev()

        with pytest.raises(RuntimeError, match="No active device to get"):
            _ = PLDB.get_active_device()

    @pytest.mark.parametrize("device_name", dev_names)
    def test_reset_active_device(self, device_name):
        """Test that we can rest the global active device list."""
        dev = qml.device(device_name, wires=2)
        PLDB.add_device(dev)
        assert PLDB.get_active_device() == dev

        PLDB.reset_active_dev()
        assert not PLDB.has_active_dev()

    def test_has_active_device(self):
        """Test that we can determine if there is an active device."""
        assert getattr(PLDB, "_PLDB__active_dev") is None

        dev = qml.device("default.qubit")
        PLDB.add_device(dev)
        assert PLDB.has_active_dev()

        PLDB.reset_active_dev()
        assert not PLDB.has_active_dev()

    tapes = (
        qml.tape.QuantumScript(
            ops=[qml.Hadamard(0), qml.CNOT([0, 1])],
            measurements=[qml.state()],
        ),
        qml.tape.QuantumScript(
            ops=[qml.Hadamard(0), qml.X(1)],
            measurements=[qml.expval(qml.Z(1))],
        ),
        qml.tape.QuantumScript(
            ops=[qml.Hadamard(0), qml.CNOT([0, 1])],
            measurements=[qml.probs()],
        ),
        qml.tape.QuantumScript(
            ops=[qml.Hadamard(0), qml.CNOT([0, 1])],
            measurements=[qml.probs(wires=[0])],
        ),
        qml.tape.QuantumScript(
            ops=[qml.Hadamard(0)],
            measurements=[qml.state()],
        ),  # Test that state expands to number of device wires
    )

    results = (
        qnp.array([1 / qnp.sqrt(2), 0, 0, 1 / qnp.sqrt(2)], dtype=complex),
        qnp.array(-1),
        qnp.array([1 / 2, 0, 0, 1 / 2]),
        qnp.array([1 / 2, 1 / 2]),
        qnp.array([1 / qnp.sqrt(2), 0, 1 / qnp.sqrt(2), 0], dtype=complex),
    )

    @pytest.mark.parametrize("tape, expected_result", zip(tapes, results))
    @pytest.mark.parametrize(
        "dev", (qml.device("default.qubit", wires=2), qml.device("lightning.qubit", wires=2))
    )
    def test_execute(self, dev, tape, expected_result):
        """Test that the _execute method works as expected."""
        PLDB.add_device(dev)
        executed_results = PLDB._execute((tape,))
        assert qnp.allclose(expected_result, executed_results)
        PLDB.reset_active_dev()


def test_tape():
    """Test that we can access the tape from the active queue."""
    with qml.queuing.AnnotatedQueue() as queue:
        qml.X(0)

        for i in range(3):
            qml.Hadamard(i)

        qml.Y(1)
        qml.Z(0)
        qml.expval(qml.Z(0))

        executed_tape = qml.debug_tape()

    expected_tape = qml.tape.QuantumScript.from_queue(queue)
    qml.assert_equal(expected_tape, executed_tape)


@pytest.mark.parametrize("measurement_process", (qml.expval(qml.Z(0)), qml.state(), qml.probs()))
@patch.object(PLDB, "_execute")
def test_measure(mock_method, measurement_process):
    """Test that the private measure function doesn't modify the active queue"""
    with qml.queuing.AnnotatedQueue() as queue:
        ops = [qml.X(0), qml.Y(1), qml.Z(0)] + [qml.Hadamard(i) for i in range(3)]
        measurements = [qml.expval(qml.X(2)), qml.state(), qml.probs(), qml.var(qml.Z(3))]
        qml.debugging.debugger._measure(measurement_process)

    executed_tape = qml.tape.QuantumScript.from_queue(queue)
    expected_tape = qml.tape.QuantumScript(ops, measurements)

    qml.assert_equal(expected_tape, executed_tape)  # no unexpected queuing

    expected_debugging_tape = qml.tape.QuantumScript(ops, measurements + [measurement_process])
    executed_debugging_tape = mock_method.call_args.args[0][0]

    qml.assert_equal(
        expected_debugging_tape, executed_debugging_tape
    )  # _execute was called with new measurements


@patch.object(PLDB, "_execute")
def test_state(_mock_method):
    """Test that the state function works as expected."""
    with qml.queuing.AnnotatedQueue() as queue:
        qml.RX(1.23, 0)
        qml.RY(0.45, 2)
        qml.sample()

        qml.debug_state()

    assert qml.state() not in queue


@patch.object(PLDB, "_execute")
def test_expval(_mock_method):
    """Test that the expval function works as expected."""
    for op in [qml.X(0), qml.Y(1), qml.Z(2), qml.Hadamard(0)]:
        with qml.queuing.AnnotatedQueue() as queue:
            qml.RX(1.23, 0)
            qml.RY(0.45, 2)
            qml.sample()

            qml.debug_expval(op)

        assert op not in queue
        assert qml.expval(op) not in queue


@patch.object(PLDB, "_execute")
def test_probs_with_op(_mock_method):
    """Test that the probs function works as expected."""

    for op in [None, qml.X(0), qml.Y(1), qml.Z(2)]:
        with qml.queuing.AnnotatedQueue() as queue:
            qml.RX(1.23, 0)
            qml.RY(0.45, 2)
            qml.sample()

            qml.debug_probs(op=op)

        assert op not in queue
        assert qml.probs(op=op) not in queue


@patch.object(PLDB, "_execute")
def test_probs_with_wires(_mock_method):
    """Test that the probs function works as expected."""

    for wires in [None, [0, 1], [2]]:
        with qml.queuing.AnnotatedQueue() as queue:
            qml.RX(1.23, 0)
            qml.RY(0.45, 2)
            qml.sample()

            qml.debug_probs(wires=wires)

        assert qml.probs(wires=wires) not in queue


@pytest.mark.parametrize("device_name", ("default.qubit", "lightning.qubit"))
def test_pldb_device_manager(device_name):
    """Test that the context manager works as expected."""
    assert not PLDB.has_active_dev()
    dev = qml.device(device_name, wires=2)

    with pldb_device_manager(dev) as _:
        assert PLDB.get_active_device() == dev

    assert not PLDB.has_active_dev()


@patch.object(PLDB, "set_trace")
def test_breakpoint_integration(mock_method):
    """Test that qml.breakpoint behaves as expected"""
    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def my_circ():
        qml.Hadamard(0)
        qml.CNOT([0, 1])
        qml.breakpoint()
        return qml.expval(qml.Z(1))

    mock_method.assert_not_called()  # Did not hit breakpoint
    my_circ()
    mock_method.assert_called_once()  # Hit breakpoint once.


@patch.object(PLDB, "set_trace")
def test_breakpoint_integration_with_valid_context_error(mock_method):
    """Test that the PLDB.valid_context() integrates well with qml.breakpoint"""
    dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev)
    def my_circ():
        qml.Hadamard(0)
        qml.CNOT([0, 1])
        qml.breakpoint()
        return qml.expval(qml.Z(1))

    with pytest.raises(TypeError, match="Breakpoints not supported on this device"):
        _ = my_circ()

    mock_method.assert_not_called()  # Error was raised before we triggered breakpoint
