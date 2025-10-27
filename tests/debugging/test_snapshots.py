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
Tests for snapshots.
"""
from contextlib import nullcontext

import numpy as np
import pytest
from scipy.stats import ttest_ind

import pennylane as qml
from pennylane.exceptions import DeviceError, PennyLaneDeprecationWarning, QuantumFunctionError
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
            qml.Snapshot(measurement=qml.probs()),
            qml.Snapshot(measurement=qml.sample(), shots=10),
        ]

        measurements = [qml.expval(qml.PauliX(0))]

        num_snapshots = len(tuple(filter(lambda x: isinstance(x, qml.Snapshot), ops)))

        expected_tapes = [
            qml.tape.QuantumTape([], [qml.state()], shots=None),
            qml.tape.QuantumTape([qml.Hadamard(0)], [qml.state()], shots=None),
            qml.tape.QuantumTape([qml.Hadamard(0), qml.CNOT((0, 1))], [qml.probs()], shots=shots),
            qml.tape.QuantumTape([qml.Hadamard(0), qml.CNOT((0, 1))], [qml.sample()], shots=10),
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

        tape_no_meas = qml.tape.QuantumScript(ops)

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
            qml.Snapshot(np.array(0.0), qml.state())

        with pytest.raises(ValueError, match="tags can only be of type 'str'"):
            qml.Snapshot(qml.state())

    @pytest.mark.parametrize(
        "dev", (qml.device("default.qubit"), qml.device("default.qutrit", wires=2))
    )
    def test_int_tag_fails_during_transform(self, dev):
        """Test ValueError is raised if user provided a int snapshot tag."""

        @qml.qnode(dev)
        def c():
            qml.Snapshot(2)
            return qml.expval(qml.Z(0))

        with pytest.raises(ValueError, match="can only be of type 'str'"):
            qml.snapshots(c)()


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
        with pytest.raises(DeviceError):
            qml.snapshots(circuit)()

    def test_non_StateMP_state_measurements_with_finite_shot_device_fails(self, dev):
        @qml.set_shots(shots=200)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.Snapshot(measurement=qml.mutual_info(0, 1))
            return qml.expval(qml.PauliZ(0))

        # Expect a DeviceError to be raised here since no shots has
        # been provided to the snapshot due to the finite-shot device
        with pytest.raises(DeviceError):
            qml.snapshots(circuit)()

    def test_StateMP_with_finite_shot_device_passes(self, dev):
        if "lightning" in dev.name or "mixed" in dev.name:
            pytest.skip()

        @qml.set_shots(shots=200)
        @qml.qnode(dev)
        def circuit():
            qml.Snapshot(measurement=qml.state())
            qml.Snapshot()

            if isinstance(dev, qml.devices.QutritDevice):
                return qml.expval(qml.GellMann(0, 1))

            return qml.expval(qml.PauliZ(0))

        _ = qml.snapshots(circuit)()

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

    def test_override_shots(self, dev):
        """Test that override shots allow snapshots to work with different numbers of measurements."""

        @qml.qnode(dev)
        def c():
            if dev.name != "default.qutrit":
                qml.H(0)
            qml.Snapshot("sample", qml.sample(wires=0), shots=5)
            qml.Snapshot("counts", qml.counts(wires=0, all_outcomes=True), shots=20)
            qml.Snapshot("probs", qml.probs(wires=0), shots=21)
            return qml.state()

        out = qml.snapshots(c)()

        assert out["sample"].shape == (5, 1)
        assert out["counts"]["0"] + out["counts"].get("1", 0) == 20
        if dev.name != "default.qutrit":
            # very rare that it will be *exactly* [0.5, 0.5] if 20 shots
            assert not qml.math.allclose(out["probs"], np.array([0.5, 0.5]), atol=1e-8)

    def test_override_analytic(self, dev):
        """Test that finite shots can be written with analytic calculations."""

        if dev.name == "default.qutrit":
            pytest.skip("hard to write generic test that works with qutrits.")

        @qml.transform
        def set_shots(tape, shots):
            return (tape.copy(shots=shots),), lambda res: res[0]

        @qml.qnode(dev, diff_method=None)
        def c():
            qml.H(0)
            qml.Snapshot("probs", qml.probs(wires=0), shots=None)
            return qml.sample(wires=0)

        out = qml.snapshots(set_shots(c, shots=10))()
        assert qml.math.allclose(out["probs"], np.array([0.5, 0.5]))
        assert out["execution_results"].shape == (10, 1)


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

        dev = qml.device("default.qutrit.mixed", wires=2)

        assert qml.debugging.snapshot._is_snapshot_compatible(dev)

        @qml.set_shots(100)
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
        result = qml.snapshots(qml.set_shots(shots=200)(circuit))(add_bad_snapshot=False)
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

        dev = qml.device("default.qubit", wires=1)

        @qml.set_shots(10)
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
            4: np.array([[0, 1, 0, 1, 0, 1, 1, 0, 0, 0]]).transpose(),
            5: np.array([0.70710678, 0.70710678]),
            "execution_results": np.array(0.2),
        }

        assert result[3]["0"] == 2
        assert result[3]["1"] == 8

        del result[3]
        del expected[3]

        _compare_numpy_dicts(result, expected)

        result = qml.snapshots(qml.set_shots(circuit, shots=200))()
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

    # Improper ttest_ind usage, but leaving it here for now;
    # should be revised and fixed soon
    # current failure rate: ~7%
    # FIXME: [sc-92966]
    @pytest.mark.local_salt(2)
    def test_lightning_qubit_finite_shots(self, seed):
        dev = qml.device("lightning.qubit", wires=2, seed=seed)

        @qml.set_shots(500)
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
        counts, _ = tuple(
            zip(*(qml.snapshots(qml.set_shots(circuit, shots=1000))().values() for _ in range(50)))
        )
        assert ttest_ind([count["0"] for count in counts], 500).pvalue >= 0.75

    @pytest.mark.parametrize("diff_method", ["backprop", "adjoint"])
    def test_lightning_qubit_fails_for_state_snapshots_with_adjoint_and_backprop(self, diff_method):
        """Test lightning with backprop and adjoint differentiation fails with default snapshot as it
        falls to qml.state() which is not supported"""

        dev = qml.device("lightning.qubit", wires=2)

        with pytest.raises(
            QuantumFunctionError,
            match=f"does not support {diff_method} with requested circuit",
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
        analytic_result = np.array([1 / 3, 0.0, 0.0, 1 / 3, 0.0, 0.0, 1 / 3, 0.0, 0.0])
        expected = {
            0: analytic_result,
            "execution_results": np.array([1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }

        assert np.allclose(result["execution_results"], expected["execution_results"])

        del result["execution_results"]  # pylint: disable=unsupported-delete-operation
        del expected["execution_results"]

        _compare_numpy_dicts(result, expected)

        # Make sure shots are overridden correctly
        with pytest.warns(
            PennyLaneDeprecationWarning,
            match="Specifying 'shots' when executing a QNode is deprecated",
        ):
            result = circuit(shots=200)
        finite_shot_result = result[0]
        assert not np.allclose(  # Since 200 does not have a factor of 3, we assert that there's no chance for finite-shot tape to reach 1/3 exactly here.
            finite_shot_result,
            analytic_result,
            atol=np.finfo(np.float64).eps,
            rtol=0,
        )


class TestSnapshotMCMS:

    def test_default_qubit_tree_traversal(self):
        """Test that tree-traversal can be used with snapshots on DQ."""

        @qml.qnode(qml.device("default.qubit"), mcm_method="tree-traversal")
        def c():
            qml.H(0)
            qml.measure(0)
            qml.Snapshot(measurement=qml.expval(qml.Z(0)))
            qml.H(0)
            qml.measure(0, reset=True)
            qml.Snapshot(measurement=qml.expval(qml.Z(0)))
            qml.H(0)
            qml.measure(0, postselect=True)
            qml.Snapshot("tag", measurement=qml.expval(qml.Z(0)))

            return qml.expval(qml.Z(0))

        results = qml.snapshots(c)()

        assert len(results[0]) == 2
        assert qml.math.allclose(results[0], [1, -1])
        assert len(results[1]) == 4
        assert qml.math.allclose(results[1], [1, 1, 1, 1])  # reset into zero state

        assert len(results["tag"]) == 4
        assert qml.math.allclose(results["tag"], [-1, -1, -1, -1])  # postselected into one state

    def test_default_qubit_one_shot(self):
        """Test that one shot can be used with snapshots."""

        @qml.qnode(qml.device("default.qubit"), mcm_method="one-shot", shots=1000)
        def c():
            qml.H(0)
            qml.measure(0)
            qml.Snapshot(measurement=qml.expval(qml.Z(0)))
            qml.H(0)
            qml.measure(0, reset=True)
            qml.Snapshot(measurement=qml.expval(qml.Z(0)))
            qml.H(0)
            qml.measure(0, postselect=True)
            qml.Snapshot("tag", measurement=qml.expval(qml.Z(0)))

            return qml.expval(qml.Z(0))

        res = qml.snapshots(c)()

        assert len(res[0]) == 1000
        assert qml.math.allclose(qml.math.mean(res[0]), 0, atol=0.1)

        assert len(res[1]) == 1000
        assert qml.math.allclose(qml.math.mean(res[1]), 1)

        # postselection not applied to snapshots
        assert len(res["tag"]) == 1000
        assert qml.math.allclose(qml.math.mean(res["tag"]), 0, atol=0.1)
