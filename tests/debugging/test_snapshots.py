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

import pennylane as qp
from pennylane.exceptions import DeviceError, PennyLaneDeprecationWarning, QuantumFunctionError
from pennylane.ops.functions.equal import assert_equal


def _compare_numpy_dicts(dict1, dict2):
    assert dict1.keys() == dict2.keys()
    assert all(np.allclose(dict1[key], dict_2_val) for key, dict_2_val in dict2.items())


class TestSnapshotTape:
    @pytest.mark.parametrize("shots", [None, 100])
    def test_snapshot_output_tapes(self, shots):
        ops = [
            qp.Snapshot(),
            qp.Hadamard(wires=0),
            qp.Snapshot("very_important_state"),
            qp.CNOT(wires=[0, 1]),
            qp.Snapshot(measurement=qp.probs()),
            qp.Snapshot(measurement=qp.sample(), shots=10),
        ]

        measurements = [qp.expval(qp.PauliX(0))]

        num_snapshots = len(tuple(filter(lambda x: isinstance(x, qp.Snapshot), ops)))

        expected_tapes = [
            qp.tape.QuantumTape([], [qp.state()], shots=None),
            qp.tape.QuantumTape([qp.Hadamard(0)], [qp.state()], shots=None),
            qp.tape.QuantumTape([qp.Hadamard(0), qp.CNOT((0, 1))], [qp.probs()], shots=shots),
            qp.tape.QuantumTape([qp.Hadamard(0), qp.CNOT((0, 1))], [qp.sample()], shots=10),
            qp.tape.QuantumTape(
                [qp.Hadamard(0), qp.CNOT((0, 1))], [qp.expval(qp.X(0))], shots=shots
            ),
        ]

        tape = qp.tape.QuantumTape(ops, measurements, shots=shots)
        tapes, _ = qp.snapshots(tape)

        assert len(tapes) == num_snapshots + 1

        for out, expected in zip(tapes, expected_tapes):
            assert_equal(out, expected)

        tape_no_meas = qp.tape.QuantumTape(ops, shots=shots)
        tapes_no_meas, _ = qp.snapshots(tape_no_meas)

        assert len(tapes_no_meas) == num_snapshots

        for out, expected in zip(tapes_no_meas, expected_tapes[:-1]):
            assert_equal(out, expected)

    def test_snapshot_postprocessing_fn(self):
        ops = [
            qp.Snapshot(),
            qp.Hadamard(wires=0),
            qp.Snapshot("very_important_state"),
            qp.CNOT(wires=[0, 1]),
            qp.Snapshot(),
        ]

        measurements = [qp.expval(qp.PauliX(0))]

        num_snapshots = len(tuple(filter(lambda x: isinstance(x, qp.Snapshot), ops)))

        tape = qp.tape.QuantumTape(ops, measurements)

        _, fn = qp.snapshots(tape)

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

        tape_no_meas = qp.tape.QuantumScript(ops)

        _, fn_no_meas = qp.snapshots(tape_no_meas)

        expected_keys.remove("execution_results")
        assert "snapshot_tags" in fn.keywords
        assert len(fn_no_meas.keywords["snapshot_tags"]) == num_snapshots
        assert all(key in fn_no_meas.keywords["snapshot_tags"] for key in expected_keys)
        assert fn(["a", 1, 2]) == {
            0: "a",
            "very_important_state": 1,
            2: 2,
        }

    def test_snapshot_fails_with_non_str_tags(self):
        with pytest.raises(ValueError, match="tags can only be of type 'str'"):
            qp.Snapshot(np.array(0.0), qp.state())

        with pytest.raises(ValueError, match="tags can only be of type 'str'"):
            qp.Snapshot(qp.state())

    @pytest.mark.parametrize(
        "dev", (qp.device("default.qubit"), qp.device("default.qutrit", wires=2))
    )
    def test_int_tag_fails_during_transform(self, dev):
        """Test ValueError is raised if user provided a int snapshot tag."""

        @qp.qnode(dev)
        def c():
            qp.Snapshot(2)
            return qp.expval(qp.Z(0))

        with pytest.raises(ValueError, match="can only be of type 'str'"):
            qp.snapshots(c)()


@pytest.mark.parametrize(
    "dev",
    [
        # Two supported devices
        qp.device("default.qubit"),
        qp.device("default.mixed", wires=2),
        # Two non-supported devices
        qp.device("default.qutrit", wires=2),
        qp.device("lightning.qubit", wires=2),
    ],
)
class TestSnapshotGeneral:
    def test_sample_measurement_with_analytical_device_fails(self, dev):
        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(0)
            qp.Snapshot(measurement=qp.sample())
            return qp.expval(qp.PauliZ(0))

        # Expect a DeviceError to be raised here since no shots has
        # been provided to the snapshot due to the analytical device
        with pytest.raises(DeviceError):
            qp.snapshots(circuit)()

    def test_non_StateMP_state_measurements_with_finite_shot_device_fails(self, dev):
        @qp.set_shots(shots=200)
        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(0)
            qp.Snapshot(measurement=qp.mutual_info(0, 1))
            return qp.expval(qp.PauliZ(0))

        # Expect a DeviceError to be raised here since no shots has
        # been provided to the snapshot due to the finite-shot device
        with pytest.raises(DeviceError):
            qp.snapshots(circuit)()

    def test_StateMP_with_finite_shot_device_passes(self, dev):
        if "lightning" in dev.name or "mixed" in dev.name:
            pytest.skip()

        @qp.set_shots(shots=200)
        @qp.qnode(dev)
        def circuit():
            qp.Snapshot(measurement=qp.state())
            qp.Snapshot()

            if isinstance(dev, qp.devices.QutritDevice):
                return qp.expval(qp.GellMann(0, 1))

            return qp.expval(qp.PauliZ(0))

        _ = qp.snapshots(circuit)()

    @pytest.mark.parametrize("diff_method", [None, "parameter-shift"])
    def test_all_state_measurement_snapshot_pure_qubit_dev(self, dev, diff_method):
        """Test that the correct measurement snapshots are returned for different measurement types."""
        if isinstance(dev, (qp.devices.default_mixed.DefaultMixed, qp.devices.QutritDevice)):
            pytest.skip()

        @qp.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qp.Snapshot(measurement=qp.expval(qp.PauliZ(0)))
            qp.RX(x, wires=0)
            qp.Snapshot(measurement=qp.var(qp.PauliZ(0)))
            qp.Snapshot(measurement=qp.probs(0))
            qp.Snapshot(measurement=qp.state())

            return qp.expval(qp.PauliZ(0))

        phi = 0.1
        with (
            pytest.warns(UserWarning, match="Snapshots are not supported for the given device")
            if "lightning" in dev.name
            else nullcontext()
        ):
            result = qp.snapshots(circuit)(phi)

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

        @qp.qnode(dev)
        def circuit():
            if isinstance(dev, qp.devices.QutritDevice):
                qp.THadamard(wires=0)
                return qp.expval(qp.GellMann(0, index=6))

            qp.Hadamard(wires=0)
            return qp.expval(qp.PauliX(0))

        result = qp.snapshots(circuit)()
        if isinstance(dev, qp.devices.QutritDevice):
            expected = {"execution_results": np.array(0.66666667)}
        else:
            expected = {"execution_results": np.array(1.0)}

        _compare_numpy_dicts(result, expected)

    def test_override_shots(self, dev):
        """Test that override shots allow snapshots to work with different numbers of measurements."""

        @qp.qnode(dev)
        def c():
            if dev.name != "default.qutrit":
                qp.H(0)
            qp.Snapshot("sample", qp.sample(wires=0), shots=5)
            qp.Snapshot("counts", qp.counts(wires=0, all_outcomes=True), shots=20)
            qp.Snapshot("probs", qp.probs(wires=0), shots=21)
            return qp.state()

        out = qp.snapshots(c)()

        assert out["sample"].shape == (5, 1)
        assert out["counts"]["0"] + out["counts"].get("1", 0) == 20
        if dev.name != "default.qutrit":
            # very rare that it will be *exactly* [0.5, 0.5] if 20 shots
            assert not qp.math.allclose(out["probs"], np.array([0.5, 0.5]), atol=1e-8)

    def test_override_analytic(self, dev):
        """Test that finite shots can be written with analytic calculations."""

        if dev.name == "default.qutrit":
            pytest.skip("hard to write generic test that works with qutrits.")

        @qp.transform
        def set_shots(tape, shots):
            return (tape.copy(shots=shots),), lambda res: res[0]

        @qp.qnode(dev, diff_method=None)
        def c():
            qp.H(0)
            qp.Snapshot("probs", qp.probs(wires=0), shots=None)
            return qp.sample(wires=0)

        out = qp.snapshots(set_shots(c, shots=10))()
        assert qp.math.allclose(out["probs"], np.array([0.5, 0.5]))
        assert out["execution_results"].shape == (10, 1)


class TestSnapshotSupportedQNode:
    """Test the Snapshot instruction for simulators."""

    # pylint: disable=protected-access
    @pytest.mark.parametrize("diff_method", ["backprop", "adjoint"])
    def test_default_qubit_with_backprob_and_adjoint(self, diff_method):
        dev = qp.device("default.qubit")

        assert qp.debugging.snapshot._is_snapshot_compatible(dev)

        @qp.qnode(dev, diff_method=diff_method)
        def circuit():
            qp.Hadamard(wires=0)
            qp.Snapshot("important_expval", measurement=qp.expval(qp.PauliX(0)))
            qp.CNOT(wires=[0, 1])
            qp.Snapshot()
            return qp.expval(qp.PauliX(0))

        circuit()
        assert dev._debugger is None
        if diff_method is not None:
            assert circuit.interface == "auto"

        result = qp.snapshots(circuit)()
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
        dev = qp.device("default.mixed", wires=2)

        assert qp.debugging.snapshot._is_snapshot_compatible(dev)

        @qp.qnode(dev, diff_method=method)
        def circuit():
            qp.Snapshot(measurement=qp.expval(qp.PauliX(0)))
            qp.Hadamard(wires=0)
            qp.Snapshot("very_important_state")
            qp.CNOT(wires=[0, 1])
            qp.Snapshot(measurement=qp.probs())
            return qp.expval(qp.PauliX(0))

        circuit()
        assert dev._debugger is None

        result = qp.snapshots(circuit)()
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
        dev = qp.device("default.gaussian", wires=2)

        assert qp.debugging.snapshot._is_snapshot_compatible(dev)

        @qp.qnode(dev, diff_method=method)
        def circuit():
            qp.Snapshot()
            qp.Displacement(0.5, 0, wires=0)
            qp.Snapshot("very_important_state")
            qp.Beamsplitter(0.5, 0.7, wires=[0, 1])
            qp.Snapshot()
            return qp.expval(qp.QuadX(0))

        circuit()
        assert dev._debugger is None

        result = qp.snapshots(circuit)()
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

        dev = qp.device("default.qutrit.mixed", wires=2)

        assert qp.debugging.snapshot._is_snapshot_compatible(dev)

        @qp.set_shots(100)
        @qp.qnode(dev, diff_method=diff_method)
        def circuit(add_bad_snapshot: bool):
            qp.THadamard(wires=0)
            qp.Snapshot(measurement=qp.counts())
            qp.TSWAP(wires=[0, 1])
            if add_bad_snapshot:
                qp.Snapshot(measurement=qp.probs())
            qp.Snapshot()
            return qp.counts()

        circuit(False)
        assert dev._debugger is None
        # This should fail since finite-shot probs() isn't supported
        with pytest.raises(NotImplementedError):
            qp.snapshots(circuit)(add_bad_snapshot=True)

        result = qp.snapshots(circuit)(add_bad_snapshot=False)
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
        result = qp.snapshots(qp.set_shots(shots=200)(circuit))(add_bad_snapshot=False)
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
        spy = mocker.spy(qp.debugging.snapshots, "default_qnode_transform")

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, interface=None)
        def circuit():
            qp.Snapshot()
            qp.Hadamard(wires=0)
            qp.Snapshot("very_important_state")
            qp.CNOT(wires=[0, 1])
            qp.Snapshot()
            if m == "expval":
                return qp.expval(qp.PauliZ(0))
            if m == "var":
                return qp.var(qp.PauliY(1))
            if m == "probs":
                return qp.probs([0, 1])
            return qp.state()

        result = qp.snapshots(circuit)()
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
        dev = qp.device("default.qubit", wires=2)

        def circuit(params, wire):
            qp.Hadamard(wire)
            qp.Snapshot()
            qp.Snapshot(measurement=qp.probs())
            qp.Rot(*params, wire)

        @qp.qnode(dev)
        def qnode(params):
            qp.Hadamard(0)
            qp.ctrl(circuit, 0)(params, wire=1)
            return qp.expval(qp.PauliZ(1))

        params = np.array([1.3, 1.4, 0.2])
        result = qp.snapshots(qnode)(params)
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

        dev = qp.device("default.qubit", wires=2)

        def circuit(params, wire):
            qp.Rot(*params, wire)
            qp.Snapshot()
            qp.Snapshot(measurement=qp.probs())
            qp.Hadamard(wire)

        @qp.qnode(dev)
        def qnode(params):
            qp.Hadamard(0)
            qp.adjoint(circuit, 0)(params, wire=1)
            return qp.expval(qp.PauliZ(1))

        params = np.array([1.3, 1.4, 0.2])
        result = qp.snapshots(qnode)(params)
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

        dev = qp.device("default.qubit", wires=1)

        @qp.set_shots(10)
        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(wires=0)
            # Shot-based measurements
            qp.Snapshot(measurement=qp.expval(qp.PauliZ(0)))
            qp.Snapshot(measurement=qp.var(qp.PauliZ(0)))
            qp.Snapshot(measurement=qp.probs(0))
            qp.Snapshot(measurement=qp.counts(wires=0))
            qp.Snapshot(measurement=qp.sample(wires=0))
            qp.Snapshot()

            return qp.expval(qp.PauliZ(0))

        result = qp.snapshots(circuit)()

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

        result = qp.snapshots(qp.set_shots(circuit, shots=200))()
        assert result[3] == {"0": 98, "1": 102}
        assert np.allclose(result[5], expected[5])

    def test_unsupported_snapshot_measurement(self):
        """Test that an exception is raised when an unsupported measurement is provided to the snapshot."""
        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(0)
            with pytest.raises(
                ValueError,
                match="The measurement PauliZ is not supported as it is not an instance "
                "of <class 'pennylane.measurements.measurements.MeasurementProcess'>",
            ):
                qp.Snapshot(measurement=qp.PauliZ(0))
            return qp.expval(qp.PauliZ(0))

        qp.snapshots(circuit)()


class TestSnapshotUnsupportedQNode:
    """Unit tests for qp.snapshots when using with qnodes with unsupported devices"""

    def test_unsupported_device_warning(self):
        """Test that a warning is raised when the device being used by a qnode does not natively support
        qp.Snapshot"""

        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(0)
            qp.Snapshot()
            qp.CNOT([0, 1])
            return qp.expval(qp.Z(1))

        with pytest.warns(UserWarning, match="Snapshots are not supported"):
            _ = qp.snapshots(circuit)

    # Improper ttest_ind usage, but leaving it here for now;
    # should be revised and fixed soon
    # current failure rate: ~7%
    # FIXME: [sc-92966]
    @pytest.mark.local_salt(2)
    def test_lightning_qubit_finite_shots(self, seed):
        dev = qp.device("lightning.qubit", wires=2, seed=seed)

        @qp.set_shots(500)
        @qp.qnode(dev, diff_method=None)
        def circuit():
            qp.Hadamard(0)
            qp.Snapshot(measurement=qp.counts(wires=0))
            return qp.expval(qp.PauliX(1))

        # TODO: fallback to simple `np.allclose` tests once `setRandomSeed` is exposed from the lightning C++ code
        counts, expvals = tuple(zip(*(qp.snapshots(circuit)().values() for _ in range(50))))
        assert ttest_ind([count["0"] for count in counts], 250).pvalue >= 0.75
        assert ttest_ind(expvals, 0.0).pvalue >= 0.75

        # Make sure shots are overridden correctly
        counts, _ = tuple(
            zip(*(qp.snapshots(qp.set_shots(circuit, shots=1000))().values() for _ in range(50)))
        )
        assert ttest_ind([count["0"] for count in counts], 500).pvalue >= 0.75

    @pytest.mark.parametrize("diff_method", ["backprop", "adjoint"])
    def test_lightning_qubit_fails_for_state_snapshots_with_adjoint_and_backprop(self, diff_method):
        """Test lightning with backprop and adjoint differentiation fails with default snapshot as it
        falls to qp.state() which is not supported"""

        dev = qp.device("lightning.qubit", wires=2)

        with pytest.raises(
            QuantumFunctionError,
            match=f"does not support {diff_method} with requested circuit",
        ):

            @qp.qnode(dev, diff_method=diff_method)
            def circuit():
                qp.Snapshot()
                return qp.expval(qp.PauliX(0))

            qp.snapshots(circuit)()

    def test_state_wire_order_preservation(self):
        """Test that the snapshots wire order reflects the wire order on the device."""

        @qp.qnode(qp.device("default.qubit", wires=2))
        def circuit():
            qp.X(1)
            qp.Snapshot()
            return qp.state()

        out = qp.snapshots(circuit)()

        assert qp.math.allclose(out[0], out["execution_results"])

    # pylint: disable=protected-access
    @pytest.mark.parametrize("method", [None, "parameter-shift"])
    def test_default_qutrit(self, method):
        """Test that multiple snapshots are returned correctly on the pure qutrit simulator."""

        dev = qp.device("default.qutrit", wires=2)

        assert not qp.debugging.snapshot._is_snapshot_compatible(dev)

        @qp.qnode(dev, diff_method=method)
        def circuit():
            qp.THadamard(wires=0)
            qp.Snapshot(measurement=qp.probs())
            qp.TSWAP(wires=[0, 1])
            return qp.probs()

        with pytest.warns(UserWarning, match="Snapshots are not supported for the given device"):
            circuit = qp.snapshots(circuit)

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

        @qp.qnode(qp.device("default.qubit"), mcm_method="tree-traversal")
        def c():
            qp.H(0)
            qp.measure(0)
            qp.Snapshot(measurement=qp.expval(qp.Z(0)))
            qp.H(0)
            qp.measure(0, reset=True)
            qp.Snapshot(measurement=qp.expval(qp.Z(0)))
            qp.H(0)
            qp.measure(0, postselect=True)
            qp.Snapshot("tag", measurement=qp.expval(qp.Z(0)))

            return qp.expval(qp.Z(0))

        results = qp.snapshots(c)()

        assert len(results[0]) == 2
        assert qp.math.allclose(results[0], [1, -1])
        assert len(results[1]) == 4
        assert qp.math.allclose(results[1], [1, 1, 1, 1])  # reset into zero state

        assert len(results["tag"]) == 4
        assert qp.math.allclose(results["tag"], [-1, -1, -1, -1])  # postselected into one state

    def test_default_qubit_one_shot(self):
        """Test that one shot can be used with snapshots."""

        @qp.qnode(qp.device("default.qubit"), mcm_method="one-shot", shots=1000)
        def c():
            qp.H(0)
            qp.measure(0)
            qp.Snapshot(measurement=qp.expval(qp.Z(0)))
            qp.H(0)
            qp.measure(0, reset=True)
            qp.Snapshot(measurement=qp.expval(qp.Z(0)))
            qp.H(0)
            qp.measure(0, postselect=True)
            qp.Snapshot("tag", measurement=qp.expval(qp.Z(0)))

            return qp.expval(qp.Z(0))

        res = qp.snapshots(c)()

        assert len(res[0]) == 1000
        assert qp.math.allclose(qp.math.mean(res[0]), 0, atol=0.1)

        assert len(res[1]) == 1000
        assert qp.math.allclose(qp.math.mean(res[1]), 1)

        # postselection not applied to snapshots
        assert len(res["tag"]) == 1000
        assert qp.math.allclose(qp.math.mean(res["tag"]), 0, atol=0.1)
