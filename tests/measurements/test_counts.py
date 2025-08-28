# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the qml.counts measurement process."""
import copy

import numpy as np
import pytest

import pennylane as qml
from pennylane.exceptions import QuantumFunctionError
from pennylane.measurements import CountsMP
from pennylane.wires import Wires


class TestCounts:
    """Tests for the counts function"""

    def test_counts_properties(self):
        """Test that the properties are correct."""
        meas1 = qml.counts(wires=0)
        meas2 = qml.counts(op=qml.PauliX(0), all_outcomes=True)
        assert meas1.samples_computational_basis is True
        assert isinstance(meas1, CountsMP)
        assert meas2.samples_computational_basis is False
        assert meas2.all_outcomes is True

    def test_queue(self):
        """Test that the right measurement class is queued."""

        with qml.queuing.AnnotatedQueue() as q:
            op = qml.PauliX(0)
            m = qml.counts(op)

        assert q.queue[0] is m
        assert isinstance(m, CountsMP)

    def test_copy(self):
        """Test that the ``__copy__`` method also copies the ``all_outcomes`` information."""
        meas = qml.counts(wires=0, all_outcomes=True)
        meas_copy = copy.copy(meas)
        assert meas_copy.wires == Wires(0)
        assert meas_copy.all_outcomes is True

    def test_providing_observable_and_wires(self):
        """Test that a ValueError is raised if both an observable is provided and wires are
        specified"""

        with pytest.raises(
            ValueError,
            match="Cannot specify the wires to sample if an observable is provided."
            " The wires to sample will be determined directly from the observable.",
        ):
            qml.counts(qml.PauliZ(0), wires=[0, 1])

    def test_hash(self):
        """Test that the hash property includes the all_outcomes property."""
        m1 = qml.counts(all_outcomes=True)
        m2 = qml.counts(all_outcomes=False)

        assert m1.hash != m2.hash

        m3 = CountsMP(eigvals=[0.5, -0.5], wires=qml.wires.Wires(0), all_outcomes=True)
        assert m3.hash != m1.hash

    def test_repr(self):
        """Test that the repr includes the all_outcomes property."""
        m1 = CountsMP(wires=Wires(0), all_outcomes=True)
        assert repr(m1) == "CountsMP(wires=[0], all_outcomes=True)"

        m2 = CountsMP(obs=qml.PauliX(0), all_outcomes=True)
        assert repr(m2) == "CountsMP(X(0), all_outcomes=True)"

        m3 = CountsMP(eigvals=(-1, 1), wires=[0], all_outcomes=False)
        assert repr(m3) == "CountsMP(eigvals=[-1  1], wires=[0], all_outcomes=False)"

        mv = qml.measure(0)
        m4 = CountsMP(obs=mv, all_outcomes=False)
        assert repr(m4) == "CountsMP(MeasurementValue(wires=[0]), all_outcomes=False)"


class TestProcessSamples:
    """Unit tests for the counts.process_samples method"""

    def test_counts_shape_single_wires(self, seed):
        """Test that the counts output is correct for single wires"""
        shots = 1000
        rng = np.random.default_rng(seed)
        samples = rng.choice([0, 1], size=(shots, 2)).astype(np.int64)

        result = qml.counts(wires=0).process_samples(samples, wire_order=[0])

        assert len(result) == 2
        assert set(result.keys()) == {"0", "1"}
        assert result["0"] == np.count_nonzero(samples[:, 0] == 0)
        assert result["1"] == np.count_nonzero(samples[:, 0] == 1)

    def test_counts_shape_multi_wires(self, seed):
        """Test that the counts function outputs counts of the right size
        for multiple wires"""
        shots = 1000
        rng = np.random.default_rng(seed)
        samples = rng.choice([0, 1], size=(shots, 2)).astype(np.int64)

        result = qml.counts(wires=[0, 1]).process_samples(samples, wire_order=[0, 1])

        assert len(result) == 4
        assert set(result.keys()) == {"00", "01", "10", "11"}
        assert result["00"] == np.count_nonzero(
            np.logical_and(samples[:, 0] == 0, samples[:, 1] == 0)
        )
        assert result["01"] == np.count_nonzero(
            np.logical_and(samples[:, 0] == 0, samples[:, 1] == 1)
        )
        assert result["10"] == np.count_nonzero(
            np.logical_and(samples[:, 0] == 1, samples[:, 1] == 0)
        )
        assert result["11"] == np.count_nonzero(
            np.logical_and(samples[:, 0] == 1, samples[:, 1] == 1)
        )

    def test_counts_with_nan_samples(self, seed):
        """Test that the counts function disregards failed measurements (samples including
        NaN values) when totalling counts"""
        shots = 1000
        rng = np.random.default_rng(seed)
        samples = rng.choice([0, 1], size=(shots, 2)).astype(np.float64)

        samples[0][0] = np.nan
        samples[17][1] = np.nan
        samples[850][0] = np.nan

        result = qml.counts(wires=[0, 1]).process_samples(samples, wire_order=[0, 1])

        # no keys with NaNs
        assert len(result) == 4
        assert set(result.keys()) == {"00", "01", "10", "11"}

        # NaNs were not converted into "0", but were excluded from the counts
        total_counts = sum(count for count in result.values())
        assert total_counts == 997

    @pytest.mark.parametrize("batch_size", [None, 1, 4])
    @pytest.mark.parametrize("n_wires", [4, 10, 65])
    @pytest.mark.parametrize("all_outcomes", [False, True])
    def test_counts_multi_wires_no_overflow(self, n_wires, all_outcomes, batch_size):
        """Test that binary strings for wire samples are not negative due to overflow."""
        if all_outcomes and n_wires == 65:
            pytest.skip("Too much memory being used, skipping")
        shots = 1000
        total_wires = 65
        shape = (batch_size, shots, total_wires) if batch_size else (shots, total_wires)
        samples = np.random.choice([0, 1], size=shape).astype(np.float64)
        result = qml.counts(wires=list(range(n_wires)), all_outcomes=all_outcomes).process_samples(
            samples, wire_order=list(range(total_wires))
        )

        if batch_size:
            assert len(result) == batch_size
            for r in result:
                assert sum(r.values()) == shots
                assert all("-" not in sample for sample in r.keys())
        else:
            assert sum(result.values()) == shots
            assert all("-" not in sample for sample in result.keys())

    def test_counts_obs(self, seed):
        """Test that the counts function outputs counts of the right size for observables"""
        shots = 1000
        rng = np.random.default_rng(seed)
        samples = rng.choice([0, 1], size=(shots, 2)).astype(np.int64)

        result = qml.counts(qml.PauliZ(0)).process_samples(samples, wire_order=[0])

        assert len(result) == 2
        assert set(result.keys()) == {1, -1}
        assert result[1] == np.count_nonzero(samples[:, 0] == 0)
        assert result[-1] == np.count_nonzero(samples[:, 0] == 1)

    def test_count_eigvals(self, seed):
        """Tests that eigvals are used instead of obs for counts"""

        shots = 100
        rng = np.random.default_rng(seed)
        samples = rng.choice([0, 1], size=(shots, 2)).astype(np.int64)
        result = CountsMP(eigvals=[1, -1], wires=0).process_samples(samples, wire_order=[0])
        assert len(result) == 2
        assert set(result.keys()) == {1, -1}
        assert result[1] == np.count_nonzero(samples[:, 0] == 0)
        assert result[-1] == np.count_nonzero(samples[:, 0] == 1)

    def test_counts_shape_single_measurement_value(self, seed):
        """Test that the counts output is correct for single mid-circuit measurement
        values."""
        shots = 1000
        rng = np.random.default_rng(seed)
        samples = rng.choice([0, 1], size=(shots, 2)).astype(np.int64)
        mv = qml.measure(0)

        result = qml.counts(mv).process_samples(samples, wire_order=[0])

        assert len(result) == 2
        assert set(result.keys()) == {0, 1}
        assert result[0] == np.count_nonzero(samples[:, 0] == 0)
        assert result[1] == np.count_nonzero(samples[:, 0] == 1)

    def test_counts_shape_composite_measurement_value(self, seed):
        """Test that the counts output is correct for composite mid-circuit measurement
        values."""
        shots = 1000
        rng = np.random.default_rng(seed)
        samples = rng.choice([0, 1], size=(shots, 2)).astype(np.int64)
        m0 = qml.measure(0)
        m1 = qml.measure(1)

        result = qml.counts(op=m0 | m1).process_samples(samples, wire_order=[0, 1])

        assert len(result) == 2
        assert set(result.keys()) == {0, 1}
        samples = np.apply_along_axis((lambda x: x[0] | x[1]), axis=1, arr=samples)
        assert result[0] == np.count_nonzero(samples == 0)
        assert result[1] == np.count_nonzero(samples == 1)

    def test_counts_shape_measurement_value_list(self, seed):
        """Test that the counts output is correct for list mid-circuit measurement
        values."""
        shots = 1000
        rng = np.random.default_rng(seed)
        samples = rng.choice([0, 1], size=(shots, 2)).astype(np.int64)
        m0 = qml.measure(0)
        m1 = qml.measure(1)

        result = qml.counts(op=[m0, m1]).process_samples(samples, wire_order=[0, 1])

        assert len(result) == 4
        assert set(result.keys()) == {"00", "01", "10", "11"}
        assert result["00"] == np.count_nonzero([np.allclose(s, [0, 0]) for s in samples])
        assert result["01"] == np.count_nonzero([np.allclose(s, [0, 1]) for s in samples])
        assert result["10"] == np.count_nonzero([np.allclose(s, [1, 0]) for s in samples])
        assert result["11"] == np.count_nonzero([np.allclose(s, [1, 1]) for s in samples])

    def test_mixed_lists_as_op_not_allowed(self):
        """Test that passing a list not containing only measurement values raises an error."""
        m0 = qml.measure(0)

        with pytest.raises(
            QuantumFunctionError,
            match="Only sequences of unprocessed MeasurementValues can be passed with the op argument",
        ):
            _ = qml.counts(op=[m0, qml.PauliZ(0)])

    def test_composed_measurement_value_lists_not_allowed(self):
        """Test that passing a list containing measurement values composed with arithmetic
        raises an error."""
        m0 = qml.measure(0)
        m1 = qml.measure(1)
        m2 = qml.measure(2)

        with pytest.raises(
            QuantumFunctionError,
            match="Only sequences of unprocessed MeasurementValues can be passed with the op argument",
        ):
            _ = qml.counts(op=[m0 + m1, m2])

    def test_processed_measurement_value_lists_not_allowed(self):
        """Test that passing a list containing measurement values composed with arithmetic
        raises an error."""
        m0 = qml.measure(0)
        m1 = qml.measure(1)

        with pytest.raises(
            QuantumFunctionError,
            match="Only sequences of unprocessed MeasurementValues can be passed with the op argument",
        ):
            _ = qml.counts(op=[2 * m0, m1])

    def test_counts_all_outcomes_wires(self):
        """Test that the counts output is correct when all_outcomes is passed"""
        shots = 1000
        samples = np.zeros((shots, 2)).astype(np.int64)

        result1 = qml.counts(wires=0, all_outcomes=False).process_samples(samples, wire_order=[0])

        assert len(result1) == 1
        assert set(result1.keys()) == {"0"}
        assert result1["0"] == shots

        result2 = qml.counts(wires=0, all_outcomes=True).process_samples(samples, wire_order=[0])

        assert len(result2) == 2
        assert set(result2.keys()) == {"0", "1"}
        assert result2["0"] == shots
        assert result2["1"] == 0

    def test_counts_all_outcomes_obs(self):
        """Test that the counts output is correct when all_outcomes is passed"""
        shots = 1000
        samples = np.zeros((shots, 2)).astype(np.int64)

        result1 = qml.counts(qml.PauliZ(0), all_outcomes=False).process_samples(
            samples, wire_order=[0]
        )

        assert len(result1) == 1
        assert set(result1.keys()) == {1}
        assert result1[1] == shots

        result2 = qml.counts(qml.PauliZ(0), all_outcomes=True).process_samples(
            samples, wire_order=[0]
        )

        assert len(result2) == 2
        assert set(result2.keys()) == {1, -1}
        assert result2[1] == shots
        assert result2[-1] == 0

    def test_counts_all_outcomes_measurement_value(self):
        """Test that the counts output is correct when all_outcomes is passed
        for mid-circuit measurement values."""
        shots = 1000
        samples = np.zeros((shots, 2)).astype(np.int64)
        mv = qml.measure(0)

        result1 = qml.counts(mv, all_outcomes=False).process_samples(samples, wire_order=[0])

        assert len(result1) == 1
        assert set(result1.keys()) == {0}
        assert result1[0] == shots

        result2 = qml.counts(mv, all_outcomes=True).process_samples(samples, wire_order=[0])

        assert len(result2) == 2
        assert set(result2.keys()) == {0, 1}
        assert result2[0] == shots
        assert result2[1] == 0

    def test_counts_all_outcomes_composite_measurement_value(self):
        """Test that the counts output is correct when all_outcomes is passed
        for composite mid-circuit measurement values."""
        shots = 1000
        samples = np.zeros((shots, 2)).astype(np.int64)
        m0 = qml.measure(0)
        m1 = qml.measure(1)
        mv = (m0 - 1) * 2 * (m1 + 1) - 2  # 00 equates to -4

        result1 = qml.counts(mv, all_outcomes=False).process_samples(samples, wire_order=[0, 1])

        assert len(result1) == 1
        assert set(result1.keys()) == {-4}
        assert result1[-4] == shots

        result2 = qml.counts(mv, all_outcomes=True).process_samples(samples, wire_order=[0, 1])

        # Possible outcomes are -4, -6, -2
        assert len(result2) == 3
        assert set(result2.keys()) == {-6, -4, -2}
        assert result2[-4] == shots
        assert result2[-6] == result2[-2] == 0

    def test_counts_all_outcomes_measurement_value_list(self):
        """Test that the counts output is correct when all_outcomes is passed
        for a list of mid-circuit measurement values."""
        shots = 1000
        samples = np.zeros((shots, 2)).astype(np.int64)
        m0 = qml.measure(0)
        m1 = qml.measure(1)

        result1 = qml.counts([m0, m1], all_outcomes=False).process_samples(
            samples, wire_order=[0, 1]
        )

        assert len(result1) == 1
        assert set(result1.keys()) == {"00"}
        assert result1["00"] == shots

        result2 = qml.counts([m0, m1], all_outcomes=True).process_samples(
            samples, wire_order=[0, 1]
        )

        assert len(result2) == 4
        assert set(result2.keys()) == {"00", "01", "10", "11"}
        assert result2["00"] == shots
        assert result2["01"] == 0
        assert result2["10"] == 0
        assert result2["11"] == 0

    def test_counts_binsize(self):
        counts = qml.counts(wires=0)
        samples = np.zeros((10, 2))
        output = counts.process_samples(
            samples, wire_order=qml.wires.Wires((0, 1)), shot_range=(0, 10), bin_size=2
        )
        assert len(output) == 5

        for r in output:
            assert r == {"0": 2}


class TestCountsIntegration:
    # pylint:disable=too-many-public-methods,not-an-iterable

    def test_counts_all_outcomes_with_mcm(self):
        """Test that all outcomes are present in results if requested."""
        n_sample = 10

        dev = qml.device("default.qubit")

        @qml.set_shots(n_sample)
        @qml.qnode(device=dev, mcm_method="one-shot")
        def single_mcm():
            m = qml.measure(0)
            return qml.counts(m, all_outcomes=True)

        res = single_mcm()

        assert list(res.keys()) == [0.0, 1.0]
        assert sum(res.values()) == n_sample
        assert res[0.0] == n_sample

        @qml.set_shots(n_sample)
        @qml.qnode(device=dev, mcm_method="one-shot")
        def double_mcm():
            m1 = qml.measure(0)
            m2 = qml.measure(1)
            return qml.counts([m1, m2], all_outcomes=True)

        res = double_mcm()

        assert list(res.keys()) == ["00", "01", "10", "11"]
        assert sum(res.values()) == n_sample
        assert res["00"] == n_sample

    def test_counts_dimension(self):
        """Test that the counts function outputs counts of the right size"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(n_sample)
        @qml.qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0)
            return qml.counts(qml.PauliZ(0)), qml.counts(qml.PauliX(1))

        sample = circuit()

        assert len(sample) == 2
        assert np.all([sum(s.values()) == n_sample for s in sample])

    def test_batched_counts_dimension(self):
        """Test that the counts function outputs counts of the right size with batching"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(n_sample)
        @qml.qnode(dev)
        def circuit():
            qml.RX([0.54, 0.65], wires=0)
            return qml.counts(qml.PauliZ(0)), qml.counts(qml.PauliX(1))

        sample = circuit()

        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert np.all([sum(s.values()) == n_sample for batch in sample for s in batch])

    def test_counts_combination(self):
        """Test the output of combining expval, var and counts"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=3)

        @qml.set_shots(n_sample)
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.RX(0.54, wires=0)

            return (
                qml.counts(qml.PauliZ(0)),
                qml.expval(qml.PauliX(1)),
                qml.var(qml.PauliY(2)),
            )

        result = circuit()

        assert len(result) == 3
        assert sum(result[0].values()) == n_sample

    def test_single_wire_counts(self):
        """Test the return type and shape of sampling counts from a single wire"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=1)

        @qml.set_shots(n_sample)
        @qml.qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.counts(qml.PauliZ(0))

        result = circuit()

        assert isinstance(result, dict)
        assert sum(result.values()) == n_sample

    def test_multi_wire_counts_regular_shape(self):
        """Test the return type and shape of sampling multiple wires
        where a rectangular array is expected"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=3)

        @qml.set_shots(n_sample)
        @qml.qnode(dev)
        def circuit():
            return (
                qml.counts(qml.PauliZ(0)),
                qml.counts(qml.PauliZ(1)),
                qml.counts(qml.PauliZ(2)),
            )

        result = circuit()

        # If all the dimensions are equal the result will end up to be a proper rectangular array
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(sum(r.values()) == n_sample for r in result)
        assert all(all(v.dtype == np.dtype("int") for v in r.values()) for r in result)

    def test_providing_no_observable_and_no_wires_counts(self):
        """Test that we can provide no observable and no wires to sample function"""
        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            res = qml.counts()
            assert res.obs is None
            assert res.wires == qml.wires.Wires([])
            return res

        circuit()

    def test_providing_no_observable_and_wires_counts(self):
        """Test that we can provide no observable but specify wires to the sample function"""
        wires = [0, 2]
        wires_obj = qml.wires.Wires(wires)
        dev = qml.device("default.qubit", wires=3)

        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            res = qml.counts(wires=wires)

            assert res.obs is None
            assert res.wires == wires_obj
            return res

        circuit()

    def test_batched_counts_work_individually(self):
        """Test that each counts call operates independently"""
        n_shots = 10
        dev = qml.device("default.qubit", wires=1)

        @qml.set_shots(n_shots)
        @qml.qnode(dev)
        def circuit():
            qml.pow(qml.PauliX(0), z=[1, 2])
            return qml.counts()

        assert circuit() == [{"1": 10}, {"0": 10}]

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("wires, basis_state", [(None, "010"), ([2, 1], "01")])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    def test_counts_no_op_finite_shots(self, interface, wires, basis_state):
        """Check all interfaces with computational basis state counts and
        finite shot"""
        n_shots = 10
        dev = qml.device("default.qubit", wires=3)

        @qml.set_shots(n_shots)
        @qml.qnode(dev, interface=interface)
        def circuit():
            qml.PauliX(1)
            return qml.counts(wires=wires)

        res = circuit()
        assert res == {basis_state: n_shots}
        assert qml.math.get_interface(res[basis_state]) == interface

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    def test_counts_operator_finite_shots(self, interface):
        """Check all interfaces with observable measurement counts and finite
        shot"""
        n_shots = 10
        dev = qml.device("default.qubit", wires=3)

        @qml.set_shots(n_shots)
        @qml.qnode(dev, interface=interface)
        def circuit():
            return qml.counts(qml.PauliZ(0))

        res = circuit()
        assert res == {1: n_shots}

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("shot_vec", [(1, 10, 10), (1, 10, 1000)])
    @pytest.mark.parametrize("wires, basis_state", [(None, "010"), ([2, 1], "01")])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    def test_counts_binned(
        self, shot_vec, interface, wires, basis_state
    ):  # pylint:disable=too-many-arguments
        """Check all interfaces with computational basis state counts and
        different shot vectors"""
        dev = qml.device("default.qubit", wires=3)

        @qml.set_shots(shot_vec)
        @qml.qnode(dev, interface=interface)
        def circuit():
            qml.PauliX(1)
            return qml.counts(wires=wires)

        res = circuit()

        assert isinstance(res, tuple)
        assert res[0] == {basis_state: shot_vec[0]}
        assert res[1] == {basis_state: shot_vec[1]}
        assert res[2] == {basis_state: shot_vec[2]}
        assert len(res) == len(shot_vec)
        assert sum(sum(res_bin.values()) for res_bin in res) == sum(shot_vec)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("shot_vec", [(1, 10, 10), (1, 10, 1000)])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    def test_counts_operator_binned(self, shot_vec, interface):
        """Check all interfaces with observable measurement counts and different
        shot vectors"""
        dev = qml.device("default.qubit", wires=3)

        @qml.set_shots(shot_vec)
        @qml.qnode(dev, interface=interface)
        def circuit():
            return qml.counts(qml.PauliZ(0))

        res = circuit()

        assert isinstance(res, tuple)
        assert res[0] == {1: shot_vec[0]}
        assert res[1] == {1: shot_vec[1]}
        assert res[2] == {1: shot_vec[2]}
        assert len(res) == len(shot_vec)
        assert sum(sum(res_bin.values()) for res_bin in res) == sum(shot_vec)

    @pytest.mark.parametrize("shot_vec", [(1, 10, 10), (1, 10, 1000)])
    def test_counts_binned_4_wires(self, shot_vec):
        """Check the autograd interface with computational basis state counts and
        different shot vectors on a device with 4 wires"""
        dev = qml.device("default.qubit", wires=4)

        @qml.set_shots(shot_vec)
        @qml.qnode(dev, interface="autograd")
        def circuit():
            qml.PauliX(1)
            qml.PauliX(2)
            qml.PauliX(3)
            return qml.counts()

        res = circuit()
        basis_state = "0111"

        assert isinstance(res, tuple)
        assert res[0][basis_state] == shot_vec[0]
        assert res[1][basis_state] == shot_vec[1]
        assert res[2][basis_state] == shot_vec[2]
        assert len(res) == len(shot_vec)
        assert sum(sum(res_bin.values()) for res_bin in res) == sum(shot_vec)

    @pytest.mark.parametrize("shot_vec", [(1, 10, 10), (1, 10, 1000)])
    def test_counts_operator_binned_4_wires(self, shot_vec):
        """Check the autograd interface with observable samples to obtain
        counts from and different shot vectors on a device with 4 wires"""
        dev = qml.device("default.qubit", wires=4)

        @qml.set_shots(shot_vec)
        @qml.qnode(dev, interface="autograd")
        def circuit():
            qml.PauliX(1)
            qml.PauliX(2)
            qml.PauliX(3)
            return qml.counts(qml.PauliZ(0))

        res = circuit()
        sample = 1

        assert isinstance(res, tuple)
        assert res[0][sample] == shot_vec[0]
        assert res[1][sample] == shot_vec[1]
        assert res[2][sample] == shot_vec[2]
        assert len(res) == len(shot_vec)
        assert sum(sum(res_bin.values()) for res_bin in res) == sum(shot_vec)

    meas2 = [
        qml.expval(qml.PauliZ(0)),
        qml.var(qml.PauliZ(0)),
        qml.probs(wires=[1, 0]),
        qml.sample(wires=1),
    ]

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    @pytest.mark.parametrize("meas2", meas2)
    @pytest.mark.parametrize("shots", [1000, (1, 10)])
    def test_counts_observable_finite_shots(self, interface, meas2, shots):
        """Check all interfaces with observable measurement counts and finite
        shot"""
        dev = qml.device("default.qubit", wires=3)

        if isinstance(shots, tuple) and interface == "torch":
            pytest.skip("Torch needs to be updated for shot vectors.")

        @qml.set_shots(shots)
        @qml.qnode(dev, interface=interface)
        def circuit():
            qml.PauliX(0)
            return qml.counts(wires=0), qml.apply(meas2)

        res = circuit()
        assert isinstance(res, tuple)

        num_shot_bins = 1 if isinstance(shots, int) else len(shots)

        if num_shot_bins == 1:
            counts_term_indices = [i * 2 for i in range(num_shot_bins)]
            for ind in counts_term_indices:
                assert isinstance(res[ind], dict)
        else:
            assert len(res) == 2

            assert isinstance(res[0], tuple)
            assert isinstance(res[0][0], dict)
            assert isinstance(res[1], tuple)
            assert isinstance(res[1][0], dict)

    def test_all_outcomes_kwarg_providing_observable(self):
        """Test that the dictionary keys *all* eigenvalues of the observable,
        including 0 count values, if observable is given and all_outcomes=True"""

        n_shots = 10
        dev = qml.device("default.qubit", wires=1)

        @qml.set_shots(n_shots)
        @qml.qnode(dev)
        def circuit():
            res = qml.counts(qml.PauliZ(0), all_outcomes=True)
            return res

        res = circuit()

        assert res == {1: n_shots, -1: 0}

    def test_all_outcomes_kwarg_no_observable_no_wires(self):
        """Test that the dictionary keys are *all* the possible combinations
        of basis states for the device, including 0 count values, if no wire
        count and no observable are given and all_outcomes=True"""

        n_shots = 10
        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(n_shots)
        @qml.qnode(dev)
        def circuit():
            return qml.counts(all_outcomes=True)

        res = circuit()

        assert res == {"00": n_shots, "01": 0, "10": 0, "11": 0}

    def test_all_outcomes_kwarg_providing_wires_and_no_observable(self):
        """Test that the dictionary keys are *all* possible combinations
        of basis states for the specified wires, including 0 count values,
        if wire count is given and all_outcomes=True"""

        n_shots = 10
        dev = qml.device("default.qubit", wires=4)

        @qml.set_shots(n_shots)
        @qml.qnode(dev)
        def circuit():
            return qml.counts(wires=[0, 2], all_outcomes=True)

        res = circuit()

        assert res == {"00": n_shots, "01": 0, "10": 0, "11": 0}

    def test_all_outcomes_hermitian(self):
        """Tests that the all_outcomes=True option for counts works with the
        qml.Hermitian observable"""

        n_shots = 10
        dev = qml.device("default.qubit", wires=2)

        A = np.array([[1, 0], [0, -1]])

        @qml.set_shots(n_shots)
        @qml.qnode(dev)
        def circuit(x):
            return qml.counts(qml.Hermitian(x, wires=0), all_outcomes=True)

        res = circuit(A)

        assert res == {-1.0: 0, 1.0: n_shots}

    def test_all_outcomes_multiple_measurements(self):
        """Tests that the all_outcomes=True option for counts works when
        multiple measurements are performed"""

        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit():
            return qml.sample(qml.PauliZ(0)), qml.counts(), qml.counts(all_outcomes=True)

        res = circuit()

        assert len(res[0]) == 10
        assert res[1] == {"00": 10}
        assert res[2] == {"00": 10, "01": 0, "10": 0, "11": 0}

    def test_batched_all_outcomes(self):
        """Tests that all_outcomes=True works with broadcasting."""
        n_shots = 10
        dev = qml.device("default.qubit", wires=1)

        @qml.set_shots(n_shots)
        @qml.qnode(dev)
        def circuit():
            qml.pow(qml.PauliX(0), z=[1, 2])
            return qml.counts(qml.PauliZ(0), all_outcomes=True)

        assert circuit() == [{1: 0, -1: n_shots}, {1: n_shots, -1: 0}]

    def test_counts_empty_wires(self):
        """Test that using ``qml.counts`` with an empty wire list raises an error."""
        with pytest.raises(ValueError, match="Cannot set an empty list of wires."):
            qml.counts(wires=[])

    @pytest.mark.parametrize("shots", [1, 100])
    def test_counts_no_arguments(self, shots):
        """Test that using ``qml.counts`` with no arguments returns the counts of all wires."""
        dev = qml.device("default.qubit", wires=3)

        @qml.set_shots(shots)
        @qml.qnode(dev)
        def circuit():
            return qml.counts()

        res = circuit()

        assert qml.math.allequal(res, {"000": shots})


@pytest.mark.all_interfaces
@pytest.mark.parametrize("wires, basis_state", [(None, "010"), ([2, 1], "01")])
@pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
def test_counts_no_op_finite_shots(interface, wires, basis_state):
    """Check all interfaces with computational basis state counts and finite shot"""
    n_shots = 10
    dev = qml.device("default.qubit", wires=3)

    @qml.set_shots(n_shots)
    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.PauliX(1)
        return qml.counts(wires=wires)

    res = circuit()
    assert res == {basis_state: n_shots}
    assert qml.math.get_interface(res[basis_state]) == interface


@pytest.mark.all_interfaces
@pytest.mark.parametrize("wires, basis_states", [(None, ("010", "000")), ([2, 1], ("01", "00"))])
@pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
def test_batched_counts_no_op_finite_shots(interface, wires, basis_states):
    """Check all interfaces with computational basis state counts and
    finite shot"""
    n_shots = 10
    dev = qml.device("default.qubit", wires=3)

    @qml.set_shots(n_shots)
    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.pow(qml.PauliX(1), z=[1, 2])
        return qml.counts(wires=wires)

    res = circuit()
    assert res == type(res)([{basis_state: n_shots} for basis_state in basis_states])


@pytest.mark.all_interfaces
@pytest.mark.parametrize("wires, basis_states", [(None, ("010", "000")), ([2, 1], ("01", "00"))])
@pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
def test_batched_counts_and_expval_no_op_finite_shots(interface, wires, basis_states):
    """Check all interfaces with computational basis state counts and
    finite shot"""
    n_shots = 10
    dev = qml.device("default.qubit", wires=3)

    @qml.set_shots(n_shots)
    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.pow(qml.PauliX(1), z=[1, 2])
        return qml.counts(wires=wires), qml.expval(qml.PauliZ(0))

    res = circuit()
    assert isinstance(res, tuple) and len(res) == 2
    for i, basis_state in enumerate(basis_states):
        assert list(res[0][i].keys()) == [basis_state]
        assert qml.math.allequal(list(res[0][i].values()), n_shots)
    # assert res[0] == [{basis_state: expected_n_shots} for basis_state in basis_states]
    assert len(res[1]) == 2 and qml.math.allequal(res[1], 1)


@pytest.mark.all_interfaces
@pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
def test_batched_counts_operator_finite_shots(interface):
    """Check all interfaces with observable measurement counts, batching and finite shots"""
    n_shots = 10
    dev = qml.device("default.qubit", wires=3)

    @qml.set_shots(n_shots)
    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.pow(qml.PauliX(0), z=[1, 2])
        return qml.counts(qml.PauliZ(0))

    res = circuit()
    assert res == type(res)([{-1: n_shots}, {1: n_shots}])


@pytest.mark.all_interfaces
@pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
def test_batched_counts_and_expval_operator_finite_shots(interface):
    """Check all interfaces with observable measurement counts, batching and finite shots"""
    n_shots = 10
    dev = qml.device("default.qubit", wires=3)

    @qml.set_shots(n_shots)
    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.pow(qml.PauliX(0), z=[1, 2])
        return qml.counts(qml.PauliZ(0)), qml.expval(qml.PauliZ(0))

    res = circuit()
    assert isinstance(res, tuple) and len(res) == 2
    assert res[0] == type(res[0])([{-1: n_shots}, {1: n_shots}])
    assert len(res[1]) == 2 and qml.math.allequal(res[1], [-1, 1])


class TestProcessCounts:
    """Unit tests for the counts.process_counts method"""

    @pytest.mark.parametrize("all_outcomes", [True, False])
    def test_should_not_modify_counts_dictionary(self, all_outcomes):
        """Tests that count dictionary is not modified"""

        counts = {"000": 100, "100": 100}
        expected = counts.copy()
        wire_order = qml.wires.Wires((0, 1, 2))

        qml.counts(wires=(0, 1), all_outcomes=all_outcomes).process_counts(counts, wire_order)

        assert counts == expected

    def test_all_outcomes_is_true(self):
        """When all_outcomes is True, 0 count should be added to missing outcomes in the counts dictionary"""

        counts_to_process = {"00": 100, "10": 100}
        wire_order = qml.wires.Wires((0, 1))

        actual = qml.counts(wires=wire_order, all_outcomes=True).process_counts(
            counts_to_process, wire_order=wire_order
        )

        expected_counts = {"00": 100, "01": 0, "10": 100, "11": 0}
        assert actual == expected_counts

    def test_all_outcomes_is_false(self):
        """When all_outcomes is True, 0 count should be removed from the counts dictionary"""

        counts_to_process = {"00": 0, "01": 0, "10": 0, "11": 100}
        wire_order = qml.wires.Wires((0, 1))

        actual = qml.counts(wires=wire_order, all_outcomes=False).process_counts(
            counts_to_process, wire_order=wire_order
        )

        expected_counts = {"11": 100}
        assert actual == expected_counts

    @pytest.mark.parametrize(
        "wires, expected",
        [
            ((0, 1), {"00": 100, "10": 100}),
            ((1, 0), {"00": 100, "01": 100}),
        ],
    )
    def test_wire_order(self, wires, expected):
        """Test impact of wires in qml.counts"""
        counts = {"000": 100, "100": 100}
        wire_order = qml.wires.Wires((0, 1, 2))

        actual = qml.counts(wires=wires, all_outcomes=False).process_counts(counts, wire_order)

        assert actual == expected

    @pytest.mark.parametrize("all_outcomes", [True, False])
    def test_process_count_returns_same_count_dictionary(self, all_outcomes):
        """
        Test that process_count returns same dictionary when all outcomes are in count dictionary and wire_order is same
        """

        expected = {"0": 100, "1": 100}
        wires = qml.wires.Wires(0)

        actual = qml.counts(wires=wires, all_outcomes=all_outcomes).process_counts(expected, wires)

        assert actual == expected

    @pytest.mark.parametrize(
        "wire_order, expected_result", [((0, 1), {-1.0: 3, 1.0: 2}), ((1, 0), {1.0: 5})]
    )
    def test_with_observable(self, wire_order, expected_result):
        """Test that processing counts to get the counts for an eigenvalue of an observable
        works as expected for an observable with a single wire."""

        counts_mp = qml.counts(qml.Z(0))

        result = counts_mp.process_counts({"00": 2, "10": 3}, wire_order)

        assert result == expected_result

    @pytest.mark.parametrize(
        "wire_order, expected_result",
        [
            ((0, 1, 2), {-1.0: 4, 1.0: 6}),
            ((0, 2, 1), {-1.0: 3, 1.0: 7}),
            ((2, 1, 0), {-1.0: 4, 1.0: 6}),
        ],
    )
    def test_with_observable_multi_wire(self, wire_order, expected_result):
        """Test that processing counts to get the counts for an eigenvalue of an observable
        works as expected for an observable with a single wire."""

        counts_mp = qml.counts(qml.Z(0) @ qml.Z(2))
        counts = {"000": 2, "001": 1, "101": 2, "010": 1, "110": 3, "111": 1}

        result = counts_mp.process_counts(counts, wire_order)

        assert result == expected_result

    @pytest.mark.parametrize(
        "all_outcomes, expected_result", [(True, {-1.0: 0, 1.0: 5}), (False, {1.0: 5})]
    )
    def test_process_counts_with_observable_all_outcomes(self, all_outcomes, expected_result):
        """Test that all-outcomes works as expected when returning observable/eigenvalue outcomes
        instead of counts in the computational basis"""

        counts_mp = qml.counts(qml.Z(0), all_outcomes=all_outcomes)

        result = counts_mp.process_counts({"00": 2, "10": 3}, [1, 0])

        assert result == expected_result
