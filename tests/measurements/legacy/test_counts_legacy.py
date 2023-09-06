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
import numpy as np
import pytest

import pennylane as qml
from pennylane.measurements import AllCounts, Counts
from pennylane.operation import Operator


# TODO: Remove this when new CustomMP are the default
def custom_measurement_process(device, spy):
    assert len(spy.call_args_list) > 0  # make sure method is mocked properly

    samples = device._samples  # pylint:disable=protected-access
    call_args_list = list(spy.call_args_list)
    for call_args in call_args_list:
        if not call_args.kwargs.get("counts", False):
            continue
        meas = call_args.args[1]
        shot_range, bin_size = (call_args.kwargs["shot_range"], call_args.kwargs["bin_size"])
        if isinstance(meas, Operator):
            all_outcomes = meas.return_type is AllCounts
            meas = qml.counts(op=meas, all_outcomes=all_outcomes)
        old_res = device.sample(call_args.args[1], **call_args.kwargs)
        new_res = meas.process_samples(
            samples=samples, wire_order=device.wires, shot_range=shot_range, bin_size=bin_size
        )
        if isinstance(old_res, dict):
            old_res = [old_res]
            new_res = [new_res]
        for old, new in zip(old_res, new_res):
            assert old.keys() == new.keys()
            assert qml.math.allequal(list(old.values()), list(new.values()))


class TestCountsIntegration:
    # pylint:disable=too-many-public-methods,not-an-iterable

    def test_counts_dimension(self, mocker):
        """Test that the counts function outputs counts of the right size"""
        n_sample = 10

        dev = qml.device("default.qubit.legacy", wires=2, shots=n_sample)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0)
            return qml.counts(qml.PauliZ(0)), qml.counts(qml.PauliX(1))

        sample = circuit()

        assert len(sample) == 2
        assert np.all([sum(s.values()) == n_sample for s in sample])

        custom_measurement_process(dev, spy)

    def test_batched_counts_dimension(self, mocker):
        """Test that the counts function outputs counts of the right size with batching"""
        n_sample = 10

        dev = qml.device("default.qubit.legacy", wires=2, shots=n_sample)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev)
        def circuit():
            qml.RX([0.54, 0.65], wires=0)
            return qml.counts(qml.PauliZ(0)), qml.counts(qml.PauliX(1))

        sample = circuit()

        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert np.all([sum(s.values()) == n_sample for batch in sample for s in batch])

        custom_measurement_process(dev, spy)

    def test_counts_combination(self, mocker):
        """Test the output of combining expval, var and counts"""
        n_sample = 10

        dev = qml.device("default.qubit.legacy", wires=3, shots=n_sample)
        spy = mocker.spy(qml.QubitDevice, "sample")

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

        custom_measurement_process(dev, spy)

    def test_single_wire_counts(self, mocker):
        """Test the return type and shape of sampling counts from a single wire"""
        n_sample = 10

        dev = qml.device("default.qubit.legacy", wires=1, shots=n_sample)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.counts(qml.PauliZ(0))

        result = circuit()

        assert isinstance(result, dict)
        assert sum(result.values()) == n_sample

        custom_measurement_process(dev, spy)

    def test_multi_wire_counts_regular_shape(self, mocker):
        """Test the return type and shape of sampling multiple wires
        where a rectangular array is expected"""
        n_sample = 10

        dev = qml.device("default.qubit.legacy", wires=3, shots=n_sample)
        spy = mocker.spy(qml.QubitDevice, "sample")

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

        custom_measurement_process(dev, spy)

    def test_observable_return_type_is_counts(self):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Counts`"""
        n_shots = 10
        dev = qml.device("default.qubit.legacy", wires=1, shots=n_shots)

        @qml.qnode(dev)
        def circuit():
            res = qml.counts(qml.PauliZ(0))
            return res

        circuit()
        assert circuit._qfunc_output.return_type is Counts  # pylint: disable=protected-access

    def test_providing_no_observable_and_no_wires_counts(self, mocker):
        """Test that we can provide no observable and no wires to sample function"""
        dev = qml.device("default.qubit.legacy", wires=2, shots=1000)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            res = qml.counts()
            assert res.obs is None
            assert res.wires == qml.wires.Wires([])
            return res

        circuit()

        custom_measurement_process(dev, spy)

    def test_providing_no_observable_and_wires_counts(self, mocker):
        """Test that we can provide no observable but specify wires to the sample function"""
        wires = [0, 2]
        wires_obj = qml.wires.Wires(wires)
        dev = qml.device("default.qubit.legacy", wires=3, shots=1000)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            res = qml.counts(wires=wires)

            assert res.obs is None
            assert res.wires == wires_obj
            return res

        circuit()

        custom_measurement_process(dev, spy)

    def test_batched_counts_work_individually(self, mocker):
        """Test that each counts call operates independently"""
        n_shots = 10
        dev = qml.device("default.qubit.legacy", wires=1, shots=n_shots)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev)
        def circuit():
            qml.pow(qml.PauliX(0), z=[1, 2])
            return qml.counts()

        assert circuit() == [{"1": 10}, {"0": 10}]
        custom_measurement_process(dev, spy)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("wires, basis_state", [(None, "010"), ([2, 1], "01")])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    def test_counts_no_op_finite_shots(self, interface, wires, basis_state, mocker):
        """Check all interfaces with computational basis state counts and
        finite shot"""
        n_shots = 10
        dev = qml.device("default.qubit.legacy", wires=3, shots=n_shots)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev, interface=interface)
        def circuit():
            qml.PauliX(1)
            return qml.counts(wires=wires)

        res = circuit()
        assert res == {basis_state: n_shots}
        assert qml.math.get_interface(res[basis_state]) == interface

        custom_measurement_process(dev, spy)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    def test_counts_operator_finite_shots(self, interface, mocker):
        """Check all interfaces with observable measurement counts and finite
        shot"""
        n_shots = 10
        dev = qml.device("default.qubit.legacy", wires=3, shots=n_shots)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev, interface=interface)
        def circuit():
            return qml.counts(qml.PauliZ(0))

        res = circuit()
        assert res == {1: n_shots}

        custom_measurement_process(dev, spy)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("shot_vec", [(1, 10, 10), (1, 10, 1000)])
    @pytest.mark.parametrize("wires, basis_state", [(None, "010"), ([2, 1], "01")])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    def test_counts_binned(
        self, shot_vec, interface, wires, basis_state, mocker
    ):  # pylint:disable=too-many-arguments
        """Check all interfaces with computational basis state counts and
        different shot vectors"""
        dev = qml.device("default.qubit.legacy", wires=3, shots=shot_vec)
        spy = mocker.spy(qml.QubitDevice, "sample")

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

        custom_measurement_process(dev, spy)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("shot_vec", [(1, 10, 10), (1, 10, 1000)])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    def test_counts_operator_binned(self, shot_vec, interface, mocker):
        """Check all interfaces with observable measurement counts and different
        shot vectors"""
        dev = qml.device("default.qubit.legacy", wires=3, shots=shot_vec)
        spy = mocker.spy(qml.QubitDevice, "sample")

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

        custom_measurement_process(dev, spy)

    @pytest.mark.parametrize("shot_vec", [(1, 10, 10), (1, 10, 1000)])
    def test_counts_binned_4_wires(self, shot_vec, mocker):
        """Check the autograd interface with computational basis state counts and
        different shot vectors on a device with 4 wires"""
        dev = qml.device("default.qubit.legacy", wires=4, shots=shot_vec)
        spy = mocker.spy(qml.QubitDevice, "sample")

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

        custom_measurement_process(dev, spy)

    @pytest.mark.parametrize("shot_vec", [(1, 10, 10), (1, 10, 1000)])
    def test_counts_operator_binned_4_wires(self, shot_vec, mocker):
        """Check the autograd interface with observable samples to obtain
        counts from and different shot vectors on a device with 4 wires"""
        dev = qml.device("default.qubit.legacy", wires=4, shots=shot_vec)
        spy = mocker.spy(qml.QubitDevice, "sample")

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

        custom_measurement_process(dev, spy)

    meas2 = [
        qml.expval(qml.PauliZ(0)),
        qml.var(qml.PauliZ(0)),
        qml.probs(wires=[1, 0]),
        qml.sample(wires=1),
    ]

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    @pytest.mark.parametrize("meas2", meas2)
    @pytest.mark.parametrize("shots", [1000, (1, 10)])
    def test_counts_observable_finite_shots(self, interface, meas2, shots, mocker):
        """Check all interfaces with observable measurement counts and finite
        shot"""
        dev = qml.device("default.qubit.legacy", wires=3, shots=shots)
        spy = mocker.spy(qml.QubitDevice, "sample")

        if isinstance(shots, tuple) and interface == "torch":
            pytest.skip("Torch needs to be updated for shot vectors.")

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

        custom_measurement_process(dev, spy)

    def test_all_outcomes_kwarg_providing_observable(self, mocker):
        """Test that the dictionary keys *all* eigenvalues of the observable,
        including 0 count values, if observable is given and all_outcomes=True"""

        n_shots = 10
        dev = qml.device("default.qubit.legacy", wires=1, shots=n_shots)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev)
        def circuit():
            res = qml.counts(qml.PauliZ(0), all_outcomes=True)
            return res

        res = circuit()

        assert res == {1: n_shots, -1: 0}

        custom_measurement_process(dev, spy)

    def test_all_outcomes_kwarg_no_observable_no_wires(self, mocker):
        """Test that the dictionary keys are *all* the possible combinations
        of basis states for the device, including 0 count values, if no wire
        count and no observable are given and all_outcomes=True"""

        n_shots = 10
        dev = qml.device("default.qubit.legacy", wires=2, shots=n_shots)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev)
        def circuit():
            return qml.counts(all_outcomes=True)

        res = circuit()

        assert res == {"00": n_shots, "01": 0, "10": 0, "11": 0}

        custom_measurement_process(dev, spy)

    def test_all_outcomes_kwarg_providing_wires_and_no_observable(self, mocker):
        """Test that the dictionary keys are *all* possible combinations
        of basis states for the specified wires, including 0 count values,
        if wire count is given and all_outcomes=True"""

        n_shots = 10
        dev = qml.device("default.qubit.legacy", wires=4, shots=n_shots)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev)
        def circuit():
            return qml.counts(wires=[0, 2], all_outcomes=True)

        res = circuit()

        assert res == {"00": n_shots, "01": 0, "10": 0, "11": 0}

        custom_measurement_process(dev, spy)

    def test_all_outcomes_hermitian(self, mocker):
        """Tests that the all_outcomes=True option for counts works with the
        qml.Hermitian observable"""

        n_shots = 10
        dev = qml.device("default.qubit.legacy", wires=2, shots=n_shots)
        spy = mocker.spy(qml.QubitDevice, "sample")

        A = np.array([[1, 0], [0, -1]])

        @qml.qnode(dev)
        def circuit(x):
            return qml.counts(qml.Hermitian(x, wires=0), all_outcomes=True)

        res = circuit(A)

        assert res == {-1.0: 0, 1.0: n_shots}

        custom_measurement_process(dev, spy)

    def test_all_outcomes_multiple_measurements(self, mocker):
        """Tests that the all_outcomes=True option for counts works when
        multiple measurements are performed"""

        dev = qml.device("default.qubit.legacy", wires=2, shots=10)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev)
        def circuit():
            return qml.sample(qml.PauliZ(0)), qml.counts(), qml.counts(all_outcomes=True)

        res = circuit()

        assert len(res[0]) == 10
        assert res[1] == {"00": 10}
        assert res[2] == {"00": 10, "01": 0, "10": 0, "11": 0}
        custom_measurement_process(dev, spy)

    def test_batched_all_outcomes(self, mocker):
        """Tests that all_outcomes=True works with broadcasting."""
        n_shots = 10
        dev = qml.device("default.qubit.legacy", wires=1, shots=n_shots)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev)
        def circuit():
            qml.pow(qml.PauliX(0), z=[1, 2])
            return qml.counts(qml.PauliZ(0), all_outcomes=True)

        assert circuit() == [{1: 0, -1: n_shots}, {1: n_shots, -1: 0}]

        custom_measurement_process(dev, spy)

    def test_counts_empty_wires(self):
        """Test that using ``qml.counts`` with an empty wire list raises an error."""
        with pytest.raises(ValueError, match="Cannot set an empty list of wires."):
            qml.counts(wires=[])

    @pytest.mark.parametrize("shots", [1, 100])
    def test_counts_no_arguments(self, shots):
        """Test that using ``qml.counts`` with no arguments returns the counts of all wires."""
        dev = qml.device("default.qubit.legacy", wires=3, shots=shots)

        @qml.qnode(dev)
        def circuit():
            return qml.counts()

        res = circuit()

        assert qml.math.allequal(res, {"000": shots})


@pytest.mark.all_interfaces
@pytest.mark.parametrize("wires, basis_state", [(None, "010"), ([2, 1], "01")])
@pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
def test_counts_no_op_finite_shots(interface, wires, basis_state, mocker):
    """Check all interfaces with computational basis state counts and finite shot"""
    n_shots = 10
    dev = qml.device("default.qubit.legacy", wires=3, shots=n_shots)
    spy = mocker.spy(qml.QubitDevice, "sample")

    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.PauliX(1)
        return qml.counts(wires=wires)

    res = circuit()
    assert res == {basis_state: n_shots}
    assert qml.math.get_interface(res[basis_state]) == interface

    custom_measurement_process(dev, spy)


@pytest.mark.all_interfaces
@pytest.mark.parametrize("wires, basis_states", [(None, ("010", "000")), ([2, 1], ("01", "00"))])
@pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
def test_batched_counts_no_op_finite_shots(interface, wires, basis_states, mocker):
    """Check all interfaces with computational basis state counts and
    finite shot"""
    n_shots = 10
    dev = qml.device("default.qubit.legacy", wires=3, shots=n_shots)
    spy = mocker.spy(qml.QubitDevice, "sample")

    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.pow(qml.PauliX(1), z=[1, 2])
        return qml.counts(wires=wires)

    assert circuit() == [{basis_state: n_shots} for basis_state in basis_states]

    custom_measurement_process(dev, spy)


@pytest.mark.all_interfaces
@pytest.mark.parametrize("wires, basis_states", [(None, ("010", "000")), ([2, 1], ("01", "00"))])
@pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
def test_batched_counts_and_expval_no_op_finite_shots(interface, wires, basis_states, mocker):
    """Check all interfaces with computational basis state counts and
    finite shot"""
    n_shots = 10
    dev = qml.device("default.qubit.legacy", wires=3, shots=n_shots)
    spy = mocker.spy(qml.QubitDevice, "sample")

    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.pow(qml.PauliX(1), z=[1, 2])
        return qml.counts(wires=wires), qml.expval(qml.PauliZ(0))

    res = circuit()
    assert isinstance(res, tuple) and len(res) == 2
    assert res[0] == [{basis_state: n_shots} for basis_state in basis_states]
    assert len(res[1]) == 2 and qml.math.allequal(res[1], 1)

    custom_measurement_process(dev, spy)


@pytest.mark.all_interfaces
@pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
def test_batched_counts_operator_finite_shots(interface, mocker):
    """Check all interfaces with observable measurement counts, batching and finite shots"""
    n_shots = 10
    dev = qml.device("default.qubit.legacy", wires=3, shots=n_shots)
    spy = mocker.spy(qml.QubitDevice, "sample")

    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.pow(qml.PauliX(0), z=[1, 2])
        return qml.counts(qml.PauliZ(0))

    assert circuit() == [{-1: n_shots}, {1: n_shots}]

    custom_measurement_process(dev, spy)


@pytest.mark.all_interfaces
@pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
def test_batched_counts_and_expval_operator_finite_shots(interface, mocker):
    """Check all interfaces with observable measurement counts, batching and finite shots"""
    n_shots = 10
    dev = qml.device("default.qubit.legacy", wires=3, shots=n_shots)
    spy = mocker.spy(qml.QubitDevice, "sample")

    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.pow(qml.PauliX(0), z=[1, 2])
        return qml.counts(qml.PauliZ(0)), qml.expval(qml.PauliZ(0))

    res = circuit()
    assert isinstance(res, tuple) and len(res) == 2
    assert res[0] == [{-1: n_shots}, {1: n_shots}]
    assert len(res[1]) == 2 and qml.math.allequal(res[1], [-1, 1])

    custom_measurement_process(dev, spy)
