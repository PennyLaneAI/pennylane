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
import pytest

import pennylane as qml
from pennylane.measurements import AllCounts
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


@pytest.mark.all_interfaces
@pytest.mark.parametrize("wires, basis_state", [(None, "010"), ([2, 1], "01")])
@pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
def test_counts_no_op_finite_shots(interface, wires, basis_state, mocker):
    """Check all interfaces with computational basis state counts and finite shot"""
    n_shots = 10
    dev = qml.device("default.qubit", wires=3, shots=n_shots)
    spy = mocker.spy(qml.QubitDevice, "sample")

    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.PauliX(1)
        return qml.counts(wires=wires)

    res = circuit()
    assert res == {basis_state: n_shots}

    custom_measurement_process(dev, spy)


@pytest.mark.all_interfaces
@pytest.mark.parametrize("wires, basis_states", [(None, ("010", "000")), ([2, 1], ("01", "00"))])
@pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
def test_batched_counts_no_op_finite_shots(interface, wires, basis_states, mocker):
    """Check all interfaces with computational basis state counts and
    finite shot"""
    n_shots = 10
    dev = qml.device("default.qubit", wires=3, shots=n_shots)
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
    dev = qml.device("default.qubit", wires=3, shots=n_shots)
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
    dev = qml.device("default.qubit", wires=3, shots=n_shots)
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
    dev = qml.device("default.qubit", wires=3, shots=n_shots)
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
