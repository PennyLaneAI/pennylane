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
"""This file tests the convert_to_numpy_parameters function."""
import numpy as np
import pytest

import pennylane as qml
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.convert_to_numpy_parameters import (
    _convert_measurement_to_numpy_data,
    _convert_op_to_numpy_data,
)

ml_frameworks_list = [
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]


@pytest.mark.parametrize("framework", ml_frameworks_list)
@pytest.mark.parametrize("shots", [None, 100])
def test_convert_arrays_to_numpy(framework, shots):
    """Tests that convert_to_numpy_parameters works with autograd arrays."""

    x = qml.math.asarray(np.array(1.234), like=framework)
    y = qml.math.asarray(np.array(0.652), like=framework)
    M = qml.math.asarray(np.eye(2), like=framework)
    state = qml.math.asarray(np.array([1, 0]), like=framework)

    numpy_data = np.array(0.62)

    ops = [
        qml.StatePrep(state, 0),
        qml.RX(x, 0),
        qml.RY(y, 1),
        qml.CNOT((0, 1)),
        qml.RZ(numpy_data, 0),
    ]
    m = [qml.state(), qml.expval(qml.Hermitian(M, 0))]

    qs = qml.tape.QuantumScript(ops, m, shots=shots)
    new_qs, fn = convert_to_numpy_parameters(qs)

    assert len(new_qs) == 1
    assert isinstance(new_qs[0], qml.tape.QuantumScript)

    new_qs = new_qs[0]
    # check ops that should be unaltered
    assert new_qs[3] is qs[3]
    assert new_qs[4] is qs[4]
    assert new_qs.measurements[0] is qs.measurements[0]
    assert fn([0.5]) == 0.5

    for ind in (0, 1, 2):
        qml.assert_equal(new_qs[ind], qs[ind], check_interface=False, check_trainability=False)
        assert qml.math.get_interface(*new_qs[ind].data) == "numpy"
    qml.assert_equal(new_qs[6], qs[6], check_interface=False, check_trainability=False)
    assert qml.math.get_interface(*new_qs[6].obs.data) == "numpy"

    # check shots attribute matches
    assert new_qs.shots == qs.shots


@pytest.mark.autograd
def test_preserves_trainable_params():
    """Test that convert_to_numpy_parameters preserves the trainable parameters property."""
    ops = [qml.RX(qml.numpy.array(2.0), 0), qml.RY(qml.numpy.array(3.0), 0)]
    qs = qml.tape.QuantumScript(ops)
    qs.trainable_params = {0}
    output, _ = convert_to_numpy_parameters(qs)
    assert output[0].trainable_params == [0]


@pytest.mark.autograd
def test_unwraps_arithmetic_op():
    """Test that the operator helper function can handle operator arithmetic objects."""
    op1 = qml.s_prod(qml.numpy.array(2.0), qml.PauliX(0))
    op2 = qml.s_prod(qml.numpy.array(3.0), qml.PauliY(0))

    sum_op = qml.sum(op1, op2)

    unwrapped_op = _convert_op_to_numpy_data(sum_op)
    assert qml.math.get_interface(*unwrapped_op.data) == "numpy"
    assert qml.math.get_interface(*unwrapped_op.data) == "numpy"


@pytest.mark.autograd
def test_unwraps_arithmetic_op_measurement():
    """Test that the measurement helper function can handle operator arithmetic objects."""
    op1 = qml.s_prod(qml.numpy.array(2.0), qml.PauliX(0))
    op2 = qml.s_prod(qml.numpy.array(3.0), qml.PauliY(0))

    sum_op = qml.sum(op1, op2)
    m = qml.expval(sum_op)

    unwrapped_m = _convert_measurement_to_numpy_data(m)
    unwrapped_op = unwrapped_m.obs
    assert qml.math.get_interface(*unwrapped_op.data) == "numpy"
    assert qml.math.get_interface(*unwrapped_op.data) == "numpy"


@pytest.mark.autograd
def test_unwraps_prod_observables():
    """Test that the measurement helper function can set data on a prod observable."""
    mat = qml.numpy.eye(2)
    obs = qml.prod(qml.PauliZ(0), qml.Hermitian(mat, 1))
    m = qml.expval(obs)

    unwrapped_m = _convert_measurement_to_numpy_data(m)
    assert qml.math.get_interface(*unwrapped_m.obs.data) == "numpy"


@pytest.mark.torch
def test_unwraps_mp_eigvals():
    """Test that a measurememnt process with autograd eigvals unwraps them to numpy."""
    import torch

    eigvals = torch.tensor([0.5, 0.5])
    m = qml.measurements.ExpectationMP(eigvals=eigvals, wires=qml.wires.Wires(0))

    unwrapped_m = _convert_measurement_to_numpy_data(m)
    assert qml.math.get_interface(unwrapped_m.eigvals) == "numpy"


def test_mp_numpy_eigvals():
    """Test that a measurement process with numpy eigvals is returned unchanged."""
    eigvals = np.array([0.5, 0.5])
    m = qml.measurements.ExpectationMP(eigvals=eigvals, wires=qml.wires.Wires(0))

    unwrapped_m = _convert_measurement_to_numpy_data(m)
    assert m is unwrapped_m
