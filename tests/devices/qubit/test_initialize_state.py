# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the :func:`~pennylane.devices.qubit.initialize_state` function.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.devices.qubit import initialize_state

pytestmark = pytest.mark.all_interfaces

tf = pytest.importorskip("tensorflow")
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")

SUPPORTED_INTERFACES_AND_TYPES = [
    ("numpy", np.tensor),
    ("scipy", np.ndarray),
    ("jax", jax.numpy.DeviceArray),
    ("torch", torch.Tensor),
    ("tf", tf.Tensor),
]
SUPPORTED_INTERFACES = [i[0] for i in SUPPORTED_INTERFACES_AND_TYPES]


@pytest.mark.parametrize("ml_framework,dtype", SUPPORTED_INTERFACES_AND_TYPES)
@pytest.mark.parametrize("num_wires", [1, 2, 3])
def test_initialize_state_no_ops(ml_framework, dtype, num_wires):
    """Tests that initialize_state works without prep operations provided."""
    actual = initialize_state(num_wires, ml_framework=ml_framework)
    expected = np.zeros((2,) * num_wires)
    expected[(0,) * num_wires] = 1
    assert qml.math.allequal(expected, actual)
    assert isinstance(actual, dtype)


@pytest.mark.parametrize("ml_framework,dtype", SUPPORTED_INTERFACES_AND_TYPES)
def test_initialize_state_with_prep(ml_framework, dtype):
    """Tests that initialize_state handles prep operations correctly."""
    ops = [qml.Hadamard(0), qml.CNOT([0, 1])]
    actual = initialize_state(3, prep_operations=ops, ml_framework=ml_framework)
    expected = np.zeros((2,) * 3)
    expected[0][0][0] = 1 / np.sqrt(2)
    expected[1][1][0] = 1 / np.sqrt(2)
    assert qml.math.allequal(expected, actual)
    assert isinstance(actual, dtype)


def test_too_many_wires_in_prep():
    """Tests that initialize_state fails when prep operations collectively have too many wires."""
    ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.PauliX(2)]
    with pytest.raises(ValueError, match="Expected no more than 2 distinct wires"):
        initialize_state(2, prep_operations=ops)


def test_unknown_framework_fails():
    """Tests that initialize_state fails when an unknown ml_framework is provided."""
    with pytest.raises(qml.QuantumFunctionError, match="Unknown framework"):
        initialize_state(1, ml_framework="nonsense")


@pytest.mark.parametrize("ml_framework", SUPPORTED_INTERFACES)
def test_initialize_state_does_not_sort_wires(ml_framework):
    """Tests that initialize_state does not try to sort wires before computing."""
    ops = [qml.Hadamard(0), qml.PauliX(2), qml.CNOT([0, 1])]
    actual = initialize_state(3, prep_operations=ops, ml_framework=ml_framework)
    expected = np.zeros((2,) * 3)
    # because "2" was the second label spotted, the wire_order was (0,2,1) while computing
    expected[0][1][0] = 1 / np.sqrt(2)
    expected[1][1][1] = 1 / np.sqrt(2)
    assert qml.math.allequal(expected, actual)


@pytest.mark.parametrize("ml_framework", SUPPORTED_INTERFACES)
@pytest.mark.parametrize(
    "wires",
    [
        (0, 1),
        ("a", "b"),
        ("a", 0),
    ],
)
def test_wires_are_padded_in_calculation(ml_framework, wires):
    """Tests that initialize_state pads wires when calculating the state to match num_wires."""
    ops = [qml.Hadamard(wires[0]), qml.CNOT([wires[0], wires[1]])]
    actual = initialize_state(3, prep_operations=ops, ml_framework=ml_framework)
    expected = np.zeros((2,) * 3)
    expected[0][0][0] = 1 / np.sqrt(2)
    expected[1][1][0] = 1 / np.sqrt(2)
    assert qml.math.allequal(expected, actual)


@pytest.mark.parametrize("ml_framework", SUPPORTED_INTERFACES)
def test_basis_state_prep_can_be_used(ml_framework):
    """Tests that qml.BasisState is an acceptable prep operation."""
    ops = [qml.BasisState([0, 1], wires=range(2))]
    actual = initialize_state(3, prep_operations=ops, ml_framework=ml_framework)
    expected = np.zeros((2,) * 3)
    expected[0][1][0] = 1
    assert qml.math.allequal(expected, actual)
