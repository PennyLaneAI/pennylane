# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for pennylane/labs/dla/dense_util.py functionality"""
import numpy as np
import pytest

import pennylane as qml
from pennylane import X, Y, Z
from pennylane.pauli import trace_inner_product


@pytest.mark.parametrize(
    "op1, op2, true_res",
    [
        (X(0), X(0), 1.0),
        (X(0), Y(0), 0.0),
        (X(0) @ X(0), X(0), 0.0),
        (X(0) @ X(0), X(0) @ X(0), 1.0),
    ],
)
def test_trace_inner_product(op1, op2, true_res):
    """Test the results from the trace inner product"""
    res = qml.pauli.trace_inner_product(op1, op2)
    assert np.allclose(res, true_res)


def test_trace_inner_product_broadcast():
    """Test the broadcasting of the trace inner product for dense inputs"""
    paulis = [qml.matrix(op, wire_order=range(2)) for op in qml.pauli.pauli_group(2)]
    res = qml.pauli.trace_inner_product(paulis, paulis)
    assert np.allclose(res, np.eye(4**2))

    res = qml.pauli.trace_inner_product(paulis, qml.matrix(X(0), wire_order=range(2)))
    assert np.allclose(res, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    res = qml.pauli.trace_inner_product(qml.matrix(X(0), wire_order=range(2)), paulis)
    assert np.allclose(res, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])


@pytest.mark.parametrize("op1", [X(0), -0.8 * X(0) @ X(1), X(0) @ Y(2), X(0) @ Z(1) + X(1) @ X(2)])
@pytest.mark.parametrize(
    "op2", [X(0), X(0) + X(0) @ X(1), 0.2 * X(0) @ Y(2), X(0) @ Z(1) + X(1) @ X(2)]
)
def test_trace_inner_product_consistency(op1, op2):
    """Test that the trace inner product norm for different operators is consistent"""
    res1 = trace_inner_product(
        qml.matrix(op1, wire_order=range(3)), qml.matrix(op2, wire_order=range(3))
    )
    res2 = trace_inner_product(op1.pauli_rep, op2.pauli_rep)
    res3 = trace_inner_product(op1, op2)
    assert np.allclose(res1, res2)
    assert np.allclose(res1, res3)


def test_trace_inner_product_raises_error_wrong_inputs():
    """Test trace_inner_product raises an error when two mismatching inputs are input"""
    with pytest.raises(TypeError, match="Both input operators need"):
        _ = qml.pauli.trace_inner_product(X(0), qml.matrix(X(0)))
