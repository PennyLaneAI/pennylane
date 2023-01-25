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
"""
Unit tests for controlled operation decompositions.
"""

from functools import reduce
import pytest
import numpy as np

import pennylane as qml
from pennylane.ops import ctrl_decomp_zyz


class TestControlledDecompositionZYZ:
    """tests for qml.ops.ctrl_decomp_zyz"""

    def test_invalid_op_error(self):
        """Tests that an error is raised when an invalid operation is passed"""
        with pytest.raises(
            ValueError, match="The target operation must be a single-qubit operation"
        ):
            qml.ops.ctrl_decomp_zyz(qml.CNOT([0, 1]), [2, 3, 4])

    @pytest.mark.parametrize("op_cls", [qml.Hadamard, qml.PauliZ, qml.S])
    def test_non_parametric_decomposition(self, op_cls, tol):
        """Tests that the decomposition of a non-parametric operation is correct"""
        op = op_cls(3)
        decomps = ctrl_decomp_zyz(op, [0, 1, 2])
        decomp_mats = (
            np.kron(np.eye(8), decomp_op.matrix())
            if not isinstance(decomp_op, qml.MultiControlledX)
            else decomp_op.matrix()
            for decomp_op in decomps
        )
        expected_op = qml.ctrl(op, [0, 1, 2])

        res = reduce(np.matmul, decomp_mats)
        expected = expected_op.matrix()
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "op",
        [
            qml.RX(0.123, wires=3),
            qml.Rot(0.123, 0.456, 0.789, wires=3),
            qml.PhaseShift(1.5, wires=3),
            qml.QubitUnitary(np.array([[0, 1], [1, 0]]), wires=3),
        ],
    )
    def test_parametric_decomposition(self, op, tol):
        """Tests that the decomposition of a parametric operation is correct"""
        decomps = ctrl_decomp_zyz(op, [0, 1, 2])
        decomp_mats = (
            np.kron(np.eye(8), decomp_op.matrix())
            if not isinstance(decomp_op, qml.MultiControlledX)
            else decomp_op.matrix()
            for decomp_op in decomps
        )
        expected_op = qml.ctrl(op, [0, 1, 2])

        res = reduce(np.matmul, decomp_mats)
        expected = expected_op.matrix()
        assert np.allclose(res, expected, atol=tol, rtol=0)
