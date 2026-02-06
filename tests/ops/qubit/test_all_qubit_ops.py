# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for the available built-in discrete-variable quantum operations. Only tests over
multiple types of operations should exist in this file. Type-specific tests should go in
the more specific file.
"""
# pylint: disable=too-few-public-methods
import pytest
from gate_data import I

import pennylane as qp


class TestOperations:
    """Tests for the operations"""

    @pytest.mark.parametrize(
        "op",
        [
            (qp.Hadamard(wires=0)),
            (qp.PauliX(wires=0)),
            (qp.PauliY(wires=0)),
            (qp.PauliZ(wires=0)),
            (qp.S(wires=0)),
            (qp.T(wires=0)),
            (qp.SX(wires=0)),
            (qp.RX(0.3, wires=0)),
            (qp.RY(0.3, wires=0)),
            (qp.RZ(0.3, wires=0)),
            (qp.PhaseShift(0.3, wires=0)),
            (qp.Rot(0.3, 0.4, 0.5, wires=0)),
        ],
    )
    def test_single_qubit_rot_angles(self, op):
        """Tests that the Rot gates yielded by single_qubit_rot_angles
        are equivalent to the true operations up to a global phase."""
        angles = op.single_qubit_rot_angles()
        obtained_mat = qp.Rot(*angles, wires=0).matrix()

        # Check whether the two matrices are each others conjugate transposes
        mat_product = qp.math.dot(op.matrix(), qp.math.conj(obtained_mat.T))
        mat_product /= mat_product[0, 0]

        assert qp.math.allclose(mat_product, I)
