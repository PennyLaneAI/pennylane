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
"""Tests for the SProd class representing the product of an operator by a scalar"""

import pytest
import numpy as np
from copy import copy
import pennylane as qml
from pennylane import math
import pennylane.numpy as qnp

from pennylane.wires import Wires
from pennylane import QuantumFunctionError
from pennylane.ops.op_math.sprod import s_prod, SProd
from pennylane.ops.op_math.sprod import _sprod  # pylint: disable=protected-access
from pennylane.operation import MatrixUndefinedError, DecompositionUndefinedError
import gate_data as gd  # a file containing matrix rep of each gate

no_mat_ops = (
    qml.Barrier,
    qml.WireCut,
)

non_param_ops = (
    (qml.Identity, gd.I),
    (qml.Hadamard, gd.H),
    (qml.PauliX, gd.X),
    (qml.PauliY, gd.Y),
    (qml.PauliZ, gd.Z),
    (qml.S, gd.S),
    (qml.T, gd.T),
    (qml.SX, gd.SX),
    (qml.CNOT, gd.CNOT),
    (qml.CZ, gd.CZ),
    (qml.CY, gd.CY),
    (qml.SWAP, gd.SWAP),
    (qml.ISWAP, gd.ISWAP),
    (qml.SISWAP, gd.SISWAP),
    (qml.CSWAP, gd.CSWAP),
    (qml.Toffoli, gd.Toffoli),
)

param_ops = (
    (qml.RX, gd.Rotx),
    (qml.RY, gd.Roty),
    (qml.RZ, gd.Rotz),
    (qml.PhaseShift, gd.Rphi),
    (qml.Rot, gd.Rot3),
    (qml.U1, gd.U1),
    (qml.U2, gd.U2),
    (qml.U3, gd.U3),
    (qml.CRX, gd.CRotx),
    (qml.CRY, gd.CRoty),
    (qml.CRZ, gd.CRotz),
    (qml.CRot, gd.CRot3),
    (qml.IsingXX, gd.IsingXX),
    (qml.IsingYY, gd.IsingYY),
    (qml.IsingZZ, gd.IsingZZ),
)

ops = (
    (qml.PauliX(wires=0), qml.PauliZ(wires=0), qml.Hadamard(wires=0)),
    (qml.CNOT(wires=[0, 1]), qml.RX(1.23, wires=1), qml.Identity(wires=0)),
    (
        qml.IsingXX(4.56, wires=[2, 3]),
        qml.Toffoli(wires=[1, 2, 3]),
        qml.Rot(0.34, 1.0, 0, wires=0),
    ),
)


class TestInitialization:
    @pytest.mark.parametrize("id", ("foo", "bar"))
    def test_init_sprod_op(self, test_id):
        sprod_op = s_prod(3.14, qml.RX(0.23, wires="a"), do_queue=True, id=test_id)

        assert sprod_op.wires == Wires((0, "a"))
        assert sprod_op.num_wires == 2
        assert sprod_op.name == "Sum"
        assert sprod_op.id == id

        assert sprod_op.data == [[], [0.23]]
        assert sprod_op.parameters == [[], [0.23]]
        assert sprod_op.num_params == 1

    def test_raise_error_fewer_then_2_summands(self):
        with pytest.raises(ValueError, match="Require at least two operators to sum;"):
            sum_op = op_sum(qml.PauliX(0))

    def test_queue_idx(self):
        sum_op = op_sum(qml.PauliX(0), qml.Identity(1))
        assert sum_op.queue_idx is None

    def test_parameters(self):
        sum_op = op_sum(qml.RX(9.87, wires=0), qml.Rot(1.23, 4.0, 5.67, wires=1))
        assert sum_op.parameters == [[9.87], [1.23, 4.0, 5.67]]

    def test_data(self):
        sum_op = op_sum(qml.RX(9.87, wires=0), qml.Rot(1.23, 4.0, 5.67, wires=1))
        assert sum_op.data == [[9.87], [1.23, 4.0, 5.67]]

    @pytest.mark.parametrize("ops_lst", ops)
    def test_terms(self, ops_lst):
        sum_op = Sum(*ops_lst)
        coeff, ops = sum_op.terms()

        assert coeff == [1.0, 1.0, 1.0]

        for op1, op2 in zip(ops, ops_lst):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert op1.data == op2.data

    def test_ndim_params_raises_error(self):
        sum_op = op_sum(qml.PauliX(0), qml.Identity(1))

        with pytest.raises(
            ValueError,
            match="Dimension of parameters is not currently implemented for Sum operators.",
        ):
            sum_op.ndim_params()

    def test_batch_size_raises_error(self):
        sum_op = op_sum(qml.PauliX(0), qml.Identity(1))

        with pytest.raises(ValueError, match="Batch size is not defined for Sum operators."):
            sum_op.batch_size()

    def test_decomposition_raises_error(self):
        sum_op = op_sum(qml.PauliX(0), qml.Identity(1))

        with pytest.raises(DecompositionUndefinedError):
            sum_op.decomposition()
