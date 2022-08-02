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

import pytest
from copy import copy

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops.op_math import Exp


class TestInitialization:
    """Test the initalization process and standard properties."""

    def test_pauli_base(self):

        base = qml.PauliX("a")

        op = Exp(base, id="something")

        assert op.base is base
        assert op.coeff == 1
        assert op.name == "Exp(1 PauliX)"
        assert op.id == "something"

        assert op.num_params == 1
        assert op.parameters == [1, []]
        assert op.data == [1, []]

        assert op.wires == qml.wires.Wires("a")

    def test_provided_coeff(self):

        base = qml.PauliZ("b") @ qml.PauliZ("c")
        coeff = np.array(1.234)

        op = Exp(base, coeff)

        assert op.base is base
        assert op.coeff is coeff
        assert op.name == "Exp(1.234 PauliZ(wires=['b']) @ PauliZ(wires=['c']))"

        assert op.num_params == 1
        assert op.parameters == [coeff, []]
        assert op.data == [coeff, []]

        assert op.wires == qml.wires.Wires(("b", "c"))

    def test_parametric_base(self):

        base_coeff = 1.23
        base = qml.RX(base_coeff, wires=5)
        coeff = np.array(-2.0)

        op = Exp(base, coeff)

        assert op.base is base
        assert op.coeff is coeff
        assert op.name == "Exp(-2.0 RX)"

        assert op.num_params == 2
        assert op.data == [coeff, [base_coeff]]

        assert op.wires == qml.wires.Wires(5)


class TestProperties:
    def test_data(self):

        phi = np.array(1.234)
        coeff = np.array(2.345)

        base = qml.RX(phi, wires=0)
        op = Exp(base, coeff)

        assert op.data == [coeff, [phi]]

        new_data = [-2.1, [-3.4]]
        op.data = new_data

        assert op.data == new_data
        assert op.coeff == -2.1
        assert base.data == [-3.4]

    def test_queue_category_ops(self):
        assert Exp(qml.PauliX(0), -1.234j)._queue_category == "_ops"

        assert Exp(qml.PauliX(0), 1 + 2j)._queue_category is None

        assert Exp(qml.RX(1.2, 0), -1.2j)._queue_category is None

    def test_is_hermitian(self):
        assert Exp(qml.PauliX(0), -1.0).is_hermitian

        assert not Exp(qml.PauliX(0), 1.0 + 2j).is_hermitian

        assert not Exp(qml.RX(1.2, wires=0)).is_hermitian


class TestMatrix:
    def test_matrix_rx(self):
        """Test the matrix comparing to the rx gate."""
        phi = np.array(1.234)
        exp_rx = Exp(qml.PauliX(0), -0.5j * phi)
        rx = qml.RX(phi, 0)

        assert qml.math.allclose(exp_rx.matrix(), rx.matrix())

    def test_sparse_matrix(self):
        """Test the sparse matrix function."""
        from scipy.sparse import csr_matrix

        format = "lil"

        H = np.array([[6 + 0j, 1 - 2j], [1 + 2j, -1]])
        H = csr_matrix(H)
        base = qml.SparseHamiltonian(H, wires=0)

        op = Exp(base, 3)

        sparse_mat = op.sparse_matrix(format=format)
        assert sparse_mat.format == format

        dense_mat = qml.matrix(op)

        assert qml.math.allclose(sparse_mat.toarray(), dense_mat)


class TestArithmetic:
    def test_pow(self):

        base = qml.PauliX(0)
        coeff = 2j
        z = 0.3

        op = Exp(base, coeff)
        pow_op = op.pow(z)

        assert isinstance(pow_op, Exp)
        assert pow_op.base is base
        assert pow_op.coeff == coeff * z
