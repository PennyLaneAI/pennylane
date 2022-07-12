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
"""
Unit tests for the Product arithmetic class of qubit operations
"""
import pytest
import numpy as np
from copy import copy
import pennylane as qml
from pennylane import math
import pennylane.numpy as qnp

from pennylane.wires import Wires
from pennylane import QuantumFunctionError
from pennylane.ops.op_math import Product,op_prod
from pennylane.ops.op_math.product import _prod  # pylint: disable=protected-access
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

ops_rep = (
    "PauliX(wires=[0]) + PauliZ(wires=[0]) + Hadamard(wires=[0])",
    "CNOT(wires=[0, 1]) + RX(1.23, wires=[1]) + Identity(wires=[0])",
    "IsingXX(4.56, wires=[2, 3]) + Toffoli(wires=[1, 2, 3]) + Rot(0.34, 1.0, 0, wires=[0])",
)


def compare_and_expand_mat(mat1, mat2):
    """Helper function which takes two square matrices (of potentially different sizes)
    and expands the smaller matrix until their shapes match."""

    if mat1.size == mat2.size:
        return mat1, mat2

    (smaller_mat, larger_mat, flip_order) = (
        (mat1, mat2, 0) if mat1.size < mat2.size else (mat2, mat1, 1)
    )

    while smaller_mat.size < larger_mat.size:
        smaller_mat = math.cast_like(math.kron(smaller_mat, math.eye(2)), smaller_mat)

    if flip_order:
        return larger_mat, smaller_mat

    return smaller_mat, larger_mat


class TestInitialization:
    @pytest.mark.parametrize("id", ("foo", "bar"))
    def test_init_prod_op(self, id):
        prod_op = op_prod(qml.PauliX(wires=0), qml.RZ(0.23, wires="a"), do_queue=True, id=id)

        assert prod_op.wires == Wires((0, "a"))
        assert prod_op.num_wires == 2
        assert prod_op.name == "Prod"
        assert prod_op.id == id

        assert prod_op.data == [[], [0.23]]
        assert prod_op.parameters == [[], [0.23]]
        assert prod_op.num_params == 1

    def test_raise_error_fewer_then_2_summands(self):
        with pytest.raises(ValueError, match="Require at least two operators to multiply;"):
            prod_op = op_prod(qml.PauliX(0))

    def test_queue_idx(self):
        prod_op = op_prod(qml.PauliX(0), qml.Identity(1))
        assert prod_op.queue_idx is None

    def test_parameters(self):
        prod_op = op_prod(qml.RX(9.87, wires=0), qml.Rot(1.23, 4.0, 5.67, wires=1))
        assert prod_op.parameters == [[9.87], [1.23, 4.0, 5.67]]

    def test_data(self):
        prod_op = op_prod(qml.RX(9.87, wires=0), qml.Rot(1.23, 4.0, 5.67, wires=1))
        assert prod_op.data == [[9.87], [1.23, 4.0, 5.67]]

    def test_ndim_params_raises_error(self):
        prod_op = op_prod(qml.PauliX(0), qml.Identity(1))

        with pytest.raises(
            ValueError,
            match="Dimension of parameters is not currently implemented for Product operators.",
        ):
            prod_op.ndim_params()

    def test_batch_size_raises_error(self):
        prod_op = op_prod(qml.PauliX(0), qml.Identity(1))

        with pytest.raises(ValueError, match="Batch size is not defined for Product operators."):
            prod_op.batch_size()

    def test_decomposition_raises_error(self):
        prod_op = op_prod(qml.PauliX(0), qml.Identity(1))

        with pytest.raises(DecompositionUndefinedError):
            prod_op.decomposition()

    


class TestMatrix:
    @pytest.mark.parametrize("op_and_mat1", non_param_ops)
    @pytest.mark.parametrize("op_and_mat2", non_param_ops)
    def test_non_parametric_ops_two_terms(self, op_and_mat1, op_and_mat2):
        """Test matrix method for a Product  of non_parametric ops"""
        op1, mat1 = op_and_mat1
        op2, mat2 = op_and_mat2
        mat1, mat2 = compare_and_expand_mat(mat1, mat2)

        prod_op = Product (op1(wires=range(op1.num_wires)), op2(wires=range(op2.num_wires)))
        prod_mat = prod_op.matrix()

        true_mat = np.dot(mat1, mat2)
        assert np.allclose(prod_mat, true_mat)

    @pytest.mark.parametrize("op_mat1", param_ops)
    @pytest.mark.parametrize("op_mat2", param_ops)
    def test_parametric_ops_two_terms(self, op_mat1, op_mat2):
        """Test matrix method for a Product  of parametric ops"""
        op1, mat1 = op_mat1
        op2, mat2 = op_mat2

        par1 = tuple(range(op1.num_params))
        par2 = tuple(range(op2.num_params))
        mat1, mat2 = compare_and_expand_mat(mat1(*par1), mat2(*par2))

        prod_op = Product (op1(*par1, wires=range(op1.num_wires)), op2(*par2, wires=range(op2.num_wires)))
        prod_mat = prod_op.matrix()

        true_mat = np.dot(mat1, mat2)
        assert np.allclose(prod_mat, true_mat)

    @pytest.mark.parametrize("op", no_mat_ops)
    def test_error_no_mat(self, op):
        """Test that an error is raised if one of the summands doesn't
        have its matrix method defined."""
        prod_op = Product (op(wires=0), qml.PauliX(wires=2), qml.PauliZ(wires=1))
        with pytest.raises(MatrixUndefinedError):
            prod_op.matrix()

    def test_prod_ops_multi_terms(self):
        """Test matrix is correct for a Product  of more then two terms."""
        prod_op = Product (qml.PauliX(wires=0), qml.Hadamard(wires=0), qml.PauliZ(wires=0))
        mat = prod_op.matrix()

        true_mat = math.array(
            [
                [1 / math.sqrt(2),  1 / math.sqrt(2)],
                [1 / math.sqrt(2), -1 / math.sqrt(2)],
            ]
        )
        assert np.allclose(mat, true_mat)


   
class TestProperties:
    @pytest.mark.parametrize("ops_lst", ops)
    def test_num_params(self, ops_lst):
        """Test num_params property updates correctly."""
        prod_op = Product(*ops_lst)
        true_num_params = 0

        for op in ops_lst:
            true_num_params += op.num_params

        assert prod_op.num_params == true_num_params

    @pytest.mark.parametrize("ops_lst", ops)
    def test_num_wires(self, ops_lst):
        """Test num_wires property updates correctly."""
        prod_op = Product(*ops_lst)
        true_wires = set()

        for op in ops_lst:
            true_wires = true_wires.union(op.wires.toset())

        assert prod_op.num_wires == len(true_wires)

    @pytest.mark.parametrize("ops_lst", ops)
    def test_is_hermitian(self, ops_lst):
        """Test is_hermitian property updates correctly."""
        prod_op = Product(*ops_lst)
        true_hermitian_state = True

        for op in ops_lst:
            true_hermitian_state = true_hermitian_state and op.is_hermitian

        assert prod_op.is_hermitian == true_hermitian_state

    @pytest.mark.parametrize("ops_lst", ops)
    def test_queue_catagory(self, ops_lst):
        """Test queue_catagory property is always None."""  # currently not supporting queuing Sum
        prod_op = Product(*ops_lst)
        assert prod_op._queue_category is None

    def test_eigendecompostion(self):
        """Test that the computed Eigenvalues and Eigenvectors are correct."""
        diag_prod_op = Product(qml.PauliZ(wires=0), qml.Identity(wires=1))
        eig_decomp = diag_prod_op.eigendecomposition
        eig_vecs = eig_decomp["eigvec"]
        eig_vals = eig_decomp["eigval"]

        true_eigvecs = qnp.tensor(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )

        true_eigvals = qnp.tensor([-1.0, -1.0, 1.0, 1.0])

        assert np.allclose(eig_vals, true_eigvals)
        assert np.allclose(eig_vecs, true_eigvecs)

    def test_eigen_caching(self):
        """Test that the eigendecomposition is stored in cache."""
        diag_prod_op = Product(qml.PauliZ(wires=0), qml.Identity(wires=1))
        eig_decomp = diag_prod_op.eigendecomposition

        eig_vecs = eig_decomp["eigvec"]
        eig_vals = eig_decomp["eigval"]

        eigs_cache = diag_prod_op._eigs[
    (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
]
        cached_vecs = eigs_cache["eigvec"]
        cached_vals = eigs_cache["eigval"]

        assert np.allclose(eig_vals, cached_vals)
        assert np.allclose(eig_vecs, cached_vecs)



class TestWrapperFunc:
    def test_op_prod_top_level(self):
        """Test that the top level function constructs an identical instance to one
        created using the class."""

        products = (qml.PauliX(wires=1), qml.RX(1.23, wires=0), qml.CNOT(wires=[0, 1]))
        op_id = "prod_op"
        do_queue = False

        product_func_op = op_prod(*products, id=op_id, do_queue=do_queue)
        product_class_op = Product(*products, id=op_id, do_queue=do_queue)

        assert product_class_op.products == product_func_op.products
        assert qnp.allclose(product_class_op.matrix(), product_func_op.matrix())
        assert product_class_op.id == product_func_op.id
        assert product_class_op.wires == product_func_op.wires
        assert product_class_op.parameters == product_func_op.parameters


class TestPrivateProduct:
    def test_product_private(self):
        """Test the sum private method generates expected matrices."""
        mats_gen = (qnp.eye(2) for _ in range(3))

        product_mat = _prod(mats_gen)
        expected_product_mat = qnp.eye(2)
        assert qnp.allclose(product_mat, expected_product_mat)

    def test_dtype(self):
        """Test dtype keyword arg casts matrix correctly"""
        dtype = "complex128"
        mats_gen = (qnp.eye(2) for _ in range(3))

        product_mat = _prod(mats_gen, dtype=dtype)
        expected_product_mat = qnp.eye(2, dtype=dtype)

        assert product_mat.dtype == "complex128"
        assert qnp.allclose(product_mat, expected_product_mat)

    def test_cast_like(self):
        """Test cast_like keyword arg casts matrix correctly"""
        cast_like = qnp.array(2, dtype="complex128")
        mats_gen = (qnp.eye(2) for _ in range(3))

        product_mat = _prod(mats_gen, cast_like=cast_like)
        expected_product_mat = qnp.eye(2, dtype="complex128")

        assert product_mat.dtype == "complex128"
        assert qnp.allclose(product_mat, expected_product_mat)
