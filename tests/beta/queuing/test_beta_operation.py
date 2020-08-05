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
Unit tests for :mod:`pennylane.operation`.
"""
import itertools
import functools
from unittest.mock import patch

import pytest
import numpy as np
from numpy.linalg import multi_dot

import pennylane as qml
import pennylane._queuing

from gate_data import I, X, Y, Rotx, Roty, Rotz, CRotx, CRoty, CRotz, CNOT, Rot3, Rphi
from pennylane.wires import Wires

# --------------------
# Beta related imports
# --------------------

# Fixture for importing beta files
from conftest import import_beta

# Need to import the beta operations
from pennylane.beta.queuing.ops import Identity
from pennylane.beta.queuing.operation import Tensor
from pennylane.beta.queuing.ops.qubit import CNOT, PauliX, PauliY, PauliZ, Hadamard, Hermitian, QubitUnitary, S, RY
import pennylane.beta.queuing.measure as beta_measure


# pylint: disable=no-self-use, no-member, protected-access, pointless-statement

# Operation subclasses to test
op_classes = [getattr(qml.ops, cls) for cls in qml.ops.__all__]
op_classes_cv = [getattr(qml.ops, cls) for cls in qml.ops._cv__all__]
op_classes_gaussian = [cls for cls in op_classes_cv if cls.supports_heisenberg]

op_classes_param_testable = op_classes.copy()
op_classes_param_testable.remove(qml.ops.PauliRot)

def U3(theta, phi, lam):
    return Rphi(phi) @ Rphi(lam) @ Rot3(lam, theta, -lam)

class TestTensor:
    """Unit tests for the Tensor class"""

    def test_construct(self):
        """Test construction of a tensor product"""
        X = PauliX(0)
        Y = PauliY(2)
        T = Tensor(X, Y)
        assert T.obs == [X, Y]

        T = Tensor(T, Y)
        assert T.obs == [X, Y, Y]

        with pytest.raises(ValueError, match="Can only perform tensor products between observables"):
            Tensor(T, CNOT(wires=[0, 1]))

    def test_name(self):
        """Test that the names of the observables are
        returned as expected"""
        X = PauliX(0)
        Y = PauliY(2)
        t = Tensor(X, Y)
        assert t.name == [X.name, Y.name]

    def test_num_wires(self):
        """Test that the correct number of wires is returned"""
        p = np.array([0.5])
        X = PauliX(0)
        Y = Hermitian(p, wires=[1, 2])
        t = Tensor(X, Y)
        assert t.num_wires == 3

    def test_wires(self):
        """Test that the correct nested list of wires is returned"""
        p = np.array([0.5])
        X = PauliX(0)
        Y = Hermitian(p, wires=[1, 2])
        t = Tensor(X, Y)
        assert t.wires == Wires([0, 1, 2])

    def test_params(self):
        """Test that the correct flattened list of parameters is returned"""
        p = np.array([0.5])
        X = PauliX(0)
        Y = Hermitian(p, wires=[1, 2])
        t = Tensor(X, Y)
        assert t.data == [p]

    def test_num_params(self):
        """Test that the correct number of parameters is returned"""
        p = np.array([0.5])
        X = PauliX(0)
        Y = Hermitian(p, wires=[1, 2])
        Z = Hermitian(p, wires=[1, 2])
        t = Tensor(X, Y, Z)
        assert t.num_params == 2

    def test_parameters(self):
        """Test that the correct nested list of parameters is returned"""
        p = np.array([0.5])
        X = PauliX(0)
        Y = Hermitian(p, wires=[1, 2])
        t = Tensor(X, Y)
        assert t.parameters == [[], [p]]

    def test_multiply_obs(self):
        """Test that multiplying two observables
        produces a tensor"""
        X = PauliX(0)
        Y = Hadamard(2)
        t = X @ Y
        assert isinstance(t, Tensor)
        assert t.obs == [X, Y]

    def test_multiply_obs_tensor(self):
        """Test that multiplying an observable by a tensor
        produces a tensor"""
        X = PauliX(0)
        Y = Hadamard(2)
        Z = PauliZ(1)

        t = X @ Y
        t = Z @ t

        assert isinstance(t, Tensor)
        assert t.obs == [Z, X, Y]

    def test_multiply_tensor_obs(self):
        """Test that multiplying a tensor by an observable
        produces a tensor"""
        X = PauliX(0)
        Y = Hadamard(2)
        Z = PauliZ(1)

        t = X @ Y
        t = t @ Z

        assert isinstance(t, Tensor)
        assert t.obs == [X, Y, Z]

    def test_multiply_tensor_tensor(self):
        """Test that multiplying a tensor by a tensor
        produces a tensor"""
        X = PauliX(0)
        Y = PauliY(2)
        Z = PauliZ(1)
        H = Hadamard(3)

        t1 = X @ Y
        t2 = Z @ H
        t = t2 @ t1

        assert isinstance(t, Tensor)
        assert t.obs == [Z, H, X, Y]

    def test_multiply_tensor_in_place(self):
        """Test that multiplying a tensor in-place
        produces a tensor"""
        X = PauliX(0)
        Y = PauliY(2)
        Z = PauliZ(1)
        H = Hadamard(3)

        t = X
        t @= Y
        t @= Z @ H

        assert isinstance(t, Tensor)
        assert t.obs == [X, Y, Z, H]

    def test_operation_multiply_invalid(self):
        """Test that an exception is raised if an observable
        is multiplied by an operation"""
        X = PauliX(0)
        Y = CNOT(wires=[0, 1])
        Z = PauliZ(0)

        with pytest.raises(ValueError, match="Can only perform tensor products between observables"):
            X @ Y

        with pytest.raises(ValueError, match="Can only perform tensor products between observables"):
            T = X @ Z
            T @ Y

        with pytest.raises(ValueError, match="Can only perform tensor products between observables"):
            T = X @ Z
            Y @ T

    def test_eigvals(self):
        """Test that the correct eigenvalues are returned for the Tensor"""
        X = PauliX(0)
        Y = PauliY(2)
        t = Tensor(X, Y)
        assert np.array_equal(t.eigvals, np.kron([1, -1], [1, -1]))

        # test that the eigvals are now cached and not recalculated
        assert np.array_equal(t._eigvals_cache, t.eigvals)

    @pytest.mark.usefixtures("tear_down_hermitian")
    def test_eigvals_hermitian(self, tol):
        """Test that the correct eigenvalues are returned for the Tensor containing an Hermitian observable"""
        X = PauliX(0)
        hamiltonian = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]])
        Herm = Hermitian(hamiltonian, wires=[1, 2])
        t = Tensor(X, Herm)
        d = np.kron(np.array([1., -1.]), np.array([-1.,  1.,  1.,  1.]))
        t = t.eigvals
        assert np.allclose(t, d, atol=tol, rtol=0)

    def test_eigvals_identity(self, tol):
        """Test that the correct eigenvalues are returned for the Tensor containing an Identity"""
        X = PauliX(0)
        Iden = Identity(1)
        t = Tensor(X, Iden)
        d = np.kron(np.array([1., -1.]), np.array([1.,  1.]))
        t = t.eigvals
        assert np.allclose(t, d, atol=tol, rtol=0)

    def test_eigvals_identity_and_hermitian(self, tol):
        """Test that the correct eigenvalues are returned for the Tensor containing
        multiple types of observables"""
        H = np.diag([1, 2, 3, 4])
        O = PauliX(0) @ Identity(2) @ Hermitian(H, wires=[4,5])
        res = O.eigvals
        expected = np.kron(np.array([1., -1.]), np.kron(np.array([1.,  1.]), np.arange(1, 5)))
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_diagonalizing_gates(self, tol):
        """Test that the correct diagonalizing gate set is returned for a Tensor of observables"""
        H = np.diag([1, 2, 3, 4])
        O = PauliX(0) @ Identity(2)  @ PauliY(1) @ Hermitian(H, [5, 6])

        res = O.diagonalizing_gates()

        # diagonalize the PauliX on wire 0 (H.X.H = Z)
        assert isinstance(res[0], Hadamard)
        assert res[0].wires == Wires([0])

        # diagonalize the PauliY on wire 1 (U.Y.U^\dagger = Z
        # where U = HSZ).
        assert isinstance(res[1], PauliZ)
        assert res[1].wires == Wires([1])
        assert isinstance(res[2], S)
        assert res[2].wires == Wires([1])
        assert isinstance(res[3], Hadamard)
        assert res[3].wires == Wires([1])

        # diagonalize the Hermitian observable on wires 5, 6
        assert isinstance(res[4], QubitUnitary)
        assert res[4].wires == Wires([5, 6])

        O = O @ Hadamard(4)
        res = O.diagonalizing_gates()

        # diagonalize the Hadamard observable on wire 4
        # (RY(-pi/4).H.RY(pi/4) = Z)
        assert isinstance(res[-1], RY)
        assert res[-1].wires == Wires([4])
        assert np.allclose(res[-1].parameters, -np.pi/4, atol=tol, rtol=0)

    def test_diagonalizing_gates_numerically_diagonalizes(self, tol):
        """Test that the diagonalizing gate set numerically
        diagonalizes the tensor observable"""

        # create a tensor observable acting on consecutive wires
        H = np.diag([1, 2, 3, 4])
        O = PauliX(0) @ PauliY(1) @ Hermitian(H, [2, 3])

        O_mat = O.matrix
        diag_gates = O.diagonalizing_gates()

        # group the diagonalizing gates based on what wires they act on
        U_list = []
        for _, g in itertools.groupby(diag_gates, lambda x: x.wires.tolist()):
            # extract the matrices of each diagonalizing gate
            mats = [i.matrix for i in g]

            # Need to revert the order in which the matrices are applied such that they adhere to the order
            # of matrix multiplication
            # E.g. for PauliY: [PauliZ(wires=self.wires), S(wires=self.wires), Hadamard(wires=self.wires)]
            # becomes Hadamard @ S @ PauliZ, where @ stands for matrix multiplication
            mats = mats[::-1]

            if len(mats) > 1:
                # multiply all unitaries together before appending
                mats = [multi_dot(mats)]

            # append diagonalizing unitary for specific wire to U_list
            U_list.append(mats[0])

        # since the test is assuming consecutive wires for each observable
        # in the tensor product, it is sufficient to Kronecker product
        # the entire list.
        U = functools.reduce(np.kron, U_list)


        res = U @ O_mat @ U.conj().T
        expected = np.diag(O.eigvals)

        # once diagonalized by U, the result should be a diagonal
        # matrix of the eigenvalues.
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_tensor_matrix(self, tol):
        """Test that the tensor product matrix method returns
        the correct result"""
        H = np.diag([1, 2, 3, 4])
        O = PauliX(0) @ PauliY(1) @ Hermitian(H, [2, 3])

        res = O.matrix
        expected = np.kron(PauliY._matrix(), H)
        expected = np.kron(PauliX._matrix(), expected)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multiplication_matrix(self, tol):
        """If using the ``@`` operator on two observables acting on the
        same wire, the tensor class should treat this as matrix multiplication."""
        O = PauliX(0) @ PauliX(0)

        res = O.matrix
        expected = PauliX._matrix() @ PauliX._matrix()

        assert np.allclose(res, expected, atol=tol, rtol=0)

    herm_matrix = np.array([
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]
                            ])

    tensor_obs = [
                    (
                    PauliZ(0) @ Identity(1) @ PauliZ(2),
                    [PauliZ(0), PauliZ(2)]
                    ),
                    (
                    Identity(0) @ PauliX(1) @ Identity(2) @ PauliZ(3) @  PauliZ(4) @ Identity(5),
                    [PauliX(1), PauliZ(3), PauliZ(4)]
                    ),

                    # List containing single observable is returned
                    (
                    PauliZ(0) @ Identity(1),
                    [PauliZ(0)]
                    ),
                    (
                    Identity(0) @ PauliX(1) @ Identity(2),
                    [PauliX(1)]
                    ),
                    (
                    Identity(0) @ Identity(1),
                    [Identity(0)]
                    ),
                    (
                    Identity(0) @ Identity(1) @ Hermitian(herm_matrix, wires=[2,3]),
                    [Hermitian(herm_matrix, wires=[2,3])]
                    )
                ]

    @pytest.mark.parametrize("tensor_observable, expected", tensor_obs)
    def test_non_identity_obs(self, tensor_observable, expected):
        """Tests that the non_identity_obs property returns a list that contains no Identity instances."""

        O = tensor_observable
        for idx, obs in enumerate(O.non_identity_obs):
            assert type(obs) == type(expected[idx])
            assert obs.wires == expected[idx].wires

    tensor_obs_pruning = [
                            (
                            PauliZ(0) @ Identity(1) @ PauliZ(2),
                            PauliZ(0) @ PauliZ(2)
                            ),
                            (
                            Identity(0) @ PauliX(1) @ Identity(2) @ PauliZ(3) @  PauliZ(4) @ Identity(5),
                            PauliX(1) @ PauliZ(3) @ PauliZ(4)
                            ),
                            # Single observable is returned
                            (
                            PauliZ(0) @ Identity(1),
                            PauliZ(0)
                            ),
                            (
                            Identity(0) @ PauliX(1) @ Identity(2),
                            PauliX(1)
                            ),
                            (
                            Identity(0) @ Identity(1),
                            Identity(0)
                            ),
                            (
                            Identity(0) @ Identity(1),
                            Identity(0)
                            ),
                            (
                            Identity(0) @ Identity(1) @ Hermitian(herm_matrix, wires=[2,3]),
                            Hermitian(herm_matrix, wires=[2,3])
                            )
                         ]

    @pytest.mark.parametrize("tensor_observable, expected", tensor_obs_pruning)
    @pytest.mark.parametrize("statistics", [beta_measure.expval, beta_measure.var, beta_measure.sample])
    def test_prune(self, tensor_observable, expected, statistics):
        """Tests that the prune method returns the expected Tensor or single non-Tensor Observable."""
        O = statistics(tensor_observable)
        O_expected = statistics(expected)

        O_pruned = O.prune()
        assert type(O_pruned) == type(expected)
        assert O_pruned.wires == expected.wires
        assert O_pruned.return_type == O_expected.return_type

    def test_append_annotating_object(self):
        """Test appending an object that writes annotations when queuing itself"""

        with qml._queuing.AnnotatedQueue() as q:
            A = PauliZ(0)
            B = PauliY(1)
            tensor_op = Tensor(A, B)

        assert q.queue == [A, B, tensor_op]
        assert q.get_info(A) == {"owner": tensor_op}
        assert q.get_info(B) == {"owner": tensor_op}
        assert q.get_info(tensor_op) == {"owns": (A, B)}
