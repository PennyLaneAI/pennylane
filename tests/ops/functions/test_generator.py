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
Unit tests for the qml.generator transform
"""
from functools import reduce
from operator import matmul
import pytest
from scipy import sparse

import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms.op_transforms import OperationTransformError

###################################################################
# Test operations
###################################################################
# The following are various custom operation classes that return
# generators using various approaches.


class CustomOp(qml.operation.Operation):
    """Base custom operation that defines the adjoint"""

    num_params = 1
    num_wires = 1

    def adjoint(self):
        op = self.__class__(*self.data, wires=self.wires)
        op.inverse = not self.inverse
        return op


class ObservableOp(CustomOp):
    """Returns the generator as a single observable"""

    coeff = -0.6
    obs = qml.PauliX

    def generator(self):
        return self.coeff * self.obs(self.wires)


class TensorOp(CustomOp):
    """Returns the generator as a tensor observable"""

    num_wires = 2
    coeff = 0.5
    obs = [qml.PauliX, qml.PauliY]

    def generator(self):
        return self.coeff * self.obs[0](self.wires[0]) @ self.obs[1](self.wires[1])


class HamiltonianOp(CustomOp):
    """Returns the generator as a Hamiltonian"""

    num_wires = 2
    coeff = [1.0, 0.5]
    obs = [[qml.PauliX, qml.Identity], [qml.PauliX, qml.PauliY]]

    def generator(self):
        obs = [reduce(matmul, [o(w) for o, w in zip(word, self.wires)]) for word in self.obs]
        return qml.Hamiltonian(self.coeff, obs)


class HermitianOp(CustomOp):
    """Returns the generator as a Hermitian observable"""

    H = np.array([[1.0, 2.0], [2.0, 3.0]])

    def generator(self):
        return qml.Hermitian(self.H, wires=self.wires[0])


class SparseOp(CustomOp):
    """Returns the generator as a SparseHamiltonian observable"""

    H = sparse.coo_matrix(np.array([[1.0, 2.0], [2.0, 3.0]]))

    def generator(self):
        return qml.SparseHamiltonian(self.H, wires=self.wires[0])


###################################################################
# Test functions
###################################################################


class TestValidation:
    def test_non_hermitian_generator(self):
        """Check that an error is raised if the generator
        returned is non-Hermitian"""

        class SomeOp(qml.operation.Operation):
            num_params = 1
            num_wires = 1

            def generator(self):
                return qml.RX(self.data[0], wires=self.wires[0])

        op = SomeOp(0.5, wires=0)

        with pytest.raises(qml.QuantumFunctionError, match="is not an observable"):
            qml.generator(op)

    def test_multi_param_op(self):
        """Test that an error is raised if the operator has more than one parameter"""

        class SomeOp(qml.operation.Operation):
            num_params = 2
            num_wires = 1

            def generator(self):
                return qml.RX(self.data[0], wires=self.wires[0])

        op = SomeOp(0.5, 0.1, wires=0)

        with pytest.raises(ValueError, match="is not written in terms of a single parameter"):
            qml.generator(op)

    def test_unknown_format(self):
        """Raise an exception is the format is unknown"""
        with pytest.raises(ValueError, match="format must be one of"):
            qml.generator(qml.RX, format=None)(0.5, wires=0)


class TestPrefactorReturn:
    """Tests for format="prefactor". This format attempts to isolate a prefactor
    (if possible) from the generator, which is useful if the generator is a Pauli word."""

    def test_observable(self):
        """Test a generator that returns a single observable is correct"""
        gen, prefactor = qml.generator(ObservableOp, format="prefactor")(0.5, wires=0)
        assert prefactor == -0.6
        assert gen.name == "PauliX"

    def test_tensor_observable(self):
        """Test a generator that returns a tensor observable is correct"""
        gen, prefactor = qml.generator(TensorOp, format="prefactor")(0.5, wires=[0, 1])
        assert prefactor == 0.5
        assert gen.name == ["PauliX", "PauliY"]

    def test_hamiltonian(self):
        """Test a generator that returns a Hamiltonian"""
        gen, prefactor = qml.generator(HamiltonianOp, format="prefactor")(0.5, wires=[0, 1])
        assert prefactor == 1.0
        assert gen.name == "Hamiltonian"

    def test_hermitian(self):
        """Test a generator that returns a Hermitian observable
        is correct"""
        gen, prefactor = qml.generator(HermitianOp, format="prefactor")(0.5, wires=0)
        assert prefactor == 1.0
        assert gen.name == "Hermitian"
        assert np.all(gen.parameters[0] == HermitianOp.H)

    def test_sparse_hamiltonian(self):
        """Test a generator that returns a SparseHamiltonian observable
        is correct"""
        gen, prefactor = qml.generator(SparseOp, format="prefactor")(0.5, wires=0)
        assert prefactor == 1.0
        assert gen.name == "SparseHamiltonian"
        assert np.all(gen.parameters[0].toarray() == SparseOp.H.toarray())

    def test_inverse(self):
        """Test an inverted generator is correct"""

        gen, prefactor = qml.generator(qml.adjoint(ObservableOp), format="prefactor")(0.5, wires=0)
        assert prefactor == 0.6
        assert gen.name == "PauliX"

        gen, prefactor = qml.generator(ObservableOp(0.5, wires=0).inv(), format="prefactor")
        assert prefactor == 0.6
        assert gen.name == "PauliX"


class TestObservableReturn:
    """Tests for format="observable". This format preserves the initial generator
    encoded in the operator."""

    def test_observable(self):
        """Test a generator that returns a single observable is correct"""
        gen = qml.generator(ObservableOp, format="observable")(0.5, wires=0)
        assert gen.name == "Hamiltonian"
        assert gen.compare(ObservableOp(0.5, wires=0).generator())

    def test_tensor_observable(self):
        """Test a generator that returns a tensor observable is correct"""
        gen = qml.generator(TensorOp, format="observable")(0.5, wires=[0, 1])
        assert gen.name == "Hamiltonian"
        assert gen.compare(TensorOp(0.5, wires=[0, 1]).generator())

    def test_hamiltonian(self):
        """Test a generator that returns a Hamiltonian"""
        gen = qml.generator(HamiltonianOp, format="observable")(0.5, wires=[0, 1])
        assert gen.name == "Hamiltonian"
        assert gen.compare(HamiltonianOp(0.5, wires=[0, 1]).generator())

    def test_hermitian(self):
        """Test a generator that returns a Hermitian observable
        is correct"""
        gen = qml.generator(HermitianOp, format="observable")(0.5, wires=0)
        assert gen.name == "Hermitian"
        assert np.all(gen.parameters[0] == HermitianOp.H)

    def test_sparse_hamiltonian(self):
        """Test a generator that returns a SparseHamiltonian observable
        is correct"""
        gen = qml.generator(SparseOp, format="observable")(0.5, wires=0)
        assert gen.name == "SparseHamiltonian"
        assert np.all(gen.parameters[0].toarray() == SparseOp.H.toarray())

    def test_hermitian_inverse(self):
        """Test a Hermitian inverted generator is correct"""
        gen = qml.generator(qml.adjoint(HermitianOp), format="observable")(0.5, wires=0)
        assert gen.name == "Hermitian"
        assert np.all(gen.parameters[0] == -HermitianOp.H)

        gen = qml.generator(HermitianOp(0.5, wires=0).inv(), format="observable")
        assert gen.name == "Hermitian"
        assert np.all(gen.parameters[0] == -HermitianOp.H)

    def test_sparse_hamiltonian_inverse(self):
        """Test a SparseHamiltonian inverted generator is correct"""
        gen = qml.generator(qml.adjoint(SparseOp), format="observable")(0.5, wires=0)
        assert gen.name == "SparseHamiltonian"
        assert np.all(gen.parameters[0].toarray() == -SparseOp.H.toarray())

        gen = qml.generator(SparseOp(0.5, wires=0).inv(), format="observable")
        assert gen.name == "SparseHamiltonian"
        assert np.all(gen.parameters[0].toarray() == -SparseOp.H.toarray())

    def test_hamiltonian_inverse(self):
        """Test a Hamiltonian inverted generator is correct"""
        gen = qml.generator(qml.adjoint(HamiltonianOp), format="observable")(0.5, wires=[0, 1])
        assert gen.name == "Hamiltonian"

        res = qml.matrix(gen)
        expected = -qml.matrix(HamiltonianOp(0.23, wires=[0, 1]).generator())
        assert np.allclose(res, expected)

        gen = qml.generator(HamiltonianOp(0.5, wires=[0, 1]).inv(), format="observable")
        assert gen.name == "Hamiltonian"

        res = qml.matrix(gen)
        assert np.allclose(res, expected)


class TestHamiltonianReturn:
    """Tests for format="hamiltonian". This format always returns the generator
    as a Hamiltonian."""

    def test_observable_no_coeff(self):
        """Test a generator that returns an observable with no coefficient is correct"""
        gen = qml.generator(qml.PhaseShift, format="hamiltonian")(0.5, wires=0)
        assert gen.name == "Hamiltonian"
        assert gen.compare(1.0 * qml.PhaseShift(0.5, wires=0).generator())

    def test_observable(self):
        """Test a generator that returns a single observable is correct"""
        gen = qml.generator(ObservableOp, format="hamiltonian")(0.5, wires=0)
        assert gen.name == "Hamiltonian"
        assert gen.compare(ObservableOp(0.5, wires=0).generator())

    def test_tensor_observable(self):
        """Test a generator that returns a tensor observable is correct"""
        gen = qml.generator(TensorOp, format="hamiltonian")(0.5, wires=[0, 1])
        assert gen.name == "Hamiltonian"
        assert gen.compare(TensorOp(0.5, wires=[0, 1]).generator())

    def test_hamiltonian(self):
        """Test a generator that returns a Hamiltonian"""
        gen = qml.generator(HamiltonianOp, format="hamiltonian")(0.5, wires=[0, 1])
        assert gen.name == "Hamiltonian"
        assert gen.compare(HamiltonianOp(0.5, wires=[0, 1]).generator())

    def test_hermitian(self):
        """Test a generator that returns a Hermitian observable
        is correct"""
        gen = qml.generator(HermitianOp, format="hamiltonian")(0.5, wires=0)
        assert gen.name == "Hamiltonian"

        expected = qml.Hamiltonian(*qml.utils.decompose_hamiltonian(HermitianOp.H))
        assert gen.compare(expected)

    def test_sparse_hamiltonian(self):
        """Test a generator that returns a SparseHamiltonian observable
        is correct"""
        gen = qml.generator(SparseOp, format="hamiltonian")(0.5, wires=0)
        assert gen.name == "Hamiltonian"

        expected = qml.Hamiltonian(*qml.utils.decompose_hamiltonian(SparseOp.H.toarray()))
        assert gen.compare(expected)

    def test_hermitian_inverse(self):
        """Test a Hermitian inverted generator is correct"""
        gen = qml.generator(qml.adjoint(HermitianOp), format="hamiltonian")(0.5, wires=0)
        assert gen.name == "Hamiltonian"

        expected = qml.Hamiltonian(*qml.utils.decompose_hamiltonian(HermitianOp.H))
        assert gen.compare(-1.0 * expected)

        gen = qml.generator(HermitianOp(0.5, wires=0).inv(), format="hamiltonian")
        assert gen.name == "Hamiltonian"
        assert gen.compare(-1.0 * expected)

    def test_sparse_hamiltonian_inverse(self):
        """Test a SparseHamiltonian inverted generator is correct"""
        gen = qml.generator(qml.adjoint(SparseOp), format="hamiltonian")(0.5, wires=0)
        assert gen.name == "Hamiltonian"

        expected = qml.Hamiltonian(*qml.utils.decompose_hamiltonian(SparseOp.H.toarray()))
        assert gen.compare(-1.0 * expected)

        gen = qml.generator(SparseOp(0.5, wires=0).inv(), format="hamiltonian")
        assert gen.name == "Hamiltonian"
        assert gen.compare(-1.0 * expected)

    def test_hamiltonian_inverse(self):
        """Test a Hamiltonian inverted generator is correct"""
        gen = qml.generator(qml.adjoint(HamiltonianOp), format="hamiltonian")(0.5, wires=[0, 1])
        assert gen.name == "Hamiltonian"

        res = qml.matrix(gen)
        expected = -qml.matrix(HamiltonianOp(0.23, wires=[0, 1]).generator())
        assert np.allclose(res, expected)

        gen = qml.generator(HamiltonianOp(0.5, wires=[0, 1]).inv(), format="hamiltonian")
        assert gen.name == "Hamiltonian"

        res = qml.matrix(gen)
        assert np.allclose(res, expected)
