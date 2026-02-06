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
Unit tests for the qp.generator transform
"""
# pylint: disable=too-few-public-methods
from functools import reduce
from operator import matmul

import pytest
from scipy import sparse

import pennylane as qp
from pennylane import numpy as np
from pennylane.exceptions import QuantumFunctionError
from pennylane.ops import Prod, SProd, Sum

###################################################################
# Test operations
###################################################################
# The following are various custom operation classes that return
# generators using various approaches.


class CustomOp(qp.operation.Operation):
    """Base custom operation."""

    num_params = 1
    num_wires = 1


class ObservableOp(CustomOp):
    """Returns the generator as a single observable"""

    coeff = -0.6
    obs = qp.PauliX

    def generator(self):
        return self.coeff * self.obs(self.wires)


class TensorOp(CustomOp):
    """Returns the generator as a tensor observable"""

    num_wires = 2
    coeff = 0.5
    obs = [qp.PauliX, qp.PauliY]

    def generator(self):
        return self.coeff * self.obs[0](self.wires[0]) @ self.obs[1](self.wires[1])


class HamiltonianOp(CustomOp):
    """Returns the generator as a Hamiltonian"""

    num_wires = 2
    coeff = [1.0, 0.5]
    obs = [[qp.PauliX, qp.Identity], [qp.PauliX, qp.PauliY]]

    def generator(self):
        obs = [reduce(matmul, [o(w) for o, w in zip(word, self.wires)]) for word in self.obs]
        return qp.Hamiltonian(self.coeff, obs)


class HamiltonianOpSameCoeff(CustomOp):
    """Returns the generator as a Hamiltonian"""

    num_wires = 2
    coeff = [0.5, 0.5]
    obs = [[qp.PauliX, qp.Identity], [qp.PauliX, qp.PauliY]]

    def generator(self):
        obs = [reduce(matmul, [o(w) for o, w in zip(word, self.wires)]) for word in self.obs]
        return qp.Hamiltonian(self.coeff, obs)


class HamiltonianOpSameAbsCoeff(CustomOp):
    """Returns the generator as a Hamiltonian"""

    num_wires = 2
    coeff = [0.5, -0.5]
    obs = [[qp.PauliX, qp.Identity], [qp.PauliX, qp.PauliY]]

    def generator(self):
        obs = [reduce(matmul, [o(w) for o, w in zip(word, self.wires)]) for word in self.obs]
        return qp.Hamiltonian(self.coeff, obs)


class SumOp(CustomOp):
    """Returns the generator as a Sum"""

    num_wires = 2

    def generator(self):
        return qp.sum(
            qp.PauliX(self.wires[0]) @ qp.Identity(self.wires[1]),
            0.5 * qp.prod(qp.PauliX(self.wires[0]), qp.PauliY(self.wires[1])),
        )


class SumOpSameCoeff(CustomOp):
    """Returns the generator as a Sum"""

    num_wires = 2

    def generator(self):
        return qp.sum(
            0.5 * qp.prod(qp.PauliX(self.wires[0]), qp.Identity(self.wires[1])),
            0.5 * qp.prod(qp.PauliX(self.wires[0]), qp.PauliY(self.wires[1])),
        )


class SumOpSameAbsCoeff(CustomOp):
    """Returns the generator as a Sum"""

    num_wires = 2

    def generator(self):
        return qp.sum(
            0.5 * qp.prod(qp.PauliX(self.wires[0]), qp.Identity(self.wires[1])),
            -0.5 * qp.prod(qp.PauliX(self.wires[0]), qp.PauliY(self.wires[1])),
        )


class HermitianOp(CustomOp):
    """Returns the generator as a Hermitian observable"""

    H = np.array([[1.0, 2.0], [2.0, 3.0]])

    def generator(self):
        return qp.Hermitian(self.H, wires=self.wires[0])


class SparseOp(CustomOp):
    """Returns the generator as a SparseHamiltonian observable"""

    H = sparse.csr_matrix(np.array([[1.0, 2.0], [2.0, 3.0]]))

    def generator(self):
        return qp.SparseHamiltonian(self.H, wires=self.wires[0])


###################################################################
# Test functions
###################################################################


class TestValidation:
    def test_non_hermitian_generator(self):
        """Check that an error is raised if the generator
        returned is non-Hermitian"""

        class SomeOp(qp.operation.Operation):
            num_params = 1
            num_wires = 1

            def generator(self):
                return qp.RX(self.data[0], wires=self.wires[0])

        op = SomeOp(0.5, wires=0)

        with pytest.raises(QuantumFunctionError, match="is not hermitian"):
            qp.generator(op)

    def test_multi_param_op(self):
        """Test that an error is raised if the operator has more than one parameter"""

        class SomeOp(qp.operation.Operation):
            num_params = 2
            num_wires = 1

            def generator(self):
                return qp.RX(self.data[0], wires=self.wires[0])

        op = SomeOp(0.5, 0.1, wires=0)

        with pytest.raises(ValueError, match="is not written in terms of a single parameter"):
            qp.generator(op)

    def test_unknown_format(self):
        """Raise an exception is the format is unknown"""
        with pytest.raises(ValueError, match="format must be one of"):
            qp.generator(qp.RX, format=None)(0.5, wires=0)


class TestBackwardsCompatibility:
    """Test that operators that provide the old style generator property
    continue to work with a deprecation warning."""

    def test_return_class(self):
        """Test that an old-style Operator that has a generator property,
        and that returns a list containing (a) operator class and (b) prefactor,
        continues to work but also raises a deprecation warning."""

        class DeprecatedClassOp(CustomOp):
            generator = [qp.PauliX, -0.6]

        op = DeprecatedClassOp(0.5, wires="a")

        with pytest.warns(UserWarning, match=r"The Operator\.generator property is deprecated"):
            gen, prefactor = qp.generator(op)

        assert isinstance(gen, qp.operation.Operator)
        assert prefactor == -0.6
        assert gen.name == "PauliX"
        assert gen.wires.tolist() == ["a"]

    def test_return_array(self):
        """Test that an old-style Operator that has a generator property,
        and that returns a list containing (a) array and (b) prefactor,
        continues to work but also raises a deprecation warning."""

        class DeprecatedClassOp(CustomOp):
            generator = [np.diag([0, 1]), -0.6]

        op = DeprecatedClassOp(0.5, wires="a")

        with pytest.warns(UserWarning, match=r"The Operator\.generator property is deprecated"):
            gen, prefactor = qp.generator(op)

        assert isinstance(gen, qp.operation.Operator)
        assert prefactor == -0.6
        assert gen.name == "Hermitian"
        assert gen.wires.tolist() == ["a"]

    def test_generator_property_old_default(self):
        """Test that if the old-style generator property is the default,
        a GeneratorUndefinedError is raised and a warning is raised about the old syntax."""

        class DeprecatedClassOp(CustomOp):
            generator = [None, 1]

        op = DeprecatedClassOp(0.5, wires="a")
        with pytest.warns(UserWarning, match=r"The Operator\.generator property is deprecated"):
            with pytest.raises(qp.operation.GeneratorUndefinedError):
                qp.generator(op)


class TestPrefactorReturn:
    """Tests for format="prefactor". This format attempts to isolate a prefactor
    (if possible) from the generator, which is useful if the generator is a Pauli word."""

    def test_observable(self):
        """Test a generator that returns a single observable is correct"""
        gen, prefactor = qp.generator(ObservableOp, format="prefactor")(0.5, wires=0)
        assert prefactor == -0.6
        assert gen.name == "PauliX"

    def test_tensor_observable(self):
        """Test a generator that returns a tensor observable is correct"""
        gen, prefactor = qp.generator(TensorOp, format="prefactor")(0.5, wires=[0, 1])
        assert prefactor == 0.5
        assert gen.name == "Prod"

    def test_hamiltonian(self):
        """Test a generator that returns a Hamiltonian"""
        gen, prefactor = qp.generator(HamiltonianOp, format="prefactor")(0.5, wires=[0, 1])
        assert prefactor == 1.0
        assert gen.name == "Sum"

    def test_hamiltonian_with_same_term(self):
        """Test a generator that returns a Hamiltonian with multiple terms, all containing the same
        coefficient."""
        gen, prefactor = qp.generator(HamiltonianOpSameCoeff, format="prefactor")(
            0.5, wires=[0, 1]
        )
        assert prefactor == 0.5
        assert isinstance(gen, Sum)
        for op in gen:
            assert isinstance(op, Prod)

    def test_hamiltonian_with_same_abs_term(self):
        """Test a generator that returns a Hamiltonian with multiple terms, all containing the same
        absolute coefficient."""
        gen, prefactor = qp.generator(HamiltonianOpSameAbsCoeff, format="prefactor")(
            0.5, wires=[0, 1]
        )
        assert prefactor == 0.5
        assert isinstance(gen, Sum)
        for op in gen:
            if isinstance(op, SProd):
                assert op.scalar == -1
            else:
                assert isinstance(op, Prod)

    def test_sum(self):
        """Test a generator that returns a Sum"""
        gen, prefactor = qp.generator(SumOp, format="prefactor")(0.5, wires=[0, 1])
        assert prefactor == 1.0
        assert isinstance(gen, Sum)

    def test_sum_with_same_term(self):
        """Test a generator that returns a Sum with multiple terms, all containing the same
        coefficient."""
        gen, prefactor = qp.generator(SumOpSameCoeff, format="prefactor")(0.5, wires=[0, 1])
        assert prefactor == 0.5
        assert isinstance(gen, Sum)
        for op in gen:
            assert isinstance(op, Prod)

    def test_sum_with_same_abs_term(self):
        """Test a generator that returns a Sum with multiple terms, all containing the same
        absolute coefficient."""
        gen, prefactor = qp.generator(SumOpSameAbsCoeff, format="prefactor")(0.5, wires=[0, 1])
        assert prefactor == 0.5
        assert isinstance(gen, Sum)
        for op in gen:
            if isinstance(op, SProd):
                assert op.scalar == -1
            else:
                assert isinstance(op, Prod)

    def test_hermitian(self):
        """Test a generator that returns a Hermitian observable
        is correct"""
        gen, prefactor = qp.generator(HermitianOp, format="prefactor")(0.5, wires=0)
        assert prefactor == 1.0
        assert gen.name == "Hermitian"
        assert np.all(gen.parameters[0] == HermitianOp.H)

    def test_sparse_hamiltonian(self):
        """Test a generator that returns a SparseHamiltonian observable
        is correct"""
        gen, prefactor = qp.generator(SparseOp, format="prefactor")(0.5, wires=0)
        assert prefactor == 1.0
        assert gen.name == "SparseHamiltonian"
        assert np.all(gen.parameters[0].toarray() == SparseOp.H.toarray())


class TestObservableReturn:
    """Tests for format="observable". This format preserves the initial generator
    encoded in the operator."""

    def test_observable(self):
        """Test a generator that returns a single observable is correct"""
        gen = qp.generator(ObservableOp, format="observable")(0.5, wires=0)
        assert gen.name == "SProd"
        qp.assert_equal(gen, ObservableOp(0.5, wires=0).generator())

    def test_tensor_observable(self):
        """Test a generator that returns a tensor observable is correct"""
        gen = qp.generator(TensorOp, format="observable")(0.5, wires=[0, 1])
        assert gen.name == "Prod"
        qp.assert_equal(gen, TensorOp(0.5, wires=[0, 1]).generator())

    def test_hamiltonian(self):
        """Test a generator that returns a Hamiltonian"""
        gen = qp.generator(HamiltonianOp, format="observable")(0.5, wires=[0, 1])
        assert isinstance(gen, type(qp.Hamiltonian([], [])))
        assert gen == HamiltonianOp(0.5, wires=[0, 1]).generator()

    def test_hermitian(self):
        """Test a generator that returns a Hermitian observable
        is correct"""
        gen = qp.generator(HermitianOp, format="observable")(0.5, wires=0)
        assert gen.name == "Hermitian"
        assert np.all(gen.parameters[0] == HermitianOp.H)

    def test_sparse_hamiltonian(self):
        """Test a generator that returns a SparseHamiltonian observable
        is correct"""
        gen = qp.generator(SparseOp, format="observable")(0.5, wires=0)
        assert gen.name == "SparseHamiltonian"
        assert np.all(gen.parameters[0].toarray() == SparseOp.H.toarray())


class TestHamiltonianReturn:
    """Tests for format="hamiltonian". This format always returns the generator
    as a qp.ops.LinearCombination."""

    def test_observable_no_coeff(self):
        """Test a generator that returns an observable with no coefficient is correct"""
        gen = qp.generator(qp.PhaseShift, format="hamiltonian")(0.5, wires=0)
        assert isinstance(gen, qp.Hamiltonian)
        assert gen == qp.Hamiltonian([1.0], [qp.PhaseShift(0.5, wires=0).generator()])

    def test_observable(self):
        """Test a generator that returns a single observable is correct"""
        gen = qp.generator(ObservableOp, format="hamiltonian")(0.5, wires=0)
        assert isinstance(gen, qp.Hamiltonian)
        assert gen == qp.Hamiltonian(*ObservableOp(0.5, wires=0).generator().terms())

    def test_tensor_observable(self):
        """Test a generator that returns a tensor observable is correct"""
        gen = qp.generator(TensorOp, format="hamiltonian")(0.5, wires=[0, 1])
        assert isinstance(gen, qp.Hamiltonian)
        assert gen == qp.Hamiltonian(*TensorOp(0.5, wires=[0, 1]).generator().terms())

    def test_hamiltonian(self):
        """Test a generator that returns a Hamiltonian"""
        gen = qp.generator(HamiltonianOp, format="hamiltonian")(0.5, wires=[0, 1])
        assert isinstance(gen, qp.Hamiltonian)
        assert gen == HamiltonianOp(0.5, wires=[0, 1]).generator()

    def test_hermitian(self):
        """Test a generator that returns a Hermitian observable
        is correct"""
        gen = qp.generator(HermitianOp, format="hamiltonian")(0.5, wires=0)
        assert isinstance(gen, qp.Hamiltonian)

        expected = qp.pauli_decompose(HermitianOp.H, hide_identity=True)
        assert gen == expected

    def test_sparse_hamiltonian(self):
        """Test a generator that returns a SparseHamiltonian observable
        is correct"""
        gen = qp.generator(SparseOp, format="hamiltonian")(0.5, wires=0)
        assert isinstance(gen, qp.Hamiltonian)

        expected = qp.pauli_decompose(SparseOp.H.toarray(), hide_identity=True)
        assert gen == expected

    def test_sum(self):
        """Test a generator that returns a Sum is correct"""
        gen = qp.generator(SumOp, format="hamiltonian")(0.5, wires=[0, 1])
        assert isinstance(gen, qp.Hamiltonian)

        expected = qp.Hamiltonian(
            [1.0, 0.5], [qp.PauliX(0) @ qp.Identity(1), qp.PauliX(0) @ qp.PauliY(1)]
        )

        assert gen == expected


class TestArithmeticReturn:
    """Tests for format="arithmetic". This format always returns the generator as an Arithmetic Operator."""

    def test_observable_no_coeff(self):
        """Test a generator that returns an observable with no coefficient is correct"""
        gen = qp.generator(qp.PhaseShift, format="arithmetic")(0.5, wires=0)
        qp.assert_equal(gen, qp.Projector(np.array([1]), wires=0))

    def test_observable(self):
        """Test a generator that returns a single observable is correct"""
        gen = qp.generator(ObservableOp, format="arithmetic")(0.5, wires=0)
        qp.assert_equal(gen, qp.s_prod(-0.6, qp.PauliX(0)))

    def test_tensor_observable(self):
        """Test a generator that returns a tensor observable is correct"""
        gen = qp.generator(TensorOp, format="arithmetic")(0.5, wires=[0, 1])
        result = qp.s_prod(0.5, qp.PauliX(0) @ qp.PauliY(1))

        assert not isinstance(gen, qp.Hamiltonian)
        assert np.allclose(
            qp.matrix(gen, wire_order=[0, 1]),
            qp.matrix(result, wire_order=[0, 1]),
        )

    def test_hamiltonian(self):
        """Test a generator that returns a Hamiltonian"""
        gen = qp.generator(HamiltonianOp, format="arithmetic")(0.5, wires=[0, 1])
        result = qp.sum(
            qp.PauliX(0) @ qp.Identity(1),
            qp.s_prod(0.5, qp.PauliX(0) @ qp.PauliY(1)),
        )
        assert not isinstance(gen, qp.Hamiltonian)
        assert np.allclose(
            qp.matrix(gen, wire_order=[0, 1]),
            qp.matrix(result, wire_order=[0, 1]),
        )

    def test_hermitian(self):
        """Test a generator that returns a Hermitian observable
        is correct"""
        gen = qp.generator(HermitianOp, format="arithmetic")(0.5, wires=0)
        expected = qp.pauli_decompose(HermitianOp.H, hide_identity=True, pauli=True).operation()

        assert not isinstance(gen, qp.Hamiltonian)
        assert np.allclose(
            qp.matrix(gen),
            qp.matrix(expected),
        )

    def test_sparse_hamiltonian(self):
        """Test a generator that returns a SparseHamiltonian observable
        is correct"""
        gen = qp.generator(SparseOp, format="arithmetic")(0.5, wires=0)
        expected = qp.pauli_decompose(
            SparseOp.H.toarray(), hide_identity=True, pauli=True
        ).operation()

        assert not isinstance(gen, qp.Hamiltonian)
        assert np.allclose(
            qp.matrix(gen),
            qp.matrix(expected),
        )
