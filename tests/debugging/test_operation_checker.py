import pytest
from scipy.stats import unitary_group
import scipy.linalg as la
import scipy as sp

import pennylane as qml
from pennylane import numpy as np

from pennylane.debugging.operation_checker import (
    equal_up_to_phase,
    is_diagonal,
    OperationChecker,
    CheckerError,
)


matrices = [
    np.diag(np.arange(4, dtype=float)),
    np.arange(25).reshape((5, 5)),
    np.ones((8, 8)) * (0.21 + 0.4j),
    np.array([[0.2, -0.5], [3.1, -0.99]]),
]
matrix_phases = [la.expm(1j * mat) for mat in matrices]
scalar_phases = [-1.0, np.exp(1j * 0.3), 1j, -1j]


class TestEqualUpToPhase:
    """Test that matrices are correctly identified as equal up to a global phase
    or not equal up to a global phase."""

    @pytest.mark.parametrize("mat", matrices)
    def test_same_matrix(self, mat):
        """Test that two copies of the same matrix are reported as equal up to a phase."""
        assert equal_up_to_phase(mat, mat, atol=1e-14)

    @pytest.mark.parametrize("mat, phase", zip(matrices, scalar_phases))
    def test_global_phase(self, mat, phase):
        """Test that a matrix with and without a global phase
        are reported as equal up to a phase."""
        assert equal_up_to_phase(mat, phase * mat, atol=1e-14)

    @pytest.mark.parametrize("mat, phase", zip(matrices, matrix_phases))
    def test_matrix_phase(self, mat, phase):
        """Test that a matrix-valued 'phase' is not reported as a global phase."""
        assert not equal_up_to_phase(mat, phase * mat, atol=1e-5)
        assert not equal_up_to_phase(mat, phase @ mat, atol=1e-5)
        assert not equal_up_to_phase(mat, mat @ phase, atol=1e-5)

    @pytest.mark.parametrize("mat, factor", zip(matrices, matrices))
    def test_matrix_prefactor(self, mat, factor):
        """Test that a matrix-valued prefactor is not reported as a global phase."""
        assert not equal_up_to_phase(mat, factor * mat, atol=1e-5)
        assert not equal_up_to_phase(mat, factor @ mat, atol=1e-5)
        assert not equal_up_to_phase(mat, mat @ factor, atol=1e-5)

    @pytest.mark.parametrize("mat, factor", zip(matrices, [0.3, 2, -0.9j]))
    def test_scalar_prefactor(self, mat, factor):
        """Test that a scalar prefactor is not reported as a global phase."""
        assert not equal_up_to_phase(mat, factor * mat, atol=1e-5)


class TestIsDiagonal:
    """Test whether diagonal and non-diagonal matrices are correctly detected."""

    @pytest.mark.parametrize(
        "mat", [np.diag(np.arange(9)), np.eye(6), np.zeros((5, 5), dtype=complex)]
    )
    def test_diagonal(self, mat):
        """Test that diagonal matrices are reported to be diagonal."""
        assert is_diagonal(mat, atol=1e-14)

    @pytest.mark.parametrize("mat", [np.arange(9).reshape((3, 3)), np.ones((6, 6))])
    def test_not_diagonal(self, mat):
        """Test that non-diagonal matrices are reported to be non-diagonal."""
        assert not is_diagonal(mat, 1e-5)


unitaries = [
    (qml.RZ(0.3, wires=0).get_matrix(),),
    (unitary_group.rvs(8),),
]
h = np.random.random((8, 8)) + 1j * np.random.random((8, 8))
hermitians = [
    (qml.PauliY(0).get_matrix(),),
    (h + h.conj().T,),
]

qmlops = {getattr(qml, op) for op in qml.ops.__all__}
qmlops_special_input = {
    qml.PauliRot: ([(0.4, "XZ"), (-1.9, "Y")], [[0, 2], [5]]),
    qml.DiagonalQubitUnitary: (
        [(np.array([-1, 1]),), (np.exp([1j, -0.4j, 0.2j, -1.8j]),)],
        [[4], [3, 1]],
    ),
    qml.QubitStateVector: (
        [(np.array([0.0, -1.0]),), (np.ones(8) / np.sqrt(8),)],
        [[2], [0, 4, 1]],
    ),
    qml.BasisState: ([([0, 0],), ([1],), ([0, 1, 1],)], [[2, 1], [0], [3, 1, 0]]),
    qml.Projector: ([([0, 0],), ([1],), ([0, 1, 1],)], [[2, 1], [0], [3, 1, 0]]),
    qml.QubitUnitary: (unitaries, [[2], [1, 9, 2]]),
    qml.PauliError: ([("X", 0.2), ("YXZ", 0.8)], [[2], [0, 3, 4]]),
}
qmlops_skip = {
    qml.Hamiltonian,
    qml.SparseHamiltonian,
    qml.MultiControlledX,
    qml.QubitChannel,
    qml.Snapshot,  # No wires argument
    qml.ControlledQubitUnitary,  # control_wires are a parameter but not included in num_params
    qml.Hermitian,  #: (hermitians, [[2], [1, 9, 2]]), # compute_diagonalizing_gates wrong
}
qmlops = qmlops.difference(set(qmlops_special_input))
qmlops = qmlops.difference(qmlops_skip)


class TestQMLOperations:
    """Test the operation checker on native PennyLane operations."""

    Checker = OperationChecker(verbosity="comment")

    @pytest.mark.parametrize("op", qmlops)
    def test_standard_qml_ops(self, op):
        """Test that the operation checker reports "pass" on all PennyLane operations
        that have standard input format. Exceptions are
        - qml.MultiRZ, which makes use of hyperparameters (This is something the
          OperationChecker comments on, setting the result status to "comment")
        - all channels, for which differentiability can not be checked yet.
        """

        if op == qml.MultiRZ:
            expected_res = "comment"
            expected_str = "but is using additional (hyper)parameters"
        elif issubclass(op, qml.operation.Channel):
            expected_res = "hint"
            expected_str = "Channels cannot be checked for the correct derivative yet"
        else:
            expected_res = "pass"
            expected_str = ""

        result, output = self.Checker(op, seed=42)
        assert result == expected_res
        if expected_res == "pass":
            assert output == ""
        else:
            assert expected_str in output

    @pytest.mark.parametrize("op", qmlops_special_input)
    def test_special_input_qml_ops(self, op):
        """Test that the operation checker reports "pass" on all PennyLane operations
        that have special input formats. Exceptions are
        - all channels, for which differentiability can not be checked yet.
        """
        all_parameters, all_wires = qmlops_special_input[op]
        for parameters, wires in zip(all_parameters, all_wires):
            result, output = self.Checker(op, parameters=parameters, wires=wires, seed=42)
            if issubclass(op, qml.operation.Channel):
                # Hint for differentiation of channels
                assert "Channels cannot be checked for the correct" in output
                assert result == "hint"
            else:
                assert output == ""
                assert result == "pass"


class OpWithNumWires(qml.operation.Operation):
    """Operation that only defines num_wires, should pass."""

    num_wires = 1


class OpWithNumParams(qml.operation.Operation):
    """Operation that only defines num_wires and num_params, should pass."""

    num_wires = 1
    num_params = 2


class OpWithAllReps(qml.operation.Operation):
    """Operation that defines all applicable representations, should pass."""

    num_wires = 2
    num_params = 1
    grad_method = None

    @staticmethod
    def compute_eigvals(theta):
        return qml.math.ones(4)

    @staticmethod
    def compute_matrix(theta):
        return qml.math.eye(4)

    @staticmethod
    def compute_sparse_matrix(theta):
        return sp.sparse.coo_matrix(qml.math.eye(4))

    @staticmethod
    def compute_terms(theta):
        return [1.0], [qml.Identity(0) @ qml.Identity(1)]

    @staticmethod
    def compute_decomposition(theta, wires):
        return [qml.Identity(wires[0]), qml.Identity(wires[1])]

    @staticmethod
    def compute_diagonalizing_gates(theta, wires):
        return []

    def generator(self):
        return qml.Hermitian(qml.math.zeros((4, 4)), wires=[0, 1])


passing_ops = [OpWithNumWires, OpWithNumParams, OpWithAllReps]


class OpWithoutNumParams(qml.operation.Operation):
    """This operation does not fix ``num_params`` although it should.
    It should receive a "hint" as result."""

    num_wires = 2

    def __init__(self, x, wires):
        super().__init__(x, wires=wires)


ops_with_hints = [
    (
        OpWithoutNumParams,
        "Instantiating OpWithoutNumParams only succeeded when using 1 parameter(s).",
    ),
]


class OpWrongMatrix(OpWithAllReps):
    """Modified OpWithAllReps with wrong hardcoded matrix."""

    @staticmethod
    def compute_matrix(theta):
        return qml.math.ones((4, 4))


class OpWrongSparseMatrix(OpWithAllReps):
    """Modified OpWithAllReps with wrong sparse matrix."""

    @staticmethod
    def compute_sparse_matrix(theta):
        return sp.sparse.coo_matrix(qml.math.ones((4, 4)))


class OpWrongGenerator(OpWithAllReps):
    """Modified OpWithAllReps with wrong generator."""

    def generator(self):
        return qml.Hermitian(qml.math.ones((4, 4)), wires=[0, 1])


class OpWrongTerms(OpWithAllReps):
    """Modified OpWithAllReps with wrong terms decomposition."""

    @staticmethod
    def compute_terms(theta):
        return [
            1.0,
        ], [qml.PauliZ(0) @ qml.PauliX(1)]


class OpWrongDecomposition(OpWithAllReps):
    """Modified OpWithAllReps with wrong gate decomposition."""

    @staticmethod
    def compute_decomposition(theta, wires):
        return [qml.PauliX(wires[0]), qml.Identity(wires[1])]


class OpWrongEigvals(OpWithAllReps):
    """Modified OpWithAllReps with wrong hardcoded eigenvalues."""

    @staticmethod
    def compute_eigvals(theta):
        return -1 * qml.math.ones(4)


class OpWrongRotAngles(qml.operation.Operation):
    """Operation that only defines a hardcoded matrix and mismatching
    single-qubit rotation angles."""

    num_wires = 1
    num_params = 0

    @staticmethod
    def compute_matrix():
        return qml.math.eye(2)

    def single_qubit_rot_angles(self):
        return [0.1, 0.4, -0.3]


class OpWrongDiagGates(qml.operation.Operation):
    """Operation that only defines a hardcoded matrix and mismatching
    diagonalizing gates."""

    num_wires = 2
    num_params = 1

    @staticmethod
    def compute_matrix(theta):
        return qml.math.ones((4, 4))

    @staticmethod
    def compute_diagonalizing_gates(theta, wires):
        return [qml.PauliX(0), qml.PauliZ(1)]


error_ops = [
    (OpWrongMatrix, "Matrices do not coincide for OpWrongMatrix"),
    (OpWrongSparseMatrix, "Matrices do not coincide for OpWrongSparseMatrix"),
    (OpWrongGenerator, "Matrices do not coincide for OpWrongGenerator"),
    (OpWrongTerms, "Matrices do not coincide for OpWrongTerms"),
    (OpWrongDecomposition, "Matrices do not coincide for OpWrongDecomposition"),
    (OpWrongRotAngles, "Matrices do not coincide for OpWrongRotAngles"),
    (
        OpWrongEigvals,
        "The eigenvalues of the matrix and the stored eigvals for OpWrongEigvals",
        "The diagonalizing gates diagonalize the matrix but produce wrong eigenvalues",
    ),
    (
        OpWrongDiagGates,
        "The diagonalizing gates do not diagonalize the matrix for OpWrongDiagGates",
    ),
]


class OpWithoutNumWires(qml.operation.Operation):
    """Failing dummy operation that does not define any properties.
    This will produce a fatal error because no num_wires are defined."""


class OpWrongMatrixShape(OpWithAllReps):
    """Modified OpWithAllReps with wrongly shaped hardcoded matrix. This will produce
    a fatal error because many parts of the check rely on the matrix being well-defined."""

    @staticmethod
    def compute_matrix(theta):
        return qml.math.ones((2, 4))


fatal_error_ops = [
    (OpWithoutNumWires, "OpWithoutNumWires does not define the number of wires"),
    (OpWrongMatrixShape, "The operation OpWrongMatrixShape defines a non-square matrix"),
]


class TestCustomOperations:
    """Test that all the custom operations defined above are reported with
    the correct result status and output."""

    Checker = OperationChecker(verbosity="comment")

    @pytest.mark.parametrize("op", passing_ops)
    def test_passing(self, op):
        """Test that custom operations that are well-defined are passing."""
        result, output = self.Checker(op)
        assert output == ""
        assert result == "pass"

    @pytest.mark.parametrize("op_with_hint", ops_with_hints)
    def test_hints(self, op_with_hint):
        """Test that custom operations correctly receive a hint if necessary."""
        op, hint = op_with_hint
        result, output = self.Checker(op)
        assert hint in output
        assert result == "hint"

    @pytest.mark.parametrize("error_op", error_ops)
    def test_errors(self, error_op):
        """Test that custom operations correctly are reported with an error."""
        op, *err_strs = error_op
        result, output = self.Checker(op)
        print(result, output)
        assert all(err_str in output for err_str in err_strs)
        assert result == "error"

    @pytest.mark.parametrize("fatal_error_op", fatal_error_ops)
    def test_fatal_errors(self, fatal_error_op):
        """Test that custom operations correctly are reported with a fatal error."""
        op, err_str = fatal_error_op
        result, output = self.Checker(op)
        assert err_str in output
        assert result == "fatal_error"
