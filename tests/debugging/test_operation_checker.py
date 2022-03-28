import pytest
from scipy.stats import unitary_group
import scipy.linalg as la

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
    @pytest.mark.parametrize("mat", matrices)
    def test_same_matrix(self, mat):
        assert equal_up_to_phase(mat, mat, atol=1e-14)

    @pytest.mark.parametrize("mat, phase", zip(matrices, scalar_phases))
    def test_global_phase(self, mat, phase):
        assert equal_up_to_phase(mat, phase * mat, atol=1e-14)

    @pytest.mark.parametrize("mat, phase", zip(matrices, matrix_phases))
    def test_matrix_phase(self, mat, phase):
        assert not equal_up_to_phase(mat, phase * mat, atol=1e-5)
        assert not equal_up_to_phase(mat, phase @ mat, atol=1e-5)
        assert not equal_up_to_phase(mat, mat @ phase, atol=1e-5)

    @pytest.mark.parametrize("mat, factor", zip(matrices, matrices))
    def test_matrix_prefactor(self, mat, factor):
        assert not equal_up_to_phase(mat, factor * mat, atol=1e-5)
        assert not equal_up_to_phase(mat, factor @ mat, atol=1e-5)
        assert not equal_up_to_phase(mat, mat @ factor, atol=1e-5)

    @pytest.mark.parametrize("mat, factor", zip(matrices, [0.3, 2, -0.9j]))
    def test_scalar_prefactor(self, mat, factor):
        assert not equal_up_to_phase(mat, factor * mat, atol=1e-5)


class TestIsDiagonal:
    @pytest.mark.parametrize(
        "mat", [np.diag(np.arange(9)), np.eye(6), np.zeros((5, 5), dtype=complex)]
    )
    def test_diagonal(self, mat):
        assert is_diagonal(mat, atol=1e-14)

    @pytest.mark.parametrize("mat", [np.arange(9).reshape((3, 3)), np.ones((6, 6))])
    def test_not_diagonal(self, mat):
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

    Checker = OperationChecker(verbosity="hint")

    @pytest.mark.parametrize("op", qmlops)
    def test_standard_qml_ops(self, op):
        if op == qml.MultiRZ:
            expected_res = "comment"
        elif issubclass(op, qml.operation.Channel):
            expected_res = "hint"
        else:
            expected_res = "pass"

        self.Checker(op, seed=42)
        assert self.Checker.output[op] == "" or expected_res == "hint"
        assert self.Checker.results[op] == expected_res

    @pytest.mark.parametrize("op", qmlops_special_input)
    def test_special_input_qml_ops(self, op):
        all_parameters, all_wires = qmlops_special_input[op]
        for parameters, wires in zip(all_parameters, all_wires):
            self.Checker(op, parameters=parameters, wires=wires, seed=42)
            if issubclass(op, qml.operation.Channel):
                # Hint for differentiation of channels
                assert "Channels can not be checked for the correct" in self.Checker.output[op]
                assert self.Checker.results[op] == "hint"
            else:
                assert self.Checker.output[op] == ""
                assert self.Checker.results[op] == "pass"


class OpWithNumWires(qml.operation.Operation):
    num_wires = 1


class OpWithNumParams(qml.operation.Operation):
    num_wires = 1
    num_params = 2


passing_ops = [OpWithNumWires, OpWithNumParams]


class OpWithoutNumParams(qml.operation.Operation):
    """This operation does not fix ``num_params`` although it should."""

    num_wires = 2

    def __init__(self, x, wires):
        super().__init__(x, wires=wires)


ops_with_hints = [OpWithoutNumParams]
hints = ["Instantiating OpWithoutNumParams only succeeded when using 1 parameter(s)."]


class OpWithoutNumWires(qml.operation.Operation):
    """Failing dummy operation."""


fatal_error_ops = [
    OpWithoutNumWires,
]
fatal_error_messages = [
    "OpWithoutNumWires does not define the number of wires",
]


class TestPassingCustomOperations:

    Checker = OperationChecker(verbosity="comment")

    def test_passing(self):
        self.Checker(passing_ops)
        for op in passing_ops:
            assert self.Checker.output[op] == ""
            assert self.Checker.results[op] == "pass"

    @pytest.mark.parametrize("op", passing_ops)
    def test_passing_serial(self, op):
        self.Checker(op)
        assert self.Checker.output[op] == ""
        assert self.Checker.results[op] == "pass"


class TestCustomOperationsWithHints:

    Checker = OperationChecker(verbosity="comment")

    def test_hints(self):
        self.Checker(ops_with_hints)
        for op in ops_with_hints:
            assert self.Checker.results[op] == "hint"

    @pytest.mark.parametrize("op, hint", zip(ops_with_hints, hints))
    def test_hints_serial(self, op, hint):
        self.Checker(op)
        assert hint in self.Checker.output[op]
        assert self.Checker.results[op] == "hint"


class TestFatalErrorCustomOperations:

    Checker = OperationChecker(verbosity="comment")

    def test_fatal_errors(self):
        self.Checker(fatal_error_ops)
        for op in fatal_error_ops:
            assert self.Checker.results[op] == "fatal_error"

    @pytest.mark.parametrize(
        "op, err_str",
        zip(fatal_error_ops, fatal_error_messages),
    )
    def test_fatal_errors_serial(self, op, err_str):
        self.Checker(op)
        assert err_str in self.Checker.output[op]
        assert self.Checker.results[op] == "fatal_error"


"""

class D(qml.operation.Operation):
    num_wires = 2

    def __init__(self, *params, wires):
        super().__init__(*params, wires=wires)


class E(qml.operation.Operation):
    num_wires = 1

    @staticmethod
    def compute_matrix(self, x):
        return None

"""
