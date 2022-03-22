import pytest
from scipy.stats import unitary_group

import pennylane as qml
from pennylane import numpy as np

from pennylane.debugging import OperationChecker, CheckerError


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
        else:
            expected_res = "pass"

        self.Checker(op, seed=42)
        assert self.Checker.output[op] == ""
        assert self.Checker.results[op] == expected_res

    @pytest.mark.parametrize("op", qmlops_special_input)
    def test_special_input_qml_ops(self, op):
        all_parameters, all_wires = qmlops_special_input[op]
        for parameters, wires in zip(all_parameters, all_wires):
            self.Checker(op, parameters=parameters, wires=wires, seed=42)
            assert self.Checker.output[op] == ""
            assert self.Checker.results[op] == "pass"


class TestDummyOperations:

    Checker = OperationChecker(verbosity="comment")

    def test_with_num_wires(self):
        class DummyOp(qml.operation.Operation):
            num_wires = 1

        self.Checker(DummyOp)
        assert self.Checker.output[DummyOp] == ""
        assert self.Checker.results[DummyOp] == "pass"

    def test_wo_num_wires(self):
        class DummyOp(qml.operation.Operation):
            """Failing dummy operation."""

        with pytest.raises(CheckerError, match="Fatal error: Subsequent checks"):
            self.Checker(DummyOp)

        assert "DummyOp does not define the number of wires" in self.Checker.output[DummyOp]
        assert self.Checker.results[DummyOp] == "error"

    def test_with_num_params(self):
        class DummyOp(qml.operation.Operation):
            num_wires = 1
            num_wires = 2

        self.Checker(DummyOp)
        assert self.Checker.output[DummyOp] == ""
        assert self.Checker.results[DummyOp] == "pass"


"""
class B(qml.operation.Operation):
    num_wires = 2
    num_params = 2


class C(qml.operation.Operation):
    num_wires = 2

    def __init__(self, x, wires):
        super().__init__(x, wires=wires)


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
