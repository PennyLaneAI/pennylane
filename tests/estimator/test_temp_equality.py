
import pytest
import pennylane.estimator as qre
from pennylane.estimator.resource_operator import ResourceOperator

class TestResourceRepEquality:
    """Test that resource_rep(**params) equals resource_rep_from_op() (which returns self)."""

    def test_simple_ops(self):
        ops = [
            qre.Hadamard(wires=0),
            qre.T(wires=1),
            qre.S(wires=2),
            qre.CNOT(wires=[0, 1]),
            qre.Toffoli(wires=[0, 1, 2]),
        ]
        for op in ops:
            rep_from_params = op.__class__.resource_rep(**op.resource_params)
            rep_from_op = op.resource_rep_from_op()

            assert rep_from_op is op
            assert rep_from_params is not op
            assert rep_from_params == rep_from_op

    def test_parametric_ops(self):
        ops = [
            qre.RX(wires=0, precision=1e-3),
            qre.RZ(wires=1, precision=1e-4),
            qre.PhaseShift(wires=2),
        ]
        for op in ops:
            rep_from_params = op.__class__.resource_rep(**op.resource_params)
            rep_from_op = op.resource_rep_from_op()

            assert rep_from_op is op
            assert rep_from_params is not op
            assert rep_from_params == rep_from_op

    def test_symbolic_ops(self):
        base = qre.RX(wires=0, precision=1e-3)
        ops = [
            qre.Adjoint(base),
            qre.Pow(base, 2),
            qre.Controlled(base, num_ctrl_wires=1, num_zero_ctrl=0),
        ]
        for op in ops:
            rep_from_params = op.__class__.resource_rep(**op.resource_params)
            rep_from_op = op.resource_rep_from_op()

            assert rep_from_op is op
            assert rep_from_params is not op
            assert rep_from_params == rep_from_op

    def test_templates(self):
        ops = [
            qre.QFT(num_wires=3),
            qre.AQFT(order=1, num_wires=3),
        ]
        for op in ops:
            rep_from_params = op.__class__.resource_rep(**op.resource_params)
            rep_from_op = op.resource_rep_from_op()

            assert rep_from_op is op
            assert rep_from_params is not op
            assert rep_from_params == rep_from_op

if __name__ == "__main__":
    pytest.main([__file__])
