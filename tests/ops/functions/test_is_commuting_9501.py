import pytest
import pennylane as qml


class TestProdSumSProdSupport:
    """Tests for Prod, Sum, SProd support in qml.is_commuting (Issue #9501)."""

    def test_sprod_pauli_base_commuting(self):
        """SProd with Pauli base that commutes (regression test)."""
        op1 = qml.s_prod(2.0, qml.X(0))
        op2 = qml.X(0)
        assert qml.is_commuting(op1, op2) is True

    def test_sprod_pauli_base_not_commuting(self):
        """SProd with Pauli base that does not commute (regression test)."""
        op1 = qml.s_prod(2.0, qml.X(0))
        op2 = qml.Z(0)
        assert qml.is_commuting(op1, op2) is False

    def test_sprod_non_pauli_base_commuting(self):
        """SProd with non-Pauli base; same rotation type always commutes."""
        op1 = qml.s_prod(2.0, qml.RX(0.5, 0))
        op2 = qml.s_prod(3.0, qml.RX(0.1, 0))
        assert qml.is_commuting(op1, op2) is True

    def test_sprod_non_pauli_base_not_commuting(self):
        """SProd with non-Pauli base that does not commute with Z."""
        op1 = qml.s_prod(2.0, qml.RX(0.5, 0))
        op2 = qml.Z(0)
        assert qml.is_commuting(op1, op2) is False

    def test_sprod_disjoint_wires(self):
        """SProd on disjoint wires always commutes."""
        op1 = qml.s_prod(2.0, qml.RX(0.5, 0))
        op2 = qml.Z(1)
        assert qml.is_commuting(op1, op2) is True

    def test_prod_non_pauli_not_commuting(self):
        """Prod containing non-Pauli op does not commute with Z on overlapping wire."""
        op1 = qml.prod(qml.RX(0.5, 0), qml.Y(1))
        op2 = qml.Z(0)
        assert qml.is_commuting(op1, op2) is False

    def test_prod_disjoint_wires(self):
        """Prod on disjoint wires always commutes."""
        op1 = qml.prod(qml.RX(0.5, 0), qml.Y(1))
        op2 = qml.Z(2)
        assert qml.is_commuting(op1, op2) is True

    def test_sum_non_pauli_not_commuting(self):
        """Sum containing non-Pauli op does not commute with Z."""
        op1 = qml.sum(qml.RX(0.5, 0), qml.Y(0))
        op2 = qml.Z(0)
        assert qml.is_commuting(op1, op2) is False

    def test_sum_disjoint_wires(self):
        """Sum on disjoint wires always commutes."""
        op1 = qml.sum(qml.X(1), qml.Z(2))
        op2 = qml.Y(0)
        assert qml.is_commuting(op1, op2) is True

    def test_no_error_for_sprod_without_pauli_rep(self):
        """Previously raised QuantumFunctionError; now returns a bool."""
        op1 = qml.s_prod(2.0, qml.RX(0.5, 0))
        op2 = qml.Z(0)
        result = qml.is_commuting(op1, op2)
        assert isinstance(result, bool)

    def test_no_error_for_prod_without_pauli_rep(self):
        """Previously raised QuantumFunctionError; now returns a bool."""
        op1 = qml.prod(qml.RX(0.5, 0), qml.Y(1))
        op2 = qml.Z(0)
        result = qml.is_commuting(op1, op2)
        assert isinstance(result, bool)

    def test_no_error_for_sum_without_pauli_rep(self):
        """Previously raised QuantumFunctionError; now returns a bool."""
        op1 = qml.sum(qml.RX(0.5, 0), qml.Y(0))
        op2 = qml.Z(0)
        result = qml.is_commuting(op1, op2)
        assert isinstance(result, bool)