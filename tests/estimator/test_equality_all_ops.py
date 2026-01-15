
import pytest
import pennylane.estimator as qre
from pennylane.estimator.resource_operator import ResourceOperator
from pennylane.estimator.compact_hamiltonian import THCHamiltonian, PauliHamiltonian

class TestResourceRepEqualityAllOps:
    """Test that resource_rep(**params) equals resource_rep_from_op() (which returns self) for all ops."""

    def check_equality(self, op):
        rep_from_params = op.__class__.resource_rep(**op.resource_params)
        rep_from_op = op.resource_rep_from_op()

        assert rep_from_op is op
        
        # Select operator avoids queueing in resource_rep, so we skip the 'is not' check for it if it wasn't queued to begin with
        # but here we instantiate ops which queue themselves.
        # However, we are checking value equality.
        
        # For Select, resource_rep returns an unqueued object. 
        # op is queued. 
        # They should be different objects.
        assert rep_from_params is not op
        assert rep_from_params == rep_from_op

    def test_non_parametric_ops(self):
        ops = [
            qre.Hadamard(wires=0),
            qre.T(wires=1),
            qre.S(wires=2),
            qre.CNOT(wires=[0, 1]),
            qre.Toffoli(wires=[0, 1, 2]),
            qre.X(wires=0),
            qre.Y(wires=1),
            qre.Z(wires=2),
            qre.SWAP(wires=[0, 1]),
            qre.CSWAP(wires=[0, 1, 2]),
            qre.CY(wires=[0, 1]),
            qre.CZ(wires=[0, 1]),
            qre.CCZ(wires=[0, 1, 2]),
            qre.CH(wires=[0, 1]),
        ]
        for op in ops:
            self.check_equality(op)

    def test_parametric_ops(self):
        ops = [
            qre.RX(wires=0, precision=1e-3),
            qre.RY(wires=1, precision=1e-3),
            qre.RZ(wires=2, precision=1e-4),
            qre.PhaseShift(wires=0, precision=1e-3),
            qre.Rot(wires=0, precision=1e-3),
            qre.CRX(wires=[0, 1], precision=1e-3),
            qre.CRY(wires=[0, 1], precision=1e-3),
            qre.CRZ(wires=[0, 1], precision=1e-3),
            qre.CRot(wires=[0, 1], precision=1e-3),
            qre.ControlledPhaseShift(wires=[0, 1], precision=1e-3),
            qre.MultiRZ(wires=[0, 1], precision=1e-3),
            qre.PauliRot(pauli_string="XYZ", wires=[0, 1, 2], precision=1e-3),
            qre.SingleExcitation(wires=[0, 1], precision=1e-3),
            qre.PCPhase(num_wires=2, dim=2, rotation_precision=1e-5, wires=[0, 1]),
        ]
        for op in ops:
            self.check_equality(op)

    def test_symbolic_ops(self):
        base = qre.RX(wires=0, precision=1e-3)
        ops = [
            qre.Adjoint(base),
            qre.Pow(base, 2),
            qre.Controlled(base, num_ctrl_wires=1, num_zero_ctrl=0),
            qre.Prod([qre.X(0), qre.Z(1)]),
            qre.ChangeOpBasis(compute_op=qre.Hadamard(0), target_op=qre.Z(0)),
        ]
        for op in ops:
            self.check_equality(op)

    def test_templates_stateprep(self):
        thc_ham = THCHamiltonian(num_orbitals=4, tensor_rank=5)
        ops = [
            qre.UniformStatePrep(num_states=4, wires=[0, 1]),
            qre.AliasSampling(num_coeffs=4, precision=1e-2, wires=[0, 1]),
            qre.QROMStatePreparation(num_state_qubits=2, precision=1e-2, wires=[0, 1]),
            qre.PrepTHC(thc_ham=thc_ham, coeff_precision=10),
            qre.MPSPrep(num_mps_matrices=2, max_bond_dim=2, wires=[0, 1]),
        ]
        for op in ops:
            self.check_equality(op)

    def test_templates_select(self):
        thc_ham = THCHamiltonian(num_orbitals=4, tensor_rank=5)
        pauli_ham = PauliHamiltonian(num_qubits=2, pauli_terms={"XY": 1, "ZZ": 1})
        ops = [
            qre.SelectTHC(thc_ham=thc_ham, rotation_precision=10),
            qre.SelectPauli(pauli_ham=pauli_ham),
        ]
        for op in ops:
            self.check_equality(op)

    def test_templates_subroutines(self):
        ops = [
            qre.QFT(num_wires=3),
            qre.AQFT(order=1, num_wires=3),
            qre.OutOfPlaceSquare(register_size=2),
            qre.IQP(num_wires=2, pattern=[[[0]], [[1]]]),
            qre.PhaseGradient(num_wires=3),
            qre.OutMultiplier(a_num_wires=2, b_num_wires=2),
            qre.SemiAdder(max_register_size=2),
            qre.ControlledSequence(base=qre.RX(0.1, wires=0), num_control_wires=2),
            qre.BasisRotation(dim=2, wires=[0, 1]),
            qre.BBQRAM(num_bitstrings=2, size_bitstring=2, num_wires=6),
            qre.QubitUnitary(num_wires=2),
        ]
        for op in ops:
            self.check_equality(op)

    def test_templates_qsp(self):
        base = qre.RX(0.1, wires=0)
        ops = [
            qre.QSP(block_encoding=base, poly_deg=2),
            qre.QSVT(block_encoding=base, encoding_dims=(2, 2), poly_deg=2),
            qre.GQSP(signal_operator=base, d_plus=2),
            qre.GQSPTimeEvolution(walk_op=base, time=1.0, one_norm=1.0, poly_approx_precision=1e-2),
        ]
        for op in ops:
            self.check_equality(op)

    def test_templates_trotter(self):
        ops = [
            qre.TrotterProduct(first_order_expansion=[qre.X(0), qre.Z(1)], num_steps=2, order=1),
        ]
        for op in ops:
            self.check_equality(op)

    def test_qubitize_thc(self):
        thc_ham = THCHamiltonian(num_orbitals=4, tensor_rank=5)
        op = qre.QubitizeTHC(thc_ham=thc_ham, coeff_precision=10, rotation_precision=10)
        self.check_equality(op)

    def test_select_subroutine(self):
        # Select from subroutines.py
        ops_list = [qre.X(0), qre.Y(0)]
        op = qre.Select(ops=ops_list)
        self.check_equality(op)

if __name__ == "__main__":
    pytest.main([__file__])
