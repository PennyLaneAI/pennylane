from itertools import permutations

import pennylane as qml
from pennylane.transforms.decompositions.relative_phases import (
    replace_iX_gate,
    replace_multi_controlled_iX_gate,
    replace_relative_phase_toffoli,
)


class TestMultiControlledPhaseXGate:

    def test_multiple_matches(self):
        def qfunc():
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        transformed_qfunc = replace_multi_controlled_iX_gate(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 20
        assert tape.operations == [
            # first instance
            qml.Hadamard(wires=3),
            qml.adjoint(qml.T(wires=3)),
            qml.MultiControlledX(wires=[2] + [3]),
            qml.T(wires=3),
            qml.MultiControlledX(wires=[0, 1] + [3]),
            qml.adjoint(qml.T(wires=3)),
            qml.MultiControlledX(wires=[2] + [3]),
            qml.T(wires=3),
            qml.MultiControlledX(wires=[0, 1] + [3]),
            qml.Hadamard(wires=3),
            # second instance
            qml.Hadamard(wires=3),
            qml.adjoint(qml.T(wires=3)),
            qml.MultiControlledX(wires=[2] + [3]),
            qml.T(wires=3),
            qml.MultiControlledX(wires=[0, 1] + [3]),
            qml.adjoint(qml.T(wires=3)),
            qml.MultiControlledX(wires=[2] + [3]),
            qml.T(wires=3),
            qml.MultiControlledX(wires=[0, 1] + [3]),
            qml.Hadamard(wires=3),
        ]

    def test_surrounded(self):
        def qfunc():
            qml.PauliZ(wires=0)
            qml.PauliY(wires=2)
            qml.PauliX(wires=3)
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.PauliZ(wires=5)
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            qml.PauliZ(wires=0)
            qml.Hadamard(wires=2)
            qml.PauliX(wires=3)
            return qml.expval(qml.Z(0))

        transformed_qfunc = replace_multi_controlled_iX_gate(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 17
        assert tape.operations == [
            qml.PauliZ(wires=0),
            qml.PauliY(wires=2),
            qml.PauliX(wires=3),
            qml.Hadamard(wires=3),
            qml.adjoint(qml.T(wires=3)),
            qml.MultiControlledX(wires=[2] + [3]),
            qml.T(wires=3),
            qml.MultiControlledX(wires=[0, 1] + [3]),
            qml.adjoint(qml.T(wires=3)),
            qml.MultiControlledX(wires=[2] + [3]),
            qml.T(wires=3),
            qml.MultiControlledX(wires=[0, 1] + [3]),
            qml.Hadamard(wires=3),
            qml.PauliZ(wires=5),
            qml.PauliZ(wires=0),
            qml.Hadamard(wires=2),
            qml.PauliX(wires=3),
        ]

    def test_incomplete_pattern(self):
        def qfunc():
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            return qml.expval(qml.Z(0))

        transformed_qfunc = replace_multi_controlled_iX_gate(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 1
        assert tape.operations == [qml.ctrl(qml.S(wires=[2]), control=[0, 1])]

    def test_wire_permutations(self):
        for first, second, third, fourth in permutations([0, 1, 2, 3]):

            def qfunc():
                qml.ctrl(qml.S(wires=[third]), control=[first, second])
                qml.MultiControlledX(wires=[first, second, third, fourth])
                return qml.expval(qml.Z(first))

            transformed_qfunc = replace_multi_controlled_iX_gate(qfunc)

            tape = qml.tape.make_qscript(transformed_qfunc)()
            assert len(tape.operations) == 10
            assert tape.operations == [
                qml.Hadamard(wires=fourth),
                qml.adjoint(qml.T(wires=fourth)),
                qml.MultiControlledX(wires=[third] + [fourth]),
                qml.T(wires=fourth),
                qml.MultiControlledX(wires=[first, second] + [fourth]),
                qml.adjoint(qml.T(wires=fourth)),
                qml.MultiControlledX(wires=[third] + [fourth]),
                qml.T(wires=fourth),
                qml.MultiControlledX(wires=[first, second] + [fourth]),
                qml.Hadamard(wires=fourth),
            ]

    def test_non_interfering_gates(self):
        def qfunc():
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.PauliX(5)
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        transformed_qfunc = replace_multi_controlled_iX_gate(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 11
        assert tape.operations == [
            qml.Hadamard(wires=3),
            qml.adjoint(qml.T(wires=3)),
            qml.MultiControlledX(wires=[2] + [3]),
            qml.T(wires=3),
            qml.MultiControlledX(wires=[0, 1] + [3]),
            qml.adjoint(qml.T(wires=3)),
            qml.MultiControlledX(wires=[2] + [3]),
            qml.T(wires=3),
            qml.MultiControlledX(wires=[0, 1] + [3]),
            qml.Hadamard(wires=3),
            qml.PauliX(wires=5),
        ]

    def test_interfering_gates(self):
        def qfunc():
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.PauliX(3)
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        transformed_qfunc = replace_multi_controlled_iX_gate(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 3
        assert tape.operations == [
            qml.ctrl(qml.S(wires=[2]), control=[0, 1]),
            qml.PauliX(3),
            qml.MultiControlledX(wires=[0, 1, 2, 3]),
        ]

    def test_basic_transform(self):
        def qfunc():
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        transformed_qfunc = replace_multi_controlled_iX_gate(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 10
        assert tape.operations == [
            qml.Hadamard(wires=3),
            qml.adjoint(qml.T(wires=3)),
            qml.MultiControlledX(wires=[2] + [3]),
            qml.T(wires=3),
            qml.MultiControlledX(wires=[0, 1] + [3]),
            qml.adjoint(qml.T(wires=3)),
            qml.MultiControlledX(wires=[2] + [3]),
            qml.T(wires=3),
            qml.MultiControlledX(wires=[0, 1] + [3]),
            qml.Hadamard(wires=3),
        ]


class TestPhaseXGate:

    def test_multiple_matches(self):
        def qfunc():
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.Toffoli(wires=[0, 1, 2])
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.Toffoli(wires=[0, 1, 2])
            return qml.expval(qml.Z(0))

        transformed_qfunc = replace_iX_gate(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 20
        assert tape.operations == [
            # first instance
            qml.Hadamard(wires=2),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.Hadamard(wires=2),
            # second instance
            qml.Hadamard(wires=2),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.Hadamard(wires=2),
        ]

    def test_surrounded(self):
        def qfunc():
            qml.PauliZ(wires=0)
            qml.PauliY(wires=2)
            qml.PauliX(wires=3)
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.PauliZ(wires=5)
            qml.Toffoli(wires=[0, 1, 2])
            qml.PauliZ(wires=0)
            qml.Hadamard(wires=2)  # cancels with H in the replacement
            qml.PauliX(wires=3)
            return qml.expval(qml.Z(0))

        transformed_qfunc = replace_iX_gate(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 15
        assert tape.operations == [
            qml.PauliY(wires=2),
            qml.Hadamard(wires=2),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.PauliZ(wires=0),
            qml.PauliX(wires=3),
            qml.PauliZ(wires=5),
            qml.PauliZ(wires=0),
            qml.PauliX(wires=3),
        ]

    def test_incomplete_pattern(self):
        def qfunc():
            qml.Toffoli(wires=[0, 1, 2])
            return qml.expval(qml.Z(0))

        transformed_qfunc = replace_iX_gate(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 11
        assert tape.operations == [
            qml.H(2),
            qml.adjoint(qml.T(2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(2),
            qml.CNOT(wires=[0, 2]),
            qml.adjoint(qml.T(2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(2),
            qml.CNOT(wires=[0, 2]),
            qml.H(2),
            qml.adjoint(qml.ctrl(qml.S(1), control=[0])),
        ]

    def test_wire_permutations(self):
        for first, second, third in permutations([0, 1, 2]):

            def qfunc():
                qml.ctrl(qml.S(wires=[second]), control=[first])
                qml.Toffoli(wires=[first, second, third])
                return qml.expval(qml.Z(first))

            transformed_qfunc = replace_iX_gate(qfunc)

            tape = qml.tape.make_qscript(transformed_qfunc)()
            assert len(tape.operations) == 10
            assert tape.operations == [
                qml.Hadamard(wires=third),
                qml.adjoint(qml.T(wires=third)),
                qml.CNOT(wires=[second, third]),
                qml.T(wires=third),
                qml.CNOT(wires=[first, third]),
                qml.adjoint(qml.T(wires=third)),
                qml.CNOT(wires=[second, third]),
                qml.T(wires=third),
                qml.CNOT(wires=[first, third]),
                qml.Hadamard(wires=third),
            ]

    def test_non_interfering_gates(self):
        def qfunc():
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.PauliX(3)
            qml.Toffoli(wires=[0, 1, 2])
            return qml.expval(qml.Z(0))

        transformed_qfunc = replace_iX_gate(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 11
        assert tape.operations == [
            qml.Hadamard(wires=2),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.Hadamard(wires=2),
            qml.PauliX(wires=3),
        ]

    def test_basic_transform(self):
        def qfunc():
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.Toffoli(wires=[0, 1, 2])
            return qml.expval(qml.Z(0))

        transformed_qfunc = replace_iX_gate(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 10
        assert tape.operations == [
            qml.Hadamard(wires=2),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.adjoint(qml.T(wires=2)),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.Hadamard(wires=2),
        ]


class TestRelativePhaseToffoli:

    def test_repeated(self):
        def qfunc():
            qml.CCZ(wires=[0, 1, 3])
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            qml.CCZ(wires=[0, 1, 3])
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        transformed_qfunc = replace_relative_phase_toffoli(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 18 * 2
        assert tape.operations == [
            # first instance
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            # second instance
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
        ]

    def test_surrounded(self):
        def qfunc():
            qml.PauliZ(wires=0)
            qml.PauliY(wires=2)
            qml.PauliX(wires=3)
            qml.CCZ(wires=[0, 1, 3])
            qml.PauliX(wires=4)
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.PauliY(wires=5)
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            qml.PauliZ(wires=0)
            qml.Hadamard(wires=2)
            qml.PauliX(wires=3)
            return qml.expval(qml.Z(0))

        transformed_qfunc = replace_relative_phase_toffoli(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 26
        assert tape.operations == [
            qml.PauliZ(wires=0),
            qml.PauliY(wires=2),
            qml.PauliX(wires=3),
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.PauliX(wires=4),
            qml.PauliY(wires=5),
            qml.PauliZ(wires=0),
            qml.Hadamard(wires=2),
            qml.PauliX(wires=3),
        ]

    def test_incomplete_pattern(self):
        def qfunc():
            qml.CCZ(wires=[0, 1, 3])
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        transformed_qfunc = replace_relative_phase_toffoli(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert tape.operations == [
            qml.CCZ(wires=[0, 1, 3]),
            qml.ctrl(qml.S(wires=[2]), control=[0, 1]),
            qml.MultiControlledX(wires=[0, 1, 2, 3]),
        ]

    def test_wire_permutations(self):
        for first, second, third, fourth in permutations([0, 1, 2, 3]):

            def qfunc():
                qml.CCZ(wires=[first, second, fourth])
                qml.ctrl(qml.S(wires=[second]), control=[first])
                qml.ctrl(qml.S(wires=[third]), control=[first, second])
                qml.MultiControlledX(wires=[first, second, third, fourth])
                return qml.expval(qml.Z(first))

            transformed_qfunc = replace_relative_phase_toffoli(qfunc)

            tape = qml.tape.make_qscript(transformed_qfunc)()
            assert len(tape.operations) == 18
            assert tape.operations == [
                qml.H(fourth),
                qml.T(fourth),
                qml.CNOT(wires=[third, fourth]),
                qml.adjoint(qml.T(fourth)),
                qml.H(fourth),
                qml.CNOT(wires=[first, fourth]),
                qml.T(fourth),
                qml.CNOT(wires=[second, fourth]),
                qml.adjoint(qml.T(fourth)),
                qml.CNOT(wires=[first, fourth]),
                qml.T(fourth),
                qml.CNOT(wires=[second, fourth]),
                qml.adjoint(qml.T(fourth)),
                qml.H(fourth),
                qml.T(fourth),
                qml.CNOT(wires=[third, fourth]),
                qml.adjoint(qml.T(fourth)),
                qml.H(fourth),
            ]

    def test_non_interfering_gates(self):
        def qfunc():
            qml.CCZ(wires=[0, 1, 3])
            qml.PauliX(wires=4)
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.PauliY(wires=5)
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        transformed_qfunc = replace_relative_phase_toffoli(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 20
        assert tape.operations == [
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.PauliX(wires=4),
            qml.PauliY(wires=5),
        ]

    def test_gates_interfering(self):
        def qfunc():
            qml.CCZ(wires=[0, 1, 3])
            qml.PauliX(wires=0)
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.PauliX(wires=3)
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        transformed_qfunc = replace_relative_phase_toffoli(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert tape.operations == [
            qml.CCZ(wires=[0, 1, 3]),
            qml.PauliX(wires=0),
            qml.ctrl(qml.S(wires=[1]), control=[0]),
            qml.PauliX(wires=3),
            qml.ctrl(qml.S(wires=[2]), control=[0, 1]),
            qml.MultiControlledX(wires=[0, 1, 2, 3]),
        ]

    def test_basic_transform(self):
        def qfunc():
            qml.CCZ(wires=[0, 1, 3])
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])
            return qml.expval(qml.Z(0))

        transformed_qfunc = replace_relative_phase_toffoli(qfunc)

        tape = qml.tape.make_qscript(transformed_qfunc)()
        assert len(tape.operations) == 18
        assert tape.operations == [
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.CNOT(wires=[0, 3]),
            qml.T(3),
            qml.CNOT(wires=[1, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
            qml.T(3),
            qml.CNOT(wires=[2, 3]),
            qml.adjoint(qml.T(3)),
            qml.H(3),
        ]
