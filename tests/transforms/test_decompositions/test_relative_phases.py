import pennylane as qml
from pennylane.transforms.decompositions.relative_phases import replace_relative_phase_toffoli


class TestRelativePhases:

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
            qml.MultiControlledX(wires=[0, 1, 2, 3])
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
            qml.H(3)
        ]
