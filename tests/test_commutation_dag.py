import pytest
import pennylane as qml
from pennylane.transforms.control import *
from pennylane.commutation_dag import *


class TestCommutation:

    def test_commutation(self):

        a = qml.PauliZ(wires=[0])
        b = qml.PauliZ(wires=[0])
        assert is_commuting(a, b) == True

        a = qml.Toffoli(wires=[1, 2, 3])
        b = qml.Toffoli(wires=[4, 5, 6])
        assert is_commuting(a, b) == True

        a = qml.Toffoli(wires=[1, 2, 3])
        b = qml.Toffoli(wires=[2, 1, 3])
        assert is_commuting(a, b) == True

        a = qml.Toffoli(wires=[1, 2, 3])
        b = qml.Toffoli(wires=[3, 2, 1])
        assert is_commuting(a, b) == False

        a = qml.Toffoli(wires=[1, 2, 3])
        b = qml.Toffoli(wires=[1, 4, 5])
        assert is_commuting(a, b) == True

        a = qml.Toffoli(wires=[1, 2, 3])
        b = qml.Toffoli(wires=[5, 4, 3])
        assert is_commuting(a, b) == True

        a = qml.CNOT(wires=[2, 3])
        b = qml.Toffoli(wires=[5, 4, 3])
        assert is_commuting(a, b) == True

        a = qml.PauliZ(wires=[0])
        b = qml.Toffoli(wires=[5, 4, 0])
        assert is_commuting(a, b) == False

        a = qml.PauliZ(wires=[4])
        b = qml.Toffoli(wires=[5, 4, 0])
        assert is_commuting(a, b) == True

        a = qml.CZ(wires=[5, 4])
        b = qml.Toffoli(wires=[5, 4, 0])
        assert is_commuting(a, b) == True

        a = qml.CZ(wires=[4, 0])
        b = qml.Toffoli(wires=[5, 4, 0])
        assert is_commuting(a, b) == False

        a = qml.CZ(wires=[1, 0])
        b = qml.CZ(wires=[0, 1])
        assert is_commuting(a, b) == True

