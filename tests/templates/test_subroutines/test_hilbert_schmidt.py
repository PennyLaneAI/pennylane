# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the Hilbert-Schmidt templates.
"""
import pytest

import pennylane as qml
from pennylane.tape import QuantumScript, make_qscript


class TestHilbertSchmidt:
    """Tests for the Hilbert-Schmidt template."""

    def test_hs_decomposition_1_qubit(self):
        """Test if the HS operation is correctly decomposed for a 1 qubit unitary."""
        U = QuantumScript([qml.Hadamard(wires=0)])

        def v_circuit(params):
            qml.RZ(params[0], wires=1)

        op = qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=[1], u_script=U)

        script_dec = make_qscript(op.decomposition)()

        expected_operations = [
            qml.Hadamard(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
            qml.RZ(-0.1, wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
        ]
        for i, j in zip(script_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.data == j.data
            assert i.wires == j.wires

    def test_hs_decomposition_2_qubits(self):
        """Test if the HS operation is correctly decomposed for 2 qubits."""
        U = QuantumScript([qml.SWAP(wires=[0, 1])])

        def v_circuit(params):
            qml.RZ(params[0], wires=2)
            qml.CNOT(wires=[2, 3])

        op = qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=[2, 3], u_script=U)

        script_dec = make_qscript(op.decomposition)()

        expected_operations = [
            qml.Hadamard(wires=[0]),
            qml.Hadamard(wires=[1]),
            qml.CNOT(wires=[0, 2]),
            qml.CNOT(wires=[1, 3]),
            qml.SWAP(wires=[0, 1]),
            qml.RZ(-0.1, wires=[2]),
            qml.CNOT(wires=[2, 3]),
            qml.CNOT(wires=[1, 3]),
            qml.CNOT(wires=[0, 2]),
            qml.Hadamard(wires=[0]),
            qml.Hadamard(wires=[1]),
        ]

        for i, j in zip(script_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.data == j.data
            assert i.wires == j.wires

    def test_hs_decomposition_2_qubits_custom_wires(self):
        """Test if the HS operation is correctly decomposed for 2 qubits with custom wires."""
        U = QuantumScript([qml.SWAP(wires=["a", "b"])])

        def v_circuit(params):
            qml.RZ(params[0], wires="c")
            qml.CNOT(wires=["c", "d"])

        op = qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=["c", "d"], u_script=U)

        script_dec = make_qscript(op.decomposition)()

        expected_operations = [
            qml.Hadamard(wires=["a"]),
            qml.Hadamard(wires=["b"]),
            qml.CNOT(wires=["a", "c"]),
            qml.CNOT(wires=["b", "d"]),
            qml.SWAP(wires=["a", "b"]),
            qml.RZ(-0.1, wires=["c"]),
            qml.CNOT(wires=["c", "d"]),
            qml.CNOT(wires=["b", "d"]),
            qml.CNOT(wires=["a", "c"]),
            qml.Hadamard(wires=["a"]),
            qml.Hadamard(wires=["b"]),
        ]

        for i, j in zip(script_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.data == j.data
            assert i.wires == j.wires

    def test_v_not_quantum_function(self):
        """Test that we cannot pass a non quantum function to the HS operation"""

        U = QuantumScript([qml.Hadamard(wires=0)])
        v_circuit = QuantumScript([qml.RZ(0.1, wires=1)])

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The argument v_function must be a callable quantum " "function.",
        ):
            qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=[1], u_script=U)

    def test_u_v_same_number_of_wires(self):
        """Test that U and V must have the same number of wires."""

        U = QuantumScript([qml.CNOT(wires=[0, 1])])

        def v_circuit(params):
            qml.RZ(params[0], wires=1)

        with pytest.raises(
            qml.QuantumFunctionError, match="U and V must have the same number of wires."
        ):
            qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=[2], u_script=U)

    def test_u_quantum_script(self):
        """Test that U must be a quantum script."""

        def u_circuit():
            qml.CNOT(wires=[0, 1])

        def v_circuit(params):
            qml.RZ(params[0], wires=1)

        with pytest.raises(
            qml.QuantumFunctionError, match="The argument u_script must be a QuantumScript."
        ):
            qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=[1], u_script=u_circuit)

    def test_v_wires(self):
        """Test that all wires in V are also in v_wires."""

        U = QuantumScript([qml.Hadamard(wires=0)])

        def v_circuit(params):
            qml.RZ(params[0], wires=2)

        with pytest.raises(
            qml.QuantumFunctionError, match="All wires in v_script must be in v_wires."
        ):
            qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=[1], u_script=U)

    def test_distinct_wires(self):
        """Test that U and V have distinct wires."""

        U = QuantumScript([qml.Hadamard(wires=0)])

        def v_circuit(params):
            qml.RZ(params[0], wires=0)

        with pytest.raises(
            qml.QuantumFunctionError, match="u_script and v_script must act on distinct wires."
        ):
            qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=[0], u_script=U)


class TestLocalHilbertSchmidt:
    """Tests for the Local Hilbert-Schmidt template."""

    def test_lhs_decomposition_1_qubit(self):
        """Test if the LHS operation is correctly decomposed"""
        U = QuantumScript([qml.Hadamard(wires=0)])

        def v_circuit(params):
            qml.RZ(params[0], wires=1)

        op = qml.LocalHilbertSchmidt([0.1], v_function=v_circuit, v_wires=[1], u_script=U)

        script_dec = make_qscript(op.decomposition)()

        expected_operations = [
            qml.Hadamard(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
            qml.RZ(-0.1, wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
        ]

        for i, j in zip(script_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.data == j.data
            assert i.wires == j.wires

    def test_lhs_decomposition_1_qubit_custom_wires(self):
        """Test if the LHS operation is correctly decomposed with custom wires."""
        U = QuantumScript([qml.Hadamard(wires="a")])

        def v_circuit(params):
            qml.RZ(params[0], wires="b")

        op = qml.LocalHilbertSchmidt([0.1], v_function=v_circuit, v_wires=["b"], u_script=U)

        script_dec = make_qscript(op.decomposition)()

        expected_operations = [
            qml.Hadamard(wires=["a"]),
            qml.CNOT(wires=["a", "b"]),
            qml.Hadamard(wires=["a"]),
            qml.RZ(-0.1, wires=["b"]),
            qml.CNOT(wires=["a", "b"]),
            qml.Hadamard(wires=["a"]),
        ]

        for i, j in zip(script_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.data == j.data
            assert i.wires == j.wires

    def test_lhs_decomposition_2_qubits(self):
        """Test if the LHS operation is correctly decomposed for 2 qubits."""
        U = QuantumScript([qml.SWAP(wires=[0, 1])])

        def v_circuit(params):
            qml.RZ(params[0], wires=2)
            qml.CNOT(wires=[2, 3])

        op = qml.LocalHilbertSchmidt([0.1], v_function=v_circuit, v_wires=[2, 3], u_script=U)

        script_dec = make_qscript(op.decomposition)()

        expected_operations = [
            qml.Hadamard(wires=[0]),
            qml.Hadamard(wires=[1]),
            qml.CNOT(wires=[0, 2]),
            qml.CNOT(wires=[1, 3]),
            qml.SWAP(wires=[0, 1]),
            qml.RZ(-0.1, wires=[2]),
            qml.CNOT(wires=[2, 3]),
            qml.CNOT(wires=[0, 2]),
            qml.Hadamard(wires=[0]),
        ]

        for i, j in zip(script_dec.operations, expected_operations):
            assert i.name == j.name
            assert i.data == j.data
            assert i.wires == j.wires

    def test_v_not_quantum_function(self):
        """Test that we cannot pass a non quantum function to the HS operation"""

        U = QuantumScript([qml.Hadamard(wires=0)])
        v_circuit = QuantumScript([qml.RZ(0.1, wires=1)])

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The argument v_function must be a callable quantum " "function.",
        ):
            qml.LocalHilbertSchmidt([0.1], v_function=v_circuit, v_wires=[1], u_script=U)

    def test_u_v_same_number_of_wires(self):
        """Test that U and V must have the same number of wires."""

        U = QuantumScript([qml.CNOT(wires=[0, 1])])

        def v_circuit(params):
            qml.RZ(params[0], wires=1)

        with pytest.raises(
            qml.QuantumFunctionError, match="U and V must have the same number of wires."
        ):
            qml.LocalHilbertSchmidt([0.1], v_function=v_circuit, v_wires=[2], u_script=U)

    def test_u_quantum_script(self):
        """Test that U must be a quantum script."""

        def u_circuit():
            qml.CNOT(wires=[0, 1])

        def v_circuit(params):
            qml.RZ(params[0], wires=1)

        with pytest.raises(
            qml.QuantumFunctionError, match="The argument u_script must be a QuantumScript."
        ):
            qml.LocalHilbertSchmidt([0.1], v_function=v_circuit, v_wires=[1], u_script=u_circuit)

    def test_v_wires(self):
        """Test that all wires in V are also in v_wires."""

        U = QuantumScript([qml.Hadamard(wires=0)])

        def v_circuit(params):
            qml.RZ(params[0], wires=2)

        with pytest.raises(
            qml.QuantumFunctionError, match="All wires in v_script must be in v_wires."
        ):
            qml.LocalHilbertSchmidt([0.1], v_function=v_circuit, v_wires=[1], u_script=U)

    def test_distinct_wires(self):
        """Test that U and V have distinct wires."""

        U = QuantumScript([qml.Hadamard(wires=0)])

        def v_circuit(params):
            qml.RZ(params[0], wires=0)

        with pytest.raises(
            qml.QuantumFunctionError, match="u_script and v_script must act on distinct wires."
        ):
            qml.LocalHilbertSchmidt([0.1], v_function=v_circuit, v_wires=[0], u_script=U)
