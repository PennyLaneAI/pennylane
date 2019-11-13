# Copyright 2019 Xanadu Quantum Technologies Inc.

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
Unit tests for the functions contained in pennylane/__init__.py.
"""
# pylint: disable=protected-access,cell-var-from-loop

import pytest
import pennylane as qml


class TestPennyLaneInit:
    """Unit tests for the functions in the pennylane/__init__.py."""

    pytest.importorskip("pennylane_qiskit")
    def test_plugin_converters(self):
        """Test that the converter from the PennyLane-Qiskit plugin entry points are included
        in the correct entry points."""
        assert 'qiskit' in qml.plugin_converters
        assert 'qasm' in qml.plugin_converters
        assert 'qasm_file' in qml.plugin_converters

    pytest.importorskip("pennylane_qiskit")
    def test_load(self):
        """Test that a converter from the PennyLane-Qiskit plugin is returned."""

        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        assert callable(qml.load(qc, name='qiskit'))

    def test_load_error_converter_does_not_exist(self):
        """Test that load raises an error if the converter does not exist."""

        some_circuit_object = qml.RX
        with pytest.raises(ValueError, match="Converter does not exist. Make sure "
                                             "the required plugin is installed "
                                             "and supports conversion."):
            qml.load(some_circuit_object, name='some_external_framework')