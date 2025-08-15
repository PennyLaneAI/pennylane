# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.io` module.
"""
from textwrap import dedent
from unittest.mock import Mock

import numpy as np
import pytest

import pennylane as qml
from pennylane import queuing
from pennylane.measurements import MeasurementValue
from pennylane.ops import RX
from pennylane.wires import Wires

has_openqasm = True
try:
    import openqasm3

    from pennylane.io.io import from_qasm3  # pylint: disable=ungrouped-imports
    from pennylane.io.qasm_interpreter import QasmInterpreter  # pylint: disable=ungrouped-imports
except (ModuleNotFoundError, ImportError) as import_error:
    has_openqasm = False


class MockPluginConverter:
    """Mocks a real plugin converter entry point."""

    def __init__(self, name):
        self.name = name
        self.mock_loader = Mock()

    def load(self):
        """Return the mocked loader function."""
        return self.mock_loader

    @property
    def called(self):
        """True if the mocked loader was called."""
        return self.mock_loader.called

    # pylint: disable=unsubscriptable-object
    @property
    def last_args(self):
        """The last call arguments of the mocked loader."""
        return self.mock_loader.call_args[0]

    @property
    def call_args(self):
        """The last call arguments of the mocked loader."""
        return self.mock_loader.call_args


load_entry_points = [
    "pyquil_program",
    "qasm_file",
    "qasm",
    "qiskit_op",
    "qiskit_noise",
    "qiskit",
    "quil_file",
    "quil",
]


@pytest.fixture(name="mock_plugin_converters")
def mock_plugin_converters_fixture(monkeypatch):
    mock_plugin_converter_dict = {
        entry_point: MockPluginConverter(entry_point) for entry_point in load_entry_points
    }
    monkeypatch.setattr(qml.io.io, "plugin_converters", mock_plugin_converter_dict)

    yield mock_plugin_converter_dict


class TestLoad:
    """Test that the convenience load functions access the correct entrypoint."""

    @pytest.mark.parametrize(
        "method, entry_point_name",
        [
            (qml.from_qiskit, "qiskit"),
            (qml.from_qiskit_op, "qiskit_op"),
            (qml.from_qiskit_noise, "qiskit_noise"),
        ],
    )
    def test_qiskit_converter_does_not_exist(self, monkeypatch, method, entry_point_name):
        """Test that a RuntimeError with an appropriate message is raised if a Qiskit convenience
        method is called but the Qiskit plugin converter is not found.
        """
        # Temporarily make a mock_converter_dict without the Qiskit entry point.
        mock_plugin_converter_dict = {
            entry_point: MockPluginConverter(entry_point) for entry_point in load_entry_points
        }
        del mock_plugin_converter_dict[entry_point_name]
        monkeypatch.setattr(qml.io, "plugin_converters", mock_plugin_converter_dict)

        # Check that the specific RuntimeError is raised as opposed to a generic ValueError.
        with pytest.raises(RuntimeError, match=r"Conversion from Qiskit requires..."):
            method("Test")

    @pytest.mark.parametrize(
        "method, entry_point_name",
        [
            (qml.from_qiskit, "qiskit"),
            (qml.from_qiskit_op, "qiskit_op"),
            (qml.from_qiskit_noise, "qiskit_noise"),
        ],
    )
    def test_qiskit_converter_load_fails(self, monkeypatch, method, entry_point_name):
        """Test that an exception which is raised while calling a Qiskit convenience method (but
        after the Qiskit plugin converter is found) is propagated correctly.
        """
        mock_plugin_converter = MockPluginConverter(entry_point_name)
        mock_plugin_converter.mock_loader.side_effect = ValueError("Some Other Error")

        mock_plugin_converter_dict = {entry_point_name: mock_plugin_converter}
        monkeypatch.setattr(qml.io.io, "plugin_converters", mock_plugin_converter_dict)

        with pytest.raises(ValueError, match=r"Some Other Error"):
            method("Test")

    @pytest.mark.parametrize(
        "method, entry_point_name",
        [
            (qml.from_qiskit, "qiskit"),
            (qml.from_qiskit_op, "qiskit_op"),
            (qml.from_qiskit_noise, "qiskit_noise"),
            (qml.from_pyquil, "pyquil_program"),
            (qml.from_quil, "quil"),
            (qml.from_quil_file, "quil_file"),
        ],
    )
    def test_convenience_functions(self, method, entry_point_name, mock_plugin_converters):
        """Test that the convenience load functions access the correct entry point."""

        method("Test")

        assert mock_plugin_converters[entry_point_name].called
        assert mock_plugin_converters[entry_point_name].last_args == ("Test",)

        for plugin_converter in mock_plugin_converters:
            if plugin_converter == entry_point_name:
                continue

            if mock_plugin_converters[plugin_converter].called:
                raise RuntimeError(f"The other plugin converter {plugin_converter} was called.")

    def test_from_qasm(self, mock_plugin_converters):
        """Tests that the correct entry point is called for from_qasm."""

        qml.from_qasm("Test")
        assert mock_plugin_converters["qasm"].called
        assert mock_plugin_converters["qasm"].last_args == ("Test",)

        for plugin_converter in mock_plugin_converters:
            if mock_plugin_converters[plugin_converter].called and plugin_converter != "qasm":
                raise RuntimeError(f"The other plugin converter {plugin_converter} was called.")

    @pytest.mark.parametrize(
        "method, entry_point_name, args, kwargs",
        [
            (qml.from_qiskit, "qiskit", ("Circuit",), {"measurements": []}),
            (qml.from_qiskit_op, "qiskit_op", ("Op",), {"params": [1, 2], "wires": [3, 4]}),
            (
                qml.from_qasm,
                "qasm",
                ("Circuit",),
                {"measurements": []},
            ),
        ],
    )
    def test_convenience_function_arguments(
        self,
        method,
        entry_point_name,
        mock_plugin_converters,
        args,
        kwargs,
    ):  # pylint: disable=too-many-arguments
        """Test that the convenience load functions access the correct entry point and forward their
        arguments correctly.
        """
        method(*args, **kwargs)

        assert mock_plugin_converters[entry_point_name].called

        called_args, called_kwargs = mock_plugin_converters[entry_point_name].call_args
        assert called_args == args
        assert called_kwargs == kwargs

        for plugin_converter in mock_plugin_converters:
            if plugin_converter == entry_point_name:
                continue

            if mock_plugin_converters[plugin_converter].called:
                raise RuntimeError(f"The other plugin converter {plugin_converter} was called.")


@pytest.mark.external
class TestOpenQasm:
    """Test the qml.to_openqasm and qml.from_qasm3 functions."""

    dev = qml.device("default.qubit", wires=2)

    @pytest.mark.skipif(not has_openqasm, reason="requires openqasm3")
    def test_return_from_qasm3(self):
        circuit = """\
            OPENQASM 3.0;
            output bit b;
            output float v;
            qubit q0;
            rx(1.2) q0;
            measure q0 -> b;
            v = 2.2;
            """

        # call the method
        b, v = from_qasm3(circuit)()  # the return order is the declaration order
        assert isinstance(b, MeasurementValue)
        assert v == 2.2

    @pytest.mark.skipif(not has_openqasm, reason="requires openqasm3")
    def test_qasm3_inputs(self):
        circuit = """\
            OPENQASM 3.0;
            qubit q0;
            input float t;
            rx(t) q0;
            """

        # call the method
        with queuing.AnnotatedQueue() as q:
            from_qasm3(circuit)(t=1.1)

        # assertions
        assert q.queue == [RX(1.1, Wires(["q0"]))]

    @pytest.mark.skipif(not has_openqasm, reason="requires openqasm3")
    def test_invalid_qasm3(self):
        circuit = """\
            OPENQASM 3.0;
            qubit q0;
            bit output = "0";
            rz(0.9) q0;
            measure q0 -> output;
            """

        with pytest.raises(
            SyntaxError, match="Something went wrong when parsing the provided OpenQASM 3.0 code"
        ):
            from_qasm3(circuit)()

    @pytest.mark.skipif(not has_openqasm, reason="requires openqasm3")
    def test_from_qasm3(self, mocker):
        circuit = """\
            OPENQASM 3.0;
            qubit q0;
            rx(1.2) q0;
            rz(0.9) q0;
            """

        # setup mocks
        parse = mocker.spy(openqasm3.parser, "parse")
        visit = mocker.spy(QasmInterpreter, "interpret")

        # call the method
        from_qasm3(circuit)()

        # assertions
        parse.assert_called_with(circuit, permissive=True)
        visit.assert_called_once()

    def test_basic_example(self):
        """Test basic usage on simple circuit with parameters."""

        @qml.set_shots(100)
        @qml.qnode(self.dev)
        def circuit(theta, phi):
            qml.RX(theta, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(phi, wires=1)
            return qml.sample()

        qasm = qml.to_openqasm(circuit)(1.2, 0.9)

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            rx(1.2) q[0];
            cx q[0],q[1];
            rz(0.9) q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            """
        )
        assert qasm == expected

    def test_measure_qubits_subset_only(self):
        """Test OpenQASM program includes measurements only over the qubits subset specified in the QNode."""

        @qml.set_shots(100)
        @qml.qnode(self.dev)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.sample(wires=1)

        qasm = qml.to_openqasm(circuit, measure_all=False)()

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0],q[1];
            measure q[1] -> c[1];
            """
        )
        assert qasm == expected

    def test_rotations_with_expval(self):
        """Test OpenQASM program includes gates that make the measured observables diagonal in the computational basis."""

        @qml.set_shots(100)
        @qml.qnode(self.dev)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(0) @ qml.PauliY(1))

        qasm = qml.to_openqasm(circuit, rotations=True)()

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0],q[1];
            h q[0];
            z q[1];
            s q[1];
            h q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            """
        )
        assert qasm == expected

    def test_precision(self):
        """Test OpenQASM program takes into account the desired numerical precision of the circuit's parameters."""

        @qml.set_shots(100)
        @qml.qnode(self.dev)
        def circuit():
            qml.RX(np.pi, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        qasm = qml.to_openqasm(circuit, precision=4)()

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            rx(3.142) q[0];
            cx q[0],q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            """
        )

        assert qasm == expected
