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
from unittest.mock import Mock

import pytest

import pennylane as qml


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
