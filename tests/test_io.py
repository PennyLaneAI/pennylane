# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
import pytest
from unittest.mock import Mock

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

    @property
    def last_args(self):
        """The last call arguments of the mocked loader."""
        return self.mock_loader.call_args[0]


load_entry_points = ["qiskit", "qasm", "qasm_file", "pyquil_program", "quil", "quil_file"]


@pytest.fixture
def mock_plugin_converters(monkeypatch):
    mock_plugin_converter_dict = {
        entry_point: MockPluginConverter(entry_point) for entry_point in load_entry_points
    }
    monkeypatch.setattr(qml.io, "plugin_converters", mock_plugin_converter_dict)

    yield mock_plugin_converter_dict


class TestLoad:
    """Test that the convenience load functions access the correct entrypoint."""

    def test_converter_does_not_exist(self):
        """Test that the proper error is raised if the converter does not exist."""
        with pytest.raises(
            ValueError, match="Converter does not exist. Make sure the required plugin is installed"
        ):
            qml.load("Test", format="some_non_existing_format")

    @pytest.mark.parametrize(
        "method,entry_point_name",
        [
            (qml.from_qiskit, "qiskit"),
            (qml.from_qasm, "qasm"),
            (qml.from_qasm_file, "qasm_file"),
            (qml.from_pyquil, "pyquil_program"),
            (qml.from_quil, "quil"),
            (qml.from_quil_file, "quil_file"),
        ],
    )
    def test_convenience_functions(self, method, entry_point_name, mock_plugin_converters):
        """Test that the convenience load functions access the correct entrypoint."""

        res = method("Test")

        assert mock_plugin_converters[entry_point_name].called
        assert mock_plugin_converters[entry_point_name].last_args == ("Test",)

        for plugin_converter in mock_plugin_converters:
            if plugin_converter == entry_point_name:
                continue

            if mock_plugin_converters[plugin_converter].called:
                raise Exception(f"The other plugin converter {plugin_converter} was called.")
