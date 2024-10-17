# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
This module contains unit tests for device capabilities and the TOML module
"""
import re

# pylint: disable=protected-access,trailing-whitespace

from os import path
from tempfile import TemporaryDirectory
from textwrap import dedent

import pytest

from pennylane.devices.capabilities import (
    ExecutionCondition,
    OperatorProperties,
    _get_compilation_flags,
    _get_measurement_processes,
    _get_observables,
    _get_operations,
    _get_options,
    load_toml_file,
    _get_toml_section,
    InvalidCapabilitiesError,
)


@pytest.fixture(scope="function")
def create_temporary_toml_file(request) -> str:
    """Create a temporary TOML file with the given content."""
    content = request.param
    with TemporaryDirectory() as temp_dir:
        toml_file = path.join(temp_dir, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(dedent(content))
        request.node.toml_file = toml_file
        yield


@pytest.mark.unit
class TestTOML:
    """Unit tests for loading and parsing TOML files."""

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file",
        [
            """
            schema = 3
            
            [operators.gates]
            
            RY = {}
            RZ = {}
            CNOT = {}
            
            [operators.observables]
            
            PauliZ = {}
            Hadamard = {}
            
            [measurement_processes]

            ExpectationMP = { }
            SampleMP = { }
            
            [compilation]

            qjit_compatible = false
            mid_circuit_measurements = false

            [options]
            
            option_key = "option_field"
            
            """
        ],
        indirect=True,
    )
    def test_load_toml_file(self, request):
        """Tests loading a TOML file."""

        document = load_toml_file(request.node.toml_file)
        assert document.get("schema") == 3

        operations = document.get("operators").get("gates")
        assert len(operations) == 3
        assert operations.get("RY") == {}
        assert operations.get("RZ") == {}
        assert operations.get("CNOT") == {}

        observables = document.get("operators").get("observables")
        assert len(observables) == 2
        assert observables.get("PauliZ") == {}
        assert observables.get("Hadamard") == {}

        measurement_processes = document.get("measurement_processes")
        assert len(measurement_processes) == 2
        assert measurement_processes.get("ExpectationMP") == {}
        assert measurement_processes.get("SampleMP") == {}

        compilation = document.get("compilation")
        assert compilation.get("qjit_compatible") is False
        assert compilation.get("mid_circuit_measurements") is False

        options = document.get("options")
        assert options.get("option_key") == "option_field"

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file",
        [
            """
            [operators.gates]

            PauliX = { properties = ["controllable", "invertible"] }
            RY = { properties = ["controllable", "invertible", "differentiable"] }
            CRY = { properties = ["invertible", "differentiable"] }
            CNOT = { properties = ["invertible"] }
            """
        ],
        indirect=True,
    )
    def test_get_operations(self, request):
        """Tests getting supported operations."""

        document = load_toml_file(request.node.toml_file)
        operations = _get_operations(document)
        assert len(operations) == 4
        assert "PauliX" in operations
        assert operations.get("PauliX") == OperatorProperties(controllable=True, invertible=True)
        assert "RY" in operations
        assert operations.get("RY") == OperatorProperties(
            controllable=True, invertible=True, differentiable=True
        )
        assert "CRY" in operations
        assert operations.get("CRY") == OperatorProperties(invertible=True, differentiable=True)
        assert "CNOT" in operations
        assert operations.get("CNOT") == OperatorProperties(invertible=True)

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file",
        [
            """
            [operators.observables]
            
            PauliX = { }
            Sum = { conditions = ["terms-commute"] }
            """
        ],
        indirect=True,
    )
    def test_get_observables(self, request):
        """Tests getting supported observables."""

        document = load_toml_file(request.node.toml_file)
        observables = _get_observables(document)
        assert len(observables) == 2
        assert "PauliX" in observables
        assert observables.get("PauliX") == OperatorProperties()
        assert "Sum" in observables
        assert observables.get("Sum") == OperatorProperties(
            conditions=[ExecutionCondition.TERMS_MUST_COMMUTE]
        )

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file",
        [
            """
            [measurement_processes]

            ExpectationMP = { }
            SampleMP = { }
            CountsMP = { conditions = ["finiteshots"] }
            StateMP = { conditions = ["analytic"] }
            """
        ],
        indirect=True,
    )
    def test_get_measurement_processes(self, request):
        """Tests getting supported measurement processes."""

        document = load_toml_file(request.node.toml_file)
        measurement_processes = _get_measurement_processes(document)
        assert len(measurement_processes) == 4
        assert "ExpectationMP" in measurement_processes
        assert measurement_processes.get("ExpectationMP") == []
        assert "SampleMP" in measurement_processes
        assert measurement_processes.get("SampleMP") == []
        assert "CountsMP" in measurement_processes
        assert measurement_processes.get("CountsMP") == [ExecutionCondition.FINITE_SHOTS_ONLY]
        assert "StateMP" in measurement_processes
        assert measurement_processes.get("StateMP") == [ExecutionCondition.ANALYTIC_MODE_ONLY]

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file",
        [
            """
            [compilation]
            
            qjit_compatible = true
            mid_circuit_measurements = true
            runtime_code_generation = false
            """
        ],
        indirect=True,
    )
    def test_get_compilation_flags(self, request):
        """Tests getting compilation flags."""

        document = load_toml_file(request.node.toml_file)
        compilation_flags = _get_compilation_flags(document)

        # Tests that specified values are correctly parsed
        assert compilation_flags.get("qjit_compatible") is True
        assert compilation_flags.get("mid_circuit_measurements") is True
        assert compilation_flags.get("runtime_code_generation") is False

        # Tests that default values are correctly populated
        assert compilation_flags.get("dynamic_qubit_management") is False
        assert compilation_flags.get("overlapping_observables") is True
        assert compilation_flags.get("non_commuting_observables") is False

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file",
        [
            """
            [options]
            
            option_key = "option_value"
            option_boolean = true
            """
        ],
        indirect=True,
    )
    def test_get_options(self, request):
        """Tests parsing options."""

        document = load_toml_file(request.node.toml_file)
        options = _get_options(document)
        assert len(options) == 2
        assert options.get("option_key") == "option_value"
        assert options.get("option_boolean") is True

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file",
        [
            """
            [pennylane.operators.gates]
            
            PauliX = {}
            PauliY = {}
            PauliZ = {}
            """
        ],
        indirect=True,
    )
    def test_get_toml_section(self, request):
        """Tests getting a section from the TOML document."""

        document = load_toml_file(request.node.toml_file)
        section = _get_toml_section(document, "operators.gates", "pennylane")
        assert len(section) == 3
        assert "PauliX" in section
        assert "PauliY" in section
        assert "PauliZ" in section

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file",
        [
            """
            [operators.gates]
        
            PauliX = {}
            PauliY = {}
            PauliZ = {}
            """
        ],
        indirect=True,
    )
    def test_get_empty_document_section(self, request):
        """Tests loading a section that does not exist."""

        document = load_toml_file(request.node.toml_file)
        section = _get_toml_section(document, "operators.observables")
        assert section == {}

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file",
        [
            """
            [operators.gates]
            
            PauliX = { invalid_attribute = ["invalid_attribute"] }
            
            [measurement_processes]
            
            CountsMP = { invalid_attribute = ["invalid_attribute"] }
            """
        ],
        indirect=True,
    )
    def test_invalid_attributes(self, request):
        """Tests loading TOML files with invalid attributes."""

        document = load_toml_file(request.node.toml_file)
        with pytest.raises(
            InvalidCapabilitiesError,
            match=re.escape("Operator 'PauliX' has unknown attributes: ['invalid_attribute']"),
        ):
            _get_operations(document)

        with pytest.raises(
            InvalidCapabilitiesError,
            match=re.escape("Measurement 'CountsMP' has unknown attributes: ['invalid_attribute']"),
        ):
            _get_measurement_processes(document)

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file",
        [
            """
            [operators.gates]
        
            PauliX = { properties = ["invalid_property"] }
            PauliY = {}
            PauliZ = {}
            """
        ],
        indirect=True,
    )
    def test_invalid_properties(self, request):
        """Tests loading TOML files with invalid operator properties."""

        document = load_toml_file(request.node.toml_file)
        with pytest.raises(
            InvalidCapabilitiesError,
            match=re.escape("Operator 'PauliX' has unknown properties: ['invalid_property']"),
        ):
            _get_operations(document)

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file",
        [
            """
            [operators.observables]
            
            Hamiltonian = { conditions = ["invalid_condition"] }
            
            [measurement_processes]
            
            CountsMP = { conditions = ["invalid_condition"] }
            """
        ],
        indirect=True,
    )
    def test_unknown_conditions(self, request):
        """Tests loading TOML files with unknown conditions."""

        document = load_toml_file(request.node.toml_file)
        with pytest.raises(
            InvalidCapabilitiesError,
            match=re.escape("Operator 'Hamiltonian' has unknown conditions: ['invalid_condition']"),
        ):
            _get_observables(document)

        with pytest.raises(
            InvalidCapabilitiesError,
            match=re.escape("Measurement 'CountsMP' has unknown conditions: ['invalid_condition']"),
        ):
            _get_measurement_processes(document)

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file",
        [
            """
            [operators.observables]
            
            PauliZ = { conditions = ["terms-commute"] }
            
            [measurement_processes]
            
            CountsMP = { conditions = ["finiteshots", "analytic"] }
            """
        ],
        indirect=True,
    )
    def test_invalid_conditions(self, request):
        """Tests loading TOML files with invalid conditions."""

        document = load_toml_file(request.node.toml_file)
        with pytest.raises(
            InvalidCapabilitiesError,
            match="'terms-commute' is only applicable to Prod, SProd, Sum, and LinearCombination.",
        ):
            _get_observables(document)

        with pytest.raises(
            InvalidCapabilitiesError,
            match="Conditions cannot contain both 'analytic' and 'finiteshots'",
        ):
            _get_measurement_processes(document)

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file",
        [
            """
            [compilation]
            
            unknown_flag = true
            """
        ],
        indirect=True,
    )
    def test_unknown_compilation_flag(self, request):
        """Tests loading TOML files with unknown compilation flags."""

        document = load_toml_file(request.node.toml_file)
        with pytest.raises(
            InvalidCapabilitiesError,
            match="The compilation section has unknown options: ",
        ):
            _get_compilation_flags(document)

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file",
        [
            """
            [compilation]
            
            overlapping_observables = false
            non_commuting_observables = true
            """
        ],
        indirect=True,
    )
    def test_invalid_combination_of_flags(self, request):
        """Tests loading TOML files with invalid combination of compilation flags."""

        document = load_toml_file(request.node.toml_file)
        with pytest.raises(
            InvalidCapabilitiesError,
            match="When overlapping_observables is False, non_commuting_observables cannot be True.",
        ):
            _get_compilation_flags(document)
