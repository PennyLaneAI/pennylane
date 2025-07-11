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

# pylint: disable=protected-access,trailing-whitespace

import re

import pytest

import pennylane as qml
from pennylane.devices.capabilities import (
    DeviceCapabilities,
    ExecutionCondition,
    OperatorProperties,
    _get_compilation_options,
    _get_measurement_processes,
    _get_observables,
    _get_operations,
    _get_toml_section,
    load_toml_file,
    observable_stopping_condition_factory,
    parse_toml_document,
    update_device_capabilities,
)
from pennylane.exceptions import InvalidCapabilitiesError


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
            supported_mcm_methods = ["one-shot", "device"]
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
        assert compilation.get("supported_mcm_methods") == ["one-shot", "device"]

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
            supported_mcm_methods = ["one-shot"]
            runtime_code_generation = false
            """
        ],
        indirect=True,
    )
    def test_get_compilation_flags(self, request):
        """Tests getting compilation flags."""

        document = load_toml_file(request.node.toml_file)
        compilation_flags = _get_compilation_options(document)

        # Tests that specified values are correctly parsed
        assert compilation_flags.get("qjit_compatible") is True
        assert compilation_flags.get("supported_mcm_methods") == ["one-shot"]
        assert compilation_flags.get("runtime_code_generation") is False

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
            _get_compilation_options(document)

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
            _get_compilation_options(document)

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize(
        "create_temporary_toml_file",
        [
            """
            schema = 3
            
            [qjit.operators.gates]
            
            PauliX = {}
            
            [compilation]
            
            qjit_compatible = false
            """
        ],
        indirect=True,
    )
    def test_qjit_incompatible_error(self, request):
        """Tests that a device with qjit-specific features is qjit-compatible."""

        document = load_toml_file(request.node.toml_file)
        capabilities = parse_toml_document(document)
        with pytest.raises(
            InvalidCapabilitiesError,
            match="qjit-specific sections are found but the device is not qjit-compatible.",
        ):
            update_device_capabilities(capabilities, document, "qjit")


def test_operator_properties():
    """Tests the OperatorProperties class."""

    prop1 = OperatorProperties(
        controllable=True,
        invertible=False,
        differentiable=True,
        conditions=[ExecutionCondition.ANALYTIC_MODE_ONLY],
    )
    prop2 = OperatorProperties(
        controllable=True,
        invertible=True,
        differentiable=False,
        conditions=[],
    )
    intersection = prop1 & prop2
    assert intersection.controllable is True
    assert intersection.invertible is False
    assert intersection.differentiable is False
    assert intersection.conditions == [ExecutionCondition.ANALYTIC_MODE_ONLY]


EXAMPLE_TOML_FILE = """
schema = 3

[operators.gates]

RY = { properties = ["controllable", "differentiable"] }
RZ = { properties = ["controllable", "invertible", "differentiable"] }
CNOT = { properties = ["invertible"] }

[operators.observables]

PauliX = { }
PauliY = { }
PauliZ = { }

[pennylane.operators.observables]

SProd = { }
Prod = { }
Sum = { conditions = ["terms-commute"] }

[measurement_processes]

ExpectationMP = { }
SampleMP = { }
StateMP = { conditions = ["analytic"] }

[pennylane.measurement_processes]

CountsMP = { conditions = ["finiteshots"] }

[compilation]

qjit_compatible = true
overlapping_observables = true
non_commuting_observables = false
initial_state_prep = true

[pennylane.compilation]

non_commuting_observables = true
supported_mcm_methods = ["one-shot"]

"""


class TestDeviceCapabilities:
    """Tests the DeviceCapabilities class."""

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [EXAMPLE_TOML_FILE], indirect=True)
    def test_load_from_toml_file(self, request):
        """Tests loading device capabilities from a TOML file."""

        document = load_toml_file(request.node.toml_file)
        device_capabilities = parse_toml_document(document)
        assert isinstance(device_capabilities, DeviceCapabilities)
        assert device_capabilities.operations == {
            "RY": OperatorProperties(differentiable=True, controllable=True),
            "RZ": OperatorProperties(invertible=True, differentiable=True, controllable=True),
            "CNOT": OperatorProperties(invertible=True),
        }
        assert device_capabilities.observables == {
            "PauliX": OperatorProperties(),
            "PauliY": OperatorProperties(),
            "PauliZ": OperatorProperties(),
        }
        assert device_capabilities.measurement_processes == {
            "ExpectationMP": [],
            "SampleMP": [],
            "StateMP": [ExecutionCondition.ANALYTIC_MODE_ONLY],
        }
        assert device_capabilities.qjit_compatible is True
        assert device_capabilities.supported_mcm_methods == []
        assert device_capabilities.dynamic_qubit_management is False
        assert device_capabilities.runtime_code_generation is False
        assert device_capabilities.overlapping_observables is True
        assert device_capabilities.non_commuting_observables is False
        assert device_capabilities.initial_state_prep is True

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [EXAMPLE_TOML_FILE], indirect=True)
    def test_update_capabilities(self, request):
        """Tests updating device capabilities for runtime interface exclusive support."""

        document = load_toml_file(request.node.toml_file)
        capabilities = parse_toml_document(document)
        update_device_capabilities(capabilities, document, "pennylane")

        assert capabilities.observables == {
            "PauliX": OperatorProperties(),
            "PauliY": OperatorProperties(),
            "PauliZ": OperatorProperties(),
            "SProd": OperatorProperties(),
            "Prod": OperatorProperties(),
            "Sum": OperatorProperties(conditions=[ExecutionCondition.TERMS_MUST_COMMUTE]),
        }
        assert capabilities.measurement_processes == {
            "ExpectationMP": [],
            "SampleMP": [],
            "StateMP": [ExecutionCondition.ANALYTIC_MODE_ONLY],
            "CountsMP": [ExecutionCondition.FINITE_SHOTS_ONLY],
        }
        assert capabilities.non_commuting_observables is True
        assert capabilities.supported_mcm_methods == ["one-shot"]

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [EXAMPLE_TOML_FILE], indirect=True)
    def test_invalid_runtime_interface(self, request):
        """Tests updating device capabilities with an invalid runtime interface."""

        document = load_toml_file(request.node.toml_file)
        capabilities = parse_toml_document(document)
        with pytest.raises(ValueError, match="Invalid runtime interface:"):
            update_device_capabilities(capabilities, document, "invalid_interface")

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [EXAMPLE_TOML_FILE], indirect=True)
    def test_filter_capabilities(self, request):
        """Tests filtering device capabilities based on execution method."""

        document = load_toml_file(request.node.toml_file)
        capabilities = parse_toml_document(document)
        update_device_capabilities(capabilities, document, "pennylane")
        shots_capabilities = capabilities.filter(finite_shots=True)
        analytic_capabilities = capabilities.filter(finite_shots=False)

        assert shots_capabilities.measurement_processes == {
            "ExpectationMP": [],
            "SampleMP": [],
            "CountsMP": [ExecutionCondition.FINITE_SHOTS_ONLY],
        }

        assert analytic_capabilities.measurement_processes == {
            "ExpectationMP": [],
            "SampleMP": [],
            "StateMP": [ExecutionCondition.ANALYTIC_MODE_ONLY],
        }

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [EXAMPLE_TOML_FILE], indirect=True)
    def test_from_toml_file_pennylane(self, request):
        """Tests loading a device capabilities from a TOML file directly."""

        capabilities = DeviceCapabilities.from_toml_file(request.node.toml_file, "pennylane")
        assert isinstance(capabilities, DeviceCapabilities)
        assert capabilities == DeviceCapabilities(
            operations={
                "RY": OperatorProperties(differentiable=True, controllable=True),
                "RZ": OperatorProperties(invertible=True, differentiable=True, controllable=True),
                "CNOT": OperatorProperties(invertible=True),
            },
            observables={
                "PauliX": OperatorProperties(),
                "PauliY": OperatorProperties(),
                "PauliZ": OperatorProperties(),
                "SProd": OperatorProperties(),
                "Prod": OperatorProperties(),
                "Sum": OperatorProperties(conditions=[ExecutionCondition.TERMS_MUST_COMMUTE]),
            },
            measurement_processes={
                "ExpectationMP": [],
                "SampleMP": [],
                "StateMP": [ExecutionCondition.ANALYTIC_MODE_ONLY],
                "CountsMP": [ExecutionCondition.FINITE_SHOTS_ONLY],
            },
            qjit_compatible=True,
            dynamic_qubit_management=False,
            runtime_code_generation=False,
            overlapping_observables=True,
            non_commuting_observables=True,
            initial_state_prep=True,
            supported_mcm_methods=["one-shot"],
        )

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [EXAMPLE_TOML_FILE], indirect=True)
    def test_from_toml_file_qjit(self, request):
        """Tests loading a device capabilities from a TOML file directly for qjit"""

        capabilities = DeviceCapabilities.from_toml_file(request.node.toml_file, "qjit")
        assert isinstance(capabilities, DeviceCapabilities)
        assert capabilities == DeviceCapabilities(
            operations={
                "RY": OperatorProperties(differentiable=True, controllable=True),
                "RZ": OperatorProperties(invertible=True, differentiable=True, controllable=True),
                "CNOT": OperatorProperties(invertible=True),
            },
            observables={
                "PauliX": OperatorProperties(),
                "PauliY": OperatorProperties(),
                "PauliZ": OperatorProperties(),
            },
            measurement_processes={
                "ExpectationMP": [],
                "SampleMP": [],
                "StateMP": [ExecutionCondition.ANALYTIC_MODE_ONLY],
            },
            qjit_compatible=True,
            dynamic_qubit_management=False,
            runtime_code_generation=False,
            overlapping_observables=True,
            non_commuting_observables=False,
            initial_state_prep=True,
            supported_mcm_methods=[],
        )

    @pytest.mark.usefixtures("create_temporary_toml_file")
    @pytest.mark.parametrize("create_temporary_toml_file", [EXAMPLE_TOML_FILE], indirect=True)
    def test_supports_operators(self, request):
        """Tests that the supports_operators method returns the correct result."""

        capabilities = DeviceCapabilities.from_toml_file(request.node.toml_file)

        for op in [
            qml.RY(0.5, wires=0),
            qml.ops.Controlled(qml.RY(0.5, wires=0), control_wires=[1]),
            qml.RZ(0.5, wires=0),
            qml.ops.Controlled(qml.RZ(0.5, wires=0), control_wires=[1]),
            qml.adjoint(qml.RZ(0.5, wires=0)),
            qml.ops.Adjoint(qml.ops.Controlled(qml.RZ(0.5, wires=0), control_wires=[1])),
            qml.ops.Controlled(qml.ops.Adjoint(qml.RZ(0.5, wires=0)), control_wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.adjoint(qml.CNOT),
        ]:
            assert capabilities.supports_operation(op.name) is True

        for op in [
            qml.X(0),
            qml.adjoint(qml.RY(0.5, wires=0)),
            qml.adjoint(qml.ops.Controlled(qml.RY(0.5, wires=0), control_wires=[1])),
            qml.ops.Controlled(qml.ops.Adjoint(qml.RY(0.5, wires=0)), control_wires=[1]),
            qml.ops.Controlled(qml.CNOT(wires=[0, 1]), control_wires=[2]),
        ]:
            assert capabilities.supports_operation(op.name) is False

        for obs in [qml.X(0), qml.Y(0), qml.Z(0)]:
            assert capabilities.supports_observable(obs.name) is True

        for obs in [
            qml.H(0),
            qml.Hamiltonian([0.5], [qml.PauliZ(0)]),
        ]:
            assert capabilities.supports_observable(obs.name) is False


def test_observable_stopping_condition_factory():
    """Tests the observable stopping condition factory."""

    capabilities = DeviceCapabilities()
    capabilities.observables = {
        "PauliX": OperatorProperties(),
        "PauliZ": OperatorProperties(),
        "SProd": OperatorProperties(),
        "Prod": OperatorProperties(),
        "Sum": OperatorProperties(),
        "LinearCombination": OperatorProperties(),
    }

    stopping_condition = observable_stopping_condition_factory(capabilities)
    assert stopping_condition(qml.X(0)) is True
    assert stopping_condition(qml.Y(0)) is False
    assert stopping_condition(0.5 * qml.X(0)) is True
    assert stopping_condition(0.5 * qml.Z(0) + 0.1 * qml.X(0) @ qml.Z(1)) is True
    assert stopping_condition(qml.Hamiltonian([0.1, 0.2], [qml.Z(0), qml.X(0) @ qml.Y(1)])) is False
