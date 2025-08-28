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
Pytest configuration file for the devices test module.
"""

import inspect
from os import path
from tempfile import TemporaryDirectory
from textwrap import dedent

import numpy as np
import pytest

import pennylane as qml
from pennylane.decomposition import disable_graph, enable_graph, enabled_graph
from pennylane.tape import QuantumScript


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


@pytest.fixture(scope="session")
def create_operation_from_name():
    """Fixture providing a function to dynamically create quantum operations using introspection.

    Returns a function that takes a gate name and returns a properly constructed operation
    instance suitable for testing.
    """

    def _create_operation_from_name(gate_name):
        """Dynamically create a quantum operation using introspection of the operation class."""
        # Get the operation class
        try:
            op_class = getattr(qml, gate_name)
        except AttributeError:
            raise ValueError(f"Operation {gate_name} not found in PennyLane")

        # Get the class signature to understand parameters
        # Use num_params attribute as the reliable source of parameter count
        num_params_attr = getattr(op_class, "num_params", 0)

        # Handle the case where num_params is a property (e.g., for template operations)
        if isinstance(num_params_attr, property):
            # For template operations, usually no parameters needed for construction
            num_params = 0
        else:
            num_params = num_params_attr

        # Fallback to signature inspection if num_params is not available
        if num_params is None:
            try:
                sig = inspect.signature(op_class.__init__)
                param_names = list(sig.parameters.keys())[1:]  # Skip 'self'
                # Filter out non-data parameters
                non_wire_params = [name for name in param_names if name not in ["wires", "id"]]
                num_params = len(non_wire_params)
            except (TypeError, ValueError):
                # Final fallback
                num_params = 0

        # Determine number of wires needed
        num_wires = getattr(op_class, "num_wires", None)
        if num_wires is None or isinstance(num_wires, property):
            # Try to infer from common patterns
            # Use class name as the primary identifier
            name = getattr(op_class, "__name__", gate_name)

            # Common patterns for wire inference
            if name in ["Toffoli", "CSWAP"]:
                num_wires = 3  # Three-qubit gates
            elif name == "MultiControlledX":  # Special case: needs at least 2 wires
                num_wires = 2  # Minimum wires for MultiControlledX
            elif any(prefix in name for prefix in ["C", "Control"]):
                num_wires = 2  # Controlled operations typically need 2 qubits
            elif name == "QFT":
                num_wires = 2  # Use 2-qubit QFT for testing
            else:
                num_wires = 1  # Default to single qubit
        wires = list(range(num_wires))

        # Handle special cases that need specific data
        if gate_name == "QubitUnitary":
            # Create a simple multi-qubit unitary (tensor product of single-qubit gates)
            single_qubit = np.array([[0, 1], [1, 0]])  # Pauli-X
            matrix = single_qubit
            for _ in range(num_wires - 1):
                matrix = np.kron(matrix, np.eye(2))
            return op_class(matrix, wires=wires)

        elif gate_name in ["BasisState", "StatePrep"]:
            # State preparation operations need state vectors
            if gate_name == "BasisState":
                state = [1] + [0] * (num_wires - 1)  # |100...>
            else:  # StatePrep
                # Simple superposition state
                state_dim = 2**num_wires
                state = np.zeros(state_dim)
                state[0] = 1.0  # |000...>
            return op_class(state, wires=wires)

        # Generate parameters dynamically based on signature analysis
        if num_params == 0:
            # No parameters needed
            return op_class(wires=wires)
        elif num_params == 1:
            return op_class(0.5, wires=wires)
        elif num_params == 2:
            return op_class(0.1, 0.2, wires=wires)
        elif num_params == 3:
            return op_class(0.1, 0.2, 0.3, wires=wires)
        else:
            params = [0.1 * (i + 1) for i in range(num_params)]
            return op_class(*params, wires=wires)

    return _create_operation_from_name


@pytest.fixture(scope="session")
def test_operation_consistency():
    """Fixture providing a function to test operation consistency between graph and non-graph systems.

    Returns a function that compares how an operation is processed by both decomposition systems.
    """

    def _test_operation_consistency(operation, stopping_condition, gate_set=None):
        """Helper to test operation consistency between graph and non-graph systems."""
        from pennylane.devices.preprocess import decompose

        tape = QuantumScript([operation], [qml.expval(qml.Z(0))])

        # Test non-graph decomposition
        disable_graph()
        processed_old = decompose(tape, stopping_condition, gate_set=gate_set)[0][0]

        # Test graph decomposition
        enable_graph()
        processed_new = decompose(tape, stopping_condition, gate_set=gate_set)[0][0]

        return processed_old, processed_new

    return _test_operation_consistency


@pytest.fixture(scope="session")
def enable_graph_decomposition():
    """Fixture that enables graph decomposition for the test session.

    This fixture ensures graph decomposition is enabled at the start of tests
    that require it, and restores the original state after the test.
    """
    # Store original state
    original_state = enabled_graph()

    # Enable graph decomposition
    enable_graph()

    yield

    # Restore original state
    if original_state:
        enable_graph()
    else:
        disable_graph()
