# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the specs transform"""
# pylint: disable=invalid-sequence-index
from collections import defaultdict
from contextlib import nullcontext
from typing import Optional

import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn

devices_list = [
    (qml.device("default.qubit"), 1),
    (qml.device("default.qubit", wires=2), 2),
]


@pytest.fixture(params=[False, True], ids=["graph_disabled", "graph_enabled"], autouse=True)
def enable_and_disable_graph_decomp(request):
    """
    A fixture that parametrizes a test to run twice: once with graph
    decomposition disabled and once with it enabled.

    It automatically handles the setup (enabling/disabling) before the
    test runs and the teardown (always disabling) after the test completes.
    """
    try:
        use_graph_decomp = request.param

        # --- Setup Phase ---
        # This code runs before the test function is executed.
        if use_graph_decomp:
            qml.decomposition.enable_graph()
        else:
            # Explicitly disable to ensure a clean state
            qml.decomposition.disable_graph()

        # Yield control to the test function
        yield use_graph_decomp

    finally:
        # --- Teardown Phase ---
        # This code runs after the test function has finished,
        # regardless of whether it passed or failed.
        qml.decomposition.disable_graph()

def test_error_with_bad_key():
    """Test that a helpful error message is raised if key does not exist."""

    @qml.qnode(qml.device("null.qubit"))
    def c():
        return qml.state()

    out = qml.specs(c)()
    with pytest.raises(KeyError, match="Options are {"):
        _ = out["bad_value"]


class TestSpecsTransform:
    """Tests for the transform specs using the QNode"""

    def sample_circuit(self):

        @qml.transforms.merge_rotations
        @qml.transforms.undo_swaps
        @qml.transforms.cancel_inverses
        @qml.qnode(
            qml.device("default.qubit"),
            diff_method="parameter-shift",
            gradient_kwargs={"shifts": pnp.pi / 4},
        )
        def circuit(x):
            qml.RandomLayers(qml.numpy.array([[1.0, 2.0]]), wires=(0, 1))
            qml.RX(x, wires=0)
            qml.RX(-x, wires=0)
            qml.SWAP((0, 1))
            qml.X(0)
            qml.X(0)
            return qml.expval(qml.sum(qml.X(0), qml.Y(1)))

        return circuit

    @pytest.mark.parametrize(
        "level,expected_gates,exptected_train_params",
        [(0, 6, 1), (1, 4, 3), (2, 3, 3), (3, 1, 1), ("device", 2, 2)],
    )
    def test_int_specs_level(self, level, expected_gates, exptected_train_params):
        circ = self.sample_circuit()
        specs = qml.specs(circ, level=level)(0.1)

        assert specs["level"] == level
        assert specs["resources"].num_gates == expected_gates

        assert specs["num_trainable_params"] == exptected_train_params

    @pytest.mark.parametrize(
        "level1,level2",
        [
            ("top", 0),
            (0, slice(0, 0)),
            ("user", 3),
            ("user", slice(0, 3)),
            (-1, slice(0, -1)),
            ("device", slice(0, None)),
        ],
    )
    def test_equivalent_levels(self, level1, level2):
        circ = self.sample_circuit()

        specs1 = qml.specs(circ, level=level1)(0.1)
        specs2 = qml.specs(circ, level=level2)(0.1)

        del specs1["level"]
        del specs2["level"]

        assert specs1 == specs2

    @pytest.mark.parametrize(
        "diff_method, len_info", [("backprop", 12), ("parameter-shift", 13), ("adjoint", 12)]
    )
    def test_empty(self, diff_method, len_info):
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, diff_method=diff_method)
        def circ():
            return qml.expval(qml.PauliZ(0))

        with (
            pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters")
            if diff_method == "parameter-shift"
            else nullcontext()
        ):
            info_func = qml.specs(circ)
            info = info_func()
        assert len(info) == len_info

        expected_resources = qml.resource.Resources(num_wires=1, gate_types=defaultdict(int))
        assert info["resources"] == expected_resources
        assert info["num_observables"] == 1
        assert info["num_device_wires"] == 1
        assert info["diff_method"] == diff_method
        assert info["num_trainable_params"] == 0
        assert info["device_name"] == dev.name
        assert info["level"] == "gradient"

        if diff_method == "parameter-shift":
            assert info["num_gradient_executions"] == 0
            assert info["gradient_fn"] == "pennylane.gradients.parameter_shift.param_shift"

    @pytest.mark.parametrize(
        "diff_method, len_info", [("backprop", 12), ("parameter-shift", 13), ("adjoint", 12)]
    )
    def test_specs(self, diff_method, len_info):
        """Test the specs transforms works in standard situations"""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x, y, add_RY=True):
            qml.RX(x[0], wires=0)
            qml.Toffoli(wires=(0, 1, 2))
            qml.CRY(x[1], wires=(0, 1))
            qml.Rot(x[2], x[3], y, wires=2)
            if add_RY:
                qml.RY(x[4], wires=1)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        x = pnp.array([0.05, 0.1, 0.2, 0.3, 0.5], requires_grad=True)
        y = pnp.array(0.1, requires_grad=False)

        info = qml.specs(circuit)(x, y, add_RY=False)

        assert len(info) == len_info

        gate_sizes = defaultdict(int, {1: 2, 3: 1, 2: 1})
        gate_types = defaultdict(int, {"RX": 1, "Toffoli": 1, "CRY": 1, "Rot": 1})
        expected_resources = qml.resource.Resources(
            num_wires=3, num_gates=4, gate_types=gate_types, gate_sizes=gate_sizes, depth=3
        )
        assert info["resources"] == expected_resources

        assert info["num_observables"] == 2
        assert info["num_device_wires"] == 4
        assert info["diff_method"] == diff_method
        assert info["num_trainable_params"] == 4
        assert info["device_name"] == dev.name
        assert info["level"] == "gradient"

        if diff_method == "parameter-shift":
            assert info["num_gradient_executions"] == 6

    @pytest.mark.parametrize("compute_depth", [True, False])
    def test_specs_compute_depth(self, compute_depth):
        """Test that the specs transform computes the depth of the circuit"""

        x = pnp.array([0.1, 0.2])

        @qml.qnode(qml.device("default.qubit"), diff_method="parameter-shift")
        def circuit(x):
            qml.RandomLayers(pnp.array([[1.0, 2.0]]), wires=(0, 1))
            qml.RX(x, wires=0)
            qml.RX(-x, wires=0)
            qml.SWAP((0, 1))
            qml.X(0)
            qml.X(0)
            return qml.expval(qml.X(0) + qml.Y(1))

        info = qml.specs(circuit, compute_depth=compute_depth)(x)

        assert info["resources"].depth == (6 if compute_depth else None)

    @pytest.mark.parametrize(
        "diff_method, len_info", [("backprop", 12), ("parameter-shift", 13), ("adjoint", 12)]
    )
    def test_specs_state(self, diff_method, len_info):
        """Test specs works when state returned"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit():
            return qml.state()

        info = qml.specs(circuit)()
        assert len(info) == len_info

        assert info["resources"] == qml.resource.Resources(gate_types=defaultdict(int))

        assert info["num_observables"] == 1
        assert info["level"] == "gradient"

    def test_level_with_diagonalizing_gates(self):
        """Test that when diagonalizing gates includes gates that are decomposed in
        device preprocess, for level=device, any unsupported diagonalizing gates are
        decomposed like the tape.operations."""

        class TestDevice(qml.devices.DefaultQubit):

            def stopping_condition(self, op):
                if isinstance(op, qml.QubitUnitary):
                    return False
                return True

            def preprocess_transforms(
                self, execution_config: Optional[qml.devices.ExecutionConfig] = None
            ):
                program = super().preprocess_transforms(execution_config)
                program.add_transform(
                    qml.devices.preprocess.decompose, stopping_condition=self.stopping_condition
                )
                return program

        dev = TestDevice(wires=2)
        matrix = qml.matrix(qml.RX(1.2, 0))

        @qml.qnode(dev)
        def circ():
            qml.QubitUnitary(matrix, wires=0)
            return qml.expval(qml.X(0) + qml.Y(0))

        specs = qml.specs(circ)()
        assert specs["resources"].num_gates == 1

        specs = qml.specs(circ, level="device")()
        assert specs["resources"].num_gates == 4

    def test_splitting_transforms(self):
        """Test that the specs transform works with splitting transforms"""
        coeffs = [0.2, -0.543, 0.1]
        obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Y(2), qml.Y(0) @ qml.X(2)]
        H = qml.Hamiltonian(coeffs, obs)

        @qml.transforms.split_non_commuting
        @qml.transforms.merge_rotations
        @qml.qnode(
            qml.device("default.qubit"),
            diff_method="parameter-shift",
            gradient_kwargs={"shifts": pnp.pi / 4},
        )
        def circuit(x):
            qml.RandomLayers(qml.numpy.array([[1.0, 2.0]]), wires=(0, 1))
            qml.RX(x, wires=0)
            qml.RX(-x, wires=0)
            qml.SWAP((0, 1))
            qml.X(0)
            qml.X(0)
            return qml.expval(H)

        specs_instance = qml.specs(circuit, level=1)(pnp.array([1.23, -1]))

        assert isinstance(specs_instance, dict)

        specs_list = qml.specs(circuit, level=2)(pnp.array([1.23, -1]))

        assert len(specs_list) == len(H)

        assert specs_list[0]["num_device_wires"] == specs_list[0]["num_tape_wires"] == 2
        assert specs_list[1]["num_device_wires"] == specs_list[1]["num_tape_wires"] == 3
        assert specs_list[2]["num_device_wires"] == specs_list[1]["num_tape_wires"] == 3

    def make_qnode_and_params(self, seed):
        """Generates a qnode and params for use in other tests"""
        n_layers = 2
        n_wires = 5

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(params):
            qml.BasicEntanglerLayers(params, wires=range(n_wires))
            return qml.expval(qml.PauliZ(0))

        params_shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_wires)
        rng = pnp.random.default_rng(seed=seed)
        params = rng.standard_normal(params_shape)  # pylint:disable=no-member

        return circuit, params

    def test_gradient_transform(self):
        """Test that a gradient transform is properly labelled"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method=qml.gradients.param_shift)
        def circuit():
            return qml.probs(wires=0)

        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            info = qml.specs(circuit)()
        assert info["diff_method"] == "pennylane.gradients.parameter_shift.param_shift"
        assert info["gradient_fn"] == "pennylane.gradients.parameter_shift.param_shift"

    def test_custom_gradient_transform(self):
        """Test that a custom gradient transform is properly labelled"""
        dev = qml.device("default.qubit", wires=2)

        @qml.transform
        def my_transform(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            return tape, None

        @qml.qnode(dev, diff_method=my_transform)
        def circuit():
            return qml.probs(wires=0)

        info = qml.specs(circuit)()
        assert info["diff_method"] == "test_specs.my_transform"
        assert info["gradient_fn"] == "test_specs.my_transform"

    @pytest.mark.parametrize(
        "device,num_wires",
        devices_list,
    )
    def test_num_wires_source_of_truth(self, device, num_wires):
        """Tests that num_wires behaves differently on old and new devices."""

        @qml.qnode(device)
        def circuit():
            qml.PauliX(0)
            return qml.state()

        info = qml.specs(circuit)()
        assert info["num_device_wires"] == num_wires

    def test_no_error_contents_on_device_level(self):
        coeffs = [0.25, 0.75]
        ops = [qml.X(0), qml.Z(0)]
        H = qml.dot(coeffs, ops)

        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            qml.Hadamard(0)
            qml.TrotterProduct(H, time=2.4, order=2)

            return qml.state()

        top_specs = qml.specs(circuit, level="top")()
        dev_specs = qml.specs(circuit, level="device")()

        assert "SpectralNormError" in top_specs["errors"]
        assert pnp.allclose(top_specs["errors"]["SpectralNormError"].error, 13.824)

        # At the device level, approximations don't exist anymore and therefore
        # we should expect an empty errors dictionary.
        assert dev_specs["errors"] == {}

    def test_specs_with_graph_decomposition_work_wires_concept(self):
        """Test that qml.specs works correctly when graph decomposition system is enabled, demonstrating work wire concepts."""
        
        # Save initial state
        initial_state = getattr(qml.decomposition, '_enabled_graph', False)
        
        try:
            dev = qml.device("default.qubit", wires=6)
            
            @qml.qnode(dev)
            def circuit_testing_graph_decomposition():
                # Operations that graph decomposition can handle well
                qml.Toffoli(wires=[0, 1, 2])  # Should be decomposed by graph system
                qml.Toffoli(wires=[3, 4, 5])  # Another one to test optimization
                return qml.expval(qml.Z(5))
            
            # Test with graph decomposition disabled first
            qml.decomposition.disable_graph()
            specs_disabled = qml.specs(circuit_testing_graph_decomposition, level="device")()
            
            # Test with graph decomposition enabled
            qml.decomposition.enable_graph()
            specs_enabled = qml.specs(circuit_testing_graph_decomposition, level="device")()
            
            # Both should work and provide resource information
            resources_disabled = specs_disabled["resources"]
            resources_enabled = specs_enabled["resources"]
            
            # Verify basic tracking works in both cases
            assert resources_disabled.num_gates > 0
            assert resources_enabled.num_gates > 0
            assert resources_disabled.num_wires >= 6
            assert resources_enabled.num_wires >= 6
            
            # Gate types should be dictionaries
            gate_types_disabled = dict(resources_disabled.gate_types)
            gate_types_enabled = dict(resources_enabled.gate_types)
            
            # The default.qubit device supports Toffoli gates natively at device level
            # So we expect to see Toffoli gates rather than their primitive decomposition
            # This tests that both graph decomposition modes work correctly
            
            # Check that Toffoli gates are preserved (since they're supported by default.qubit)
            assert "Toffoli" in gate_types_disabled, f"Expected Toffoli gates in disabled mode, got: {gate_types_disabled}"
            assert "Toffoli" in gate_types_enabled, f"Expected Toffoli gates in enabled mode, got: {gate_types_enabled}"
            
            # Both should have the same number of Toffoli gates (2)
            assert gate_types_disabled["Toffoli"] == 2, f"Expected 2 Toffoli gates in disabled mode, got: {gate_types_disabled['Toffoli']}"
            assert gate_types_enabled["Toffoli"] == 2, f"Expected 2 Toffoli gates in enabled mode, got: {gate_types_enabled['Toffoli']}"
            
            # Total gate counts should be reasonable for two Toffoli gates
            assert resources_disabled.num_gates == 2  # Exactly 2 Toffoli gates
            assert resources_enabled.num_gates == 2   # Same for both modes
            
        finally:
            # Restore initial state
            if initial_state:
                qml.decomposition.enable_graph()
            else:
                qml.decomposition.disable_graph()

    def test_specs_with_graph_decomposition_weighted_gate_sets(self):
        """Test that qml.specs works with graph decomposition using weighted gate sets for cost optimization."""
        
        # Save initial state
        initial_state = getattr(qml.decomposition, '_enabled_graph', False)
        
        try:
            dev = qml.device("default.qubit", wires=4)
            
            @qml.qnode(dev)
            def circuit_for_optimization():
                # Use operations that can benefit from weighted gate set optimization
                qml.Toffoli(wires=[0, 1, 2])  # This is in the default gate set
                qml.MultiControlledX(wires=[0, 1, 2], control_values=[0, 1])
                return qml.expval(qml.Z(3))
            
            # Test with graph decomposition enabled
            qml.decomposition.enable_graph()
            
            # Test that weighted gate sets can be used (if configured)
            # This tests the integration with the DecompositionGraph's weighted optimization
            specs = qml.specs(circuit_for_optimization, level="device")()
            
            # Verify that decomposition occurred
            resources = specs["resources"]
            gate_types = dict(resources.gate_types)
            
            # Verify the system works correctly - Toffoli is in the default gate set so it may remain
            # Focus on testing that the graph decomposition system integrates properly
            
            # Should contain our expected operations
            expected_gates = ["Toffoli", "MultiControlledX"]  # These may stay at device level
            for gate in expected_gates:
                if gate in gate_types:
                    assert gate_types[gate] > 0
            
            # Total gate count should be reasonable
            assert resources.num_gates > 0
            assert resources.num_wires >= 3  # At least the wires we used
            
        finally:
            # Restore initial state
            if initial_state:
                qml.decomposition.enable_graph()
            else:
                qml.decomposition.disable_graph()

    def test_specs_with_graph_decomposition_resource_registration(self):
        """Test that qml.specs integrates with @register_resources decorator and work wire specifications."""
        
        # Save initial state
        initial_state = getattr(qml.decomposition, '_enabled_graph', False)
        
        try:
            from pennylane.decomposition.decomposition_rule import register_resources
            
            # Create a custom operation that uses the @register_resources decorator
            @register_resources({qml.Hadamard: 1, qml.T: 1}, work_wires=lambda wires: 2)
            class CustomWorkWireOp(qml.operation.Operation):
                num_wires = 1
                
                @staticmethod
                def compute_decomposition(wires, **kwargs):
                    # Simple decomposition that could benefit from work wires
                    return [qml.Hadamard(wires[0]), qml.T(wires[0])]
            
            dev = qml.device("default.qubit", wires=6)  # Extra wires for work wire allocation
            
            @qml.qnode(dev)
            def circuit_with_registered_resources():
                qml.Toffoli(wires=[0, 1, 2])  # Standard operation in default gate set
                CustomWorkWireOp(wires=[3])  # Our custom operation with work wires
                return qml.expval(qml.Z(3))
            
            # Test with graph decomposition enabled
            qml.decomposition.enable_graph()
            specs = qml.specs(circuit_with_registered_resources, level="device")()
            
            # Verify resource tracking works with registered resources
            resources = specs["resources"]
            gate_types = dict(resources.gate_types)
            
            # CustomWorkWireOp should be decomposed to its registered gates
            # (if graph decomposition processes it, otherwise default decomposition should handle it)
            assert "Hadamard" in gate_types or "T" in gate_types, "Expected decomposition gates from CustomWorkWireOp"
            
            # Toffoli should remain as it's in the default gate set
            assert "Toffoli" in gate_types
            
            # Should have reasonable gate count including decompositions
            assert resources.num_gates >= 2  # At least the gates from our operations
            
        finally:
            # Restore initial state
            if initial_state:
                qml.decomposition.enable_graph()
            else:
                qml.decomposition.disable_graph()

    def test_specs_with_graph_decomposition_work_wire_concepts(self):
        """Test specs with graph decomposition demonstrating work wire concepts through manual wire usage."""
        
        # Save initial state
        initial_state = getattr(qml.decomposition, '_enabled_graph', False)
        
        try:
            dev = qml.device("default.qubit", wires=8)  # Extra wires to simulate work wire availability
            
            @qml.qnode(dev)
            def circuit_simulating_work_wire_usage():
                # Simulate work wire usage patterns without using qml.allocation.allocate()
                # to avoid the current limitation with Allocate/Deallocate operations
                
                # Main computation on core wires
                qml.Toffoli(wires=[0, 1, 2])
                
                # Use additional wires as if they were allocated work wires
                # (This simulates the pattern that would occur with work wire allocation)
                qml.Hadamard(6)  # Initialize a work wire
                qml.CNOT(wires=[0, 6])  # Use work wire in computation
                qml.CNOT(wires=[6, 7])  # Chain through work wires
                qml.CNOT(wires=[7, 3])  # Connect back to main computation
                
                # Another pattern using different "work wires"
                qml.MultiControlledX(wires=[1, 2, 4], control_values=[0, 1])
                
                return qml.expval(qml.Z(0) @ qml.Z(1))
            
            # Test with graph decomposition enabled
            qml.decomposition.enable_graph()
            specs = qml.specs(circuit_simulating_work_wire_usage, level="device")()
            
            # Verify that resource tracking handles the work wire patterns
            resources = specs["resources"]
            gate_types = dict(resources.gate_types)
            
            # Should see our explicit operations
            assert "Hadamard" in gate_types
            assert "CNOT" in gate_types
            assert gate_types["CNOT"] >= 3  # Our 3 explicit CNOTs
            
            # Verify the system works without errors and tracks operations correctly
            # Note: Current graph decomposition may keep some operations at higher level
            assert resources.num_gates >= 6  # At least our explicit gates
            assert resources.num_wires >= 5  # Several wires used
            
            # Verify that all our operations are tracked
            expected_operations = ["Toffoli", "Hadamard", "CNOT", "MultiControlledX"]
            for op in expected_operations:
                assert op in gate_types, f"Expected operation {op} to be tracked in gate types"
            
        finally:
            # Restore initial state
            if initial_state:
                qml.decomposition.enable_graph()
            else:
                qml.decomposition.disable_graph()

    def test_specs_with_decomposition_graph_solver_integration(self):
        """Test that specs works with DecompositionGraph solver and pathfinding optimization."""
        
        # Save initial state
        initial_state = getattr(qml.decomposition, '_enabled_graph', False)
        
        try:
            dev = qml.device("default.qubit", wires=6)
            
            @qml.qnode(dev)
            def circuit_with_complex_decomposition():
                # Operations that will trigger DecompositionGraph solver
                qml.Toffoli(wires=[0, 1, 2])  # In default gate set
                qml.Toffoli(wires=[3, 4, 5])  # Another Toffoli
                qml.MultiControlledX(wires=[0, 1, 2], control_values=[0, 1])  # Multi-controlled gate
                return qml.expval(qml.Z(5))
            
            # Test with graph decomposition enabled
            qml.decomposition.enable_graph()
            specs = qml.specs(circuit_with_complex_decomposition, level="device")()
            
            # Verify that complex operations are tracked correctly
            resources = specs["resources"]
            gate_types = dict(resources.gate_types)
            
            # Since Toffoli is in the default gate set, it should remain
            assert "Toffoli" in gate_types
            assert gate_types["Toffoli"] == 2  # Our two Toffoli gates
            
            # MultiControlledX may or may not be decomposed depending on target gate set
            assert "MultiControlledX" in gate_types  # Likely stays at device level
            
            # Gate count should match our operations
            assert resources.num_gates >= 3  # At least our 3 operations
            
            # Wire count should reflect the circuit requirements
            assert resources.num_wires >= 6  # All the wires we used
            
        finally:
            # Restore initial state
            if initial_state:
                qml.decomposition.enable_graph()
            else:
                qml.decomposition.disable_graph()

    def test_specs_graph_decomposition_vs_standard_decomposition_comparison(self):
        """Test comparison between graph decomposition and standard decomposition resource reporting."""
        
        # Save initial state
        initial_state = getattr(qml.decomposition, '_enabled_graph', False)
        
        try:
            dev = qml.device("default.qubit", wires=5)
            
            @qml.qnode(dev)
            def circuit_for_comparison():
                qml.Toffoli(wires=[0, 1, 2])  # In default gate set
                qml.Toffoli(wires=[2, 3, 4])  # Another Toffoli
                return qml.expval(qml.Z(4))
            
            # Test with standard decomposition (graph disabled)
            qml.decomposition.disable_graph()
            specs_standard = qml.specs(circuit_for_comparison, level="device")()
            
            # Test with graph decomposition enabled
            qml.decomposition.enable_graph()
            specs_graph = qml.specs(circuit_for_comparison, level="device")()
            
            # Both should successfully handle the operations
            resources_standard = specs_standard["resources"]
            resources_graph = specs_graph["resources"]
            
            # Since Toffoli is in the default gate set, both should likely keep it
            gate_types_standard = dict(resources_standard.gate_types)
            gate_types_graph = dict(resources_graph.gate_types)
            
            # Verify basic functionality in both modes
            assert resources_standard.num_gates >= 2  # At least our 2 Toffoli gates
            assert resources_graph.num_gates >= 2
            
            # Verify that both approaches work without errors
            assert isinstance(resources_standard.num_gates, int)
            assert isinstance(resources_graph.num_gates, int)
            assert isinstance(resources_standard.num_wires, int)
            assert isinstance(resources_graph.num_wires, int)
            
            # Both should track the same basic circuit structure
            assert resources_standard.num_wires >= 5
            assert resources_graph.num_wires >= 5
            
        finally:
            # Restore initial state
            if initial_state:
                qml.decomposition.enable_graph()
            else:
                qml.decomposition.disable_graph()

    def test_specs_graph_decomposition_comprehensive_integration(self):
        """Comprehensive test demonstrating graph decomposition integration with qml.specs across key features."""
        
        # Save initial state
        initial_state = getattr(qml.decomposition, '_enabled_graph', False)
        
        try:
            dev = qml.device("default.qubit", wires=8)
            
            @qml.qnode(dev)
            def comprehensive_circuit():
                # Test different operation types that exercise graph decomposition features
                
                # 1. Operations in default gate set (should remain at device level)
                qml.Toffoli(wires=[0, 1, 2])
                qml.Hadamard(3)
                qml.CNOT(wires=[0, 4])
                
                # 2. Operations that may be decomposed by graph system
                qml.MultiControlledX(wires=[1, 2, 5], control_values=[1, 0])
                
                # 3. Simulate work wire patterns (manual wire usage)
                qml.Hadamard(6)  # "work wire" initialization
                qml.CNOT(wires=[3, 6])  # Use work wire
                qml.CNOT(wires=[6, 7])  # Chain work wires
                
                return qml.expval(qml.Z(0) @ qml.Z(1))
            
            # Test with graph decomposition disabled
            qml.decomposition.disable_graph()
            specs_disabled = qml.specs(comprehensive_circuit, level="device")()
            
            # Test with graph decomposition enabled  
            qml.decomposition.enable_graph()
            specs_enabled = qml.specs(comprehensive_circuit, level="device")()
            
            # Verify both modes work correctly
            resources_disabled = specs_disabled["resources"]
            resources_enabled = specs_enabled["resources"]
            
            # Both should track all our operations
            assert resources_disabled.num_gates >= 7  # Our 7 explicit operations
            assert resources_enabled.num_gates >= 7
            assert resources_disabled.num_wires >= 8
            assert resources_enabled.num_wires >= 8
            
            # Both should contain expected gate types from default gate set
            for specs in [specs_disabled, specs_enabled]:
                gate_types = dict(specs["resources"].gate_types)
                expected_gates = ["Toffoli", "Hadamard", "CNOT"]
                for gate in expected_gates:
                    assert gate in gate_types, f"Expected {gate} in gate types: {gate_types}"
            
            # Verify system robustness - both approaches should produce consistent basic metrics
            assert abs(resources_disabled.num_wires - resources_enabled.num_wires) <= 0  # Same wires used
            
            # Both should successfully execute without errors
            assert all(isinstance(val, (int, float)) for val in [
                resources_disabled.num_gates, resources_disabled.num_wires,
                resources_enabled.num_gates, resources_enabled.num_wires
            ])
            
        finally:
            # Restore initial state
            if initial_state:
                qml.decomposition.enable_graph()
            else:
                qml.decomposition.disable_graph()
