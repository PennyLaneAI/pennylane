# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=no-name-in-module, no-self-use, protected-access
"""Tests for E2E pipeline, to be updated as the final passes are implemented"""
import pytest
from catalyst.ftqc.pipelines import mbqc_pipeline
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import (
    convert_to_mbqc_formalism_pass,
    decompose_graph_state_pass,
    diagonalize_final_measurements_pass,
    measurements_from_samples_pass,
)
from pennylane.ftqc.decomposition import convert_to_mbqc_gateset_pass

# ToDo: add filecheck statements to validate MLIR
# ToDo: parametrize to test on both lightning and null for all


@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.usefixtures("enable_disable_plxpr")
class TestBasicCircuits:

    def test_simple_circuit(self):
        """Test that a basic circuit is converted as expected and can be executed in the
        E2E pipeline"""

        @qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()], pipelines=mbqc_pipeline())
        @decompose_graph_state_pass
        @convert_to_mbqc_formalism_pass
        @convert_to_mbqc_gateset_pass
        @measurements_from_samples_pass
        @diagonalize_final_measurements_pass
        # @split_non_commuting_pass
        # @outline_state_evolution_pass
        @qml.qnode(qml.device("lightning.qubit", wires=4), shots=3000)
        def circ(x: float, y: float):
            qml.RZ(x, 0)
            qml.RZ(y, 1)
            qml.RX(x, 0)
            qml.RY(y, 0)
            return qml.expval(
                qml.PauliY(0)
            )  # , qml.expval(qml.PauliX(0)) (requires split_non_commuting)

        circ(0.84, 0.74)
        print(circ.mlir)

    def test_if_stmt_circuit(self):
        """Test that a circuit with an if-statement is converted as expected and can be
        executed in the E2E pipeline"""

        @qml.qjit(
            pass_plugins=[getXDSLPluginAbsolutePath()], pipelines=mbqc_pipeline(), autograph=True
        )
        @decompose_graph_state_pass
        @convert_to_mbqc_formalism_pass
        @convert_to_mbqc_gateset_pass
        @measurements_from_samples_pass
        @diagonalize_final_measurements_pass
        # @split_non_commuting_pass
        # @outline_state_evolution_pass
        @qml.qnode(qml.device("lightning.qubit", wires=3), shots=3000)
        def circ(x: float, y: float):
            qml.RX(x, 0)
            m = qml.measure(0)
            if m:
                qml.RY(y, 1)
            return qml.expval(
                qml.PauliX(1)
            )  # , qml.expval(qml.PauliY(1)) (requires split_non_commuting)

        circ(0.84, 0.74)

    def test_while_loop_circuit(self):
        """Test that a circuit with a while-loop is converted as expected and can be
        executed in the E2E pipeline"""

        @qml.qjit(
            pass_plugins=[getXDSLPluginAbsolutePath()], pipelines=mbqc_pipeline(), autograph=True
        )
        @decompose_graph_state_pass
        @convert_to_mbqc_formalism_pass
        @convert_to_mbqc_gateset_pass
        @measurements_from_samples_pass
        @diagonalize_final_measurements_pass
        # @split_non_commuting_pass
        # @outline_state_evolution_pass
        @qml.qnode(qml.device("lightning.qubit", wires=3), shots=3000)
        def circ(x: float):
            idx = 0
            while idx < 5:
                qml.RX(x, 0)
                idx += 1

            return qml.expval(
                qml.PauliY(1)
            )  # , qml.expval(qml.PauliY(0)) (requires split_non_commuting)

        circ(0.84)

    def test_for_loop_circuit(self):
        """Test that a circuit with a for-loop is converted as expected and can be
        executed in the E2E pipeline"""

        @qml.qjit(
            pass_plugins=[getXDSLPluginAbsolutePath()], pipelines=mbqc_pipeline(), autograph=True
        )
        @decompose_graph_state_pass
        @convert_to_mbqc_formalism_pass
        @convert_to_mbqc_gateset_pass
        @measurements_from_samples_pass
        @diagonalize_final_measurements_pass
        # @split_non_commuting_pass
        # @outline_state_evolution_pass
        @qml.qnode(qml.device("lightning.qubit", wires=3), shots=3000)
        def circ(x: int):
            for i in range(x):
                qml.H(i)
            return qml.expval(
                qml.PauliX(1)
            )  # , qml.expval(qml.PauliY(1)) (requires split_non_commuting)

        circ(3)

    def test_mock_xas_workflow(self):
        """Test that a circuit that mimics the structure and return of the XAS workflow is working. To be updated to
        test a small-scale verison of the actual XAS workflow instead."""

        @qml.qjit(
            pass_plugins=[getXDSLPluginAbsolutePath()], pipelines=mbqc_pipeline(), autograph=True
        )
        @decompose_graph_state_pass
        @convert_to_mbqc_formalism_pass
        @convert_to_mbqc_gateset_pass
        @measurements_from_samples_pass
        @diagonalize_final_measurements_pass
        # @split_non_commuting_pass
        # @outline_state_evolution_pass
        @qml.qnode(qml.device("lightning.qubit", wires=3), shots=3000)
        def circ(num_steps: int):

            qml.Hadamard(0)

            for _ in range(num_steps):
                qml.Rot(0.12, 0.56, 0.78, 0)
                qml.Rot(2.3, 0.34, 0.67, 1)
                qml.Rot(1.2, 0.89, 1.9, 2)
                qml.ctrl(qml.MultiRZ(0.345, wires=[1, 2]), control=0)
                qml.PhaseShift(-0.567, wires=0)

            return qml.expval(
                qml.PauliX(0)
            )  # , qml.expval(qml.PauliY(0)) # requires split_non_commuting

        circ(100)
