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
"""Unit test module for the outline state evolution transform"""
import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")
catalyst = pytest.importorskip("catalyst")

# pylint: disable=wrong-import-position
from catalyst.ftqc import mbqc_pipeline
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import (
    OutlineStateEvolutionPass,
    convert_to_mbqc_formalism_pass,
    decompose_graph_state_pass,
    measurements_from_samples_pass,
    outline_state_evolution_pass,
)
from pennylane.ftqc import RotXZX


class TestOutlineStateEvolutionPass:
    """Unit tests for OutlineStateEvolutionPass."""

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_gates_in_mbqc_gate_set_e2e_wo_for_loop(self):
        """Test that the outline_state_evolution_pass end to end on null.qubit."""
        dev = qml.device("null.qubit", wires=1000)

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
            pipelines=mbqc_pipeline(),
            autograph=True,
        )
        @decompose_graph_state_pass
        @convert_to_mbqc_formalism_pass
        @measurements_from_samples_pass
        @outline_state_evolution_pass
        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            qml.H(0)
            qml.S(1)
            RotXZX(0.1, 0.2, 0.3, wires=[2])
            qml.RZ(phi=0.1, wires=[3])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Z(wires=0))

        res = circuit()
        assert res == 1.0

    # TODOs: we need to support the program structure maintain.
    @pytest.mark.xfail(
        reason="program structure is not support for the current outline_state_evolution_pass"
    )
    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_gates_in_mbqc_gate_set_e2e_for_loop(self):
        """Test that the outline_state_evolution_pass end to end on null.qubit."""
        dev = qml.device("null.qubit", wires=1000)

        @qml.for_loop(0, 1000, 1)
        def loop_func(i):
            qml.H(i)
            qml.S(i)
            RotXZX(0.1, 0.2, 0.3, wires=[i])
            qml.RZ(phi=0.1, wires=[i])

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
            pipelines=mbqc_pipeline(),
            autograph=True,
        )
        @decompose_graph_state_pass
        @convert_to_mbqc_formalism_pass
        @measurements_from_samples_pass
        @outline_state_evolution_pass
        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            loop_func()
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Z(wires=0))

        res = circuit()
        assert res == 1.0
