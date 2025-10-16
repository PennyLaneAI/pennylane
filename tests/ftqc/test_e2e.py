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

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")
catalyst = pytest.importorskip("catalyst")

# pylint: disable=wrong-import-position
from catalyst.ftqc import mbqc_pipeline
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import (
    convert_to_mbqc_formalism_pass,
    decompose_graph_state_pass,
    diagonalize_final_measurements_pass,
    measurements_from_samples_pass,
)
from pennylane.ftqc.decomposition import convert_to_mbqc_gateset_pass


def e2e_mbqc_pipeline(qnode):
    """All the transforms currently in the E2E pipeline for MBQC test workload"""
    # qnode = outline_state_evolution_pass(qnode)
    # qnode = split_non_commuting_pass(qnode)
    qnode = diagonalize_final_measurements_pass(qnode)
    qnode = measurements_from_samples_pass(qnode)
    qnode = convert_to_mbqc_gateset_pass(qnode)
    qnode = convert_to_mbqc_formalism_pass(qnode)
    qnode = decompose_graph_state_pass(qnode)
    return qnode


@pytest.mark.usefixtures("enable_graph_decomposition")
@pytest.mark.usefixtures("enable_disable_plxpr")
class TestCircuits:

    @pytest.mark.parametrize("dev_name", ["null.qubit", "lightning.qubit"])
    def test_simple_circuit(self, dev_name, run_filecheck_qjit):
        """Test that a basic circuit is converted as expected and can be executed in the
        E2E pipeline"""

        @qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()], pipelines=mbqc_pipeline())
        @e2e_mbqc_pipeline
        @qml.qnode(qml.device(dev_name, wires=4), shots=3000)
        def circ(x: float, y: float):
            # sanity check that we converted to the MBQC formalism:
            # CHECK-NOT: quantum.custom "RX"
            # CHECK-NOT: quantum.custom "RY"
            # CHECK-NOT: quantum.custom "RZ"
            # CHECK: quantum.custom "Hadamard"
            # CHECK: quantum.custom "CZ"
            # -----------------
            # sanity check that measurements have been updated
            # CHECK-NOT: quantum.namedobs
            # CHECK: quantum.compbasis
            # CHECK: quantum.sample
            qml.RZ(x, 0)
            qml.RZ(y, 1)
            qml.RX(x, 0)
            qml.RY(y, 0)
            return qml.expval(
                qml.PauliY(0)
            )  # , qml.expval(qml.PauliX(0)) (requires split_non_commuting)

        # the MLIR looks reasonable and is executable
        _ = circ(0.84, 0.74)
        run_filecheck_qjit(circ)

    @pytest.mark.parametrize("dev_name", ["null.qubit", "lightning.qubit"])
    def test_if_stmt_circuit(self, dev_name, run_filecheck_qjit):
        """Test that a circuit with an if-statement is converted as expected and can be
        executed in the E2E pipeline"""

        @qml.qjit(
            pass_plugins=[getXDSLPluginAbsolutePath()], pipelines=mbqc_pipeline(), autograph=True
        )
        @e2e_mbqc_pipeline
        @qml.qnode(qml.device(dev_name, wires=3), shots=3000)
        def circ(x: float, y: float):
            # structure is preserved
            # CHECK: scf.if
            # -----------------
            # we converted to the MBQC formalism:
            # CHECK-NOT: quantum.custom "RX"
            # CHECK: quantum.custom "CZ"
            # -----------------
            # measurements have been updated
            # CHECK-NOT: quantum.namedobs
            # CHECK: quantum.sample
            qml.RX(x, 0)
            m = qml.measure(0)
            if m:
                qml.RY(y, 1)
            return qml.expval(
                qml.PauliX(1)
            )  # , qml.expval(qml.PauliY(1)) (requires split_non_commuting)

        # the MLIR looks reasonable and is executable
        _ = circ(0.84, 0.74)
        run_filecheck_qjit(circ)

    @pytest.mark.parametrize("dev_name", ["null.qubit", "lightning.qubit"])
    def test_while_loop_circuit(self, dev_name, run_filecheck_qjit):
        """Test that a circuit with a while-loop is converted as expected and can be
        executed in the E2E pipeline"""

        @qml.qjit(
            pass_plugins=[getXDSLPluginAbsolutePath()], pipelines=mbqc_pipeline(), autograph=True
        )
        @e2e_mbqc_pipeline
        @qml.qnode(qml.device(dev_name, wires=3), shots=3000)
        def circ(x: float):
            # structure is preserved
            # CHECK: scf.while
            # -----------------
            # we converted to the MBQC formalism:
            # CHECK-NOT: quantum.custom "RX"
            # CHECK: quantum.custom "CZ"
            # -----------------
            # measurements have been updated
            # CHECK-NOT: quantum.namedobs
            # CHECK: quantum.sample
            idx = 0
            while idx < 5:
                qml.RX(x, 0)
                idx += 1
            return qml.expval(
                qml.PauliY(1)
            )  # , qml.expval(qml.PauliY(0)) (requires split_non_commuting)

        # the MLIR looks reasonable and is executable
        _ = circ(0.84)
        run_filecheck_qjit(circ)

    @pytest.mark.parametrize("dev_name", ["null.qubit", "lightning.qubit"])
    def test_for_loop_circuit(self, dev_name, run_filecheck_qjit):
        """Test that a circuit with a for-loop is converted as expected and can be
        executed in the E2E pipeline"""

        @qml.qjit(
            pass_plugins=[getXDSLPluginAbsolutePath()], pipelines=mbqc_pipeline(), autograph=True
        )
        @e2e_mbqc_pipeline
        @qml.qnode(qml.device(dev_name, wires=3), shots=3000)
        def circ(x: int):
            # structure is preserved
            # CHECK: scf.for
            # -----------------
            # we converted to the MBQC formalism:
            # CHECK-NOT: quantum.custom "RX"
            # CHECK: quantum.custom "CZ"
            # -----------------
            # measurements have been updated
            # CHECK-NOT: quantum.namedobs
            # CHECK: quantum.sample
            for i in range(x):
                qml.RX(0.12, i)
            return qml.expval(
                qml.PauliX(1)
            )  # , qml.expval(qml.PauliY(1)) (requires split_non_commuting)

        # the MLIR looks reasonable and is executable
        _ = circ(3)
        run_filecheck_qjit(circ)

    @pytest.mark.parametrize("dev_name", ["null.qubit", "lightning.qubit"])
    def test_pretend_xas_workflow(self, dev_name, run_filecheck_qjit):
        """Test that a circuit that mimics the structure and return of the XAS workflow is working. Could be updated to
        test a small-scale verison of the actual XAS workflow instead."""

        @qml.qjit(
            pass_plugins=[getXDSLPluginAbsolutePath()], pipelines=mbqc_pipeline(), autograph=True
        )
        @e2e_mbqc_pipeline
        @qml.qnode(qml.device(dev_name, wires=3), shots=3000)
        def circ(num_steps: int):
            # structure is preserved
            # CHECK: scf.for
            # -----------------
            # we converted to the MBQC formalism:
            # CHECK-NOT: quantum.custom "Rot"
            # CHECK: quantum.custom "CZ"
            # -----------------
            # measurements have been updated
            # CHECK-NOT: quantum.namedobs
            # CHECK: quantum.sample
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

        # the MLIR looks reasonable and is executable
        _ = circ(100)
        run_filecheck_qjit(circ)
