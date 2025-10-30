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
    diagonalize_final_measurements_pass,
    measurements_from_samples_pass,
    outline_state_evolution_pass,
)
from pennylane.ftqc import RotXZX


@qml.while_loop(lambda i: i < 1000)
def _while_for(i):
    qml.H(i)
    qml.S(i)
    RotXZX(0.1, 0.2, 0.3, wires=[i])
    qml.RZ(phi=0.1, wires=[i])
    i = i + 1
    return i


class TestOutlineStateEvolutionPass:
    """Unit tests for OutlineStateEvolutionPass."""

    def test_func_wo_qnode_attr(self, run_filecheck):
        """Test outline state evolution pass is not applied to a func without a qnode attribute."""
        program = """
            module @module_circuit {
                func.func public @circuit() -> tensor<f64> {
                    %0 = arith.constant 0 : i64
                    quantum.device shots(%0) ["", "", ""]
                    %1 = quantum.alloc( 50) : !quantum.reg
                    %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
                    // CHECK: quantum.custom "PauliX"()
                    // CHECK-NOT: call @circuit.state_evolution
                    %out_qubits = quantum.custom "PauliX"() %2 : !quantum.bit
                    %3 = quantum.namedobs %out_qubits[ PauliX] : !quantum.obs
                    %4 = quantum.expval %3 : f64
                    %from_elements = tensor.from_elements %4 : tensor<f64>
                    %5 = quantum.insert %1[ 0], %out_qubits : !quantum.reg, !quantum.bit
                    quantum.dealloc %5 : !quantum.reg
                    quantum.device_release
                    return %from_elements : tensor<f64>
                }
                // CHECK-NOT: func.func public @circuit.state_evolution
            }
        """

        pipeline = (OutlineStateEvolutionPass(),)
        run_filecheck(program, pipeline)

    def test_func_w_qnode_attr(self, run_filecheck):
        """Test outline state evolution pass would be applied to a func with a qnode attribute."""
        program = """
            module @module_circuit {
                func.func public @circuit() -> tensor<f64> attributes {qnode} {
                    %0 = arith.constant 0 : i64
                    quantum.device shots(%0) ["", "", ""]
                    %1 = quantum.alloc( 50) : !quantum.reg
                    %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
                    // CHECK-NOT: quantum.custom "PauliX"()
                    // CHECK: call @circuit.state_evolution
                    %out_qubits = quantum.custom "PauliX"() %2 : !quantum.bit
                    %3 = quantum.namedobs %out_qubits[ PauliX] : !quantum.obs
                    %4 = quantum.expval %3 : f64
                    %from_elements = tensor.from_elements %4 : tensor<f64>
                    %5 = quantum.insert %1[ 0], %out_qubits : !quantum.reg, !quantum.bit
                    quantum.dealloc %5 : !quantum.reg
                    quantum.device_release
                    return %from_elements : tensor<f64>
                }
                // CHECK: func.func public @circuit.state_evolution
            }
        """

        pipeline = (OutlineStateEvolutionPass(),)
        run_filecheck(program, pipeline)

    def test_multiple_func_w_qnode_attr(self, run_filecheck):
        """Test outline state evolution pass would be applied to a func with a qnode attribute."""
        program = """
            module @module_circuit {
                func.func public @circuit1() -> tensor<f64> attributes {qnode} {
                    %0 = arith.constant 0 : i64
                    quantum.device shots(%0) ["", "", ""]
                    %1 = quantum.alloc( 50) : !quantum.reg
                    %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
                    // CHECK: call @circuit1.state_evolution
                    %out_qubits = quantum.custom "PauliX"() %2 : !quantum.bit
                    %3 = quantum.namedobs %out_qubits[ PauliX] : !quantum.obs
                    %4 = quantum.expval %3 : f64
                    %from_elements = tensor.from_elements %4 : tensor<f64>
                    %5 = quantum.insert %1[ 0], %out_qubits : !quantum.reg, !quantum.bit
                    quantum.dealloc %5 : !quantum.reg
                    quantum.device_release
                    return %from_elements : tensor<f64>
                }
                func.func public @circuit2() -> tensor<f64> attributes {qnode} {
                    %0 = arith.constant 0 : i64
                    quantum.device shots(%0) ["", "", ""]
                    %1 = quantum.alloc( 50) : !quantum.reg
                    %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
                    // CHECK: call @circuit2.state_evolution
                    %out_qubits = quantum.custom "PauliX"() %2 : !quantum.bit
                    %3 = quantum.namedobs %out_qubits[ PauliX] : !quantum.obs
                    %4 = quantum.expval %3 : f64
                    %from_elements = tensor.from_elements %4 : tensor<f64>
                    %5 = quantum.insert %1[ 0], %out_qubits : !quantum.reg, !quantum.bit
                    quantum.dealloc %5 : !quantum.reg
                    quantum.device_release
                    return %from_elements : tensor<f64>
                }
                // CHECK: func.func public @circuit1.state_evolution
                // CHECK: func.func public @circuit2.state_evolution
            }
        """

        pipeline = (OutlineStateEvolutionPass(),)
        run_filecheck(program, pipeline)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_outline_state_evolution_no_error(self):
        """Test outline_state_evolution_pass does not raise error for circuit with classical operations only."""

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
        )
        @outline_state_evolution_pass
        def circuit(x, y):
            return x * y + 5

        circuit(1, 4)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_outline_state_evolution_no_terminal_op_error(self):
        """Test outline_state_evolution_pass raises error when no terminal_boundary_op is found in circuit with quantum operation."""
        # TODOs: we can resolve this issue if the boundary op is inserted when the program is captured.
        dev = qml.device("null.qubit", wires=10)

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
        )
        @outline_state_evolution_pass
        @qml.qnode(dev)
        def circuit():
            return qml.state()

        with pytest.raises(
            RuntimeError, match="A terminal_boundary_op op is not found in the circuit."
        ):
            circuit()

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_outline_state_evolution_pass_only(self, run_filecheck_qjit):
        """Test the outline_state_evolution_pass only."""
        dev = qml.device("lightning.qubit", wires=1000)

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
        )
        @outline_state_evolution_pass
        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            # CHECK-LABEL: func.func public @circuit()
            # CHECK-NOT: scf.while
            # CHECK-NOT: quantum.custom "Hadamard"()
            # CHECK-NOT: quantum.custom "S"()
            # CHECK-NOT: quantum.custom "RotXZX"
            # CHECK-NOT: quantum.custom "RZ"
            # CHECK-NOT: quantum.custom "CNOT"
            # CHECK-NOT: func.func public @circuit.state_evolution
            # CHECK: quantum.alloc
            # CHECK-NEXT: func.call @circuit.state_evolution
            # CHECK-LABEL: func.func public @circuit.state_evolution
            # CHECK-NOT: quantum.alloc
            # CHECK-NOT: quantum.namedobs
            # CHECK: scf.while
            # CHECK: quantum.custom "Hadamard"()
            # CHECK: quantum.custom "S"()
            # CHECK: quantum.custom "RotXZX"
            # CHECK: quantum.custom "RZ"
            # CHECK: quantum.custom "CNOT"
            _while_for(0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Z(wires=0))

        run_filecheck_qjit(circuit)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_outline_state_evolution_pass_with_convert_to_mbqc_formalism(self, run_filecheck_qjit):
        """Test if the outline_state_evolution_pass works with the convert-to-mbqc-formalism pass on lightning.qubit."""
        dev = qml.device("lightning.qubit", wires=1000)

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
            pipelines=mbqc_pipeline(),
        )
        @decompose_graph_state_pass
        @convert_to_mbqc_formalism_pass
        @outline_state_evolution_pass
        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            # CHECK-LABEL: func.func public @circuit()
            # CHECK-NOT: quantum.custom "Hadamard"()
            # CHECK-NOT: quantum.custom "S"()
            # CHECK-NOT: quantum.custom "RotXZX"
            # CHECK-NOT: quantum.custom "RZ"
            # CHECK-NOT: quantum.custom "CNOT"()
            # CHECK-NOT: mbqc.measure_in_basis
            # CHECK-NOT: scf.if
            # CHECK-NOT: quantum.dealloc_qb
            # CHECK-LABEL: func.func public @circuit.state_evolution
            # CHECK-NOT: quantum.custom "S"()
            # CHECK-NOT: quantum.custom "RotXZX"
            # CHECK-NOT: quantum.custom "RZ"
            # CHECK-NOT: quantum.custom "CNOT"()
            # CHECK-NOT: quantum.namedobs
            # CHECK: scf.while
            # CHECK: quantum.custom "Hadamard"()
            # CHECK: quantum.custom "CZ"()
            # CHECK: mbqc.measure_in_basis
            # CHECK: scf.if
            # CHECK: quantum.custom "PauliX"()
            # CHECK: quantum.custom "PauliZ"()
            # CHECK: quantum.dealloc_qb
            _while_for(0)
            qml.H(0)
            qml.S(1)
            RotXZX(0.1, 0.2, 0.3, wires=[2])
            qml.RZ(phi=0.1, wires=[3])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.X(wires=0))

        run_filecheck_qjit(circuit)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_outline_state_evolution_pass_with_mbqc_pipeline(self, run_filecheck_qjit):
        """Test if the outline_state_evolution_pass works with all mbqc transform pipeline on null.qubit."""
        dev = qml.device("null.qubit", wires=1000)

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
            pipelines=mbqc_pipeline(),
        )
        @decompose_graph_state_pass
        @convert_to_mbqc_formalism_pass
        @measurements_from_samples_pass
        @diagonalize_final_measurements_pass
        @outline_state_evolution_pass
        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            # CHECK-LABEL: func.func public @circuit()
            # NOTE: There is scf.if, mbqc.measure_in_basis in the circuit()
            # scope as X obs is decomposed into H@Z and H is converted to MBQC formalism.
            # CHECK-NOT: quantum.custom "S"()
            # CHECK-NOT: quantum.custom "RotXZX"
            # CHECK-NOT: quantum.custom "RZ"
            # CHECK-NOT: quantum.custom "CNOT"()
            # CHECK-LABEL: func.func public @circuit.state_evolution
            # CHECK-NOT: quantum.custom "S"()
            # CHECK-NOT: quantum.custom "RotXZX"
            # CHECK-NOT: quantum.custom "RZ"
            # CHECK-NOT: quantum.custom "CNOT"()
            # CHECK-NOT: quantum.namedobs
            # CHECK: scf.while
            # CHECK: quantum.custom "Hadamard"()
            # CHECK: quantum.custom "CZ"()
            # CHECK: mbqc.measure_in_basis
            # CHECK: scf.if
            # CHECK: quantum.custom "PauliX"()
            # CHECK: quantum.custom "PauliZ"()
            # CHECK: quantum.dealloc_qb
            _while_for(0)
            qml.H(0)
            qml.S(1)
            RotXZX(0.1, 0.2, 0.3, wires=[2])
            qml.RZ(phi=0.1, wires=[3])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.X(wires=0))

        run_filecheck_qjit(circuit)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_outline_state_evolution_pass_with_mbqc_pipeline_run_on_nullqubit(self):
        """Test if a circuit can be transfored with the outline_state_evolution_pass and all mbqc transform pipeline can be executed on null.qubit."""
        dev = qml.device("null.qubit", wires=1000)

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
            pipelines=mbqc_pipeline(),
        )
        @decompose_graph_state_pass
        @convert_to_mbqc_formalism_pass
        @measurements_from_samples_pass
        @diagonalize_final_measurements_pass
        @outline_state_evolution_pass
        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            _while_for(0)
            qml.H(0)
            qml.S(1)
            RotXZX(0.1, 0.2, 0.3, wires=[2])
            qml.RZ(phi=0.1, wires=[3])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.X(wires=0))

        res = circuit()
        assert res == 1.0

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_lightning_execution_with_structure(self):
        """Test that the outline_state_evolution_pass on lightning.qubit for a circuit with program structure is executable and returns results as expected."""
        dev = qml.device("lightning.qubit", wires=10)

        @qml.for_loop(0, 10, 1)
        def for_fn(i):
            qml.H(i)
            qml.S(i)
            qml.RZ(phi=0.1, wires=[i])

        @qml.while_loop(lambda i: i < 10)
        def while_fn(i):
            qml.H(i)
            qml.S(i)
            qml.RZ(phi=0.1, wires=[i])
            i = i + 1
            return i

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
        )
        @outline_state_evolution_pass
        @qml.qnode(dev)
        def circuit():
            for_fn()
            while_fn(0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.prod(qml.X(0), qml.Z(1)))

        res = circuit()

        @qml.qjit(
            target="mlir",
        )
        @qml.qnode(dev)
        def circuit_ref():
            for_fn()
            while_fn(0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.prod(qml.X(0), qml.Z(1)))

        res_ref = circuit_ref()
        assert res == res_ref
