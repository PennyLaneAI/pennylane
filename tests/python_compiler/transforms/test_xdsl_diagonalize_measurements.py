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
"""Unit test module for the xDSL implementation of the diagonalize_measurements transform"""


# pylint: disable=wrong-import-position

from functools import partial

import numpy as np
import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")
from xdsl.dialects import arith, builtin, func, tensor

catalyst = pytest.importorskip("catalyst")
from catalyst.passes import xdsl_plugin

import pennylane as qml
from pennylane.compiler.python_compiler import quantum_dialect as quantum
from pennylane.compiler.python_compiler.transforms import (
    DiagonalizeFinalMeasurementsPass,
    diagonalize_measurements_pass,
)


@pytest.fixture(name="context_and_pipeline", scope="function")
def fixture_context_and_pipeline():
    """A fixture that prepares the context and pipeline for unit tests of the
    measurements-from-samples pass.
    """
    ctx = xdsl.context.Context(allow_unregistered=True)
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(tensor.Tensor)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(quantum.QuantumDialect)

    pipeline = xdsl.passes.PipelinePass((DiagonalizeFinalMeasurementsPass(),))

    yield ctx, pipeline


class TestDiagonalizeFinalMeasurementsPass:
    """Unit tests for the measurements-from-samples pass."""

    def test_unsupported_observable_raises_error(self, context_and_pipeline):
        """Test that an unsupported observable raises an error. At the time of
        implementation, the only observable that is supported in MLIR but not
        in this transform is Hadamard."""

        program = """
            func.func @test_func() {
                %0 = "test.op"() : () -> !quantum.bit
                %1 = quantum.namedobs %0[Hadamard] : !quantum.obs
                %2 = quantum.expval %1 : f64
                return
            }
            """

        ctx, pipeline = context_and_pipeline
        module = xdsl.parser.Parser(ctx, program).parse_module()
        with pytest.raises(NotImplementedError, match="not supported for diagonalization"):
            pipeline.apply(ctx, module)

    def test_with_pauli_z(self, context_and_pipeline, run_filecheck):
        """Test that a PauliZ observable is not affected by diagonalization"""

        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit

                // CHECK: [[q0_1:%.*]] =  quantum.namedobs [[q0]][PauliZ]
                %1 = quantum.namedobs %0[Identity] : !quantum.obs

                // CHECK: quantum.expval [[q0_1]]
                %2 = quantum.expval %1 : f64
                return
            }
            """

        ctx, pipeline = context_and_pipeline
        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_with_identity(self, context_and_pipeline, run_filecheck):
        """Test that an Identity observable is not affected by diagonalization."""

        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit

                // CHECK: [[q0_1:%.*]] =  quantum.namedobs [[q0]][Identity]
                %1 = quantum.namedobs %0[Identity] : !quantum.obs

                // CHECK: quantum.var [[q0_1]]
                %2 = quantum.expval %1 : f64
                return
            }
            """

        ctx, pipeline = context_and_pipeline
        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_with_pauli_x(self, context_and_pipeline, run_filecheck):
        """Test that when diagonalizing a PauliX observable, the expected diagonalizing
        gates are inserted and the observable becomes PauliZ."""

        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit

                // CHECK: [[q_obs:%.*]] = quantum.custom "Hadamard"() [[q0]]
                // CHECK-NEXT: [[q_measure:%.*]] =  quantum.namedobs [[q_obs]][PauliZ]
                // CHECK-NOT: quantum.namedobs [[q:%.+]][PauliX]
                %1 = quantum.namedobs %0[PauliX] : !quantum.obs

                // CHECK: quantum.var [[q_measure]]
                %2 = quantum.expval %1 : f64
                return
            }
            """

        ctx, pipeline = context_and_pipeline()
        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_with_pauli_y(self, context_and_pipeline, run_filecheck):
        """Test that when diagonalizing a PauliY observable, the expected diagonalizing
        gates are inserted and the observable becomes PauliZ."""

        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit

                // CHECK: [[q0_1:%.*]] = quantum.custom "PauliZ"() [[q0]]
                // CHECK: [[q0_2:%.*]] = quantum.custom "S"() [[q0_1]]
                // CHECK: [[q0_3:%.*]] = quantum.custom "Hadamard"() [[q0_2]]
                // CHECK-NEXT: [[q0_4:%.*]] =  quantum.namedobs [[q0_3]][PauliZ]
                // CHECK-NOT: quantum.namedobs [[q:%.+]][PauliY]
                %1 = quantum.namedobs %0[PauliY] : !quantum.obs

                // CHECK: quantum.expval [[q0_4]]
                %2 = quantum.expval %1 : f64
                return
            }
            """

        ctx, pipeline = context_and_pipeline
        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_with_composite_observable(self, context_and_pipeline, run_filecheck):
        """Test transform on a measurement process with a composite observable. In this
        case, the simplified program is based on the MLIR generated by the circuit

        @qml.qjit(target="mlir")
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            return qml.expval(qml.Y(0)@qml.X(1) + qml.Z(2))
        """

        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q2:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit

                // CHECK: [[q0_1:%.*]] = quantum.custom "PauliZ"() [[q0]]
                // CHECK: [[q0_2:%.*]] = quantum.custom "S"() [[q0_1]]
                // CHECK: [[q0_3:%.*]] = quantum.custom "Hadamard"() [[q0_2]]
                // CHECK: [[q_y:%.*]] =  quantum.namedobs [[q0_3]][PauliZ]
                // CHECK-NOT: quantum.namedobs [[q:%.+]][PauliY]
                %3 = quantum.namedobs %0[PauliY]: !quantum.obs
                
                // CHECK: [[q1_1:%.*]] = quantum.custom "Hadamard"() [[q1]]
                // CHECK: [[q_x:%.*]] = custom.namedobs [[q1_1]][PauliZ]
                // CHECK-NOT: quantum.namedobs [[q:%.+]][PauliX]
                %4 = quantum.namedobs %1[PauliX]: !quantum.obs
                
                
                // CHECK: [[t0:%.*]] = quantum.tensor [[q_x]], [[q_y]]: !quantum.obs
                %5 = quantum.tensor %3, %4: !quantum.obs
                
                // CHECK: [[q_z:%.*]] = quantum.namedobs [[q2]][PauliZ]: !quantum.obs
                %6 = quantum.namedobs %2[PauliZ]: !quantum.obs
                
                // CHECK: [[size:%.*]] = "test.op"() : () -> tensor<2xf64>
                %size_info = "test.op"() : () -> tensor<2xf64>
                
                // CHECK: quantum.hamiltonian([[size]]: tensor<2xf64>) [[t0]], [[q_z]] : !quantum.obs
                %7 = quantum.hamiltonian(%size_info : tensor<2xf64>) %5, %6 : !quantum.obs

                // CHECK: quantum.expval
                %8 = quantum.expval %7 : f64
                return
            }
            """

        ctx, pipeline = context_and_pipeline
        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_with_multiple_measurements(self, context_and_pipeline, run_filecheck):
        """Test diagonalizing a circuit with multiple measurements. The simplified program
        for this test is based on the circuit

        @qml.qjit(target="mlir")
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def circuit():
            return qml.var(qml.Y(0)), qml.var(qml.X(1))
        """

        program = """
            func.func @test_func() {
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
    
                // CHECK: quantum.custom "PauliZ"()
                // CHECK-NEXT: quantum.custom "S"()
                // CHECK-NEXT: quantum.custom "Hadamard"()
                // CHECK-NEXT: quantum.namedobs [[q:%.+]][PauliZ]
                %2 = quantum.namedobs %0[PauliY] : !quantum.obs
                // CHECK: quantum.var
                %3 = quantum.var %2 : f64
                
    
                // CHECK: quantum.custom "Hadamard"()
                // CHECK-NEXT: quantum.namedobs [[q:%.+]][PauliZ]
                %4 = quantum.namedobs %1[PauliX] : !quantum.obs
    
                // CHECK: quantum.expval
                %5 = quantum.expval %4 : f64
                return
            }
            """

        ctx, pipeline = context_and_pipeline
        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline.apply(ctx, module)

        run_filecheck(program, module)


class TestDiagonalizeFinalMeasurementsProgramCaptureExecution:

    @pytest.mark.usefixtures("enable_disable_plxpr")
    @pytest.mark.parametrize("obs", [qml.PauliZ(1), qml.Identity(0), qml.Identity()])
    def test_with_1_diagonalized_observable(self, obs, context_and_pipeline):
        raise RuntimeError

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_with_1_undiagonalized_observable(self, obs, diag_gates, context_and_pipeline):
        raise RuntimeError

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_with_multiple_observables(self, context_and_pipeline, run_filecheck):
        raise RuntimeError

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_with_multiple_measurements(self, context_and_pipeline, run_filecheck):
        raise RuntimeError


class TestDiagonalizeFinalMeasurementsExecution:

    @pytest.mark.parametrize("obs", [qml.PauliZ(1), qml.Identity(0), qml.Identity()])
    def test_with_1_diagonalized_observable(self, obs, context_and_pipeline):
        raise RuntimeError

    def test_with_1_undiagonalized_observable(self, obs, diag_gates, context_and_pipeline):
        raise RuntimeError

    def test_with_multiple_observables(self, context_and_pipeline, run_filecheck):
        raise RuntimeError

    def test_with_overlapping_observables(self, context_and_pipeline, run_filecheck):
        raise RuntimeError

    def test_with_multiple_measurements(self, context_and_pipeline, run_filecheck):
        raise RuntimeError
