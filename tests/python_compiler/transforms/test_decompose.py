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
"""Unit test module for the ``decompose`` transform pass in the PennyLane Python compiler."""

import io

import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")
filecheck = pytest.importorskip("filecheck")

# pylint: disable=wrong-import-position
from filecheck.finput import FInput
from filecheck.matcher import Matcher
from filecheck.options import parse_argv_options
from filecheck.parser import Parser as FCParser
from filecheck.parser import pattern_for_opts
from xdsl.dialects.test import Test

from pennylane.compiler.python_compiler.quantum_dialect import QuantumDialect
from pennylane.compiler.python_compiler.transforms.decompose import DecompositionTransformPass


def test_single_rot_decomposition():
    """Test that the Rot gate is decomposed into RZ and RY gates."""

    # qml.Rot(0.5, 0.5, 0.5, wires=0)
    # ---->
    # RZ(0.5, wires=0)
    # RY(0.5, wires=0)
    # RZ(0.5, wires=0)

    module_source = """
builtin.module @circuit {
    // CHECK: func.func public @circuit()
    func.func public @circuit() -> tensor<8xcomplex<f64>> attributes {qnode} {

      %0 = arith.constant 5.000000e-01 : f64
      %1 = arith.constant 0 : i64
      "quantum.device_init"(%1) <{kwargs = "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", lib = "...", name = "LightningSimulator"}> : (i64) -> ()
      %2 = "quantum.alloc"() <{nqubits_attr = 3 : i64}> : () -> !quantum.reg
      %3 = "quantum.extract"(%2) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit

      // CHECK: [[VALUE:%.*]] = arith.constant 5.000000e-01 : f64
      // CHECK: [[QUBITREGISTER:%.*]] = "quantum.alloc"() <{nqubits_attr = 3 : i64}> : () -> !quantum.reg
      // CHECK: [[QUBIT1:%.*]] = "quantum.extract"(%2) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit
      // CHECK: [[QUBIT2:%.*]] = "quantum.custom"([[VALUE]], [[QUBIT1]]) <{gate_name = "RZ"
      // CHECK: [[QUBIT3:%.*]] = "quantum.custom"([[VALUE]], [[QUBIT2]]) <{gate_name = "RY"
      // CHECK: [[LASTQUBIT:%.*]] = "quantum.custom"([[VALUE]], [[QUBIT3]]) <{gate_name = "RZ"
      // CHeCK: [[___]] = "quantum.insert"([[QUBITREGISTER]], [[LASTQUBIT]]) <{idx_attr = 0 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
      %4 = "quantum.custom"(%0, %0, %0, %3) <{gate_name = "Rot", operandSegmentSizes = array<i32: 3, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, f64, f64, !quantum.bit) -> !quantum.bit
    }
}
"""

    ctx = xdsl.context.Context()
    ctx.allow_unregistered_dialects = True
    ctx.load_dialect(xdsl.dialects.builtin.Builtin)
    ctx.load_dialect(xdsl.dialects.func.Func)
    ctx.load_dialect(xdsl.dialects.transform.Transform)
    ctx.load_dialect(xdsl.dialects.arith.Arith)
    ctx.load_dialect(Test)
    ctx.load_dialect(QuantumDialect)

    module = xdsl.parser.Parser(ctx, module_source).parse_module()

    pipeline = xdsl.passes.PipelinePass((DecompositionTransformPass(),))
    pipeline.apply(ctx, module)

    opts = parse_argv_options(["filecheck", __file__])
    matcher = Matcher(
        opts,
        FInput("no-name", str(module)),
        FCParser(opts, io.StringIO(module_source), *pattern_for_opts(opts)),
    )
    assert matcher.run() == 0


def test_multiple_rot_decomposition():
    """Test that multiple Rot gates are decomposed correctly into RZ and RY gates."""

    # qml.Rot(0.1, 0.2, 0.3, wires=0)
    # qml.Rot(0.4, 0.5, 0.6, wires=0)
    # qml.Rot(0.1, 0.1, 0.1, wires=1)
    # ---->
    # RZ(0.1, wires=0)
    # RY(0.2, wires=0)
    # RZ(0.3, wires=0)
    # RZ(0.4, wires=0)
    # RY(0.5, wires=0)
    # RZ(0.6, wires=0)
    # RZ(0.1, wires=1)
    # RY(0.1, wires=1)
    # RZ(0.1, wires=1)

    module_source = """
builtin.module @circuit {
    // CHECK: func.func public @circuit()
    func.func public @circuit() -> tensor<8xcomplex<f64>> attributes {qnode} {

        %0 = arith.constant 6.000000e-01 : f64
        %1 = arith.constant 5.000000e-01 : f64
        %2 = arith.constant 4.000000e-01 : f64
        %3 = arith.constant 3.000000e-01 : f64
        %4 = arith.constant 2.000000e-01 : f64
        %5 = arith.constant 1.000000e-01 : f64
        %6 = arith.constant 0 : i64

        "quantum.device_init"(%6) <{kwargs = "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", lib = "...", name = "LightningSimulator"}> : (i64) -> ()
        %7 = "quantum.alloc"() <{nqubits_attr = 3 : i64}> : () -> !quantum.reg
        %8 = "quantum.extract"(%7) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit

        // CHECK: [[QUBITREGISTER:%.*]] = "quantum.alloc"() <{nqubits_attr = 3 : i64}> : () -> !quantum.reg
        // CHECK: [[QUBIT1:%.*]] = "quantum.extract"([[QUBITREGISTER]]) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit
        // CHECK: [[QUBIT2:%.*]] = "quantum.custom"(%5, [[QUBIT1]]) <{gate_name = "RZ"
        // CHECK: [[QUBIT3:%.*]] = "quantum.custom"(%4, [[QUBIT2]]) <{gate_name = "RY"
        // CHECK: [[QUBIT4:%.*]] = "quantum.custom"(%3, [[QUBIT3]]) <{gate_name = "RZ"
        // CHECK: [[QUBIT5:%.*]] = "quantum.custom"(%2, [[QUBIT4]]) <{gate_name = "RZ"
        // CHECK: [[QUBIT6:%.*]] = "quantum.custom"(%1, [[QUBIT5]]) <{gate_name = "RY"
        // CHECK: [[QUBIT7:%.*]] = "quantum.custom"(%0, [[QUBIT6]]) <{gate_name = "RZ"
        // CHECK: [[QUBIT8:%.*]] = "quantum.extract"([[QUBITREGISTER]]) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit
        // CHECK: [[QUBIT9:%.*]] = "quantum.custom"(%5, [[QUBIT8]]) <{gate_name = "RZ"
        // CHECK: [[QUBIT10:%.*]] = "quantum.custom"(%5, [[QUBIT9]]) <{gate_name = "RY"
        // CHECK: [[QUBIT11:%.*]] = "quantum.custom"(%5, [[QUBIT10]]) <{gate_name = "RZ"
        // CHECK: [[QUBIT12:%.*]] = "quantum.insert"([[QUBITREGISTER]], [[QUBIT7]]) <{idx_attr = 0 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
        // CHECK: [[___:%.*]] = "quantum.insert"([[QUBIT12]], [[QUBIT11]]) <{idx_attr = 1 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg

        %9 = "quantum.custom"(%5, %8) <{gate_name = "RZ", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
        %10 = "quantum.custom"(%4, %9) <{gate_name = "RY", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
        %11 = "quantum.custom"(%3, %10) <{gate_name = "RZ", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
        %12 = "quantum.custom"(%2, %11) <{gate_name = "RZ", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
        %13 = "quantum.custom"(%1, %12) <{gate_name = "RY", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
        %14 = "quantum.custom"(%0, %13) <{gate_name = "RZ", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
        %15 = "quantum.extract"(%7) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit
        %16 = "quantum.custom"(%5, %15) <{gate_name = "RZ", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
        %17 = "quantum.custom"(%5, %16) <{gate_name = "RY", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
        %18 = "quantum.custom"(%5, %17) <{gate_name = "RZ", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
        %19 = "quantum.insert"(%7, %14) <{idx_attr = 0 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
        %20 = "quantum.insert"(%19, %18) <{idx_attr = 1 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg

    }
}
"""

    ctx = xdsl.context.Context()
    ctx.allow_unregistered_dialects = True
    ctx.load_dialect(xdsl.dialects.builtin.Builtin)
    ctx.load_dialect(xdsl.dialects.func.Func)
    ctx.load_dialect(xdsl.dialects.transform.Transform)
    ctx.load_dialect(xdsl.dialects.arith.Arith)
    ctx.load_dialect(Test)
    ctx.load_dialect(QuantumDialect)

    module = xdsl.parser.Parser(ctx, module_source).parse_module()

    pipeline = xdsl.passes.PipelinePass((DecompositionTransformPass(),))
    pipeline.apply(ctx, module)

    opts = parse_argv_options(["filecheck", __file__])
    matcher = Matcher(
        opts,
        FInput("no-name", str(module)),
        FCParser(opts, io.StringIO(module_source), *pattern_for_opts(opts)),
    )
    assert matcher.run() == 0


def test_rot_cnot_decomposition():
    """Test that a circuit with Rot and CNOT gates is decomposed correctly."""

    # qml.CNOT(wires=[0, 1])
    # qml.CNOT(wires=[0, 2])
    # qml.Rot(0.1, 0.2, 0.3, wires=0)
    # qml.Rot(0.4, 0.5, 0.6, wires=1)
    # qml.CNOT(wires=[1, 0])
    # qml.CNOT(wires=[1, 2])
    # ---->
    # qml.CNOT(wires=[0, 1])
    # qml.CNOT(wires=[0, 2])
    # RZ(0.1, wires=0)
    # RY(0.2, wires=0)
    # RZ(0.3, wires=0)
    # RZ(0.4, wires=1)
    # RY(0.5, wires=1)
    # RZ(0.6, wires=1)
    # qml.CNOT(wires=[1, 0])
    # qml.CNOT(wires=[1, 2])

    module_source = """
builtin.module @circuit {
    // CHECK: func.func public @circuit()
    func.func public @circuit() -> tensor<8xcomplex<f64>> attributes {qnode} {

        %0 = arith.constant 6.000000e-01 : f64
        %1 = arith.constant 5.000000e-01 : f64
        %2 = arith.constant 4.000000e-01 : f64
        %3 = arith.constant 3.000000e-01 : f64
        %4 = arith.constant 2.000000e-01 : f64
        %5 = arith.constant 1.000000e-01 : f64
        %6 = arith.constant 0 : i64

        "quantum.device_init"(%6) <{kwargs = "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", lib = "...", name = "LightningSimulator"}> : (i64) -> ()
        %7 = "quantum.alloc"() <{nqubits_attr = 3 : i64}> : () -> !quantum.reg
        %8 = "quantum.extract"(%7) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit
        %9 = "quantum.extract"(%7) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit

        // CHECK: [[QUBITREGISTER:%.*]] = "quantum.alloc"() <{nqubits_attr = 3 : i64}> : () -> !quantum.reg
        // CHECK: [[QUBIT1:%.*]] = "quantum.extract"([[QUBITREGISTER]]) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit
        // CHECK: [[QUBIT2:%.*]] = "quantum.extract"([[QUBITREGISTER]]) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit
        // CHECK: [[CNOT1:%.*]], [[CNOT2:%.*]] = "quantum.custom"([[QUBIT1]], [[QUBIT2]]) <{gate_name = "CNOT", 
        // CHECK: [[QUBIT3:%.*]] = "quantum.extract"([[QUBITREGISTER]]) <{idx_attr = 2 : i64}> : (!quantum.reg) -> !quantum.bit
        // CHECK: [[CNOT3:%.*]], [[CNOT4:%.*]] = "quantum.custom"([[CNOT1]], [[QUBIT3]]) <{gate_name = "CNOT",
        // CHECK: [[QUBIT4:%.*]] = "quantum.custom"(%5, [[CNOT3]]) <{gate_name = "RZ"
        // CHECK: [[QUBIT5:%.*]] = "quantum.custom"(%4, [[QUBIT4]]) <{gate_name = "RY"
        // CHECK: [[QUBIT6:%.*]] = "quantum.custom"(%3, [[QUBIT5]]) <{gate_name = "RZ"
        // CHECK: [[QUBIT7:%.*]] = "quantum.custom"(%2, [[CNOT2]]) <{gate_name = "RZ"
        // CHECK: [[QUBIT8:%.*]] = "quantum.custom"(%1, [[QUBIT7]]) <{gate_name = "RY"
        // CHECK: [[QUBIT9:%.*]] = "quantum.custom"(%0, [[QUBIT8]]) <{gate_name = "RZ"
        // CHECK: [[CNOT5:%.*]], [[CNOT6:%.*]] = "quantum.custom"([[QUBIT9]], [[QUBIT6]]) <{gate_name = "CNOT",
        // CHECK: [[CNOT7:%.*]], [[CNOT8:%.*]] = "quantum.custom"([[CNOT5]], [[CNOT4]]) <{gate_name = "CNOT",
        // CHECK: [[QUBIT10:%.*]] = "quantum.insert"([[QUBITREGISTER]], [[CNOT6]]) <{idx_attr = 0 : i64}> 
        // CHECK: [[QUBIT11:%.*]] = "quantum.insert"([[QUBIT10]], [[CNOT7]]) <{idx_attr = 1 : i64}>
        // CHECK: [[___:%.*]] = "quantum.insert"([[QUBIT11]], [[CNOT8]]) <{idx_attr = 2 : i64}>
        
        %10, %11 = "quantum.custom"(%8, %9) <{gate_name = "CNOT", operandSegmentSizes = array<i32: 0, 2, 0, 0>, resultSegmentSizes = array<i32: 2, 0>}> : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        %12 = "quantum.extract"(%7) <{idx_attr = 2 : i64}> : (!quantum.reg) -> !quantum.bit
        %13, %14 = "quantum.custom"(%10, %12) <{gate_name = "CNOT", operandSegmentSizes = array<i32: 0, 2, 0, 0>, resultSegmentSizes = array<i32: 2, 0>}> : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        %15 = "quantum.custom"(%5, %4, %3, %13) <{gate_name = "Rot", operandSegmentSizes = array<i32: 3, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, f64, f64, !quantum.bit) -> !quantum.bit
        %16 = "quantum.custom"(%2, %1, %0, %11) <{gate_name = "Rot", operandSegmentSizes = array<i32: 3, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, f64, f64, !quantum.bit) -> !quantum.bit
        %17, %18 = "quantum.custom"(%16, %15) <{gate_name = "CNOT", operandSegmentSizes = array<i32: 0, 2, 0, 0>, resultSegmentSizes = array<i32: 2, 0>}> : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        %19, %20 = "quantum.custom"(%17, %14) <{gate_name = "CNOT", operandSegmentSizes = array<i32: 0, 2, 0, 0>, resultSegmentSizes = array<i32: 2, 0>}> : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        %21 = "quantum.insert"(%7, %18) <{idx_attr = 0 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
        %22 = "quantum.insert"(%21, %19) <{idx_attr = 1 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
        %23 = "quantum.insert"(%22, %20) <{idx_attr = 2 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg

    }
}
"""

    ctx = xdsl.context.Context()
    ctx.allow_unregistered_dialects = True
    ctx.load_dialect(xdsl.dialects.builtin.Builtin)
    ctx.load_dialect(xdsl.dialects.func.Func)
    ctx.load_dialect(xdsl.dialects.transform.Transform)
    ctx.load_dialect(xdsl.dialects.arith.Arith)
    ctx.load_dialect(Test)
    ctx.load_dialect(QuantumDialect)

    module = xdsl.parser.Parser(ctx, module_source).parse_module()

    pipeline = xdsl.passes.PipelinePass((DecompositionTransformPass(),))
    pipeline.apply(ctx, module)

    opts = parse_argv_options(["filecheck", __file__])
    matcher = Matcher(
        opts,
        FInput("no-name", str(module)),
        FCParser(opts, io.StringIO(module_source), *pattern_for_opts(opts)),
    )
    assert matcher.run() == 0
