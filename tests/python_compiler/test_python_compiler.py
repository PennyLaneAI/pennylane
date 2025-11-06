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

"""Unit test module for pennylane/compiler/python_compiler/impl.py"""

from dataclasses import dataclass

# pylint: disable=wrong-import-position
import pytest

pytestmark = pytest.mark.catalyst

catalyst = pytest.importorskip("catalyst")
jax = pytest.importorskip("jax")
jaxlib = pytest.importorskip("jaxlib")
xdsl = pytest.importorskip("xdsl")

from catalyst import CompileError
from catalyst.passes import apply_pass
from catalyst.passes import cancel_inverses as catalyst_cancel_inverses
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath
from xdsl import passes
from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.interpreters import Interpreter

import pennylane as qml
from pennylane.capture import enabled as capture_enabled
from pennylane.compiler.python_compiler import Compiler
from pennylane.compiler.python_compiler.conversion import (
    mlir_from_docstring,
    mlir_module,
    xdsl_from_docstring,
)
from pennylane.compiler.python_compiler.dialects import transform
from pennylane.compiler.python_compiler.pass_api import (
    ApplyTransformSequence,
    TransformFunctionsExt,
    TransformInterpreterPass,
    available_passes,
    compiler_transform,
)
from pennylane.compiler.python_compiler.transforms import (
    iterative_cancel_inverses_pass,
    merge_rotations_pass,
)


@dataclass(frozen=True)
class HelloWorldPass(passes.ModulePass):
    """A simple pass that prints 'hello world' when run."""

    name = "hello-world"

    def apply(self, _ctx: Context, _module: builtin.ModuleOp) -> None:
        print("hello world")


hello_world_pass = compiler_transform(HelloWorldPass)


def test_compiler():
    """Test that we can pass a jax module into the compiler.

    In this particular case, the compiler is not doing anything
    because this module does not contain nested modules which is what
    is expected of Catalyst.

    So, it just tests that Compiler.run does not trigger an assertion
    and returns a valid
    """

    @mlir_module
    @jax.jit
    def identity(x):
        return x

    input_module = identity(1)
    retval = Compiler.run(input_module)
    assert isinstance(retval, jaxlib.mlir.ir.Module)
    assert str(retval) == str(input_module)


def test_generic_catalyst_program():
    """
    test that actually will trigger the transform interpreter
    """

    @mlir_from_docstring
    def program():
        """
        "builtin.module"() <{sym_name = "circuit"}> ({
          "func.func"() <{function_type = () -> tensor<2xcomplex<f64>>, sym_name = "jit_circuit", sym_visibility = "public"}> ({
            %8 = "catalyst.launch_kernel"() <{callee = @module_circuit::@circuit}> : () -> tensor<2xcomplex<f64>>
            "func.return"(%8) : (tensor<2xcomplex<f64>>) -> ()
          }) {llvm.emit_c_interface} : () -> ()
          "builtin.module"() <{sym_name = "module_circuit"}> ({
            "builtin.module"() ({
              "transform.named_sequence"() <{function_type = (!transform.op<"builtin.module">) -> (), sym_name = "__transform_main"}> ({
              ^bb0(%arg0: !transform.op<"builtin.module">):
                "transform.yield"() : () -> ()
              }) : () -> ()
            }) {transform.with_named_sequence} : () -> ()
            "func.func"() <{function_type = () -> tensor<2xcomplex<f64>>, sym_name = "circuit", sym_visibility = "public"}> ({
              %0 = "arith.constant"() <{value = 0 : i64}> : () -> i64
              "quantum.device"(%0) <{kwargs = "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}", lib = "/usr/local/lib/python3.11/dist-packages/pennylane_lightning/liblightning_qubit_catalyst.so", name = "LightningSimulator"}> : (i64) -> ()
              %1 = "quantum.alloc"() <{nqubits_attr = 1 : i64}> : () -> !quantum.reg
              %2 = "quantum.extract"(%1) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit
              %3 = "quantum.custom"(%2) <{gate_name = "Hadamard", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
              %4 = "quantum.custom"(%3) <{gate_name = "Hadamard", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
              %5 = "quantum.insert"(%1, %4) <{idx_attr = 0 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
              %6 = "quantum.compbasis"(%5) <{operandSegmentSizes = array<i32: 0, 1>}> : (!quantum.reg) -> !quantum.obs
              %7 = "quantum.state"(%6) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!quantum.obs) -> tensor<2xcomplex<f64>>
              "quantum.dealloc"(%5) : (!quantum.reg) -> ()
              "quantum.device_release"() : () -> ()
              "func.return"(%7) : (tensor<2xcomplex<f64>>) -> ()
            }) {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} : () -> ()
          }) : () -> ()
          "func.func"() <{function_type = () -> (), sym_name = "setup"}> ({
            "quantum.init"() : () -> ()
            "func.return"() : () -> ()
          }) : () -> ()
          "func.func"() <{function_type = () -> (), sym_name = "teardown"}> ({
            "quantum.finalize"() : () -> ()
            "func.return"() : () -> ()
          }) : () -> ()
        }) : () -> ()
        """

    retval = Compiler.run(program())
    assert isinstance(retval, jaxlib.mlir.ir.Module)


def test_raises_error_when_pass_does_not_exists():
    """Attempts to run pass "this-pass-does-not-exists" on an empty module.

    This should raise an error
    """

    @xdsl_from_docstring
    def empty_module():
        """
        builtin.module {}
        """

    @xdsl_from_docstring
    def schedule_module():
        """
        builtin.module {
          builtin.module {
            transform.named_sequence @__transform_main(%arg0 : !transform.op<"builtin.module">) {
              %0 = transform.apply_registered_pass "this-pass-does-not-exists" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
              transform.yield
            }
          }
        }
        """

    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(transform.Transform)
    schedule = TransformInterpreterPass.find_transform_entry_point(
        schedule_module(), "__transform_main"
    )
    interpreter = Interpreter(empty_module())
    interpreter.register_implementations(TransformFunctionsExt(ctx, {}))
    with pytest.raises(CompileError):
        interpreter.call_op(schedule, (empty_module(),))


def test_decorator():
    """Test that the decorator has modified the available_passes dictionary"""
    assert "hello-world" in available_passes
    assert available_passes["hello-world"]() == HelloWorldPass


def test_integration_for_transform_interpreter(capsys):
    """Test that a pass is run via the transform interpreter"""

    # The hello-world pass is in the IR
    @xdsl_from_docstring
    def program():
        """
        builtin.module {
          builtin.module {
            transform.named_sequence @__transform_main(%arg0 : !transform.op<"builtin.module">) {
              %0 = transform.apply_registered_pass "hello-world" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
              transform.yield
            }
          }
        }
        """

    ctx = xdsl.context.Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(transform.Transform)

    pipeline = xdsl.passes.PassPipeline((ApplyTransformSequence(),))
    pipeline.apply(ctx, program())
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"


class TestCatalystIntegration:
    """Tests for integration of the Python compiler with Catalyst"""

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_integration_catalyst_no_passes_with_capture(self):
        """Test that the xDSL plugin can be used even when no passes are applied
        when capture is enabled."""

        assert capture_enabled()

        @catalyst.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def f(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        out = f(1.5)
        assert jax.numpy.allclose(out, jax.numpy.cos(1.5))

    def test_integration_catalyst_no_passes_no_capture(self):
        """Test that the xDSL plugin can be used even when no passes are applied
        when capture is disabled."""

        assert not capture_enabled()

        @catalyst.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def f(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        out = f(1.5)
        assert jax.numpy.allclose(out, jax.numpy.cos(1.5))

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_integration_catalyst_xdsl_pass_with_capture(self, capsys):
        """Test that a pass is run via the transform interpreter when using with a
        qjit workflow and capture is enabled."""

        assert capture_enabled()

        @catalyst.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
        @hello_world_pass
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def f(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        out = f(1.5)
        assert jax.numpy.allclose(out, jax.numpy.cos(1.5))
        captured = capsys.readouterr()
        assert captured.out.strip() == "hello world"

    def test_integration_catalyst_xdsl_pass_no_capture(self, capsys):
        """Test that a pass is run via the transform interpreter when using with a
        qjit workflow and capture is disabled."""

        assert not capture_enabled()

        @catalyst.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
        @apply_pass("hello-world")
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def f(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        out = f(1.5)
        assert jax.numpy.allclose(out, jax.numpy.cos(1.5))
        captured = capsys.readouterr()
        assert captured.out.strip() == "hello world"

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_integration_catalyst_mixed_passes_with_capture(self, capsys):
        """Test that both Catalyst and Python compiler passes can be used with qjit
        when capture is enabled."""

        assert capture_enabled()

        @catalyst.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
        @hello_world_pass
        @qml.transforms.cancel_inverses
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def f(x):
            qml.RX(x, 0)
            qml.X(0)
            qml.X(0)
            return qml.expval(qml.Z(0))

        out = f(1.5)
        assert jax.numpy.allclose(out, jax.numpy.cos(1.5))
        captured = capsys.readouterr()
        assert captured.out.strip() == "hello world"

    def test_integration_catalyst_mixed_passes_no_capture(self, capsys):
        """Test that both Catalyst and Python compiler passes can be used with qjit
        when capture is disabled."""

        assert not capture_enabled()

        @catalyst.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
        @apply_pass("hello-world")
        @catalyst_cancel_inverses
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def f(x):
            qml.RX(x, 0)
            qml.X(0)
            qml.X(0)
            return qml.expval(qml.Z(0))

        out = f(1.5)
        assert jax.numpy.allclose(out, jax.numpy.cos(1.5))
        captured = capsys.readouterr()
        assert captured.out.strip() == "hello world"


class TestCallbackIntegration:
    """Test the integration of the callback functionality"""

    def test_callback_integration(self, capsys):
        """Test that the callback mechanism works with the transform interpreter"""

        @compiler_transform
        @dataclass(frozen=True)
        class _(passes.ModulePass):
            name = "none-pass"

            def apply(self, _ctx: Context, _module: builtin.ModuleOp) -> None: ...

        def print_between_passes(*_, pass_level=0):
            if pass_level == 0:
                return
            print("hello world")

        @xdsl_from_docstring
        def program():
            """
            builtin.module {
              builtin.module {
                transform.named_sequence @__transform_main(%arg0 : !transform.op<"builtin.module">) {
                  %0 = transform.apply_registered_pass "none-pass" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
                  transform.yield
                }
              }
            }
            """

        ctx = Context()
        ctx.load_dialect(builtin.Builtin)
        pipeline = xdsl.passes.PassPipeline(
            (ApplyTransformSequence(callback=print_between_passes),)
        )
        pipeline.apply(ctx, program())
        captured = capsys.readouterr()
        assert captured.out.strip() == "hello world"

    def test_callback_prints_module_after_each_pass(self, capsys):
        """Test that the callback prints the module after each pass"""

        # pylint: disable=redefined-outer-name
        def print_between_passes(_, module, __, pass_level=0):
            if pass_level == 0:
                return
            print("=== Between Pass ===")
            print(module)

        @xdsl_from_docstring
        def program_2_passes():
            """
            builtin.module {
              builtin.module @module_foo {
                transform.named_sequence @__transform_main(%arg0 : !transform.op<"builtin.module">) {
                  %0 = transform.apply_registered_pass "xdsl-cancel-inverses" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
                  %1 = transform.apply_registered_pass "xdsl-merge-rotations" to %0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
                  transform.yield %1 : !transform.op<"builtin.module">
                  func.func public @foo() {
                    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                    %3 = tensor.extract %2[] : tensor<i64>
                    %4 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                    %5 = quantum.alloc(1) : !quantum.reg
                    %6 = tensor.extract %2[] : tensor<i64>
                    %7 = quantum.extract %5[%6] : !quantum.reg -> !quantum.bit
                    %8 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
                    %9 = tensor.extract %8[] : tensor<f64>
                    %10 = quantum.custom "RX"(%9) %7 : !quantum.bit
                    %11 = tensor.extract %8[] : tensor<f64>
                    %12 = quantum.custom "RX"(%11) %10 : !quantum.bit
                    %13 = quantum.custom "Hadamard"() %12 : !quantum.bit
                    %14 = quantum.custom "Hadamard"() %13 : !quantum.bit
                    %15 = tensor.extract %1[] : tensor<i64>
                    %16 = quantum.insert %5[%15], %14 : !quantum.reg, !quantum.bit
                    %17 = quantum.compbasis qreg %16 : !quantum.obs
                    %18 = quantum.probs %17 : tensor<2xf64>
                  }
                }
              }
            }
            """

        ctx = Context()
        ctx.load_dialect(builtin.Builtin)
        pipeline = xdsl.passes.PassPipeline(
            (ApplyTransformSequence(callback=print_between_passes),)
        )
        pipeline.apply(ctx, program_2_passes())

        out = capsys.readouterr().out
        printed_modules = out.split("=== Between Pass ===")[1:]

        assert (
            len(printed_modules) == 2
        ), "Callback should have been called twice (after each pass)."

        # callback after cancel-inverses
        assert 'quantum.custom "RX"' in printed_modules[0]
        assert 'quantum.custom "Hadamard"' not in printed_modules[0]

        # callback after merge-rotations
        # We expect an `arith.addf` if rotations were merged
        assert "arith.addf" in printed_modules[1], "Expected merged RX gates into a single rotation"
        assert 'quantum.custom "RX"' in printed_modules[1]

        assert printed_modules[0] != printed_modules[1], "IR should differ between passes"

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_callback_run_integration(self, capsys):
        """Test that the callback is integrated into the pass pipeline with the Compiler.run() method"""

        # pylint: disable=redefined-outer-name
        def print_between_passes(_, module, __, pass_level=0):
            if pass_level == 0:
                return
            print("=== Between Pass ===")
            print(module)

        @qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
        @iterative_cancel_inverses_pass
        @merge_rotations_pass
        @qml.qnode(qml.device("null.qubit", wires=2))
        def circuit():
            qml.RX(0.1, 0)
            qml.RX(2.0, 0)
            qml.Hadamard(1)
            qml.Hadamard(1)
            return qml.state()

        Compiler.run(circuit.mlir_module, callback=print_between_passes)
        out = capsys.readouterr().out
        printed_modules = out.split("=== Between Pass ===")[1:]

        assert (
            len(printed_modules) == 2
        ), "Callback should have been called twice (after each pass)."

        # callback after merge-rotations
        # We expect an `arith.addf` if rotations were merged
        assert "arith.addf" in printed_modules[0], "Expected merged RX gates into a single rotation"
        assert 'quantum.custom "RX"' in printed_modules[0]
        assert 'quantum.custom "Hadamard"' in printed_modules[0]

        # callback after cancel-inverses
        assert "arith.addf" in printed_modules[1], "Expected merged RX gates into a single rotation"
        assert 'quantum.custom "RX"' in printed_modules[1]
        assert 'quantum.custom "Hadamard"' not in printed_modules[1]

        assert printed_modules[0] != printed_modules[1], "IR should differ between passes"


if __name__ == "__main__":
    pytest.main(["-x", __file__])
