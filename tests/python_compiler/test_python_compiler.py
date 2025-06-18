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

pytestmark = pytest.mark.external

catalyst = pytest.importorskip("catalyst")
jax = pytest.importorskip("jax")
jaxlib = pytest.importorskip("jaxlib")
xdsl = pytest.importorskip("xdsl")

from catalyst import CompileError
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath
from xdsl import passes
from xdsl.context import Context
from xdsl.dialects import builtin, transform
from xdsl.interpreters import Interpreter

from pennylane.compiler.python_compiler import Compiler
from pennylane.compiler.python_compiler.jax_utils import (
    jax_from_docstring,
    module,
    xdsl_from_docstring,
)
from pennylane.compiler.python_compiler.transforms.api import (
    ApplyTransformSequence,
    TransformFunctionsExt,
    TransformInterpreterPass,
    available_passes,
    xdsl_transform,
)


def test_compiler():
    """Test that we can pass a jax module into the compiler.

    In this particular case, the compiler is not doing anything
    because this module does not contain nested modules which is what
    is expected of Catalyst.

    So, it just tests that Compiler.run does not trigger an assertion
    and returns a valid
    """

    @module
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

    @jax_from_docstring
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

    @dataclass(frozen=True)
    class PrintModule(passes.ModulePass):
        name = "print-module"

        def apply(self, _ctx: Context, _module: builtin.ModuleOp) -> None:
            print("hello")

    xdsl_transform(PrintModule)
    assert "print-module" in available_passes
    assert available_passes["print-module"]() == PrintModule


def test_integration_for_transform_interpreter(capsys):
    """Test that a pass is run via the transform interpreter"""

    @xdsl_transform
    @dataclass(frozen=True)
    class _HelloWorld(passes.ModulePass):
        name = "hello-world"

        def apply(self, _ctx: Context, _module: builtin.ModuleOp) -> None:
            print("hello world")

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

    pipeline = xdsl.passes.PipelinePass((ApplyTransformSequence(),))
    pipeline.apply(ctx, program())
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"


def test_integration_but_run_with_catalyst():

    @catalyst.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
    def none(): ...

    none()


if __name__ == "__main__":
    pytest.main(["-x", __file__])
