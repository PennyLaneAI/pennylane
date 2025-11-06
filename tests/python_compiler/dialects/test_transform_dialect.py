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

"""Unit test module for pennylane/compiler/python_compiler/transform.py."""

from dataclasses import dataclass

import pytest

# pylint: disable=wrong-import-position

xdsl = pytest.importorskip("xdsl")
filecheck = pytest.importorskip("filecheck")

pytestmark = pytest.mark.external

from xdsl import passes
from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects.builtin import DictionaryAttr, IntegerAttr, i64
from xdsl.dialects.transform import AnyOpType
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value

from pennylane.compiler.python_compiler.conversion import xdsl_from_docstring
from pennylane.compiler.python_compiler.dialects import transform
from pennylane.compiler.python_compiler.dialects.transform import ApplyRegisteredPassOp
from pennylane.compiler.python_compiler.pass_api import (
    ApplyTransformSequence,
    compiler_transform,
)


def test_dict_options():
    """Test ApplyRegisteredPassOp constructor with dict options."""
    target = create_ssa_value(AnyOpType())
    options = {"option1": 1, "option2": True}

    op = ApplyRegisteredPassOp("canonicalize", target, options)

    assert op.pass_name.data == "canonicalize"
    assert isinstance(op.options, DictionaryAttr)
    assert op.options == DictionaryAttr({"option1": 1, "option2": True})
    assert op.verify_() is None


def test_attr_options():
    """Test ApplyRegisteredPassOp constructor with DictionaryAttr options."""
    target = create_ssa_value(AnyOpType())
    options = DictionaryAttr({"test-option": IntegerAttr(42, i64)})

    # This should trigger the __init__ method
    op = ApplyRegisteredPassOp("canonicalize", target, options)

    assert op.pass_name.data == "canonicalize"
    assert isinstance(op.options, DictionaryAttr)
    assert op.options == DictionaryAttr({"test-option": IntegerAttr(42, i64)})
    assert op.verify_() is None


def test_none_options():
    """Test ApplyRegisteredPassOp constructor with None options."""
    target = create_ssa_value(AnyOpType())

    # This should trigger the __init__ method
    op = ApplyRegisteredPassOp("canonicalize", target, None)

    assert op.pass_name.data == "canonicalize"
    assert isinstance(op.options, DictionaryAttr)
    assert op.options == DictionaryAttr({})
    assert op.verify_() is None


def test_invalid_options():
    """Test ApplyRegisteredPassOp constructor with invalid options type."""
    target = create_ssa_value(AnyOpType())

    with pytest.raises(
        VerifyException, match="invalid_options should be of base attribute dictionary"
    ):
        ApplyRegisteredPassOp("canonicalize", target, "invalid_options").verify_()


def test_transform_dialect_filecheck(run_filecheck):
    """Test that the transform dialect operations are parsed correctly."""
    program = """
        "builtin.module"() ({
            "transform.named_sequence"() <{function_type = (!transform.any_op) -> (), sym_name = "__transform_main"}> ({
            ^bb0(%arg0: !transform.any_op):
                %0 = "transform.structured.match"(%arg0) <{ops = ["func.func"]}> : (!transform.any_op) -> !transform.any_op
                // CHECK: options = {"invalid-option" = 1 : i64}
                %1 = "transform.apply_registered_pass"(%0) <{options = {"invalid-option" = 1 : i64}, pass_name = "canonicalize"}> : (!transform.any_op) -> !transform.any_op
                "transform.yield"() : () -> ()
            }) : () -> ()
        }) {transform.with_named_sequence} : () -> ()
    """

    run_filecheck(program)


def test_integration_for_transform_interpreter(capsys):
    """Test that a pass with options is run via the transform interpreter"""

    @compiler_transform
    @dataclass(frozen=True)
    class _HelloWorld(passes.ModulePass):
        name = "test-hello-world"

        custom_print: str | None = None

        def apply(self, _ctx: Context, _module: builtin.ModuleOp) -> None:
            if self.custom_print:
                print(self.custom_print)
            else:
                print("hello world")

    @xdsl_from_docstring
    def program():
        """
        builtin.module {
          builtin.module {
            transform.named_sequence @__transform_main(%arg0 : !transform.op<"builtin.module">) {
              %0 = "transform.apply_registered_pass"(%arg0) <{options = {"custom_print" = "Hello from custom option!"}, pass_name = "test-hello-world"}> : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
              transform.yield
            }
          }
        }
        """

    ctx = xdsl.context.Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(transform.Transform)

    mod = program()
    pipeline = xdsl.passes.PassPipeline((ApplyTransformSequence(),))
    pipeline.apply(ctx, mod)

    assert "Hello from custom option!" in capsys.readouterr().out
