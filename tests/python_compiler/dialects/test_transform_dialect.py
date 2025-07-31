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

import pytest

# pylint: disable=wrong-import-position

xdsl = pytest.importorskip("xdsl")
filecheck = pytest.importorskip("filecheck")

pytestmark = pytest.mark.external


def test_transform_dialect_update(run_filecheck):
    """Test that the transform dialect is updated correctly."""

    program = """
        "builtin.module"() ({
            "transform.named_sequence"() <{function_type = (!transform.any_op) -> (), sym_name = "__transform_main"}> ({
            ^bb0(%arg0: !transform.any_op):
                %0 = "transform.structured.match"(%arg0) <{ops = ["func.func"]}> : (!transform.any_op) -> !transform.any_op
                // CHECK: "invalid-option"
                %1 = "transform.apply_registered_pass"(%0) <{options = {"invalid-option" = 1 : i64}, pass_name = "canonicalize"}> : (!transform.any_op) -> !transform.any_op
                "transform.yield"() : () -> ()
            }) : () -> ()
        }) {transform.with_named_sequence} : () -> ()
    """

    run_filecheck(program, ())
