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
"""Unit test module for the convert to MBQC formalism transform"""
import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")

# pylint: disable=wrong-import-position
from xdsl.context import Context
from xdsl.dialects import func, test

from pennylane.compiler.python_compiler.quantum_dialect import QuantumDialect as Quantum
from pennylane.compiler.python_compiler.transforms import ConvertToMBQCFormalismPass


class TestConvertToMBQCFormalismPass:
    """Unit tests for ConvertToMBQCFormalismPass."""

    def test_convert_to_mbqc_formalism_allocop_aux_wires(self, run_filecheck):
        """Test that ConvertToMBQCFormalismPass can pre-allocate 13 aux wires. Note that
        this is a temporal solution.
        """
        program = """
            func.func @test_func() {
              // CHECK: %0 = "quantum.alloc"() <{nqubits_attr = 15 : i64}> : () -> !quantum.reg
              %0 = "quantum.alloc"() <{nqubits_attr = 2 : i64}> : () -> !quantum.reg
              return
            }
        """

        ctx = Context()
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((ConvertToMBQCFormalismPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)
