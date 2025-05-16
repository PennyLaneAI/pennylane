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

"""
An custom implementation of the interpreter functions used for the transform dialect.
"""

import io
from typing import Callable

from xdsl.context import Context
from xdsl.dialects import transform
from xdsl.interpreter import Interpreter, PythonValues, impl, register_impls
from xdsl.interpreters.transform import TransformFunctions
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PipelinePass
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter
from xdsl.utils import parse_pipeline


# pylint: disable=too-few-public-methods
@register_impls
class TransformFunctionsExt(TransformFunctions):
    """
    Unlike the implementation available in xDSL, this implementation overrides
    the semantics of the `transform.apply_registered_pass` operation by
    first always attempting to apply the xDSL pass, but if it isn't found
    then it will try to run this pass in Catalyst.
    """

    ctx: Context
    passes: dict[str, Callable[[], type[ModulePass]]]

    def __init__(
        self, ctx: Context, passes: dict[str, Callable[[], type[ModulePass]]]
    ):  # pragma: no cover

        self.ctx = ctx  # pragma: no cover
        self.passes = passes  # pragma: no cover

    @impl(transform.ApplyRegisteredPassOp)
    def run_apply_registered_pass_op(  # pragma: no cover
        self,
        _interpreter: Interpreter,
        op: transform.ApplyRegisteredPassOp,
        args: PythonValues,
    ) -> PythonValues:
        """Try to run the pass in xDSL, if it can't run on catalyst"""

        pass_name = op.pass_name.data  # pragma: no cover
        requested_by_user = PipelinePass.build_pipeline_tuples(
            self.passes, parse_pipeline.parse_pipeline(pass_name)
        )

        try:
            # pragma: no cover
            schedule = tuple(
                pass_type.from_pass_spec(spec) for pass_type, spec in requested_by_user
            )
            pipeline = PipelinePass(schedule)
            pipeline.apply(self.ctx, args[0])
            return (args[0],)
        except:  # pylint: disable=bare-except
            # pragma: no cover
            # Yes, we are importing it here because we don't want to import it
            # and have an import time dependency on catalyst.
            # This allows us to test without having a hard requirement on Catalyst
            from catalyst.compiler import _quantum_opt  # pylint: disable=import-outside-toplevel

            buffer = io.StringIO()

            Printer(stream=buffer, print_generic_format=True).print(args[0])
            schedule = f"--{pass_name}"
            modified = _quantum_opt(schedule, "-mlir-print-op-generic", stdin=buffer.getvalue())

            module = args[0]
            data = Parser(self.ctx, modified).parse_module()
            rewriter = Rewriter()
            rewriter.replace_op(module, data)
            return (data,)
