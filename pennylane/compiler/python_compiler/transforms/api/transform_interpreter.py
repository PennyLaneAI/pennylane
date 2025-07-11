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

"""Custom Transform Dialect Interpreter Pass

Differs from xDSL's upstream implementation by allowing passes
to be passed in as options.
"""

import io
from collections.abc import Callable

from catalyst.compiler import _quantum_opt  # pylint: disable=protected-access
from xdsl.context import Context
from xdsl.dialects import builtin, transform
from xdsl.interpreter import Interpreter, PythonValues, impl, register_impls
from xdsl.interpreters.transform import TransformFunctions
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PipelinePass
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter
from xdsl.utils import parse_pipeline
from xdsl.utils.exceptions import PassFailedException


# pylint: disable=too-few-public-methods
@register_impls
class TransformFunctionsExt(TransformFunctions):
    """
    Unlike the implementation available in xDSL, this implementation overrides
    the semantics of the `transform.apply_registered_pass` operation by
    first always attempting to apply the xDSL pass, but if it isn't found
    then it will try to run this pass in Catalyst.
    """

    @impl(transform.ApplyRegisteredPassOp)
    def run_apply_registered_pass_op(  # pragma: no cover
        self,
        _interpreter: Interpreter,
        op: transform.ApplyRegisteredPassOp,
        args: PythonValues,
    ) -> PythonValues:
        """Try to run the pass in xDSL, if it can't run on catalyst"""

        pass_name = op.pass_name.data  # pragma: no cover
        if pass_name in self.passes:
            # pragma: no cover
            pipeline = PipelinePass(
                tuple(
                    PipelinePass.iter_passes(self.passes, parse_pipeline.parse_pipeline(pass_name))
                )
            )
            pipeline.apply(self.ctx, args[0])
            return (args[0],)

        # pragma: no cover
        buffer = io.StringIO()

        Printer(stream=buffer, print_generic_format=True).print(args[0])
        schedule = f"--{pass_name}"
        modified = _quantum_opt(schedule, "-mlir-print-op-generic", stdin=buffer.getvalue())

        module = args[0]
        data = Parser(self.ctx, modified).parse_module()
        rewriter = Rewriter()
        rewriter.replace_op(module, data)
        return (data,)


class TransformInterpreterPass(ModulePass):
    """Transform dialect interpreter"""

    passes: dict[str, Callable[[], type[ModulePass]]]
    name = "transform-interpreter"

    entry_point: str = "__transform_main"

    def __init__(self, passes):
        self.passes = passes

    @staticmethod
    def find_transform_entry_point(
        root: builtin.ModuleOp, entry_point: str
    ) -> transform.NamedSequenceOp:
        """Find the entry point of the program"""
        for op in root.walk():
            if isinstance(op, transform.NamedSequenceOp) and op.sym_name.data == entry_point:
                return op
        raise PassFailedException(  # pragma: no cover
            f"{root} could not find a nested named sequence with name: {entry_point}"
        )

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """Run the interpreter with op."""
        schedule = TransformInterpreterPass.find_transform_entry_point(op, self.entry_point)
        interpreter = Interpreter(op)
        interpreter.register_implementations(TransformFunctionsExt(ctx, self.passes))
        schedule.parent_op().detach()
        interpreter.call_op(schedule, (op,))
