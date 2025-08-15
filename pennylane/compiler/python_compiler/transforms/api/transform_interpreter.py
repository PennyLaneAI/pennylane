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
to be passed in as options and for the pipeline to have a callback and apply it
after every pass is run. The callback differs from how xDSL callback mechanism
is integrated into the PassPipeline object since PassPipeline only runs
if there are more than two passes. Here we are running one pass at a time
which will prevent the callback from being called.


See here (link valid with xDSL 0.46): https://github.com/xdslproject/xdsl/blob/334492e660b1726bc661efc7afb927e74bac48f4/xdsl/passes.py#L211-L222
"""

import io
from collections.abc import Callable

from catalyst.compiler import _quantum_opt  # pylint: disable=protected-access
from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects.transform import NamedSequenceOp
from xdsl.interpreter import Interpreter, PythonValues, impl, register_impls
from xdsl.interpreters.transform import TransformFunctions
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PassPipeline
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter
from xdsl.utils.exceptions import PassFailedException

from ...dialects.transform import ApplyRegisteredPassOp


# pylint: disable=too-few-public-methods
@register_impls
class TransformFunctionsExt(TransformFunctions):
    """
    Unlike the implementation available in xDSL, this implementation overrides
    the semantics of the `transform.apply_registered_pass` operation by
    first always attempting to apply the xDSL pass, but if it isn't found
    then it will try to run this pass in Catalyst.
    """

    def __init__(self, ctx, passes, callback=None):
        super().__init__(ctx, passes)
        # The signature of the callback function is assumed to be
        # the one used in xDSL:
        # def callback(previous_pass: ModulePass, module: ModuleOp, next_pass: ModulePass) -> None:
        self.callback = callback

    @impl(ApplyRegisteredPassOp)
    def run_apply_registered_pass_op(  # pragma: no cover
        self,
        _interpreter: Interpreter,
        op: ApplyRegisteredPassOp,
        args: PythonValues,
    ) -> PythonValues:
        """Try to run the pass in xDSL, if it can't run on catalyst"""

        pass_name = op.pass_name.data  # pragma: no cover
        module = args[0]

        if pass_name in self.passes:
            # pragma: no cover
            pass_class = self.passes[pass_name]()
            pass_instance = pass_class(**op.options.data)
            pipeline = PassPipeline((pass_instance,))
            pipeline.apply(self.ctx, module)
            if self.callback:
                next_pass = None
                self.callback(pass_instance, module, next_pass)
            return (module,)

        # pragma: no cover
        buffer = io.StringIO()

        Printer(stream=buffer, print_generic_format=True).print_op(module)
        schedule = f"--{pass_name}"
        modified = _quantum_opt(schedule, "-mlir-print-op-generic", stdin=buffer.getvalue())

        data = Parser(self.ctx, modified).parse_module()
        rewriter = Rewriter()
        rewriter.replace_op(module, data)
        if self.callback:
            previous_pass = None
            next_pass = None
            self.callback(previous_pass, data, next_pass)
        return (data,)


class TransformInterpreterPass(ModulePass):
    """Transform dialect interpreter"""

    passes: dict[str, Callable[[], type[ModulePass]]]
    name = "transform-interpreter"
    callback: Callable[[ModulePass, builtin.ModuleOp, ModulePass], None] | None = None

    entry_point: str = "__transform_main"

    def __init__(self, passes, callback):
        self.passes = passes
        self.callback = callback

    @staticmethod
    def find_transform_entry_point(root: builtin.ModuleOp, entry_point: str) -> NamedSequenceOp:
        """Find the entry point of the program"""
        for op in root.walk():
            if isinstance(op, NamedSequenceOp) and op.sym_name.data == entry_point:
                return op
        raise PassFailedException(  # pragma: no cover
            f"{root} could not find a nested named sequence with name: {entry_point}"
        )

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """Run the interpreter with op."""
        schedule = TransformInterpreterPass.find_transform_entry_point(op, self.entry_point)
        interpreter = Interpreter(op)
        interpreter.register_implementations(TransformFunctionsExt(ctx, self.passes, self.callback))
        schedule.parent_op().detach()
        interpreter.call_op(schedule, (op,))
