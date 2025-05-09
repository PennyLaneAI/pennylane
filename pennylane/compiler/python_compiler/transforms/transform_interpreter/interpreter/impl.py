import io
from typing import Callable

from xdsl.interpreter import Interpreter, PythonValues, impl, register_impls
from xdsl.interpreters.transform import TransformFunctions
from xdsl.context import Context
from xdsl.dialects import transform
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PipelinePass
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter
from xdsl.utils import parse_pipeline

from catalyst.compiler import _quantum_opt

@register_impls
class TransformFunctionsExt(TransformFunctions):
    ctx: Context
    passes: dict[str, Callable[[], type[ModulePass]]]

    def __init__(
        self, ctx: Context, passes : dict[str, Callable[[], type[ModulePass]]]
    ):
        self.ctx = ctx
        self.passes = passes

    @impl(transform.ApplyRegisteredPassOp)
    def run_apply_registered_pass_op(
        self,
        interpreter: Interpreter,
        op: transform.ApplyRegisteredPassOp,
        args: PythonValues,
    ) -> PythonValues:
        pass_name = op.pass_name.data
        requested_by_user = PipelinePass.build_pipeline_tuples(
            self.passes, parse_pipeline.parse_pipeline(pass_name)
        )

        try:
            schedule = tuple(
                pass_type.from_pass_spec(spec) for pass_type, spec in requested_by_user
            )
            pipeline = PipelinePass(schedule)
            pipeline.apply(self.ctx, args[0])
            return (args[0],)
        except:
            buffer = io.StringIO()

            Printer(stream=buffer, print_generic_format=True).print(args[0])
            schedule = f"--{pass_name}"
            modified = _quantum_opt(schedule, "-mlir-print-op-generic", stdin=buffer.getvalue())

            module = args[0]
            data = Parser(self.ctx, modified).parse_module()
            rewriter = Rewriter()
            rewriter.replace_op(module, data)
            return (data,)
