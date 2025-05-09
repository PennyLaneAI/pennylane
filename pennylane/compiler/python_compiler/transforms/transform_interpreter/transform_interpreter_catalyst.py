from dataclasses import dataclass
from typing import Callable

from xdsl.context import Context
from xdsl.dialects import builtin, transform
from xdsl.interpreters import Interpreter
from xdsl.passes import ModulePass

from .interpreter import TransformFunctionsExt

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
        for op in root.walk():
            if (
                isinstance(op, transform.NamedSequenceOp)
                and op.sym_name.data == entry_point
            ):
                return op
        raise PassFailedException(
            f"{root} could not find a nested named sequence with name: {entry_point}"
        )

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        schedule = TransformInterpreterPass.find_transform_entry_point(
            op, self.entry_point
        )
        interpreter = Interpreter(op)
        interpreter.register_implementations(TransformFunctionsExt(ctx, self.passes))
        schedule.parent_op().detach()
        interpreter.call_op(schedule, (op,))
