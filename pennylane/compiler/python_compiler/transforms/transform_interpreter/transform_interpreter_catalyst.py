from dataclasses import dataclass

from xdsl.dialects import builtin, transform
from xdsl.interpreters import Interpreter
from xdsl.passes import ModulePass

from interpreter import TransformFunctionsExt

@dataclass(frozen=True)
class TransformInterpreterPass(ModulePass):
    """Transform dialect interpreter"""

    name = "transform-interpreter"

    entry_point: str = "__transform_main"
    passes: dict[str, Callable[[], type[ModulePass]]]

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
        interpreter.register_implementations(TransformFunctionsExt(ctx, passes))
        schedule.parent_op().detach()
        interpreter.call_op(schedule, (op,))
