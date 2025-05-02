from dataclasses import dataclass

from xdsl.dialects import builtin
from xdsl.context import Context
from xdsl.passes import ModulePass, PipelinePass

from .transform_interpreter import TransformInterpreterPass

available_passes = {}

def register_pass(name, _callable):
    available_passes[name] = _callable

@dataclass(frozen=True)
class ApplyTransformSequence(ModulePass):
    name = "apply-transform-sequence"

    def apply(self, ctx: Context, module: builtin.ModuleOp) -> None:
        nested_modules = []
        for region in module.regions:
            for block in region.blocks:
                for op in block.ops:
                    if isinstance(op, builtin.ModuleOp):
                        nested_modules.append(op)

        pipeline = PipelinePass((TransformInterpreterPass(passes=available_passes),))
        for op in nested_modules:
            pipeline.apply(ctx, op)

        for op in nested_modules:
            for region in op.regions:
                for block in region.blocks:
                    for op in block.ops:
                        if isinstance(op, builtin.ModuleOp) and op.get_attr_or_prop("transform.with_named_sequence"):
                            block.erase_op(op)
