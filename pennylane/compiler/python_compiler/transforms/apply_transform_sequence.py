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
"""This file contains the pass that applies all passes present in the program representation."""

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
