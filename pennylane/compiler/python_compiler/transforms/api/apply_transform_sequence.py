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

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass, PipelinePass

from .interpreter import TransformInterpreterPass  # pylint: disable=no-name-in-module

available_passes = {}


def register_pass(name, _callable):
    """Registers the passes available in the dictionary"""
    available_passes[name] = _callable  # pragma: no cover


# pylint: disable=too-few-public-methods
@dataclass(frozen=True)
class ApplyTransformSequence(ModulePass):
    """
    Looks for nested modules. Nested modules in this context are guaranteed to correspond
    to qnodes. These modules are already annotated with which passes are to be executed.
    The pass ApplyTransformSequence will run passes annotated in the qnode modules.

    At the end, we delete the list of passes as they have already been applied.
    """

    name = "apply-transform-sequence"

    def apply(  # pylint: disable=arguments-renamed,no-self-use
        self, ctx: Context, module: builtin.ModuleOp
    ) -> None:
        """Applies the transformation"""
        nested_modules = []
        for region in module.regions:
            for block in region.blocks:
                for op in block.ops:
                    if isinstance(op, builtin.ModuleOp):
                        nested_modules.append(op)

        pipeline = PipelinePass(
            # pylint: disable-next=unexpected-keyword-arg
            (TransformInterpreterPass(passes=available_passes),)
        )
        for op in nested_modules:
            pipeline.apply(ctx, op)

        for mod in nested_modules:
            for region in mod.regions:
                for block in region.blocks:
                    for op in block.ops:
                        if isinstance(op, builtin.ModuleOp) and op.get_attr_or_prop(
                            "transform.with_named_sequence"
                        ):
                            block.erase_op(op)  # pragma: no cover
