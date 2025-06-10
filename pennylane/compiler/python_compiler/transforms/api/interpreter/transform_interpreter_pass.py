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

from typing import Callable

from xdsl.context import Context
from xdsl.dialects import builtin, transform
from xdsl.interpreters import Interpreter
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import PassFailedException

from .transform_functions import TransformFunctionsExt


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
