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

"""Extension to TransformInterpreterPass"""

from dataclasses import dataclass
from typing import Callable

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.interpreters import Interpreter
from xdsl.passes import ModulePass
from xdsl.transforms.transform_interpreter import TransformInterpreterPass

from .interpreter import TransformFunctionsExt


@dataclass(frozen=True)
class TransformInterpreterPassExt(TransformInterpreterPass):
    """Extension of TransformInterpreterPass in xDSL.

    Instead of running the usual transform interpreter semantics,
    we use the semantics implemented in TransformFunctionsExt.
    """

    passes: dict[str, Callable[[], type[ModulePass]]]

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        schedule = TransformInterpreterPass.find_transform_entry_point(op, self.entry_point)
        interpreter = Interpreter(op)
        interpreter.register_implementations(TransformFunctionsExt(ctx, self.passes))
        schedule.parent_op().detach()
        interpreter.call_op(schedule, (op,))
