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

"""Utilities for translating JAX to xDSL"""

from collections.abc import Sequence

from xdsl.context import Context as xContext
from xdsl.dialects import arith as xarith
from xdsl.dialects import builtin as xbuiltin
from xdsl.dialects import func as xfunc
from xdsl.dialects import scf as xscf
from xdsl.dialects import tensor as xtensor
from xdsl.ir import Dialect as xDialect
from xdsl.parser import Parser as xParser

from .dialects import MBQC, QEC, Catalyst, Quantum, StableHLO, Transform


class QuantumParser(xParser):  # pylint: disable=abstract-method,too-few-public-methods
    """A subclass of ``xdsl.parser.Parser`` that automatically loads relevant dialects
    into the input context.

    Args:
        ctx (xdsl.context.Context): Context to use for parsing.
        input (str): Input program string to parse.
        name (str): The name for the input. ``"<unknown>"`` by default.
        extra_dialects (Sequence[xdsl.ir.Dialect]): Any additional dialects
            that should be loaded into the context before parsing.
    """

    default_dialects: tuple[xDialect] = (
        xarith.Arith,
        xbuiltin.Builtin,
        xfunc.Func,
        xscf.Scf,
        StableHLO,
        xtensor.Tensor,
        Transform,
        Quantum,
        MBQC,
        Catalyst,
        QEC,
    )

    def __init__(
        self,
        ctx: xContext,
        input: str,
        name: str = "<unknown>",
        extra_dialects: Sequence[xDialect] | None = (),
    ) -> None:
        super().__init__(ctx, input, name)

        extra_dialects = extra_dialects or ()
        for dialect in self.default_dialects + tuple(extra_dialects):
            if self.ctx.get_optional_dialect(dialect.name) is None:
                self.ctx.load_dialect(dialect)
