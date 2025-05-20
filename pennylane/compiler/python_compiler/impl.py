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
"""This file contains the implementation of the PennyLane-xDSL integration API."""


import io

from jax._src.interpreters import mlir
from jaxlib.mlir.dialects import stablehlo
from jaxlib.mlir.ir import Context as jaxContext  # pylint: disable=no-name-in-module
from jaxlib.mlir.ir import Module as jaxModule  # pylint: disable=no-name-in-module
from xdsl.context import Context as xdslContext
from xdsl.dialects import arith, builtin, func, scf
from xdsl.dialects import stablehlo as xdslstablehlo
from xdsl.dialects import tensor, transform
from xdsl.parser import Parser
from xdsl.passes import PipelinePass
from xdsl.printer import Printer

from .quantum_dialect import QuantumDialect as Quantum
from .transforms import ApplyTransformSequence


# pylint: disable=too-few-public-methods
class Compiler:
    """Compiler namespace"""

    @staticmethod
    def run(jmod: jaxModule) -> jaxModule:
        """Runs the apply-transform-sequence pass.

        The apply-transform-sequence pass is a "meta-pass". In other words,
        it is a pass that runs other passes.
        """

        gentxtmod: str = jmod.operation.get_asm(
            binary=False, print_generic_op_form=True, assume_verified=True
        )

        ctx = xdslContext(allow_unregistered=True)
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(builtin.Builtin)
        ctx.load_dialect(func.Func)
        ctx.load_dialect(scf.Scf)
        ctx.load_dialect(xdslstablehlo.StableHLO)
        ctx.load_dialect(tensor.Tensor)
        ctx.load_dialect(transform.Transform)
        ctx.load_dialect(Quantum)

        xmod: builtin.ModuleOp = Parser(ctx, gentxtmod).parse_module()
        pipeline = PipelinePass((ApplyTransformSequence(),))
        # xmod is modified in place
        pipeline.apply(ctx, xmod)

        buffer = io.StringIO()
        Printer(stream=buffer, print_generic_format=True).print(xmod)
        with jaxContext() as ctx:
            ctx.allow_unregistered_dialects = True
            ctx.append_dialect_registry(mlir.upstream_dialects)
            stablehlo.register_dialect(ctx)  # pylint: disable=no-member
            newmod: jaxModule = jaxModule.parse(buffer.getvalue())

        return newmod
