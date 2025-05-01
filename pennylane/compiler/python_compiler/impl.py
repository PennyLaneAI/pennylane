import io

from jaxlib.mlir.ir import Context as jaxContext, Module as jaxModule
from jaxlib.mlir.dialects import stablehlo
from jax._src.interpreters import mlir

from xdsl.context import Context as xdslContext
from xdsl.dialects import arith, builtin, func, scf, tensor, transform
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.passes import PipelinePass

from .quantum_dialect import QuantumDialect as Quantum

class Compiler:

    def run(self, jmod: jaxModule) -> jaxModule:
        """Runs the apply-transform-sequence pass.

        The apply-transform-sequence pass is a "meta-pass". In other words,
        it is a pass that runs other passes.
        """

        gentxtmod: str = jmod.operation.get_asm(binary=False, print_generic_op_form=True, assume_verified=True)

        ctx = xdslContext(allow_unregistered=True)
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(builtin.Builtin)
        ctx.load_dialect(func.Func)
        ctx.load_dialect(scf.Scf)
        ctx.load_dialect(tensor.Tensor)
        ctx.load_dialect(transform.Transform)
        ctx.load_dialect(Quantum)

        xmod: builtin.ModuleOp = Parser(ctx, gentxtmod).parse_module()
        pipeline = passes.PipelinePass((ApplyTransformSequence(),))
        # xmod is modified in place
        pipeline.apply(ctx, xmod)

        buffer = io.StringIO()
        Printer(stream=buffer, print_generic_format=True).print(xmod)
        with jaxContext() as ctx:
            ctx.allow_unregistered_dialects = True
            ctx.append_dialect_registry(mlir.upstream_dialects)
            stablehlo.register_dialect(ctx)
            newmod: jaxModule = Module.parse(buffer.getvalue())

        return newmod
