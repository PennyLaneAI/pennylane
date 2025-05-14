from dataclasses import dataclass
from typing import Callable

import numpy as np
import xdsl
from catalyst.compiler import _quantum_opt
from xdsl import context, passes, pattern_rewriter
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, scf, tensor, transform
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import Float64Type, FloatAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.ir import BlockArgument, OpResult
from xdsl.rewriter import InsertPoint
from xdsl.utils import parse_pipeline

import pennylane as qml
from pennylane.compiler.python_compiler.quantum_dialect import QuantumDialect, QubitType
from pennylane.ops.op_math.decompositions import one_qubit_decomposition, two_qubit_decomposition

ctx = Context(allow_unregistered=True)
ctx.load_dialect(arith.Arith)
ctx.load_dialect(builtin.Builtin)
ctx.load_dialect(func.Func)
ctx.load_dialect(scf.Scf)
ctx.load_dialect(tensor.Tensor)
ctx.load_dialect(transform.Transform)
ctx.load_dialect(QuantumDialect)


@dataclass(frozen=True)
# All passes inherit from passes.ModulePass
class PrintModule(passes.ModulePass):
    # All passes require a name field
    name = "print"

    # All passes require an apply method with this signature.
    def apply(self, ctx: context.MLContext, module: builtin.ModuleOp) -> None:
        print("Hello from inside the pass\n", module)


CustomOp = QuantumDialect._operations[4]
QubitUnitaryOp = QuantumDialect._operations[20]

U = np.array(
    [
        [-0.17111489, -0.69352236],
        [0.25053735, 0.60700543],
    ]
)


@qml.qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def circuit():
    qml.QubitUnitary(U, wires=2)
    return qml.state()


mlir_string = circuit.mlir
print(mlir_string)

generic = _quantum_opt(
    ("--pass-pipeline", "builtin.module(canonicalize)"), "-mlir-print-op-generic", stdin=mlir_string
)
m = xdsl.parser.Parser(ctx, generic).parse_module()
print(m)


available_passes: dict[str, Callable[[], type[passes.ModulePass]]] = {}
available_passes["print"] = lambda: PrintModule
user_requested_pass = "print"  # just for example
requested_by_user = passes.PipelinePass.build_pipeline_tuples(
    available_passes, parse_pipeline.parse_pipeline(user_requested_pass)
)
schedule = tuple(pass_type.from_pass_spec(spec) for pass_type, spec in requested_by_user)
pipeline = passes.PipelinePass(schedule)
pipeline.apply(ctx, m)


# This function is still a bit of a mess and right now doesn't work
def extract_matrix_value(rewriter, op, module):

    # TODO: This function doesn't work yet.
    # To move on, let's just return a random matrix
    return qml.numpy.random.rand(2, 2)

    convert_op = op.matrix.op
    op_name = convert_op.op_name.data

    if op_name != "stablehlo.convert":
        return None

    convert_operand = convert_op.operands[0]

    if isinstance(convert_operand, OpResult):
        input_op = convert_operand.op
        if input_op.op_name.data == "stablehlo.constant":
            return input_op.attributes["value"].value.data
        else:
            raise NotImplementedError(
                "The input operand is not a constant. We don't know how to handle this case."
            )

    elif isinstance(convert_operand, BlockArgument):

        def get_parent_of_type(op, typ):
            """Recursive call parent until we find a parent of type `typ`."""
            if isinstance(op, typ):
                return op
            if isinstance(op, BlockArgument):
                return get_parent_of_type(op.owner.parent_op(), typ)
            return get_parent_of_type(op.parent_op(), typ)

        funcOp = get_parent_of_type(convert_operand, func.FuncOp)
        deviceProgram = get_parent_of_type(funcOp, builtin.ModuleOp)
        hostProgram = get_parent_of_type(deviceProgram.parent_op(), builtin.ModuleOp)

        arg_index = funcOp.body.blocks[0].args.index(convert_operand)

        for operation in hostProgram.walk():

            if not operation.name == "catalyst.launch_kernel":
                continue

            pass

    return None


class UnitaryToRotPattern(pattern_rewriter.RewritePattern):
    def __init__(self, module):
        self.module = module
        super().__init__()

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):
        for op in funcOp.body.walk():
            if not isinstance(op, QubitUnitaryOp):
                continue
            matrix_value = extract_matrix_value(rewriter, op, self.module)
            extract_op = op.in_qubits[0].owner
            wire_index = extract_op.idx_attr.parameters[0].data
            matrix_shape = qml.math.shape(matrix_value)
            if matrix_shape == (2, 2):
                ops = one_qubit_decomposition(matrix_value, wire_index)
            elif matrix_shape == (4, 4):
                ops = two_qubit_decomposition(matrix_value, wire_index)
            else:
                ops = [op]

            for qml_op in ops:
                angle = qml_op.parameters[0].item()
                wire = qml_op.wires[0]

                # Build constant op for angle
                # and inserts the constant op corresponding to the angle
                # into the IR, before QubitUnitaryOp
                # (right now know it is a float)
                angle_attr = FloatAttr(angle, Float64Type())
                angle_const_op = ConstantOp(value=angle_attr)
                rewriter.insert_op(angle_const_op, InsertPoint.before(op))

                # And we do the same for the wire
                wire_attr = IntegerAttr(wire, IntegerType(64))
                wire_const_op = ConstantOp(value=wire_attr)
                rewriter.insert_op(wire_const_op, InsertPoint.before(op))

                # angle_const_op.result is the SSA value of the constant op
                # TODO: here I am simply parsing the properties and successors of the QubitUnitaryOp
                # and passing them to the custom op. But it would be probably better to it differently.
                custom_op = CustomOp(
                    operands=(angle_const_op.result, wire_const_op.result, None, None),
                    properties={
                        "gate_name": StringAttr(qml_op.name),
                        **op.properties,
                    },
                    attributes={},
                    successors=op.successors,
                    regions=(),
                    result_types=(QubitType(), []),
                )

                # This line inserts the custom op into the IR, before QubitUnitaryOp
                rewriter.insert_op(custom_op, InsertPoint.before(op))

                # This line replaces the uses (SSA values) of the QubitUnitaryOp with the new custom op
                # This is needed because the QubitUnitaryOp is being erased at the end
                for old_res, new_res in zip(op.results, custom_op.results):
                    old_res.replace_by(new_res)

            # We finally erase the QubitUnitaryOp
            rewriter.erase_op(op)


@dataclass(frozen=True)
class UnitaryToRotPass(passes.ModulePass):
    name = "unitary-to-rot"

    def apply(self, ctx: context.MLContext, module: builtin.ModuleOp) -> None:
        pattern = UnitaryToRotPattern(module)
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([pattern])
        ).rewrite_module(module)


print(f"\n\n\n\n\n\n\n\n")
print(f"Running the pass")


available_passes["unitary-to-rot"] = lambda: UnitaryToRotPass
user_requested_pass = "unitary-to-rot"
requested_by_user = passes.PipelinePass.build_pipeline_tuples(
    available_passes, parse_pipeline.parse_pipeline(user_requested_pass)
)
schedule = tuple(pass_type.from_pass_spec(spec) for pass_type, spec in requested_by_user)
pipeline = passes.PipelinePass(schedule)
pipeline.apply(ctx, m)
print(m)
