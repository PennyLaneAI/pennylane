import numpy as np
import xdsl
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, scf, tensor, transform

import pennylane as qml

# from pennylane.compiler.python_compiler.quantum_dialect import QuantumDialect
from pennylane.compiler.python_compiler.quantum_dialect import QuantumDialect
from pennylane.ops.op_math.decompositions import one_qubit_decomposition, two_qubit_decomposition

# The allow_unregistered option is important.
# It says that the program will contain things the context doesn't understand
# This is needed for the catalyst.launch_kernel operation
ctx = Context(allow_unregistered=True)

# Load the dialects that the context does understand.
ctx.load_dialect(arith.Arith)
ctx.load_dialect(builtin.Builtin)
ctx.load_dialect(func.Func)
ctx.load_dialect(scf.Scf)
ctx.load_dialect(tensor.Tensor)
ctx.load_dialect(transform.Transform)
ctx.load_dialect(QuantumDialect)

from dataclasses import dataclass

from xdsl import context, passes
from xdsl.utils import parse_pipeline


@dataclass(frozen=True)
# All passes inherit from passes.ModulePass
class PrintModule(passes.ModulePass):
    # All passes require a name field
    name = "print"

    # All passes require an apply method with this signature.
    def apply(self, ctx: context.MLContext, module: builtin.ModuleOp) -> None:
        print("Hello from inside the pass\n", module)


from typing import Callable

CustomOp = QuantumDialect._operations[4]


from xdsl import pattern_rewriter
from xdsl.rewriter import InsertPoint

U = np.array(
    [
        [-0.17111489, -0.69352236],
        [0.25053735, 0.60700543],
    ]
)


@qml.qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def circuit():
    qml.RX(0.1, wires=0)
    qml.QubitUnitary(U, wires=2)
    return qml.state()


mlir_string = circuit.mlir
print(mlir_string)


from catalyst.compiler import _quantum_opt

generic = _quantum_opt(
    ("--pass-pipeline", "builtin.module(canonicalize)"), "-mlir-print-op-generic", stdin=mlir_string
)
print(generic)


# The allow_unregistered option is important.
# It says that the program will contain things the context doesn't understand
# This is needed for the catalyst.launch_kernel operation
ctx = Context(allow_unregistered=True)

# Load the dialects that the context does understand.
ctx.load_dialect(arith.Arith)
ctx.load_dialect(builtin.Builtin)
ctx.load_dialect(func.Func)
ctx.load_dialect(scf.Scf)
ctx.load_dialect(tensor.Tensor)
ctx.load_dialect(transform.Transform)
ctx.load_dialect(QuantumDialect)

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


QubitUnitaryOp = QuantumDialect._operations[20]


from xdsl import pattern_rewriter
from xdsl.ir import BlockArgument, Operation, OpResult
from xdsl.rewriter import InsertPoint


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

            # breakpoint()

            # ops = [RZ(tensor(0., requires_grad=True), wires=[2]), RY(tensor(1.05043081, requires_grad=True), wires=[2]), RZ(tensor(0., requires_grad=True), wires=[2])]
            # Now I just need to create a 'CustomOp' for every op in ops
            # and replace the QubitUnitaryOp with the new ops.

            # 'CustomOp' inherits from the 'IRDLOperation' class defined in xdsl.
            # How can I create a new 'CustomOp' and replace the QubitUnitaryOp with it?

            # new_op = CustomOp(
            #    operands=...
            #    properties=...
            #    attributes=...
            #    successors=...
            #    regions=...
            #    result_types=...
            # )

            from xdsl.dialects.arith import ConstantOp
            from xdsl.dialects.builtin import (
                AnyAttr,
                AnyOf,
                BaseAttr,
                Float64Type,
                FloatAttr,
                IntegerType,
                StringAttr,
                UnitAttr,
            )

            # angle = ops[0].parameters[0].item()

            # Convert raw float to a FloatAttr (with correct type)
            # angle_attr = FloatAttr(angle, Float64Type())

            # Create a constant op from the attribute
            # angle_const_op.result gets the SSA value (result) from the constant op
            # angle_const_op = ConstantOp(value=angle_attr)

            # Insert the constant op into the IR BEFORE `op`
            # rewriter.insert_op(angle_const_op, InsertPoint.before(op))

            # Operand is the original qubit being used:
            current_input = op.in_qubits[0]  # SSAValue

            for qml_op in ops:
                angle = qml_op.parameters[0].item()
                gate_name = qml_op.name  # e.g., "RX", "RY", "RZ"

                # Build constant op for angle
                angle_attr = FloatAttr(angle, Float64Type())
                angle_const_op = ConstantOp(value=angle_attr)
                rewriter.insert_op(angle_const_op, InsertPoint.before(op))  # insert constant

                angle_ssa = angle_const_op.result

                # Create and insert the custom op
                custom_op = CustomOp(
                    operands=(angle_ssa, current_input, None, None),
                    properties={
                        "gate_name": StringAttr(gate_name),
                        **op.properties,
                    },
                    attributes=op.attributes,
                    successors=op.successors,
                    regions=op.regions,
                    result_types=(op.result_types, []),
                )
                rewriter.insert_op(custom_op, InsertPoint.before(op))

                for old_res, new_res in zip(op.results, custom_op.results):
                    old_res.replace_by(new_res)

                # Update current input for next gate in chain
                current_input = custom_op.in_qubits[0]

            breakpoint()

            # new_op = CustomOp(
            #    operands=(angle_const_op.result, qubit_operand, None, None),
            #    properties={
            #        "gate_name": StringAttr("RX"),
            #        **op.properties,
            #    },
            #    attributes=op.attributes,
            #    successors=op.successors,
            #    regions=op.regions,
            #    result_types=(op.result_types, []),
            # )

            # breakpoint()

            # Step 6: Insert before the original op
            # rewriter.insert_op(new_op, InsertPoint.before(op))

            # Step 7: Replace uses if needed
            # for old_res, new_res in zip(op.results, new_op.results):
            #    old_res.replace_by(new_res)

            # Step 8: Erase original op
            rewriter.erase_op(op)

            # Continue from here
            pass


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

user_requested_pass = "unitary-to-rot"  # just for example

requested_by_user = passes.PipelinePass.build_pipeline_tuples(
    available_passes, parse_pipeline.parse_pipeline(user_requested_pass)
)
schedule = tuple(pass_type.from_pass_spec(spec) for pass_type, spec in requested_by_user)
pipeline = passes.PipelinePass(schedule)
pipeline.apply(ctx, m)
print(m)


# op.matrix = <OpResult[tensor<2x2xcomplex<f64>>] index: 0, operation: builtin.unregistered, uses: 1>
# op.matrix.owner = UnregisteredOp.with_name.<locals>.UnregisteredOpWithNameOp(%0 = "stablehlo.convert"(%arg0) : (tensor<2x2xf64>) -> tensor<2x2xcomplex<f64>>)
# op.matrix.op = UnregisteredOp.with_name.<locals>.UnregisteredOpWithNameOp(%0 = "stablehlo.convert"(%arg0) : (tensor<2x2xf64>) -> tensor<2x2xcomplex<f64>>)
# matrix_op = UnregisteredOp.with_name.<locals>.UnregisteredOpWithNameOp(%0 = "stablehlo.convert"(%arg0) : (tensor<2x2xf64>) -> tensor<2x2xcomplex<f64>>)
# matrix_op.name = builtin.unregistered
