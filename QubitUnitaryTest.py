# %% [markdown]
# # Testing the Python Compiler

# %% [markdown]
# The purpose of this notebook is to test the Quantum dialect for the Python Compiler project.
#
# The idea is to have something similar to [this notebook](https://colab.research.google.com/drive/1qKRTRgxjDxYE_MDTfpqImxrx0fRhwICA#scrollTo=3GKCogPId4I8), but using the Quantum dialect generated with [xDSL](https://github.com/xdslproject/xdsl) provided in this module.

# %%
import pennylane as qml
import numpy as np

# from pennylane.compiler.python_compiler.quantum_dialect import QuantumDialect
from pennylane.compiler.python_compiler.quantum_dialect import QuantumDialect

from pennylane.ops.op_math.decompositions import one_qubit_decomposition, two_qubit_decomposition

# %% [markdown]
# ## Generating string representation of the program from Catalyst


# %%
@qml.qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit():
    qml.PauliX(wires=[0])
    qml.Hadamard(wires=[0])
    qml.Hadamard(wires=[0])
    qml.PauliX(wires=[0])
    return qml.state()


mlir_string = circuit.mlir
print(mlir_string)

# %%
from catalyst.compiler import _quantum_opt

generic = _quantum_opt(
    ("--pass-pipeline", "builtin.module(canonicalize)"), "-mlir-print-op-generic", stdin=mlir_string
)
print(generic)

# %% [markdown]
# ## Loading dialect

# %%
import xdsl
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, scf, tensor, transform

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

# %%
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


# %%
from typing import Callable

available_passes: dict[str, Callable[[], type[passes.ModulePass]]] = {}
available_passes["print"] = lambda: PrintModule

user_requested_pass = "print"  # just for example

requested_by_user = passes.PipelinePass.build_pipeline_tuples(
    available_passes, parse_pipeline.parse_pipeline(user_requested_pass)
)
schedule = tuple(pass_type.from_pass_spec(spec) for pass_type, spec in requested_by_user)
pipeline = passes.PipelinePass(schedule)
pipeline.apply(ctx, m)

# %%
QuantumDialect._operations

# %%
CustomOp = QuantumDialect._operations[4]

# %% [markdown]
# ## Implementing the CancelInverses transform

# %%
from xdsl.rewriter import InsertPoint
from xdsl import pattern_rewriter

self_inverses = ("PauliZ", "PauliX", "PauliY", "Hadamard", "Identity")


def cancel_ops(rewriter, op, next_op):
    rewriter._replace_all_uses_with(next_op.results[0], op.in_qubits[0])
    rewriter.erase_op(next_op)
    rewriter.erase_op(op)
    owner = op.in_qubits[0].owner

    if isinstance(owner, CustomOp) and owner.gate_name.data in self_inverses:
        next_user = None

        for use in owner.results[0].uses:
            user = use.operation
            if isinstance(user, CustomOp) and user.gate_name.data == owner.gate_name.data:
                next_user = user
                break

        if next_user is not None:
            cancel_ops(rewriter, owner, next_user)


class DeepCancelInversesSingleQubitPattern(pattern_rewriter.RewritePattern):
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):
        """Deep Cancel for Self Inverses"""
        for op in funcOp.body.walk():
            if not isinstance(op, CustomOp):
                continue

            if op.gate_name.data not in self_inverses:
                continue

            next_user = None
            for use in op.results[0].uses:
                user = use.operation
                if isinstance(user, CustomOp) and user.gate_name.data == op.gate_name.data:
                    next_user = user
                    break

            if next_user is not None:
                cancel_ops(rewriter, op, next_user)


@dataclass(frozen=True)
class DeepCancelInversesSingleQubitPass(passes.ModulePass):
    name = "deep-cancel-inverses-single-qubit"

    def apply(self, ctx: context.MLContext, module: builtin.ModuleOp) -> None:
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([DeepCancelInversesSingleQubitPattern()])
        ).rewrite_module(module)


# %%
available_passes["deep-cancel-inverses-single-qubit"] = lambda: DeepCancelInversesSingleQubitPass

user_requested_pass = "deep-cancel-inverses-single-qubit"  # just for example

requested_by_user = passes.PipelinePass.build_pipeline_tuples(
    available_passes, parse_pipeline.parse_pipeline(user_requested_pass)
)
schedule = tuple(pass_type.from_pass_spec(spec) for pass_type, spec in requested_by_user)
pipeline = passes.PipelinePass(schedule)
pipeline.apply(ctx, m)

# %%
print(m)

# %% [markdown]
# ## Trying with `unitary_to_rot`

# %%
U = np.array(
    [
        [-0.17111489, -0.69352236],
        [0.25053735, 0.60700543],
    ]
)


# %%
@qml.qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def circuit():
    qml.QubitUnitary(U, wires=2)
    return qml.state()


mlir_string = circuit.mlir
print(mlir_string)

# %%
from catalyst.compiler import _quantum_opt

generic = _quantum_opt(
    ("--pass-pipeline", "builtin.module(canonicalize)"), "-mlir-print-op-generic", stdin=mlir_string
)
print(generic)

# %%
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

# %%
available_passes: dict[str, Callable[[], type[passes.ModulePass]]] = {}
available_passes["print"] = lambda: PrintModule

user_requested_pass = "print"  # just for example

requested_by_user = passes.PipelinePass.build_pipeline_tuples(
    available_passes, parse_pipeline.parse_pipeline(user_requested_pass)
)
schedule = tuple(pass_type.from_pass_spec(spec) for pass_type, spec in requested_by_user)
pipeline = passes.PipelinePass(schedule)
pipeline.apply(ctx, m)

# %%
QubitUnitaryOp = QuantumDialect._operations[20]

# %%
QubitUnitaryOp

# %%
from xdsl.rewriter import InsertPoint
from xdsl import pattern_rewriter
from xdsl.ir import OpResult, BlockArgument, Operation


def deep_walk(op):
    """Yield all ops recursively, including inside functions."""
    yield op
    if hasattr(op, "regions"):
        for region in op.regions:
            for block in region.blocks:
                for inner_op in block.ops:
                    yield from deep_walk(inner_op)


def extract_matrix_value_1(rewriter, op, module):
    convert_op = op.matrix.op
    op_name = convert_op.op_name.data
    print(f"op_name = {op_name}")

    if op_name != "stablehlo.convert":
        return None

    convert_operand = convert_op.operands[0]

    if isinstance(convert_operand, OpResult):
        input_op = convert_operand.op
        if input_op.op_name.data == "stablehlo.constant":
            return input_op.attributes["value"].value.data
        else:
            print("Input op is not a constant")

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

        # Determine the argument index
        arg_index = funcOp.body.blocks[0].args.index(convert_operand)

        print(f"Looking for launch_kernel with argument index {arg_index}")

        # Recursively walk all nested operations
        def deep_walk(op):
            yield op
            if hasattr(op, "regions"):
                for region in op.regions:
                    for block in region.blocks:
                        for inner_op in block.ops:
                            yield from deep_walk(inner_op)

        for operation in deep_walk(hostProgram):
            if not hasattr(operation, "op_name"):
                continue

            if operation.op_name.data != "catalyst.launch_kernel":
                continue

            callee = operation.attributes.get("callee")
            if not callee:
                continue

            # Check if the call matches our funcOp
            if (
                callee.root_reference.data == deviceProgram.sym_name.data
                and callee.nested_reference.data == funcOp.sym_name.data
            ):
                print(f"Matched launch_kernel for {funcOp.sym_name.data}")
                actual_arg = operation.operands[arg_index]

                # Trace the argument back to a constant
                if isinstance(actual_arg, OpResult):
                    input_op = actual_arg.op
                    if input_op.op_name.data == "stablehlo.constant":
                        return input_op.attributes["value"].value.data
                    else:
                        print("Launch arg is not a constant")
                else:
                    print("Launch arg is not an OpResult")

    return None


def extract_matrix_value_2(rewriter, op, module):
    convert_op = op.matrix.op
    print(f"op_name = {convert_op.op_name.data}")
    if convert_op.op_name.data != "stablehlo.convert":
        return None

    convert_operand = convert_op.operands[0]

    if isinstance(convert_operand, OpResult):
        input_op = convert_operand.op
        if input_op.op_name.data == "stablehlo.constant":
            return input_op.attributes["value"].value.data
        else:
            print("Input to convert is not constant")
            return None

    elif isinstance(convert_operand, BlockArgument):
        funcOp = convert_operand.owner.parent_op()
        arg_index = funcOp.body.blocks[0].args.index(convert_operand)
        print(f"Tracing argument {arg_index} of function {funcOp.sym_name.data}")

        # Go to the top-level module
        root_module = funcOp.parent_op().parent_op()

        def deep_walk(op):
            yield op
            if hasattr(op, "regions"):
                for region in op.regions:
                    for block in region.blocks:
                        for inner_op in block.ops:
                            yield from deep_walk(inner_op)

        for op in deep_walk(root_module):
            if not hasattr(op, "op_name") or op.op_name.data != "catalyst.launch_kernel":
                continue

            callee_attr = op.attributes.get("callee")
            if not callee_attr:
                continue

            if callee_attr.nested_reference.data == funcOp.sym_name.data:
                print(f"Found call to {funcOp.sym_name.data}")
                actual_arg = op.operands[arg_index]
                if isinstance(actual_arg, OpResult):
                    input_op = actual_arg.op
                    if input_op.op_name.data == "stablehlo.constant":
                        return input_op.attributes["value"].value.data
                    else:
                        print("Launch argument is not a constant")
                else:
                    print("Launch argument is not an OpResult")

    return None


def extract_matrix_value_3(rewriter, op, module):
    convert_op = op.matrix.op
    op_name = convert_op.op_name.data
    print(f"op_name = {op_name}")

    if op_name == "stablehlo.convert":
        convert_operand = convert_op.operands[0]

        if isinstance(convert_operand, OpResult):
            input_op = convert_operand.op
            if input_op.op_name.data == "stablehlo.constant":
                return input_op.attributes["value"].value.data
            else:
                print("Input op is not a constant")
        elif isinstance(convert_operand, BlockArgument):
            # This is a little bit hard coded.
            # We know we are inside of a function
            # And the question is how can I relate this block
            # argument to the function.

            # We are dealing with a function argument, we want to retrieve the actual value
            def get_parent_of_type(op, typ):
                """Recursive call parent until we find a parent of type `typ`."""
                if isinstance(op, typ):
                    return op
                if isinstance(op, BlockArgument):
                    if isinstance(op.owner.parent_op(), typ):
                        return op.owner.parent_op()
                    else:
                        return get_parent_of_type(op.owner.parent_op(), typ)
                else:
                    return get_parent_of_type(op.parent_op(), typ)

            funcOp = get_parent_of_type(convert_operand, func.FuncOp)
            deviceProgram = get_parent_of_type(funcOp, builtin.ModuleOp)
            hostProgram = get_parent_of_type(deviceProgram.parent_op(), builtin.ModuleOp)

            # Get the argument index of the block argument
            arg_index = funcOp.body.blocks[0].args.index(convert_operand)

            breakpoint()

            for operation in hostProgram.walk():

                if not hasattr(operation, "op_name"):
                    print(f"Skipping: {type(operation)}")
                    continue

                if operation.op_name.data == "catalyst.launch_kernel":

                    breakpoint()

                    callee = operation.attributes.get("callee")

                    if not callee:
                        continue

                    breakpoint()

                    # Make sure it's calling our function
                    if (
                        callee.root_reference.data == deviceProgram.sym_name.data
                        and callee.nested_reference.data == funcOp.sym_name.data
                    ):
                        # We found the right launch call

                        actual_arg = operation.operands[arg_index]

                # TODO: complete this!

    return None


def extract_matrix_value_N(rewriter, qubit_unitary_op, top_module):
    conv = qubit_unitary_op.matrix.op
    if conv.op_name.data != "stablehlo.convert":
        return None

    src = conv.operands[0]

    # Case A: direct constant
    if isinstance(src, OpResult):
        c = src.op
        if c.op_name.data == "stablehlo.constant":
            return c.attributes["value"].value.data
        return None

    # Case B: came in as function arg → must scan the host entry-point
    if isinstance(src, BlockArgument):
        device_fn = src.owner.parent_op()  # func.func @circuit
        arg_idx = device_fn.body.blocks[0].args.index(src)

        # climb up TWO levels: func.func → inner module → outer host module
        inner_mod = device_fn.parent_op()  # builtin.module @module_circuit
        host_mod = inner_mod.parent_op()  # builtin.module @circuit (the host)

        for op in host_mod.walk():
            print(f"op = {op}")
            if getattr(op, "op_name", None) != "catalyst.launch_kernel":
                continue

            print(f"op = {op}")
            callee = op.attributes.get("callee")
            if not callee:
                continue

            print(f"callee = {callee}")
            if callee.nested_reference.data != device_fn.sym_name.data:
                continue

            kernel_arg = op.operands[arg_idx]
            if isinstance(kernel_arg, OpResult):
                kc = kernel_arg.op
                if kc.op_name.data == "stablehlo.constant":
                    return kc.attributes["value"].value.data
            return None

    return None


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

        # print(f"funcOp = {funcOp}")
        # print(f"deviceProgram = {deviceProgram}")
        # print(f"hostProgram = {hostProgram}")

        # Determine the argument index (not used yet)
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
            from xdsl.dialects.arith import ConstantOp

            angle = ops[0].parameters[0].item()

            # Convert raw float to a FloatAttr (with correct type)
            angle_attr = FloatAttr(angle, Float64Type())

            # Create a constant op from the attribute
            angle_const_op = ConstantOp(value=angle_attr)

            # Get the SSA value (result) from the constant op
            angle_ssa = angle_const_op.result

            # Insert the constant op into the IR BEFORE `op`
            rewriter.insert_op(angle_const_op, InsertPoint.before(op))

            # Operand is the original qubit being used:
            qubit_operand = op.in_qubits[0]  # SSAValue

            # breakpoint()

            new_op = CustomOp(
                operands=(angle_ssa, qubit_operand, None, None),
                properties=op.properties,
                attributes=op.attributes,
                successors=op.successors,
                regions=op.regions,
                result_types=(op.result_types, []),
            )

            # breakpoint()

            # Step 6: Insert before the original op
            rewriter.insert_op(new_op, InsertPoint.before(op))

            # Step 7: Replace uses if needed
            for old_res, new_res in zip(op.results, new_op.results):
                old_res.replace_by(new_res)

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

# %%
available_passes["unitary-to-rot"] = lambda: UnitaryToRotPass

user_requested_pass = "unitary-to-rot"  # just for example

requested_by_user = passes.PipelinePass.build_pipeline_tuples(
    available_passes, parse_pipeline.parse_pipeline(user_requested_pass)
)
schedule = tuple(pass_type.from_pass_spec(spec) for pass_type, spec in requested_by_user)
pipeline = passes.PipelinePass(schedule)
pipeline.apply(ctx, m)
print(m)

# %% [markdown]
#

# %% [markdown]
#

# %% [markdown]
#

# %% [markdown]
# op.matrix = <OpResult[tensor<2x2xcomplex<f64>>] index: 0, operation: builtin.unregistered, uses: 1>
# op.matrix.owner = UnregisteredOp.with_name.<locals>.UnregisteredOpWithNameOp(%0 = "stablehlo.convert"(%arg0) : (tensor<2x2xf64>) -> tensor<2x2xcomplex<f64>>)
# op.matrix.op = UnregisteredOp.with_name.<locals>.UnregisteredOpWithNameOp(%0 = "stablehlo.convert"(%arg0) : (tensor<2x2xf64>) -> tensor<2x2xcomplex<f64>>)
# matrix_op = UnregisteredOp.with_name.<locals>.UnregisteredOpWithNameOp(%0 = "stablehlo.convert"(%arg0) : (tensor<2x2xf64>) -> tensor<2x2xcomplex<f64>>)
# matrix_op.name = builtin.unregistered

# %%


# %%
