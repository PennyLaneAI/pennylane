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
from pennylane.compiler.python_compiler.quantum_dialect import (
    CustomOp,
    ExtractOp,
    QuantumDialect,
    QubitType,
    QubitUnitaryOp,
)
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


@qml.qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def circuit():
    """A simple circuit to test the pass."""
    # qml.CNOT(wires=[0, 1])
    qml.Rot(0.1, 0.2, 0.3, wires=1)
    # qml.Rot(0.4, 0.5, 0.6, wires=1)
    qml.Rot(0.4, 0.5, 0.6, wires=2)
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

from_str_to_PL_gate = {
    "RX": qml.RX,
    "RY": qml.RY,
    "RZ": qml.RZ,
    "Rot": qml.Rot,
    "CNOT": qml.CNOT,
}


def resolve_constant_params(op):

    while hasattr(op, "owner"):
        op = op.owner
    if isinstance(op, ConstantOp):
        val = op.value
        if isinstance(val, (FloatAttr, IntegerAttr)):
            return val.value.data
    if isinstance(op, ExtractOp):
        return op.idx_attr.parameters[0].data
    if isinstance(op, CustomOp):
        raise NotImplementedError("Cannot resolve params from CustomOp")
    raise NotImplementedError(f"Cannot resolve params from {op}")


def resolve_constant_wire(op):

    while hasattr(op, "owner"):
        op = op.owner
    if isinstance(op, ConstantOp):
        val = op.value
        if isinstance(val, IntegerAttr):
            return val.value.data
    if isinstance(op, ExtractOp):
        return op.idx_attr.parameters[0].data
    if isinstance(op, CustomOp):
        raise NotImplementedError("Cannot resolve wires from CustomOp")
    raise NotImplementedError(f"Cannot resolve wires from {op}")


def get_parameters(op) -> list[float | int]:
    return [resolve_constant_params(p) for p in op.params if p is not None]


def get_wires(op) -> list[int]:
    return [resolve_constant_wire(w) for w in op.in_qubits if w is not None]


def get_op_name(op) -> str:
    return op.properties["gate_name"].data


def resolve_gate(name: str):
    try:
        return from_str_to_PL_gate[name]
    except KeyError:
        raise ValueError(f"Unsupported gate: {name}")


def reconstruct_gate(op: CustomOp):
    gate_name = get_op_name(op)
    parameters = get_parameters(op)
    wires = get_wires(op)
    print(f"Reconstructing gate: {gate_name} with parameters: {parameters} and wires: {wires}")
    return resolve_gate(gate_name)(*parameters, wires=wires)


class UnitaryToRotPattern(pattern_rewriter.RewritePattern):
    def __init__(self, module):
        self.module = module
        super().__init__()

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):
        for op in funcOp.body.walk():

            if not isinstance(op, CustomOp):
                continue

            concrete_op = reconstruct_gate(op)

            if not concrete_op.has_decomposition:
                continue

            decomp_ops = concrete_op.decomposition()

            last_custom_op = None

            for qml_op in decomp_ops:

                angle = qml_op.parameters[0]
                angle = angle.item() if not isinstance(angle, float) else angle

                wire = qml_op.wires[0]

                # Right now, we know that the angle is a float.
                if isinstance(angle, int):
                    angle_attr = IntegerAttr(angle, IntegerType(64))
                elif isinstance(angle, float):
                    angle_attr = FloatAttr(angle, Float64Type())
                else:
                    raise TypeError(f"Unsupported angle type: {type(angle)}")

                angle_const_op = ConstantOp(value=angle_attr)
                rewriter.insert_op(angle_const_op, InsertPoint.before(op))

                wire_attr = IntegerAttr(wire, IntegerType(64))
                wire_const_op = ConstantOp(value=wire_attr)
                rewriter.insert_op(wire_const_op, InsertPoint.before(op))

                # TODO: here I am simply parsing the properties and successors of the QubitUnitaryOp
                # and passing them to the custom op. But it would be probably better to it differently.
                custom_op = CustomOp(
                    operands=(angle_const_op.result, wire_const_op.result, None, None),
                    properties={
                        "gate_name": StringAttr(qml_op.name),
                        # **op.properties,
                    },
                    attributes={},
                    successors=op.successors,
                    regions=(),
                    result_types=(QubitType(), []),
                )

                rewriter.insert_op(custom_op, InsertPoint.before(op))

                last_custom_op = custom_op

                # for old_res, new_res in zip(op.results, custom_op.results):
                #    old_res.replace_by(new_res)

            if op.results:
                op.results[0].replace_by(last_custom_op.results[0])

            rewriter.erase_op(op)


@dataclass(frozen=True)
class DummyTransformPass(passes.ModulePass):
    name = "dummy-transform"

    def apply(self, ctx: context.MLContext, module: builtin.ModuleOp) -> None:
        pattern = UnitaryToRotPattern(module)
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([pattern])
        ).rewrite_module(module)


print(f"\n\n\n\n\n\n\n\n")
print(f"Running the pass")


available_passes["dummy-transform"] = lambda: DummyTransformPass
user_requested_pass = "dummy-transform"
requested_by_user = passes.PipelinePass.build_pipeline_tuples(
    available_passes, parse_pipeline.parse_pipeline(user_requested_pass)
)
schedule = tuple(pass_type.from_pass_spec(spec) for pass_type, spec in requested_by_user)
pipeline = passes.PipelinePass(schedule)
pipeline.apply(ctx, m)
print(m)
