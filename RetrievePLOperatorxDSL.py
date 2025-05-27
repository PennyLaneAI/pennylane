from dataclasses import dataclass
from typing import Callable

import xdsl
from catalyst.compiler import _quantum_opt
from xdsl import context, passes, pattern_rewriter
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, scf, tensor, transform
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import Float64Type, FloatAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.rewriter import InsertPoint
from xdsl.utils import parse_pipeline

import pennylane as qml
from pennylane.compiler.python_compiler.quantum_dialect import (
    CustomOp,
    ExtractOp,
    GlobalPhaseOp,
    QuantumDialect,
    QubitType,
)
from pennylane.operation import Operator

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
    """A simple pass that prints the module."""

    # All passes require a name field
    name = "print"

    # All passes require an apply method with this signature.
    def apply(self, ctx: context.MLContext, module: builtin.ModuleOp) -> None:
        """Print the module."""
        print("Hello from inside the pass\n", module)


@qml.qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def circuit():
    """A simple circuit to test the pass."""
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(0.5, wires=1)
    qml.RY(0.5, wires=2)
    qml.RZ(0.5, wires=0)
    qml.Rot(0.5, 0.5, 0.5, wires=1)
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
    "Hadamard": qml.Hadamard,
    "PhaseShift": qml.PhaseShift,
}


def flatten(nested):
    """Flatten a nested list."""
    flat = []
    for item in nested:
        if isinstance(item, list):
            flat.extend(flatten(item))
        else:
            flat.append(item)
    return flat


def resolve_constant_params(op: Operator):
    """Resolve the constant parameters that this operation originates from."""

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


def resolve_constant_wire(operand):
    """Resolve the integer wire index that this operand originates from."""

    # Traverse to producing op if this is a result
    while hasattr(operand, "owner"):
        result_index = operand.index
        operand = operand.owner  # the producing Operation

        # Since GlobalPhaseOp does not have a wire
        if isinstance(operand, GlobalPhaseOp):
            return

        if isinstance(operand, ConstantOp):
            val = operand.value
            if isinstance(val, IntegerAttr):
                return val.value.data

        elif isinstance(operand, ExtractOp):
            return operand.idx_attr.parameters[0].data

        elif isinstance(operand, CustomOp):
            # CustomOp result[i] is assumed to correspond to in_qubits[i]
            # so we use the index of the result to retrieve the right input
            if result_index < len(operand.in_qubits):
                input_operand = operand.in_qubits[result_index]
                return resolve_constant_wire(input_operand)
            else:
                raise IndexError(
                    f"Result index {result_index} out of bounds for CustomOp with {len(operand.in_qubits)} in_qubits."
                )

        else:
            raise NotImplementedError(f"Cannot resolve wires from operation type: {type(operand)}")

    raise TypeError(f"Operand {operand} is not a result or constant-producing op")


def get_parameters(op: Operator) -> list[float | int]:
    """Get the parameters from the operation."""
    return [resolve_constant_params(p) for p in op.params if p is not None]


def get_wires(op: Operator) -> list[int]:
    """Get the wires from the operation."""
    if not hasattr(op, "in_qubits"):
        return []
    return [resolve_constant_wire(w) for w in op.in_qubits if w is not None]


def get_op_name(op: Operator) -> str:
    """Get the name of the operation from the properties."""
    return op.properties["gate_name"].data


def resolve_gate(name: str):
    """Resolve the gate from the name."""
    try:
        return from_str_to_PL_gate[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported gate: {name}") from exc


def reconstruct_gate(op: CustomOp):
    """Reconstruct the gate from the operation."""
    gate_name = get_op_name(op)
    parameters = get_parameters(op)
    wires = get_wires(op)
    wires = flatten(wires)
    return resolve_gate(gate_name)(*parameters, wires=wires)


class DummyDecompositionTransform(pattern_rewriter.RewritePattern):
    """A pattern that rewrites CustomOps to their decomposition."""

    def __init__(self, module):
        self.module = module
        super().__init__()

    # pylint:disable=arguments-differ
    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):
        """Rewrite the function by replacing CustomOps with their decomposition."""
        for op in funcOp.body.walk():

            if not isinstance(op, CustomOp):
                continue

            concrete_op = reconstruct_gate(op)

            if not concrete_op.has_decomposition:
                continue

            decomp_ops = concrete_op.decomposition()
            last_custom_op = None

            for qml_op in decomp_ops:
                parameters_xdsl = []
                wires_xdsl = []

                # We convert the parameters to xDSL constants
                for param in qml_op.parameters:
                    val = param.item() if hasattr(param, "item") else param
                    if isinstance(val, int):
                        attr = IntegerAttr(val, IntegerType(64))
                    elif isinstance(val, float):
                        attr = FloatAttr(val, Float64Type())
                    else:
                        raise TypeError(f"Unsupported parameter type: {type(val)}")

                    const_op = ConstantOp(value=attr)
                    rewriter.insert_op(const_op, InsertPoint.before(op))
                    parameters_xdsl.append(const_op.result)

                # We convert the wires to xDSL constants
                for wire in qml_op.wires:
                    wire_attr = IntegerAttr(wire, IntegerType(64))
                    wire_const_op = ConstantOp(value=wire_attr)
                    rewriter.insert_op(wire_const_op, InsertPoint.before(op))
                    wires_xdsl.append(wire_const_op.result)

                # At this stage, this is the only 'special' operator we handle (GlobalPhase does not have wires)
                if qml_op.name == "GlobalPhase":

                    custom_op = GlobalPhaseOp(
                        operands=(*parameters_xdsl, None, None),
                        properties={
                            "gate_name": StringAttr(qml_op.name),
                            # Do we need to pass more properties here?
                            # **op.properties,
                        },
                        attributes={},
                        successors=op.successors,
                        regions=(),
                        # This should probably be generalized
                        result_types=(QubitType(),),
                    )

                else:

                    custom_op = CustomOp(
                        operands=(*parameters_xdsl, *wires_xdsl, None, None),
                        # operands=(angle_const_op.result, wire_const_op.result, None, None),
                        properties={
                            "gate_name": StringAttr(qml_op.name),
                            # Do we need to pass more properties here?
                            # **op.properties,
                        },
                        attributes={},
                        successors=op.successors,
                        regions=(),
                        # This should probably be generalized
                        result_types=(QubitType(), []),
                    )

                rewriter.insert_op(custom_op, InsertPoint.before(op))

                last_custom_op = custom_op

            if op.results:
                op.results[0].replace_by(last_custom_op.results[0])

            rewriter.erase_op(op)


@dataclass(frozen=True)
class DummyDecompositionTransformPass(passes.ModulePass):
    """A pass that applies the DummyTransform pattern to a module."""

    name = "dummy-decomposition-transform"

    def apply(self, ctx: context.MLContext, module: builtin.ModuleOp) -> None:
        pattern = DummyDecompositionTransform(module)
        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([pattern])
        ).rewrite_module(module)


print("\n\n\n\n\n\n\n\n")
print("Running the pass")


available_passes["dummy-decomposition-transform"] = lambda: DummyDecompositionTransformPass
user_requested_pass = "dummy-decomposition-transform"
requested_by_user = passes.PipelinePass.build_pipeline_tuples(
    available_passes, parse_pipeline.parse_pipeline(user_requested_pass)
)
schedule = tuple(pass_type.from_pass_spec(spec) for pass_type, spec in requested_by_user)
pipeline = passes.PipelinePass(schedule)
pipeline.apply(ctx, m)
print(m)
