#############################################
## xDSL Graph-Based Decomposition Prototype #
#############################################

# This is a prototype for the xDSL graph-based decomposition pass
# for PennyLane circuits using the xDSL framework.

# TODOs:
# - [Required] Convert the decomposition rule to xDSL subprogram
# - [Required] Replace the original CustomOp with the decomposed ops
# - Use multiple match_and_rewrite methods to handle different decomposition steps
# - Reduce the number of walks through the function body
# - Use a dict of supported gates to better handle PL's operators
# - Handle dynamic AllocOp and ExtractOp
# - Handle abstracted gate parameters
# - Handle multiple-qubit gates
# - Handle the case when the graph is not solved for the operation
# - Handle control-flow operations in the circuit

import struct

import catalyst
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath
from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin, func
from xdsl.dialects.tensor import ExtractOp

import pennylane as qml
import pennylane.compiler.python_compiler.quantum_dialect as quantum


def xdsl_transform(_klass):
    """TODO: Add to PennyLane"""

    def graph_decomposition_transform(tape):
        # TODO: Check if plxpr is enabled if enabled return this
        # if it is not enabled, and we are inside Catalyst
        # I think we can just use apply_pass internally here
        # to avoid using apply_pass directly in plxpr
        return tape, lambda args: args[0]

    graph_decomposition_transform.__name__ = "xdsl_transform" + _klass.__name__
    transform = qml.transform(graph_decomposition_transform)
    catalyst.from_plxpr.register_transform(transform, _klass.name, False)
    from pennylane.compiler.python_compiler.transforms import register_pass

    register_pass(_klass.name, lambda: _klass())

    return transform


# Step 1. Define the visitor pattern to collect ops
def convert_to_operator(op: quantum.CustomOp) -> qml.operation.Operator:
    """Convert a CustomOp to a PennyLane operator."""

    if op.gate_name.data == "PauliX":
        return qml.decomposition.CompressedResourceOp(qml.PauliX)
    elif op.gate_name.data == "CNOT":
        return qml.decomposition.CompressedResourceOp(qml.CNOT)
    if op.gate_name.data == "Hadamard":
        return qml.decomposition.CompressedResourceOp(qml.Hadamard)
    elif op.gate_name.data == "RX":
        return qml.decomposition.CompressedResourceOp(qml.RX)
    else:
        raise ValueError(f"Unsupported gate: {op.gate_name.data}")


###########################
# Graph-based decomposition
###########################


# Define a Rewrite pattern to apply the decomposiiton rules
class GBDecompositionPattern(pattern_rewriter.RewritePattern):
    """Rewrite pattern for graph-based decomposition."""

    def __init__(self):
        super().__init__()
        self.graph = None
        self.collected_ops = set()
        self.pl_ops = []

    def device_gateset(self):
        """Return the supported operations for the decomposition."""

        # TODO: get the device name from module
        # dev = qml.device("lightning.qubit")
        # dev_ops = dev.capabilities.supported_operations
        # return dev_ops

        # FIXME: remove this after testing
        return {"RY", "RZ", "Hadamard"}

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter):
        """Apply decomposition rules using the graph."""

        # Step 1. First walk to visit the func and collect the ops
        for op in funcOp.body.walk():
            if isinstance(op, quantum.CustomOp):
                self.collected_ops.add(convert_to_operator(op))

        # Step 2. construct the graph-decomposition and find the decomposition rules:
        # - Ops are collected using the visitor pattern
        # - Gateset is defined based on the device capabilities and passed as a list of ops strings to the graph
        print("Collected ops: ", self.collected_ops)
        print("Device gateset: ", self.device_gateset())

        self.graph = qml.decomposition.DecompositionGraph(
            operations=self.collected_ops,
            gate_set=self.device_gateset(),
        )

        self.graph.solve()
        print("Graph solved!")

        # Step 3. walk through the function body to collect the number of wires
        # and create a dict of qregs to wire values. This dict will be used
        # to define a mapping from wires to values to construct the gate instance
        # and get the decomposition rule from the graph.
        qregs_map = dict()
        for op in funcOp.body.walk():
            if isinstance(op, quantum.AllocOp):
                try:
                    n_qubits = int(op.nqubits_attr.value.data)
                    print("n_qubits: ", n_qubits)
                except AttributeError:
                    raise ValueError("Dynamic AllocOp is not supported yet.")
            if isinstance(op, quantum.ExtractOp):
                # print("ExtractOp found: ", op)
                # TODO: Handle multiple-qubit gates if this qregs_map turns out to be useful :)
                value = op.prev_op.operands[0].op.properties["value"]
                data = struct.unpack("<q", value.data.data)[0]
                qregs_map[op.qubit] = data
            if isinstance(op, quantum.CustomOp):
                if hasattr(op, "in_qubits"):
                    if op.in_qubits[0] in qregs_map:
                        qregs_map[op.out_qubits[0]] = qregs_map[op.in_qubits[0]]

        # Step 4. Walk through the function body again to replace the CustomOp with the decomposed ops
        # This is where we will use the graph to get the decomposition rule
        # and replace the CustomOp with the decomposed ops
        for op in funcOp.body.walk():
            if isinstance(op, quantum.CustomOp):
                # We need to convert the CustomOp to the PennyLane operator type
                # to be able to use graph.is_solved_for and graph.decomposition
                op_name = op.gate_name.data
                params = []
                if hasattr(op, "params"):
                    for param in op.params:
                        if param and isinstance(param.op, ExtractOp):
                            value = param.op.prev_op.properties["value"]
                            data = struct.unpack("<d", value.data.data)[0]
                            # print("param value: ", data)
                            params.append(data)

                wires = []
                if hasattr(op, "in_qubits"):
                    for wire in op.in_qubits:
                        if wire and wire in qregs_map:
                            wires.append(qregs_map[wire])

                # FIXME: Use a dict of supported gates
                if op_name == "PauliX":
                    assert len(wires) == 1, "PauliX gate should have one wire"
                    pl_op_instance = qml.PauliX(wires=wires)
                elif op_name == "CNOT":
                    assert len(wires) == 2, "CNOT gate should have two wires"
                    pl_op_instance = qml.CNOT(wires=wires)
                elif op_name == "Hadamard":
                    assert len(wires) == 1, "Hadamard gate should have one wire"
                    pl_op_instance = qml.Hadamard(wires=wires)
                elif op_name == "RX":
                    assert len(wires) == 1, "RX gate should have one wire"
                    assert len(params) == 1, "RX gate should have one parameter"
                    pl_op_instance = qml.RX(params[0], wires=wires)
                else:
                    raise ValueError(f"Unsupported gate: {op_name}")

                self.pl_ops.append(pl_op_instance)

                if (
                    self.graph
                    and self.graph.is_solved_for(pl_op_instance)
                    and pl_op_instance.name not in self.device_gateset()
                ):
                    rule = self.graph.decomposition(pl_op_instance)  # Example for RX gate

                    # TODO: Compile the rule to xdsl subprogram
                    # TODO: Replace the op with the decomposed ops

                    # Remove the original CustomOp
                    # rewriter.erase_op(op)


@xdsl_transform
class GBDecompositionPass(passes.ModulePass):
    name = "graph-based-decomposition"
    description = "Decomposes quantum circuits using the graph-based framework."

    def apply(self, ctx: context.Context, module: builtin.ModuleOp) -> None:

        print("module in: ", module)

        pattern_rewriter.PatternRewriteWalker(
            pattern_rewriter.GreedyRewritePatternApplier([GBDecompositionPattern()]),
        ).rewrite_module(module)

        print("module out: ", module)


qml.capture.enable()


# @catalyst.qjit(keep_intermediate=True, pass_plugins=[getXDSLPluginAbsolutePath()])
@catalyst.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
@GBDecompositionPass
@qml.qnode(qml.device("lightning.qubit", wires=2))
def captured_circuit():
    qml.Hadamard(wires=1)
    for _ in range(2):
        qml.RX(0.2, wires=0)
    return qml.state()


print(captured_circuit())
qml.capture.disable()
