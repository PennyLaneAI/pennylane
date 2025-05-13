from functools import partial
from typing import Callable
import re
import pennylane
from pennylane import QNode, device, Identity, Hadamard, PauliX, PauliY, PauliZ, S, T, SX, RX, RY, RZ, \
    PhaseShift, U1, U2, U3, CNOT, CY, CZ, CH, SWAP, CSWAP, CPhase, CRX, CRY, CRZ, \
    Toffoli
from openqasm3.visitor import QASMVisitor, QASMNode

SINGLE_QUBIT_GATES = {
    "ID": Identity,
    "H": Hadamard,
    "X": PauliX,
    "Y": PauliY,
    "Z": PauliZ,
    "S": S,
    "T": T,
    "SX": SX
}

PARAMETERIZED_SIGNLE_QUBIT_GATES = {
    "RX": RX,
    "RY": RY,
    "RZ": RZ,
    "P": PhaseShift,
    "PHASE": PhaseShift,
    "U1": U1,
    "U2": U2,
    "U3": U3,
}

TWO_QUBIT_GATES = {
    "CX": CNOT,
    "CY": CY,
    "CZ": CZ,
    "CH": CH,
    "SWAP": SWAP,
    "CSWAP": CSWAP,
    "CP": CPhase,
    "CPHASE": CPhase,
    "CRX": CRX,
    "CRY": CRY,
    "CRZ": CRZ,
}

MULTI_QUBIT_GATES = {
    "CCX": Toffoli,
}

class QasmInterpreter(QASMVisitor):
    """
    Inherits generic_visit(self, node: QASMNode, context: Optional[T]) which takes the
    top level node of the AST as a parameter and recursively descends the AST, calling the
    user-defined visitor function on each node.
    """

    def visit(self, node: QASMNode, context: dict):
        """
        Applied to each node in the AST.
        """
        match node.__class__.__name__:
            case "Identifier":
                self.identifier(node, context)
            case "QubitDeclaration":
                self.qubit_declaration(node, context)
            case "ClassicalDeclaration":
                self.classical_declaration(node, context)
            case "QuantumGate":
                self.quantum_gate(node, context)
            # TODO: call appropriate handler methods here
            case _:
                print(f"An unrecognized QASM instruction was encountered: {node.__class__.__name__}")
        return context

    def generic_visit(self, node: QASMNode, context: dict):
        """Wraps the provided generic_visit method to make the context a required parameter
        and return the context."""
        super().generic_visit(node, context)
        self.construct_qnode(context)
        return context

    @staticmethod
    def construct_qnode(context: dict):
        if "device" not in context:
            context["device"] = device("default.qubit", wires=len(context["wires"]))
        context["qnode"] = QNode(lambda: [gate() for gate in context["gates"]], device=context["device"])

    @staticmethod
    def identifier(node: QASMNode, context: dict):
        """
        Registers an identifier in the current context.
        """
        if not hasattr(context, "identifiers"):
            context["identifiers"] = []
        context["identifiers"].append(node.name.name)

    @staticmethod
    def qubit_declaration(node: QASMNode, context: dict):
        """
        Registers a qubit declaration. Named qubits are mapped to numbered wires by their indices
        in context["wires"].
        """
        if "wires" not in context:
            context["wires"] = []
        context["wires"].append(node.qubit.name)

    @staticmethod
    def classical_declaration(node: QASMNode, context: dict):
        """
        Registers a classical declaration.
        """
        if "vars" not in context:
            context["vars"] = {}
        context["vars"][node.identifier.name] = {
            'ty': node.type.__class__.__name__,
            'val': node.init_expression.value,
            'line': node.init_expression.span.start_line
        }

    def quantum_gate(self, node: QASMNode, context: dict):
        """
        Registers a quantum gate application. TODO: support modifiers
        """
        if "gates" not in context:
            context["gates"] = []
        if node.name.name.upper() in SINGLE_QUBIT_GATES:
            gate = self.non_parameterized_gate(SINGLE_QUBIT_GATES, node, context)
        elif node.name.name.upper() in PARAMETERIZED_SIGNLE_QUBIT_GATES:
            gate = self.param_single_qubit_gate(node, context)
        elif node.name.name.upper() in TWO_QUBIT_GATES:
            gate = self.non_parameterized_gate(TWO_QUBIT_GATES, node, context)
        elif node.name.name.upper() in MULTI_QUBIT_GATES:
            gate = self.non_parameterized_gate(MULTI_QUBIT_GATES, node, context)
        else:
            print(f"Unsupported gate encountered in QASM: {node.name}")

        if len(node.modifiers) > 0:
            gate = self.modifiers(gate, node, context)

        context["gates"].append(gate)

    def modifiers(self, gate: Callable, node: QASMNode, context: dict):
        """
        Registers a modifier on a gate.
        """
        call_stack = [gate]
        for mod in node.modifiers:
            if mod.modifier.name == 'inv':
                wrapper = pennylane.adjoint
            elif mod.modifier.name == 'pow':
                if re.search('Literal', mod.argument.__class__.__name__) is not None:
                    wrapper = partial(pennylane.pow, z=mod.argument.value)
                elif mod.argument.name in context["vars"]:
                    wrapper = partial(pennylane.pow, z=context["vars"][mod.argument.name]["val"])
            elif mod.modifier.name == 'ctrl':
                wrapper = partial(pennylane.ctrl, control=gate.keywords["wires"][0:-1])
            call_stack = [wrapper] + call_stack

        def call():
            res = None
            for callable in call_stack[::-1]:
                if ('partial' == call_stack[0].__class__.__name__ and 'control' in call_stack[0].keywords):
                    if 'control' in callable.keywords:
                        res.keywords["wires"] = [res.keywords["wires"][-1]]
                    # i.e. qml.ctrl(qml.RX, (1))(2, wires=0)
                    res = callable(res.func)(**res.keywords) if res is not None else callable
                else:
                    # i.e. qml.pow(qml.RX(1.5, wires=0), z=4)
                    res = callable(res) if res is not None else callable()

        return call

    @staticmethod
    def param_single_qubit_gate(node: QASMNode, context: dict):
        """
        Registers a parameterized single qubit gate application.
        """
        args = []
        for arg in node.arguments:
            if hasattr(arg, "name") and arg.name in context["vars"]:
                # the context at this point should reflect the states of the
                # variables as evaluated in the correct (current) scope.
                args.append(context["vars"][arg.name]["val"])
            elif re.search('Literal', arg.__class__.__name__) is not None:
                args.append(arg.value)
            else:
                raise NameError(
                    f"Uninitialized variable {arg.name if hasattr(arg, 'name') else arg.__class__.__name__} "
                    f"encountered in QASM."
                )
        return partial(
            PARAMETERIZED_SIGNLE_QUBIT_GATES[node.name.name.upper()],
            *args,
            wires=[context["wires"].index(node.qubits[0].name)]
        )

    @staticmethod
    def non_parameterized_gate(gates_dict: dict, node: QASMNode, context: dict):
        """
        Registers a multi qubit gate application.
        """
        return partial(
            gates_dict[node.name.name.upper()],
            wires=[
                context["wires"].index(node.qubits[q].name)
                for q in range(len(node.qubits))
            ]
        )
