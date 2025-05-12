from functools import partial
from pennylane import QNode, device, Identity, Hadamard, PauliX, PauliY, PauliZ, S, T, SX, RX, RY, RZ, \
    PhaseShift, U1, U2, U3, CNOT, CY, CZ, CH, SWAP, CSWAP, CPhase, CRX, CRY, CRZ, \
    Toffoli, MultiControlledX, Barrier
from openqasm3.visitor import QASMVisitor, QASMNode

SINGLE_QUBIT_GATES = {
    "ID": Identity,  # TODO: translate all other QASM std lib gates to equivalent series of pennylane gates
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
        QNode(lambda _: [gate() for gate in context["gates"]], device=context["device"])

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
            gate = self.single_qubit_gate(node, context)
        elif node.name.name.upper() in PARAMETERIZED_SIGNLE_QUBIT_GATES:
            gate = self.param_single_qubit_gate(node, context)
        elif node.name.name.upper() in TWO_QUBIT_GATES:
            gate = self.two_qubit_gate(node, context)
        elif node.name.name.upper() in MULTI_QUBIT_GATES:
            gate = self.multi_qubit_gate(node, context)
        else:
            print(f"Unsupported gate encountered in QASM: {node.name}")
        context["gates"].append(gate)

    @staticmethod
    def single_qubit_gate(node: QASMNode, context: dict):
        """
        Registers a single qubit gate application.
        """
        return partial(SINGLE_QUBIT_GATES[node.name.name.upper()], wires=[context["wires"].index(node.qubits[0].name)])

    @staticmethod
    def param_single_qubit_gate(node: QASMNode, context: dict):
        """
        Registers a parameterized single qubit gate application.
        """
        keyword_args = {}
        for arg in node.arguments:
            if arg.name in context["vars"]:
                # the context at this point should reflect the states of the
                # variables as evaluated in the correct (current) scope.
                keyword_args[arg.name] = context["vars"][arg.name]["val"]
        return partial(
            PARAMETERIZED_SIGNLE_QUBIT_GATES[node.name.name.upper()],
            **keyword_args,
            wires=[context["wires"].index(node.qubits[0].name)]
        )

    @staticmethod
    def two_qubit_gate(node: QASMNode, context: dict):
        """
        Registers a two qubit gate application.
        """
        return partial(
            TWO_QUBIT_GATES[node.name.name.upper()],
            wires=[
                context["wires"].index(node.qubits[0].name),
                context["wires"].index(node.qubits[1].name),
            ]
        )

    @staticmethod
    def multi_qubit_gate(node: QASMNode, context: dict):
        """
        Registers a multi qubit gate application.
        """
        return partial(
            MULTI_QUBIT_GATES[node.name.name.upper()],
            wires=[
                context["wires"].index(node.qubits[q].name)
                for q in range(len(node.qubits))
            ]
        )
