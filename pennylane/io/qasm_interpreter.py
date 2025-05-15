import re
from functools import partial
from typing import Callable

from openqasm3.visitor import QASMNode, QASMVisitor

from pennylane import (
    CH,
    CNOT,
    CRX,
    CRY,
    CRZ,
    CSWAP,
    CY,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    SX,
    U1,
    U2,
    U3,
    CPhase,
    Hadamard,
    Identity,
    PauliX,
    PauliY,
    PauliZ,
    PhaseShift,
    S,
    T,
    Toffoli,
    device,
    adjoint,
    pow,
    ctrl
)

SINGLE_QUBIT_GATES = {
    "ID": Identity,
    "H": Hadamard,
    "X": PauliX,
    "Y": PauliY,
    "Z": PauliZ,
    "S": S,
    "T": T,
    "SX": SX,
}

PARAMETERIZED_SINGLE_QUBIT_GATES = {
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
    "SWAP": SWAP
}

PARAMETERIZED_TWO_QUBIT_GATES = {
    "CP": CPhase,
    "CPHASE": CPhase,
    "CRX": CRX,
    "CRY": CRY,
    "CRZ": CRZ
}

MULTI_QUBIT_GATES = {
    "CCX": Toffoli,
    "CSWAP": CSWAP,
}


class QasmInterpreter(QASMVisitor):
    """
    Overrides generic_visit(self, node: QASMNode, context: Optional[T]) which takes the
    top level node of the AST as a parameter and recursively descends the AST, calling the
    overriden visitor function on each node.
    """

    def visit(self, node: QASMNode, context: dict):
        """
        Visitor function is called on each node in the AST, which is traversed using recursive descent.
        The purpose of this function is to pass each node to the appropriate handler.

        Args:
            node (QASMNode): the QASMNode to visit next.
            context (dict): the current context populated with any locally available variables, etc.

        Returns:
            dict: The context updated after the compilation of the current node into Callables
                to queue into a QNode.

        Raises:
            NameError: When a (so far) unsupported node type is encountered.
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
                raise NotImplementedError(
                    f"An unsupported QASM instruction was encountered: {node.__class__.__name__}"
                )
        return context

    def generic_visit(self, node: QASMNode, context: dict):
        """
        Wraps the provided generic_visit method to make the context a required parameter
        and return the context for testability. Constructs the QNode after all of the nodes
        have been visited.

        Args:
            node (QASMNode): The top-most QASMNode.
            context (dict): The initial context populated with the name of the program (the outermost scope).

        Returns:
            dict: The context updated after the compilation of all nodes by the visitor. Contains a QNode
                with a list of Callables that are queued into it.
        """
        super().generic_visit(node, context)
        self.construct_callable(context)
        return context

    @staticmethod
    def construct_callable(context: dict):
        """
        Constructs a callable that can be queued into a QNode.

        Args:
            context (dict): The final context populated with the Callables (called gates) to queue in the QNode.
        """
        if "device" not in context:
            context["device"] = device("default.qubit", wires=len(context["wires"]))
        context["callable"] = lambda: [gate() for gate in context["gates"]]

    @staticmethod
    def identifier(node: QASMNode, context: dict):
        """
        Registers an identifier in the current context.

        Args:
            node (QASMNode): The Identifier QASMNode.
            context (dict): The current context.
        """
        if not hasattr(context, "identifiers"):
            context["identifiers"] = []
        context["identifiers"].append(node.name.name)

    @staticmethod
    def qubit_declaration(node: QASMNode, context: dict):
        """
        Registers a qubit declaration. Named qubits are mapped to numbered wires by their indices
        in context["wires"]. TODO: this should be changed to have greater specificity. Coming in a follow-up PR.

        Args:
            node (QASMNode): The QubitDeclaration QASMNode.
            context (dict): The current context.
        """
        if "wires" not in context:
            context["wires"] = []
        context["wires"].append(node.qubit.name)

    @staticmethod
    def classical_declaration(node: QASMNode, context: dict):
        """
        Registers a classical declaration. Traces data flow through the context, transforming QASMNodes into Python
        type variables that can be readily used in expression evaluation, for example.

        Args:
            node (QASMNode): The ClassicalDeclaration QASMNode.
            context (dict): The current context.
        """
        if "vars" not in context:
            context["vars"] = {}
        if node.init_expression is not None:
            context["vars"][node.identifier.name] = {
                "ty": node.type.__class__.__name__,
                "val": node.init_expression.value,
                "line": node.init_expression.span.start_line,
            }
        else:
            context["vars"][node.identifier.name] = {
                "ty": node.type.__class__.__name__,
                "val": None,
                "line": node.span.start_line,
            }

    def quantum_gate(self, node: QASMNode, context: dict):
        """
        Registers a quantum gate application. Calls the appropriate handler based on the sort of gate
        (parameterized or non-parameterized).

        Args:
            node (QASMNode): The QuantumGate QASMNode.
            context (dict): The current context.
        """
        gate = None
        if "gates" not in context:
            context["gates"] = []
        if node.name.name.upper() in SINGLE_QUBIT_GATES:
            gate = self.non_parameterized_gate(SINGLE_QUBIT_GATES, node, context)
        elif node.name.name.upper() in PARAMETERIZED_SINGLE_QUBIT_GATES:
            gate = self.parameterized_gate(PARAMETERIZED_SINGLE_QUBIT_GATES, node, context)
        elif node.name.name.upper() in TWO_QUBIT_GATES:
            gate = self.non_parameterized_gate(TWO_QUBIT_GATES, node, context)
        elif node.name.name.upper() in PARAMETERIZED_TWO_QUBIT_GATES:
            gate = self.parameterized_gate(PARAMETERIZED_TWO_QUBIT_GATES, node, context)
        elif node.name.name.upper() in MULTI_QUBIT_GATES:
            gate = self.non_parameterized_gate(MULTI_QUBIT_GATES, node, context)
        else:
            # TODO: turn into warning when we have supported all we would like to
            print(f"Unsupported gate encountered in QASM: {node.name}")

        if len(node.modifiers) > 0:
            gate = self.modifiers(gate, node, context)

        context["gates"].append(gate)

    def modifiers(self, gate: Callable, node: QASMNode, context: dict):
        """
        Registers a modifier on a gate. Modifiers are applied to gates differently in Pennylane
        depending on the type of modifier. We build a Callable that applies the modifier appropriately
        at execution time, evaluating the gate Callable appropriately as well.

        Args:
            gate (Callable): The Callable partial built for the gate we wish to modify.
            node (QASMNode): The original QquantumGate QASMNode.
            context (dict): The current context.

        Returns:
            Callable: The callable which will appropriately apply the modifier and execute the gate.
        """
        call_stack = [gate]
        for mod in node.modifiers:
            if mod.modifier.name == "inv":
                wrapper = adjoint
            elif mod.modifier.name == "pow":
                if re.search("Literal", mod.argument.__class__.__name__) is not None:
                    wrapper = partial(pow, z=mod.argument.value)
                elif mod.argument.name in context["vars"]:
                    wrapper = partial(pow, z=context["vars"][mod.argument.name]["val"])
            elif mod.modifier.name == "ctrl":
                wrapper = partial(ctrl, control=gate.keywords["wires"][0:-1])
            call_stack = [wrapper] + call_stack

        def call():
            res = None
            for callable in call_stack[::-1]:
                # if there is a control in the stack
                if (
                    "partial" == call_stack[0].__class__.__name__
                    and "control" in call_stack[0].keywords
                ):
                    # if we are processing the control now
                    if "control" in callable.keywords:
                        res.keywords["wires"] = [res.keywords["wires"][-1]]
                    # i.e. qml.ctrl(qml.RX, (1))(2, wires=0)
                    res = callable(res.func)(**res.keywords) if res is not None else callable
                else:
                    # i.e. qml.pow(qml.RX(1.5, wires=0), z=4)
                    res = callable(res) if res is not None else callable()

        return call

    @staticmethod
    def parameterized_gate(gates_dict: dict, node: QASMNode, context: dict):
        """
        Registers a parameterized gate application. Builds a Callable partial
        that can be executed when the QNode is called. The gate will be executed at that time
        with the appropriate arguments.

        Args:
            node (QASMNode): The QuantumGate QASMNode.
            context (dict): The current context.

        Returns:
            Callable: The Callable partial that will execute the gate with the appropriate arguments at
                "runtime".

        Raises:
            NameError: If an argument is not found in the current context.
        """
        args = []
        for arg in node.arguments:
            if hasattr(arg, "name") and arg.name in context["vars"]:
                # the context at this point should reflect the states of the
                # variables as evaluated in the correct (current) scope.
                args.append(context["vars"][arg.name]["val"])
            elif re.search("Literal", arg.__class__.__name__) is not None:
                args.append(arg.value)
            else:
                raise NameError(
                    f"Uninitialized variable {arg.name if hasattr(arg, 'name') else arg.__class__.__name__} "
                    f"encountered in QASM."
                )
        return partial(
            gates_dict[node.name.name.upper()],
            *args,
            wires=[
                context["wires"].index(
                    node.qubits[q].name
                    if isinstance(node.qubits[q].name, str)
                    else node.qubits[q].name.name
                )
                for q in range(len(node.qubits))
            ],
        )

    @staticmethod
    def non_parameterized_gate(gates_dict: dict, node: QASMNode, context: dict):
        """
        Registers a multi qubit gate application. Builds a Callable partial that
        will execute the gate with the appropriate arguments at "runtime".

        Args:
            gates_dict (dict): A mapping from QASM std lib naming to Pennylane gate class.
            node (QASMNode): The QuantumGate QASMNode.
            context (dict): The current context.

        Returns:
            Callable: The Callable partial that will execute the gate on the appropriate qubits.
        """
        return partial(
            gates_dict[node.name.name.upper()],
            wires=[
                context["wires"].index(
                    node.qubits[q].name
                    if isinstance(node.qubits[q].name, str)
                    else node.qubits[q].name.name
                )
                for q in range(len(node.qubits))
            ],
        )
