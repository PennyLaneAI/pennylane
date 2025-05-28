"""
This submodule contains the interpreter for QASM 3.0.
"""

import functools
import re

from openqasm3.ast import ClassicalDeclaration, QuantumGate, QubitDeclaration
from openqasm3.visitor import QASMNode

from pennylane import ops
from pennylane.operation import Operator

NON_PARAMETERIZED_GATES = {
    "ID": ops.Identity,
    "H": ops.Hadamard,
    "X": ops.PauliX,
    "Y": ops.PauliY,
    "Z": ops.PauliZ,
    "S": ops.S,
    "T": ops.T,
    "SX": ops.SX,
    "CX": ops.CNOT,
    "CY": ops.CY,
    "CZ": ops.CZ,
    "CH": ops.CH,
    "SWAP": ops.SWAP,
    "CCX": ops.Toffoli,
    "CSWAP": ops.CSWAP,
}

PARAMETERIZED_GATES = {
    "RX": ops.RX,
    "RY": ops.RY,
    "RZ": ops.RZ,
    "P": ops.PhaseShift,
    "PHASE": ops.PhaseShift,
    "U1": ops.U1,
    "U2": ops.U2,
    "U3": ops.U3,
    "CP": ops.CPhase,
    "CPHASE": ops.CPhase,
    "CRX": ops.CRX,
    "CRY": ops.CRY,
    "CRZ": ops.CRZ,
}


class QasmInterpreter:
    """
    Overrides generic_visit(self, node: QASMNode, context: Optional[T]) which takes the
    top level node of the AST as a parameter and recursively descends the AST, calling the
    overriden visitor function on each node.
    """

    @functools.singledispatchmethod
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
        raise NotImplementedError(
            f"An unsupported QASM instruction was encountered: {node.__class__.__name__}"
        )

    def interpret(self, node: QASMNode, context: dict):
        """
        Entry point for visiting the QASMNodes of a parsed QASM 3.0 program.

        Args:
            node (QASMNode): The top-most QASMNode.
            context (dict): The initial context populated with the name of the program (the outermost scope).

        Returns:
            dict: The context updated after the compilation of all nodes by the visitor.
        """

        context.update({"wires": [], "vars": {}, "gates": [], "callable": None})

        # begin recursive descent traversal
        for value in node.__dict__.values():
            if not isinstance(value, list):
                value = [value]
            for item in value:
                if isinstance(item, QASMNode):
                    self.visit(item, context)
        return context

    # needs to have same signature as visit()
    @visit.register(QubitDeclaration)
    def visit_qubit_declaration(self, node: QASMNode, context: dict):  # pylint: disable=no-self-use
        """
        Registers a qubit declaration. Named qubits are mapped to numbered wires by their indices
        in context["wires"]. TODO: this should be changed to have greater specificity. Coming in a follow-up PR.

        Args:
            node (QASMNode): The QubitDeclaration QASMNode.
            context (dict): The current context.
        """
        context["wires"].append(node.qubit.name)

    # needs to have same signature as visit()
    @visit.register(ClassicalDeclaration)
    def visit_classical_declaration(self, node: QASMNode, context: dict):  # pylint: disable=no-self-use
        """
        Registers a classical declaration. Traces data flow through the context, transforming QASMNodes into Python
        type variables that can be readily used in expression evaluation, for example.

        Args:
            node (QASMNode): The ClassicalDeclaration QASMNode.
            context (dict): The current context.
        """
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

    @visit.register(QuantumGate)
    def visit_quantum_gate(self, node: QASMNode, context: dict):
        """
        Registers a quantum gate application. Calls the appropriate handler based on the sort of gate
        (parameterized or non-parameterized).

        Args:
            node (QASMNode): The QuantumGate QASMNode.
            context (dict): The current context.
        """
        self._require_wires(context)

        name = node.name.name.upper()
        if name in PARAMETERIZED_GATES:
            if not node.arguments:
                raise TypeError(
                    f"Missing required argument(s) for parameterized gate {node.name.name}"
                )
            gates_dict = PARAMETERIZED_GATES
        elif name in NON_PARAMETERIZED_GATES:
            gates_dict = NON_PARAMETERIZED_GATES
        else:
            raise NotImplementedError(f"Unsupported gate encountered in QASM: {node.name.name}")

        # setup arguments
        args = []
        for arg in node.arguments:
            if re.search("Literal", arg.__class__.__name__) is not None:
                args.append(arg.value)
            else:
                args.append(self.retrieve_variable(arg.name, context))

        # retrieve gate method
        gate = gates_dict[node.name.name.upper()]

        # setup wires
        wires = [
            # parser will sometimes represent as a str and sometimes as an Identifier
            (
                context["qubit_mapping"][
                    (
                        node.qubits[q].name
                        if isinstance(node.qubits[q].name, str)
                        else node.qubits[q].name.name
                    )
                ]
                if context["qubit_mapping"] is not None
                else (
                    node.qubits[q].name
                    if isinstance(node.qubits[q].name, str)
                    else node.qubits[q].name.name
                )
            )
            for q in range(len(node.qubits))
        ]

        if len(node.modifiers) > 0:
            num_control = sum(mod.modifier.name == "ctrl" or mod.modifier.name == "negctrl" for mod in node.modifiers)
            op_wires = wires[num_control:]
            control_wires = wires[:num_control]
            if node.modifiers[-1].modifier.name == "ctrl" or node.modifiers[-1].modifier.name == "negctrl":
                prev, wires = self.apply_modifier(
                    node.modifiers[-1], gate(*args, wires=op_wires), context, control_wires
                )
            else:
                prev, wires = self.apply_modifier(
                    node.modifiers[-1], gate(*args, wires=wires), context, wires
                )

            for mod in node.modifiers[::-1][1:]:
                prev, wires = self.apply_modifier(mod, prev, context, wires)
        else:
            gate(*args, wires=wires)

    @staticmethod
    def apply_modifier(mod: QASMNode, previous: Operator, context: dict, wires: list):
        """
        Applies a modifier to the previous gate or modified gate.

        Args:
            mod (QASMNode): The modifier QASMNode.
            previous (Operator): The previous (called) operator.
            context (dict): The current context.
            wires (list): The wires that the operator is applied to.
        """
        # the parser will raise when a modifier name is anything but the three modifiers (inv, pow, ctrl)
        # in the QASM 3.0 spec. i.e. if we change `pow(power) @` to `wop(power) @` it will raise:
        # `no viable alternative at input 'wop(power)@'`, long before we get here.
        assert mod.modifier.name in ("inv", "pow", "ctrl", "negctrl")
        next = None

        if mod.modifier.name == "inv":
            next = ops.adjoint(previous)
        elif mod.modifier.name == "pow":
            if re.search("Literal", mod.argument.__class__.__name__) is not None:
                next = ops.pow(previous, z=mod.argument.value)
            elif "vars" in context and mod.argument.name in context["vars"]:
                next = ops.pow(previous, z=context["vars"][mod.argument.name]["val"])
        elif mod.modifier.name == "ctrl":
            next = ops.ctrl(previous, control=wires[-1])
            wires = wires[:-1]
        elif mod.modifier.name == "negctrl":
            next = ops.ctrl(previous, control=wires[-1], control_values=[0])
            wires = wires[:-1]

        return next, wires

    @staticmethod
    def retrieve_variable(name: str, context: dict):
        """
        Attempts to retrieve a variable from the current context by name.

        Args:
            name (str): the name of the variable to retrieve.
            context (dict): the current context.
        """
        if name in context["vars"]:
            # the context at this point should reflect the states of the
            # variables as evaluated in the correct (current) scope.
            if context["vars"][name]["val"] is not None:
                return context["vars"][name]["val"]
            raise NameError(f"Attempt to reference uninitialized parameter {name}!")
        raise NameError(f"Undeclared variable {name} encountered in QASM.")

    @staticmethod
    def _require_wires(context):
        """
        Simple helper that checks if we have wires in the current context.

        Args:
            context (dict): The current context.

        Raises:
            NameError: If the context is missing a wire.
        """
        if len(context["wires"]) == 0:
            raise NameError(
                f"Attempt to reference wires that have not been declared in {context['name']}"
            )
