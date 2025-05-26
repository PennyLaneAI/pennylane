"""
This submodule contains the interpreter for QASM 3.0.
"""

import re
from functools import partial
from typing import Callable

from pennylane import ops

has_openqasm = True
try:
    from openqasm3.visitor import QASMNode, QASMVisitor
except (ModuleNotFoundError, ImportError) as import_error:  # pragma: no cover
    has_openqasm = False  # pragma: no cover

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


class QasmInterpreter(QASMVisitor):
    """
    Overrides generic_visit(self, node: QASMNode, context: Optional[T]) which takes the
    top level node of the AST as a parameter and recursively descends the AST, calling the
    overriden visitor function on each node.
    """

    def __init__(self):
        """
        Checks that the openqasm3 package is available, otherwise raises an error.

        Raises:
            ImportError: if the openqasm3 package is not available.
        """
        if not has_openqasm:  # pragma: no cover
            raise ImportError(
                "QASM interpreter requires openqasm3 to be installed"
            )  # pragma: no cover
        super().__init__()

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
        handler_name = "visit_" + node.__class__.__name__
        if hasattr(self, handler_name):
            getattr(self, handler_name)(node, context)
        else:
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
            dict: The context updated after the compilation of all nodes by the visitor.
        """

        context.update({"wires": [], "vars": {}, "gates": [], "callable": None})

        # begin recursive descent traversal
        super().generic_visit(node, context)
        self.construct_callable(context)
        return context

    @staticmethod
    def _execute_all(context: dict):
        """
        Executes all the gates in the context.

        Args:
            context (dict): the current context populated with the gates.
        """
        for func in context["gates"]:
            func()

    def construct_callable(self, context: dict):
        """
        Constructs a callable that can be queued into a QNode.

        Args:
            context (dict): The final context populated with the Callables (called gates) to queue in the QNode.
        """
        context["callable"] = partial(self._execute_all, context)

    @staticmethod
    def visit_QubitDeclaration(node: QASMNode, context: dict):
        """
        Registers a qubit declaration. Named qubits are mapped to numbered wires by their indices
        in context["wires"]. TODO: this should be changed to have greater specificity. Coming in a follow-up PR.

        Args:
            node (QASMNode): The QubitDeclaration QASMNode.
            context (dict): The current context.
        """
        context["wires"].append(node.qubit.name)

    @staticmethod
    def visit_ClassicalDeclaration(node: QASMNode, context: dict):
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

    def visit_QuantumGate(self, node: QASMNode, context: dict):
        """
        Registers a quantum gate application. Calls the appropriate handler based on the sort of gate
        (parameterized or non-parameterized).

        Args:
            node (QASMNode): The QuantumGate QASMNode.
            context (dict): The current context.
        """
        name = node.name.name.upper()
        if name in PARAMETERIZED_GATES:
            if not node.arguments:
                raise TypeError(
                    f"Missing required argument(s) for parameterized gate {node.name.name}"
                )
            gate = self.gate(PARAMETERIZED_GATES, node, context)
        elif name in NON_PARAMETERIZED_GATES:
            gate = self.gate(NON_PARAMETERIZED_GATES, node, context)
        else:
            raise NotImplementedError(f"Unsupported gate encountered in QASM: {node.name.name}")

        if len(node.modifiers) > 0:
            gate = self.modifiers(gate, node, context)

        context["gates"].append(gate)

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
    def modifiers(gate: Callable, node: QASMNode, context: dict):
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
            # the parser will raise when a modifier name is anything but the three modifiers (inv, pow, ctrl)
            # in the QASM 3.0 spec. i.e. if we change `pow(power) @` to `wop(power) @` it will raise:
            # `no viable alternative at input 'wop(power)@'`, long before we get here.
            assert mod.modifier.name in ("inv", "pow", "ctrl")

            if mod.modifier.name == "inv":
                wrapper = ops.adjoint
            elif mod.modifier.name == "pow":
                if re.search("Literal", mod.argument.__class__.__name__) is not None:
                    wrapper = partial(ops.pow, z=mod.argument.value)
                elif "vars" in context and mod.argument.name in context["vars"]:
                    wrapper = partial(ops.pow, z=context["vars"][mod.argument.name]["val"])
            elif mod.modifier.name == "ctrl":
                wrapper = partial(ops.ctrl, control=gate.keywords["wires"][0:-1])

            call_stack.append(wrapper)

        def call():
            res = None
            for func in call_stack:
                # if there is a control in the stack
                if (
                    call_stack[-1].__class__.__name__ == "partial"
                    and "control" in call_stack[-1].keywords
                ):
                    # if we are processing the control now
                    if "control" in func.keywords:
                        res.keywords["wires"] = [res.keywords["wires"][-1]]
                    # i.e. qml.ctrl(qml.RX, (1))(2, wires=0)
                    res = func(res.func)(**res.keywords) if res is not None else func
                else:
                    # i.e. qml.pow(qml.RX(1.5, wires=0), z=4)
                    res = func(res) if res is not None else func()

        return call

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

    def gate(self, gates_dict: dict, node: QASMNode, context: dict):
        """
        Registers a gate application. Builds a Callable partial
        that can be executed when the QNode is called. The gate will be executed at that time
        with the appropriate arguments.
        TODO: a robust method for retrieving vars from context will be provided in follow-up PR according to [sc-90383]

        Args:
            node (QASMNode): The QuantumGate QASMNode.
            context (dict): The current context.

        Returns:
            Callable: The Callable partial that will execute the gate with the appropriate arguments at
                "runtime".

        Raises:
            NameError: If an argument is not found in the current context.
        """
        self._require_wires(context)
        args = []
        for arg in node.arguments:
            if re.search("Literal", arg.__class__.__name__) is not None:
                args.append(arg.value)
            else:
                args.append(self.retrieve_variable(arg.name, context))
        return partial(
            gates_dict[node.name.name.upper()],
            *args,
            wires=[
                # parser will sometimes represent as a str and sometimes as a Identifier
                (
                    node.qubits[q].name
                    if isinstance(node.qubits[q].name, str)
                    else node.qubits[q].name.name
                )
                for q in range(len(node.qubits))
            ],
        )
