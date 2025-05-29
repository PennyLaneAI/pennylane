"""
This submodule contains the interpreter for OpenQASM 3.0.
"""

import functools
import re
from typing import Callable

from openqasm3.visitor import QASMNode
from openqasm3.ast import BitstringLiteral, ArrayLiteral, ClassicalDeclaration, QuantumGate, QubitDeclaration, \
    ConstantDeclaration, ClassicalAssignment, Cast, Identifier, IndexExpression, RangeDefinition, UnaryExpression, \
    BinaryExpression

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

EQUALS = '='
ARROW = '->'
PLUS = '+'
DOUBLE_PLUS = '++'
MINUS = '-'
ASTERISK = '*'
DOUBLE_ASTERISK = '**'
SLASH = '/'
PERCENT = '%'
PIPE = '|'
DOUBLE_PIPE = '||'
AMPERSAND = '&'
DOUBLE_AMPERSAND = '&&'
CARET = '^'
AT = '@'
TILDE = '~'
EXCLAMATION_POINT = '!'
EQUALITY_OPERATORS = ['==', '!=']
COMPOUND_ASSIGNMENT_OPERATORS = ['+=', '-=', '*=', '/=', '&=', '|=', '~=', '^=', '<<=', '>>=', '%=', '**=']
COMPARISON_OPERATORS = ['>', '<', '>=', '<=']
BIT_SHIFT_OPERATORS = ['>>', '<<']

NON_ASSIGNMENT_CLASSICAL_OPERATORS = EQUALITY_OPERATORS + COMPARISON_OPERATORS + BIT_SHIFT_OPERATORS \
    + [PLUS, DOUBLE_PLUS, MINUS, ASTERISK, DOUBLE_ASTERISK, SLASH, PERCENT, PIPE, DOUBLE_PIPE,
       AMPERSAND, DOUBLE_AMPERSAND, CARET, AT, TILDE, EXCLAMATION_POINT]

ASSIGNMENT_CLASSICAL_OPERATORS = [ARROW, EQUALS, COMPOUND_ASSIGNMENT_OPERATORS]

class QasmInterpreter:
    """
    Takes the top level node of the AST as a parameter and recursively descends the AST, calling the
    visitor function on each node.
    """

    def __init__(self, permissive=False):
        """
        Initializes the QASM interpreter.
        """
        self.permissive = permissive

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
        if self.permissive:
            print(
                f"An unrecognized QASM instruction {node.__class__.__name__} "
                f"was encountered on line {node.span.start_line}, in {context['name']}."
            )
        else:
            raise NotImplementedError(
                f"An unsupported QASM instruction {node.__class__.__name__} "
                f"was encountered on line {node.span.start_line}, in {context['name']}."
            )

        return context

    def interpret(self, node: QASMNode, context: dict):
        """
        Entry point for visiting the QASMNodes of a parsed OpenQASM 3.0 program.

        Args:
            node (QASMNode): The top-most QASMNode.
            context (dict): The initial context populated with the name of the program (the outermost scope).

        Returns:
            dict: The context updated after the compilation of all nodes by the visitor.
        """

        context.update({"wires": [], "vars": {}})

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
    def visit_qubit_declaration(
        self, node: QubitDeclaration, context: dict
    ):  # pylint: disable=no-self-use
        """
        Registers a qubit declaration. Named qubits are mapped to numbered wires by their indices
        in context["wires"]. Note: Qubit declarations must be global.

        Args:
            node (QASMNode): The QubitDeclaration QASMNode.
            context (dict): The current context.
        """
        context["wires"].append(node.qubit.name)

    def _update_var(self, value: any, name: str, node: QASMNode, context: dict):
        """
        Updates a variable, or raises if it is constant.
        Args:
            value (any): the value to set.
            name (str): the name of the variable.
            node (QASMNode): the QASMNode that corresponds to the update.
            context (dict): the current context.
        """
        context["vars"][name]["val"] = value
        if context["vars"][name]["constant"]:
            raise ValueError(
                f"Attempt to mutate a constant {name} on line {node.span.start_line} that was "
                f"defined on line {context['vars'][name]['line']}"
            )
        context["vars"][name]["line"] = node.span.start_line

    @visit.register(ClassicalAssignment)
    def visit_classical_assignment(self, node: QASMNode, context: dict):
        """
        Registers a classical assignment.
        Args:
            node (QASMNode): the assignment QASMNode.
            context (dict): the current context.
        """
        # references to an unresolved value see a func for now
        name = (
            node.lvalue.name if isinstance(node.lvalue.name, str) else node.lvalue.name.name
        )  # str or Identifier
        res = self.eval_expr(node.rvalue, context)
        self._update_var(res, name, node, context)

    def visit_alias_statement(self, node: QASMNode, context: dict):
        """
        Registers an alias statement.
        Args:
            node (QASMNode): the alias QASMNode.
            context (dict): the current context.
        """
        self._init_aliases(context)
        context["aliases"][node.target.name] = self.eval_expr(node.value, context, aliasing=True)

    def retrieve_variable(self, name: str, context: dict):
        """
        Attempts to retrieve a variable from the current context by name.
        Args:
            name (str): the name of the variable to retrieve.
            context (dict): the current context.
        """

        if "vars" in context and context["vars"] is not None and name in context["vars"]:
            res = context["vars"][name]
            if res["val"] is not None:
                return res
            else:
                raise NameError(f"Attempt to reference uninitialized parameter {name}!")
        elif (
            "wires" in context
            and context["wires"] is not None
            and name in context["wires"]
            or "outer_wires" in context
            and name in context["outer_wires"]
        ):
            return name
        elif "aliases" in context and context["wires"] is not None and name in context["aliases"]:
            res = context["aliases"][name](context)  # evaluate the alias and de-reference
            if isinstance(res, str):
                return res
            if res["val"] is not None:
                return res
            else:
                raise NameError(f"Attempt to reference uninitialized parameter {name}!")
        else:
            raise TypeError(
                f"Attempt to use unevaluated variable {name} in {context['name']}, "
                f"last updated on line {context['vars'][name]['line'] if name in context['vars'] else 'unknown'}."
            )

    def _init_vars(self, context: dict):
        """
        context["callable"] = partial(self._execute_all, context)
        Inits the vars dict on the current context.
        Args:
            context (dict): the current context.
        """
        if "vars" not in context:
            context["vars"] = dict()

    def _init_aliases(self, context: dict):
        """
        Inits the aliases dict on the current context.
        Args:
            context (dict): the current context.
        """
        if "aliases" not in context:
            context["aliases"] = dict()

    @staticmethod
    def _get_bit_type_val(var):
        return bin(var["val"])[2:].zfill(var["size"])

    @visit.register(ConstantDeclaration)
    def visit_constant_declaration(self, node: QASMNode, context: dict):
        """
        Registers a constant declaration. Traces data flow through the context, transforming QASMNodes into
        Python type variables that can be readily used in expression eval, etc.
        Args:
            node (QASMNode): The constant QASMNode.
            context (dict): The current context.
        """
        self.visit_classical_declaration(node, context, constant=True)

    @visit.register(ClassicalDeclaration)
    def visit_classical_declaration(self, node: QASMNode, context: dict, constant:bool=False):
        """
        Registers a classical declaration. Traces data flow through the context, transforming QASMNodes into Python
        type variables that can be readily used in expression evaluation, for example.
        Args:
            node (QASMNode): The ClassicalDeclaration QASMNode.
            context (dict): The current context.
            constant (bool): Whether the classical variable is a constant.
        """

        self._init_vars(context)
        if node.init_expression is not None:
            if isinstance(node.init_expression, BitstringLiteral):
                context["vars"][node.identifier.name] = {
                    "ty": node.type.__class__.__name__,
                    "val": self.eval_expr(node.init_expression, context),
                    "size": node.init_expression.width,
                    "line": node.init_expression.span.start_line,
                    "constant": constant,
                }
            elif not isinstance(node.init_expression, ArrayLiteral):
                context["vars"][node.identifier.name] = {
                    "ty": node.type.__class__.__name__,
                    "val": self.eval_expr(node.init_expression, context),
                    "line": node.init_expression.span.start_line,
                    "constant": constant,
                }
            else:
                context["vars"][node.identifier.name] = {
                    "ty": node.type.__class__.__name__,
                    "val": [
                        self.eval_expr(literal, context) for literal in node.init_expression.values
                    ],
                    "line": node.init_expression.span.start_line,
                    "constant": constant,
                }
        else:
            # the var is declared but uninitialized
            context["vars"][node.identifier.name] = {
                "ty": node.type.__class__.__name__,
                "val": None,
                "line": node.span.start_line,
                "constant": constant,
            }

    @visit.register(QuantumGate)
    def visit_quantum_gate(self, node: QuantumGate, context: dict):
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
            gates_dict = PARAMETERIZED_GATES
        elif name in NON_PARAMETERIZED_GATES:
            gates_dict = NON_PARAMETERIZED_GATES
        else:
            raise NotImplementedError(f"Unsupported gate encountered in QASM: {node.name.name}")

        gate, args, wires = self._gate_setup_helper(node, gates_dict, context)

        if len(node.modifiers) > 0:
            num_control = sum("ctrl" in mod.modifier.name for mod in node.modifiers)
            op_wires = wires[num_control:]
            control_wires = wires[:num_control]
            if "ctrl" in node.modifiers[-1].modifier.name:
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

    def _gate_setup_helper(self, node: QuantumGate, gates_dict: dict, context: dict):
        """
        Helper to setup the quantum gate call, also resolving arguments and wires.

        Args:
            node (QuantumGate): The QuantumGate QASMNode.
            gates_dict (dict): the gates dictionary.
            context (dict): the current context.

        Returns:
            QuantumGate: The gate to execute.
            list: The list of arguments to the QuantumGate.
            list: The wires the gate applies to.
        """
        # setup arguments
        args = []
        for arg in node.arguments:
            args.append(self.eval_expr(arg, context))

        # retrieve gate method
        gate = gates_dict[node.name.name.upper()]

        # setup wires
        wires = [
            # parser will sometimes represent as a str and sometimes as an Identifier
            (
                node.qubits[q].name
                if isinstance(node.qubits[q].name, str)
                else node.qubits[q].name.name
            )
            for q in range(len(node.qubits))
        ]

        self._require_wires(wires, context)

        if context["wire_map"] is not None:
            wires = list(map(lambda wire: context["wire_map"][wire], wires))

        return gate, args, wires

    @staticmethod
    def apply_modifier(mod: QuantumGate, previous: Operator, context: dict, wires: list):
        """
        Applies a modifier to the previous gate or modified gate.

        Args:
            mod (QASMNode): The modifier QASMNode.
            previous (Operator): The previous (called) operator.
            context (dict): The current context.
            wires (list): The wires that the operator is applied to.

        Raises:
            NotImplementedError: If the modifier has a param of an as-yet unsupported type.
        """
        # the parser will raise when a modifier name is anything but the three modifiers (inv, pow, ctrl)
        # in the OpenQASM 3.0 spec. i.e. if we change `pow(power) @` to `wop(power) @` it will raise:
        # `no viable alternative at input 'wop(power)@'`, long before we get here.
        assert mod.modifier.name in ("inv", "pow", "ctrl", "negctrl")
        next = None

        if mod.modifier.name == "inv":
            next = ops.adjoint(previous)
        elif mod.modifier.name == "pow":
            if re.search("Literal", mod.argument.__class__.__name__) is not None:
                power = mod.argument.value
            elif (
                "vars" in context
                and hasattr(mod.argument, "name")
                and mod.argument.name in context["vars"]
            ):
                power = context["vars"][mod.argument.name]["val"]
            else:
                raise NotImplementedError(
                    f"Unable to handle {mod.argument.__class__.__name__} expression at this time"
                )
            next = ops.pow(previous, z=power)
        elif mod.modifier.name == "ctrl":
            next = ops.ctrl(previous, control=wires[-1])
            wires = wires[:-1]
        elif mod.modifier.name == "negctrl":
            next = ops.ctrl(previous, control=wires[-1], control_values=[0])
            wires = wires[:-1]

        return next, wires

    def eval_expr(self, node: QASMNode, context: dict, aliasing: bool = False):
        """
        Constructs a callable that can be queued into a QNode.
        Evaluates an expression.
        Args:
            context (dict): The final context populated with the Callables (called gates) to queue in the QNode.
            node (QASMNode): the expression QASMNode.
            context (dict): the current context.
            aliasing (bool): whether to use aliases or not.
        """
        res = None
        if isinstance(node, Cast):
            return self.retrieve_variable(node.argument.name, context)["val"]
        elif isinstance(node, BinaryExpression):
            lhs = self.eval_expr(node.lhs, context)
            rhs = self.eval_expr(node.rhs, context)
            if node.op.name in NON_ASSIGNMENT_CLASSICAL_OPERATORS:  # makes sure we are not executing anything malicious
                res = eval(f"{lhs}{node.op.name}{rhs}")
            elif node.op.name in ASSIGNMENT_CLASSICAL_OPERATORS:
                raise SyntaxError(f"{node.op.name} assignment operators should only be used in classical assignments,"
                                  f"not in binary expressions.")
            else:
                raise SyntaxError(f"Invalid operator {node.op.name} encountered in binary expression "
                                  f"on line {node.span.start_line}.")
        elif isinstance(node, UnaryExpression):
            if node.op.name in NON_ASSIGNMENT_CLASSICAL_OPERATORS:  # makes sure we are not executing anything malicious
                res = eval(f"{node.op.name}{self.eval_expr(node.expression, context)}")
            elif node.op.name in ASSIGNMENT_CLASSICAL_OPERATORS:
                raise SyntaxError(f"{node.op.name} assignment operators should only be used in classical assignments,"
                                  f"not in unary expressions.")
            else:
                raise SyntaxError(f"Invalid operator {node.op.name} encountered in unary expression "
                                  f"on line {node.span.start_line}.")
        elif isinstance(node, IndexExpression):

            def _index_into_var(var):
                if var["ty"] == "BitType":
                    var = bin(var["val"])[2:].zfill(var["size"])
                else:
                    var = var["val"]
                if isinstance(node.index[0], RangeDefinition):
                    return var[node.index[0].start.value : node.index[0].end.value]
                elif re.search("Literal", node.index[0].__class__.__name__):
                    return var[node.index[0].value]
                else:
                    raise TypeError(
                        f"Array index is not a RangeDefinition or Literal at line {node.span.start_line}."
                    )

            if aliasing:

                def alias(context):
                    try:
                        var = self.retrieve_variable(node.collection.name, context)
                        return _index_into_var(var)
                    except NameError:
                        raise NameError(
                            f"Attempt to alias an undeclared variable "
                            f"{node.collection.name} in {context['name']}."
                        ) from e

                res = alias
            else:
                var = self.retrieve_variable(node.collection.name, context)
                return _index_into_var(var)
        elif isinstance(node, Identifier):
            if aliasing:

                def alias(context):
                    try:
                        return self.retrieve_variable(node.collection.name, context)
                    except NameError as e:
                        raise NameError(
                            f"Attempt to alias an undeclared variable "
                            f"{node.name} in {context['name']}."
                        ) from e

                res = alias
            else:
                try:
                    var = self.retrieve_variable(node.name, context)
                    value = var["val"] if isinstance(var, dict) and "val" in var else var
                    if isinstance(value, Callable):
                        var["val"] = value(context)
                        var["line"] = node.span.start_line
                        value = var["val"]
                    return value
                except NameError as e:
                    raise NameError(
                        str(e) or f"Reference to an undeclared variable {node.name} in {context['name']}."
                    ) from e
        elif isinstance(node, Callable):
            res = node()
        elif re.search("Literal", node.__class__.__name__):
            res = node.value
        # TODO: include all other cases here
        return res

    @staticmethod
    def _require_wires(wires: list, context: dict):
        """
        Simple helper that checks if we have wires in the current context.

        Args:
            context (dict): The current context.
            wires (list): The wires that are required.

        Raises:
            NameError: If the context is missing a wire.
        """
        missing_wires = []
        for wire in wires:
            if wire not in context["wires"]:
                missing_wires.append(wire)
        if len(missing_wires) > 0:
            raise NameError(
                f"Attempt to reference wire(s): {missing_wires} that have not been declared in {context['name']}"
            )
