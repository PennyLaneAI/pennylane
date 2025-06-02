"""
This submodule contains the interpreter for OpenQASM 3.0.
"""

import functools
import re
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from openqasm3.ast import (
    AliasStatement,
    ArrayLiteral,
    BinaryExpression,
    Cast,
    ClassicalAssignment,
    ClassicalDeclaration,
    ConstantDeclaration,
    EndStatement,
    Expression,
    ExpressionStatement,
    Identifier,
    IndexExpression,
    QuantumGate,
    QubitDeclaration,
    RangeDefinition,
    UnaryExpression,
)
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

EQUALS = "="
ARROW = "->"
PLUS = "+"
DOUBLE_PLUS = "++"
MINUS = "-"
ASTERISK = "*"
DOUBLE_ASTERISK = "**"
SLASH = "/"
PERCENT = "%"
PIPE = "|"
DOUBLE_PIPE = "||"
AMPERSAND = "&"
DOUBLE_AMPERSAND = "&&"
CARET = "^"
AT = "@"
TILDE = "~"
EXCLAMATION_POINT = "!"
EQUALITY_OPERATORS = ["==", "!="]
COMPOUND_ASSIGNMENT_OPERATORS = [
    "+=",
    "-=",
    "*=",
    "/=",
    "&=",
    "|=",
    "~=",
    "^=",
    "<<=",
    ">>=",
    "%=",
    "**=",
]
COMPARISON_OPERATORS = [">", "<", ">=", "<="]
BIT_SHIFT_OPERATORS = [">>", "<<"]

NON_ASSIGNMENT_CLASSICAL_OPERATORS = (
    EQUALITY_OPERATORS
    + COMPARISON_OPERATORS
    + BIT_SHIFT_OPERATORS
    + [
        PLUS,
        DOUBLE_PLUS,
        MINUS,
        ASTERISK,
        DOUBLE_ASTERISK,
        SLASH,
        PERCENT,
        PIPE,
        DOUBLE_PIPE,
        AMPERSAND,
        DOUBLE_AMPERSAND,
        CARET,
        AT,
        TILDE,
        EXCLAMATION_POINT,
    ]
)

ASSIGNMENT_CLASSICAL_OPERATORS = [ARROW, EQUALS, COMPOUND_ASSIGNMENT_OPERATORS]


@dataclass
class Variable:
    ty: str
    val: Any
    size: int
    line: int
    constant: bool


class Context:
    """Class with helper methods for managing, updating, checking context."""

    def __init__(self, context):
        if "vars" not in context:
            context["vars"] = {}
        if "aliases" not in context:
            context["aliases"] = {}
        self.context = context
        if "wires" not in context:
            context["wires"] = []

    def update_var(
        self,
        value: any,
        name: str,
        node: QASMNode,
    ):
        """
        Updates a variable, or raises if it is constant.
        Args:
            value (any): the value to set.
            name (str): the name of the variable.
            node (QASMNode): the QASMNode that corresponds to the update.
        """
        if self.vars[name].constant:
            raise ValueError(
                f"Attempt to mutate a constant {name} on line {node.span.start_line} that was "
                f"defined on line {self.vars[name].line}"
            )
        self.vars[name].val = value
        self.vars[name].line = node.span.start_line

    def require_wires(self, wires: list):
        """
        Simple helper that checks if we have wires in the current context.

        Args:
            wires (list): The wires that are required.

        Raises:
            NameError: If the context is missing a wire.
        """
        missing_wires = []
        for wire in wires:
            if wire not in self.wires:
                missing_wires.append(wire)
        if len(missing_wires) > 0:
            raise NameError(
                f"Attempt to reference wire(s): {missing_wires} that have not been declared in {self.name}"
            )

    def __getattr__(self, name: str):
        """
        If the attribute is not found on the class, instead uses the attr name as an index
        into the context dictionary, for easy access.
        """
        if name in self.context:
            return self.context[name]
        else:
            raise NameError(
                f"No attribute {name} on Context and no {name} key found on context {self.context}"
            )


def _get_bit_type_val(var):
    return bin(var.val)[2:].zfill(var.size)


def _index_into_var(var: dict, node: IndexExpression):
    """
    Index into a variable using an IndexExpression.

    Args:
        var (Variable): The data structure representing the variable to index.

    Returns:
        The indexed slice of the variable.
    """
    if var.ty == "BitType":
        var = bin(var.val)[2:].zfill(var.size)
    else:
        var = var.val
    if isinstance(node.index[0], RangeDefinition):
        return var[node.index[0].start.value : node.index[0].end.value]
    if re.search("Literal", node.index[0].__class__.__name__):
        return var[node.index[0].value]
    raise TypeError(
        f"Array index is not a RangeDefinition or Literal at line {node.span.start_line}."
    )


class EndProgram(Exception):
    """Exception raised when it encounters an end statement in the QASM circuit."""


class QasmInterpreter:
    """
    Takes the top level node of the AST as a parameter and recursively descends the AST, calling the
    visitor function on each node.
    """

    @functools.singledispatchmethod
    def visit(
        self, node: QASMNode, context: Context, aliasing: bool = False
    ):  # pylint: disable=unused-argument
        """
        Visitor function is called on each node in the AST, which is traversed using recursive descent.
        The purpose of this function is to pass each node to the appropriate handler.

        Args:
            node (QASMNode): the QASMNode to visit next.
            context (Context): the current context populated with any locally available variables, etc.
            aliasing (bool): whether we are aliasing a variable in the context.

        Raises:
            NotImplementedError: When a (so far) unsupported node type is encountered.
        """
        if re.search("Literal", node.__class__.__name__):  # There is no single "Literal" base class
            return self.visit_literal(node)
        if callable(node):
            return self.visit_callable(node, context)  # cannot register Callable
        raise NotImplementedError(
            f"An unsupported QASM instruction {node.__class__.__name__} "
            f"was encountered on line {node.span.start_line}, in {context.name}."
        )

    def interpret(self, node: QASMNode, context: dict):
        """
        Entry point for visiting the QASMNodes of a parsed OpenQASM 3.0 program.

        Args:
            node (QASMNode): The top-most QASMNode.
            context (dict): The initial context populated with the name of the program (the outermost scope).

        Returns:
            dict: The context updated after the compilation of all nodes by the visitor.
        """
        context = Context(context)

        # begin recursive descent traversal
        try:
            for value in node.__dict__.values():
                if not isinstance(value, list):
                    value = [value]
                for item in value:
                    if isinstance(item, QASMNode):
                        self.visit(item, context)
        except EndProgram:
            pass
        return context

    @visit.register(EndStatement)
    def visit_end_statement(self, node: QASMNode, context: Context):  # pylint: disable=no-self-use
        """
        Ends the program.
        Args:
            node (QASMNode): The end statement QASMNode.
            context (Context): the current context.
        """
        raise EndProgram(
            f"The QASM program was terminated om line {node.span.start_line}."
            f"There may be unprocessed QASM code."
        )

    # needs to have same signature as visit()
    @visit.register(QubitDeclaration)
    def visit_qubit_declaration(
        self, node: QubitDeclaration, context: Context
    ):  # pylint: disable=no-self-use
        """
        Registers a qubit declaration. Named qubits are mapped to numbered wires by their indices
        in context.wires. Note: Qubit declarations must be global.

        Args:
            node (QASMNode): The QubitDeclaration QASMNode.
            context (Context): The current context.
        """
        context.wires.append(node.qubit.name)

    @visit.register(ClassicalAssignment)
    def visit_classical_assignment(self, node: QASMNode, context: Context):
        """
        Registers a classical assignment.
        Args:
            node (QASMNode): the assignment QASMNode.
            context (Context): the current context.
        """
        # references to an unresolved value see a func for now
        name = (
            node.lvalue.name if isinstance(node.lvalue.name, str) else node.lvalue.name.name
        )  # str or Identifier
        res = self.visit(node.rvalue, context)
        context.update_var(res, name, node)

    @visit.register(AliasStatement)
    def visit_alias_statement(self, node: QASMNode, context: Context):
        """
        Registers an alias statement.
        Args:
            node (QASMNode): the alias QASMNode.
            context (Context): the current context.
        """
        context.aliases[node.target.name] = self.visit(node.value, context, aliasing=True)

    @staticmethod
    def retrieve_variable(name: str, context: Context):
        """
        Attempts to retrieve a variable from the current context by name.
        Args:
            name (str): the name of the variable to retrieve.
            context (Context): the current context.
        """

        if name in context.vars:
            res = context.vars[name]
            if res.val is not None:
                return res
            raise NameError(f"Attempt to reference uninitialized parameter {name}!")
        if name in context["wires"]:
            return name
        if name in context.aliases:
            res = context.aliases[name](context)  # evaluate the alias and de-reference
            if isinstance(res, str):
                return res
            if res.val is not None:
                return res
            raise NameError(f"Attempt to reference uninitialized parameter {name}!")
        raise TypeError(
            f"Attempt to use unevaluated variable {name} in {context.name}, "
            f"last updated on line {context.vars[name]['line'] if name in context.vars else 'unknown'}."
        )

    @visit.register(ConstantDeclaration)
    def visit_constant_declaration(self, node: QASMNode, context: Context):
        """
        Registers a constant declaration. Traces data flow through the context, transforming QASMNodes into
        Python type variables that can be readily used in expression eval, etc.
        Args:
            node (QASMNode): The constant QASMNode.
            context (Context): The current context.
        """
        self.visit_classical_declaration(node, context, constant=True)

    @visit.register(ClassicalDeclaration)
    def visit_classical_declaration(self, node: QASMNode, context: Context, constant: bool = False):
        """
        Registers a classical declaration. Traces data flow through the context, transforming QASMNodes into Python
        type variables that can be readily used in expression evaluation, for example.
        Args:
            node (QASMNode): The ClassicalDeclaration QASMNode.
            context (Context): The current context.
            constant (bool): Whether the classical variable is a constant.
        """

        context.vars[node.identifier.name] = Variable(
            node.type.__class__.__name__,
            (
                self.visit(node.init_expression, context)
                if hasattr(node, "init_expression") and node.init_expression is not None
                else None
            ),
            (
                node.init_expression.width
                if hasattr(node, "init_expression") and hasattr(node.init_expression, "width")
                else None
            ),
            (
                node.init_expression.span.start_line
                if hasattr(node, "init_expression") and node.init_expression is not None
                else node.span.start_line
            ),
            constant,
        )

    @visit.register(ArrayLiteral)
    def visit_array_literal(self, node: ArrayLiteral, context: Context):
        """
        Evaluates an array literal.

        Args:
            node (ArrayLiteral): The array literal QASMNode.
            context (Context): The current context.

        Returns:
            list: The evaluated array.
        """
        return [self.visit(literal, context) for literal in node.values]

    @visit.register(QuantumGate)
    def visit_quantum_gate(self, node: QuantumGate, context: Context):
        """
        Registers a quantum gate application. Calls the appropriate handler based on the sort of gate
        (parameterized or non-parameterized).

        Args:
            node (QASMNode): The QuantumGate QASMNode.
            context (Context): The current context.
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
        num_control = sum("ctrl" in mod.modifier.name for mod in node.modifiers)
        op_wires = wires[num_control:]
        control_wires = wires[:num_control]

        op = gate(*args, wires=op_wires)
        for mod in reversed(node.modifiers):
            op, control_wires = self.apply_modifier(mod, op, context, control_wires)

    def _gate_setup_helper(self, node: QuantumGate, gates_dict: dict, context: Context):
        """
        Helper to setup the quantum gate call, also resolving arguments and wires.

        Args:
            node (QuantumGate): The QuantumGate QASMNode.
            gates_dict (dict): the gates dictionary.
            context (Context): the current context.

        Returns:
            QuantumGate: The gate to execute.
            list: The list of arguments to the QuantumGate.
            list: The wires the gate applies to.
        """
        # setup arguments
        args = [self.visit(arg, context) for arg in node.arguments]

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

        context.require_wires(wires)

        if context.wire_map is not None:
            wires = list(map(lambda wire: context.wire_map[wire], wires))

        return gate, args, wires

    def apply_modifier(self, mod: QuantumGate, previous: Operator, context: Context, wires: list):
        """
        Applies a modifier to the previous gate or modified gate.

        Args:
            mod (QASMNode): The modifier QASMNode.
            previous (Operator): The previous (called) operator.
            context (Context): The current context.
            wires (list): The wires that the operator is applied to.

        Raises:
            NotImplementedError: If the modifier has a param of an as-yet unsupported type.
        """
        # the parser will raise when a modifier name is anything but the three modifiers (inv, pow, ctrl)
        # in the OpenQASM 3.0 spec. i.e. if we change `pow(power) @` to `wop(power) @` it will raise:
        # `no viable alternative at input 'wop(power)@'`, long before we get here.
        assert mod.modifier.name in ("inv", "pow", "ctrl", "negctrl")

        if mod.modifier.name == "inv":
            next = ops.adjoint(previous)
        elif mod.modifier.name == "pow":
            power = self.visit(mod.argument, context)
            next = ops.pow(previous, z=power)
        elif mod.modifier.name == "ctrl":
            next = ops.ctrl(previous, control=wires[-1])
            wires = wires[:-1]
        elif mod.modifier.name == "negctrl":
            next = ops.ctrl(previous, control=wires[-1], control_values=[0])
            wires = wires[:-1]
        else:
            raise ValueError(f"Unknown modifier {mod}")  # pragma: no cover

        return next, wires

    @visit.register(ExpressionStatement)
    def visit_expression_statement(self, node: ExpressionStatement, context: Context):
        """
        Registers an expression statement.
        Args:
            node (ExpressionStatement): The expression statement.
            context (Context): The current context.
        """
        return self.visit(node.expression, context)

    @visit.register(Cast)
    def visit_cast(self, node: Cast, context: Context):
        """
        Registers a Cast expression.

        Args:
            node (Cast): The Cast expression.
            context (Context): The current context.
        """
        return self.retrieve_variable(node.argument.name, context).val

    @visit.register(BinaryExpression)
    def visit_binary_expression(self, node: BinaryExpression, context: Context):
        """
        Registers a binary expression.

        Args:
            node (BinaryExpression): The binary expression.
            context (Context): The current context.

        Returns:
            The result of the evaluated expression.
        """
        lhs = float(self.visit(node.lhs, context))
        rhs = float(self.visit(node.rhs, context))
        if (
            node.op.name in NON_ASSIGNMENT_CLASSICAL_OPERATORS
        ):  # makes sure we are not executing anything malicious
            return eval(f"{lhs}{node.op.name}{rhs}")  # pylint: disable=eval-used
        if node.op.name in ASSIGNMENT_CLASSICAL_OPERATORS:
            raise SyntaxError(
                f"{node.op.name} assignment operators should only be used in classical assignments,"
                f"not in binary expressions."
            )
        raise SyntaxError(
            f"Invalid operator {node.op.name} encountered in binary expression "
            f"on line {node.span.start_line}."
        )

    @visit.register(UnaryExpression)
    def visit_unary_expression(self, node: UnaryExpression, context: Context):
        """
        Registers a unary expression.

        Args:
            node (UnaryExpression): The unary expression.
            context (Context): The current context.

        Returns:
            The result of the evaluated expression.
        """
        if (
            node.op.name in NON_ASSIGNMENT_CLASSICAL_OPERATORS
        ):  # makes sure we are not executing anything malicious
            return eval(  # pylint: disable=eval-used
                f"{node.op.name}{float(self.visit(node.expression, context))}"
            )
        if node.op.name in ASSIGNMENT_CLASSICAL_OPERATORS:
            raise SyntaxError(
                f"{node.op.name} assignment operators should only be used in classical assignments,"
                f"not in unary expressions."
            )
        raise SyntaxError(
            f"Invalid operator {node.op.name} encountered in unary expression "
            f"on line {node.span.start_line}."
        )

    @visit.register(IndexExpression)
    def visit_index_expression(
        self, node: IndexExpression, context: Context, aliasing: bool = False
    ):
        """
        Registers an index expression.

        Args:
            node (IndexExpression): The index expression.
            context (Context): The current context.
            aliasing (bool): If ``True``, the expression will be treated as an alias.

        Returns:
            The slice of the indexed value.
        """

        if aliasing:  # we are registering an alias
            return lambda cntxt: _index_into_var(self._alias(node, cntxt), node)

        # else we are just evaluating an index
        var = self.retrieve_variable(node.collection.name, context)
        return _index_into_var(var, node)

    def _alias(self, node: Identifier | IndexExpression, context: Context):
        """
        An alias is registered as a callable since we need to be able to
        evaluate it at a later time.

        Args:
            context (Context): The current context.

        Returns:
            The de-referenced alias.
        """
        try:
            return self.retrieve_variable(node.collection.name, context)
        except NameError as e:
            raise NameError(
                f"Attempt to alias an undeclared variable " f"{node.name} in {context.name}."
            ) from e

    @visit.register(Identifier)
    def visit_identifier(self, node: Identifier, context: Context, aliasing: bool = False):
        """
        Registers an identifier.

        Args:
            node (Identifier): The identifier.
            context (Context): The current context.
            aliasing (bool): If ``True``, the Identifier will be treated as an alias.

        Returns:
            The de-referenced identifier.
        """
        if aliasing:  # we are registering an alias
            return partial(self._alias, node)
        # else we are evaluating an alias
        try:
            var = self.retrieve_variable(node.name, context)
            value = var.val if isinstance(var, Variable) else var
            if callable(value):
                var.val = value(context)
                var.line = node.span.start_line
                value = var.val
            return value
        except NameError as e:
            raise NameError(
                str(e) or f"Reference to an undeclared variable {node.name} in {context.name}."
            ) from e

    @staticmethod
    def visit_callable(func: Callable, context: Context):
        """
        Visits a Callable.

        Args:
            func (Callable): The callable.
            context (Context): The current context.

        Returns:
            The result of the called callable.
        """
        return func(context)

    @staticmethod
    def visit_literal(node: Expression):
        """
        Visits a literal.

        Args:
            node (Literal): The literal.

        Returns:
            The value of the literal.
        """
        return node.value
