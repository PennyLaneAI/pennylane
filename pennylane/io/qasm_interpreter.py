"""
This submodule contains the interpreter for OpenQASM 3.0.
"""

import copy
import functools
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Iterable

from numpy import uint
from openqasm3.ast import (
    AliasStatement,
    ArrayLiteral,
    BinaryExpression,
    BitstringLiteral,
    BitType,
    BooleanLiteral,
    BoolType,
    BranchingStatement,
    BreakStatement,
    Cast,
    ClassicalAssignment,
    ClassicalDeclaration,
    ComplexType,
    ConstantDeclaration,
    ContinueStatement,
    DiscreteSet,
    DurationLiteral,
    EndStatement,
    Expression,
    ExpressionStatement,
    FloatLiteral,
    FloatType,
    ForInLoop,
    FunctionCall,
    Identifier,
    ImaginaryLiteral,
    IndexExpression,
    IntegerLiteral,
    IntType,
    IODeclaration,
    IOKeyword,
    QuantumArgument,
    QuantumGate,
    QuantumGateDefinition,
    QuantumMeasurementStatement,
    QuantumReset,
    QubitDeclaration,
    RangeDefinition,
    ReturnStatement,
    SubroutineDefinition,
    SwitchStatement,
    UintType,
    UnaryExpression,
    WhileLoop,
)
from openqasm3.visitor import QASMNode

from pennylane import ops
from pennylane.control_flow import for_loop, while_loop
from pennylane.measurements import MeasurementValue, measure
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


def _eval_unary_op(operand: any, operator: str, line: int):
    """
    Evaluates a unary operator.

    Args:
        operand (any): The only operand.
        operator (str): The unary operation.
        line (int): The line number.
    """
    if operator == "!":
        return not operand
    if operator == "-":
        return -operand  # pylint: disable=invalid-unary-operand-type
    if operator == "~":
        return ~operand  # pylint: disable=invalid-unary-operand-type
    # we shouldn't ever get thi error if the parser did its job right
    raise SyntaxError(  # pragma: no covers
        f"Invalid operator {operator} encountered in unary expression " f"on line {line}."
    )  # pragma: no cover


def _eval_assignment(lhs: any, operator: str, value: any, line: int):
    """
    Evaluates an assignment.

    Args:
        lhs (any): The variable to update.
        operator (str): The assignment operator.
        value (any): The value to assign.
        line (int): The line number.

    Returns:
        any: The updated left hand side.
    """
    match operator:
        case "=":
            lhs = value
        case "+=":
            lhs += value
        case "-=":
            lhs -= value
        case "*=":
            lhs = lhs * value
        case "/=":
            lhs = lhs / value
        case "&=":
            lhs = lhs & value
        case "|=":
            lhs = lhs | value
        case "^=":
            lhs = lhs ^ value
        case "<<=":
            lhs = lhs << value
        case ">>=":
            lhs = lhs >> value
        case "%=":
            lhs = lhs % value
        case "**=":
            lhs = lhs**value
        case _:  # pragma: no cover
            # we shouldn't ever get this error if the parser did its job right
            raise SyntaxError(  # pragma: no cover
                f"Invalid operator {operator} encountered in assignment expression "
                f"on line {line}."
            )  # pragma: no cover
    return lhs


def _eval_binary_op(lhs: any, operator: str, rhs: any, line: int):
    """
    Evaluates a binary operator.

    Args:
         lhs (any): the first operand, usually a variable or a MeasurementValue.
         operator (str): the operation.
         rhs (any): the second operand.
         line (int): the line number the operation occurs on.
    """
    match operator:
        case "==":
            return lhs == rhs
        case "!=":
            return lhs != rhs
        case ">":
            return lhs > rhs
        case "<":
            return lhs < rhs
        case ">=":
            return lhs >= rhs
        case "<=":
            return lhs <= rhs
        case ">>":
            return lhs >> rhs
        case "<<":
            return lhs << rhs
        case "+":
            return lhs + rhs
        case "-":
            return lhs - rhs
        case "*":
            return lhs * rhs
        case "**":
            return lhs**rhs
        case "/":
            return lhs / rhs
        case "%":
            return lhs % rhs
        case "|":
            return lhs | rhs
        case "||":
            return lhs or rhs
        case "&":
            return lhs & rhs
        case "&&":
            return lhs and rhs
        case "^":
            return lhs ^ rhs
        case _:  # pragma: no cover
            # we shouldn't ever get this error if the parser did its job right
            raise SyntaxError(  # pragma: no cover
                f"Invalid operator {operator} encountered in binary expression " f"on line {line}."
            )  # pragma: no cover


@dataclass
class Variable:
    """
    A data class that represents a variables.

    Args:
        ty (type): The type of the variable.
        val (any): The value of the variable.
        size (int): The size of the variable if it has a size, like an array.
        line (int): The line number at which the variable was most recently updated.
        constant (bool): Whether the variable is a constant.
        scope (str): The name of the scope of the variable.
    """

    ty: str
    val: Any
    size: int
    line: int
    constant: bool
    scope: str = "global"


class Context:
    """Class with helper methods for managing, updating, checking context."""

    def __init__(self, context: dict):
        """
        Initializes the context.

        Args:
            context (dict): A dictionary that contains some information about the context.
        """
        if "vars" not in context:
            context["vars"] = {}
        if "aliases" not in context:
            context["aliases"] = {}
        if "wires" not in context:
            context["wires"] = []
        if "scopes" not in context:
            context["scopes"] = {"subroutines": {}, "custom_gates": {}}
        if "wire_map" not in context or context["wire_map"] is None:
            context["wire_map"] = {}
        if "return" not in context:
            context["return"] = {}
        self.context = context

    def init_loops_scope(self, node: ForInLoop | WhileLoop):
        """
        Inits the loops scope on the current context.

        Args:
            node (ForInLoop | WhileLoop): the loop node.
        """
        if "loops" not in self.scopes:
            self.scopes["loops"] = dict()

        # the namespace is shared with the outer scope, but we need to keep track of the gates separately
        if isinstance(node, WhileLoop):
            self.scopes["loops"][f"while_{node.span.start_line}"] = (
                self.init_clause_in_same_namespace(self, f"while_{node.span.start_line}")
            )

        elif isinstance(node, ForInLoop):
            self.scopes["loops"][f"for_{node.span.start_line}"] = (
                self.init_clause_in_same_namespace(self, f"for_{node.span.start_line}")
            )

    def init_switches_scope(self, node: QASMNode):
        """
        Inits the switches scope on the current context.

        Args:
            node (QASMNode): the switch node.
        """
        if "switches" not in self.scopes:
            self.scopes["switches"] = dict()

        self.scopes["switches"][f"switch_{node.span.start_line}"] = dict()

    def init_branches_scope(self, node: QASMNode):
        """
        Inits the branches scope on the current context.

        Args:
            node (BranchingStatement): the branch node.
        """
        if "branches" not in self.scopes:
            self.scopes["branches"] = dict()

        self.scopes["branches"][f"branch_{node.span.start_line}"] = dict()

    def init_custom_gate_scope(self, node: QuantumGateDefinition):
        """
        Initializes a context for a custom quantum gate.

        Args:
            node (QuantumGateDefinition): the custom quantum gate definition.
        """
        self.scopes["custom_gates"][node.name.name] = self.init_clause_in_same_namespace(
            self, node.name.name
        )
        self.scopes["custom_gates"][node.name.name].update(
            {
                "vars": {k: v for k, v in self.vars.items() if v.constant},
                "body": node.body,
                "params": [_resolve_name(param) for param in node.arguments]
                + [_resolve_name(qubit) for qubit in node.qubits],
            }
        )

    def init_subroutine_scope(self, node: SubroutineDefinition):
        """
        Initializes a sub context with all the params, constants, subroutines and qubits it has access to.

        Args:
            node (SubroutineDefinition): the subroutine definition.
        """

        # outer scope variables are available to inner scopes... but not vice versa!
        # names prefixed with outer scope names for specificity
        self.scopes["subroutines"][node.name.name] = self.init_clause_in_same_namespace(
            self, node.name.name
        )
        self.scopes["subroutines"][node.name.name].update(
            {
                "vars": {k: v for k, v in self.vars.items() if v.constant},
                "body": node.body,
                "params": [param.name.name for param in node.arguments],
            }
        )

    @staticmethod
    def init_clause_in_same_namespace(outer_context, name: str):
        """
        Initializes a clause that shares the namespace of the outer scope, but contains its own
        set of gates, operations, expressions, logic, etc.
        Args:
            outer_context (Context): the context of the outer scope.
            name (str): the name of the clause.
        Returns:
            dict: the inner context.
        """
        # we want wires declared in outer scopes to be available
        context = {
            "vars": outer_context.vars,  # same namespace
            "wire_map": {},
            "wires": outer_context.wires,
            "name": name,
            # we want subroutines declared in the global scope to be available
            "scopes": {
                "subroutines": outer_context.scopes["subroutines"],
                "custom_gates": outer_context.scopes["custom_gates"],
            },
        }

        return Context(context)

    def retrieve_variable(self, name: str):
        """
        Attempts to retrieve a variable from the current context by name.

        Args:
            name (str): the name of the variable to retrieve.

        Returns:
            The value of the variable in the current context.

        Raises:
             NameError: if the variable is not initialized.
             TypeError: if the variable is not declared.
        """

        if name in self.vars:
            res = self.vars[name]
            if res.val is not None:
                return res
            raise ValueError(f"Attempt to reference uninitialized parameter {name}!")
        if name in self.wires:
            return name
        if name in self.aliases:
            return self.aliases[name](self)  # evaluate the alias and de-reference
        raise TypeError(f"Attempt to use undeclared variable {name} in {self.name}")

    def process_measurement(self, operator: str, line: int, lhs: Variable, rhs: Variable = None):
        """
        Updates a MeasurementValue's processing function to reflect classical logic applied to
        the MeasurementValue.

        Args:
            operator (str): The operator to apply to the MeasurementValue.
            line (int): The line number at which the operator occurs.
            lhs (Variable): The first operand of the operation involving the MeasurementValue(s).
            rhs (Optional[Variable]): The second operand of the operation involving the MeasurementValue(s).
        """
        lhs_call = copy.deepcopy(lhs.val.processing_fn)
        rhs_call = copy.deepcopy(rhs.val.processing_fn)

        def new_processing_fn(*args):
            left = lhs_call(*args) if isinstance(lhs.val, MeasurementValue) else lhs.val
            if rhs is not None:
                right = rhs_call(*args) if isinstance(rhs.val, MeasurementValue) else rhs.val
                return _eval_binary_op(left, operator, right, line)
            else:
                return _eval_unary_op(left, operator, line)

        if isinstance(lhs.val, MeasurementValue) and isinstance(rhs.val, MeasurementValue):
            new_measurements = lhs.val.measurements + rhs.val.measurements
        elif isinstance(lhs.val, MeasurementValue):
            new_measurements = lhs.val.measurements
        else:  # isinstance(rhs.val, MeasurementValue) == True
            new_measurements = rhs.val.measurements

        return MeasurementValue(new_measurements, new_processing_fn)

    def update_measurement(self, value: any, prev: Variable, operator: str, line: int):
        """
        Updates a MeasurementValue's processing function to reflect classical logic applied to
        the MeasurementValue.

        Args:
            value (any): The second operand of the operation involving the MeasurementValue.
            prev (Variable): The Variables whose value is a MeasurementValue.
            operator (str): The operator to apply to the MeasurementValue.
            line (int): The line number at which the operator occurs.
        """
        prev_call = copy.deepcopy(prev.val.processing_fn)

        def new_processing_fn(*args):
            return _eval_assignment(prev_call(*args), operator, value, line).processing_fn()

        prev.val = MeasurementValue(prev.val.measurements, new_processing_fn)

    def update_var(
        self, value: any, name: str, operator: str, line: int
    ):  # pylint: disable=too-many-branches
        """
        Updates a variable, or raises if it is constant.
        Args:
            value (any): the value to set.
            name (str): the name of the variable.
            operator (str): the assignment operator.
            line (int): the line number at which we encountered the assignment node.
        """
        if name not in self.vars:
            raise TypeError(f"Attempt to use undeclared variable {name} in {self.name}")
        if self.vars[name].constant:
            raise ValueError(
                f"Attempt to mutate a constant {name} on line {line} that was "
                f"defined on line {self.vars[name].line}"
            )
        self.vars[name].val = _eval_assignment(self.vars[name].val, operator, value, line)
        self.vars[name].line = line

    def require_wires(self, wires: list):
        """
        Simple helper that checks if we have wires in the current context.

        Args:
            wires (list): The wires that are required.

        Raises:
            NameError: If the context is missing a wire.
        """
        missing_wires = set(wires) - set(self.wires)
        if len(missing_wires) > 0:
            raise NameError(
                f"Attempt to reference wire(s): {missing_wires} that have not been declared in {self.name}"
            )

    def __getattr__(self, name: str):
        """
        If the attribute is not found on the class, instead uses the attr name as an index
        into the context dictionary, for easy access.

        Args:
            name (str): the name of the attribute.

        Returns:
            Any: the value of the attribute.

        Raises:
            KeyError: if the attribute is not found on the context.
        """
        if name in self.context:
            return self.context[name]
        if hasattr(self.context, name):
            return getattr(self.context, name)
        raise KeyError(
            f"No attribute {name} on Context and no {name} key found on context {self.name}"
        )

    def __getitem__(self, item):
        """
        Allows accessing items on the context by subscripting.

        Args:
            item: the name of the key to retrieve.

        Returns:
            Any: the value corresponding to the key.
        """
        return self.context[item]


def _get_bit_type_val(var):
    return bin(var.val)[2:].zfill(var.size)


def _resolve_name(node: QASMNode):
    """
    Fully resolves the name of a node which may be provided as an Identifier or string,
    and therefore may require referencing different attributes.

    Args:
        node (QASMNode): the QASMNode whose name is being resolved.

    Returns:
        str: the resolved name.
    """
    # parser will sometimes represent a name as a str and sometimes as an Identifier
    return node.name if isinstance(node.name, str) else node.name.name


def preprocess_operands(operand):
    """
    Interprets a string operand as an appropriate type.

    Args:
        operand (str): the string operand to interpret.

    Returns:
        The interpreted operand as an appropriate type.
    """
    if isinstance(operand, str):
        if operand.isdigit():
            operand = int(operand)
        elif operand.replace(".", "").isnumeric():
            operand = float(operand)
    return operand


class BreakException(Exception):  # pragma: no cover
    """Exception raised when encountering a break statement."""


class ContinueException(Exception):  # pragma: no cover
    """Exception raised when encountering a continue statement."""


class EndProgram(Exception):
    """Exception raised when it encounters an end statement in the QASM circuit."""


# pylint: disable=unused-argument, no-self-use, too-many-public-methods
class QasmInterpreter:
    """
    Takes the top level node of the AST as a parameter and recursively descends the AST, calling the
    visitor function on each node.
    """

    def __init__(self):
        """
        Initializes the QASM interpreter.
        """
        self.inputs = {}
        self.outputs = []

    @functools.singledispatchmethod
    def visit(self, node: QASMNode, context: Context, aliasing: bool = False):
        """
        Visitor function is called on each node in the AST, which is traversed using recursive descent.
        The purpose of this function is to pass each node to the appropriate handler.

        Args:
            node (QASMNode): the QASMNode to visit next.
            context (Context): the current context populated with any locally available variables, etc.
            aliasing (bool): whether we are aliasing a variable in the context.

        Raises:
            NotImplementedError: when an unsupported QASMNode type is found.
        """
        raise NotImplementedError(
            f"An unsupported QASM instruction {node.__class__.__name__} "
            f"was encountered on line {node.span.start_line}, in {context.name}."
        )

    @visit.register(list)
    def visit_list(self, node_list: list, context: Context):
        """
        Visits a list of QASMNodes.

        Args:
            node_list (list): the list of QASMNodes to visit.
            context (Context): the current context.
        """
        for sub_node in node_list:
            self.visit(sub_node, context)

    def interpret(self, node: QASMNode, context: dict, **inputs):
        """
        Entry point for visiting the QASMNodes of a parsed OpenQASM 3.0 program.

        Args:
            node (QASMNode): The top-most QASMNode.
            context (dict): The initial context populated with the name of the program (the outermost scope).
            inputs (dict): Additional inputs to the OpenQASM 3.0 program.

        Returns:
            dict: The context updated after the compilation of all nodes by the visitor.
        """
        context = Context(context)
        self.inputs = inputs

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

        for output in self.outputs:
            context["return"][output] = context.retrieve_variable(output)

        return context

    @visit.register(BreakStatement)
    def visit_break_statement(self, node: QASMNode, context: Context):
        """
        Registers a break statement.

        Args:
            node (QASMNode): the break QASMNode.
            context (Context): the current context.
        """

        raise BreakException(f"Break statement encountered in {context.name}")

    @visit.register(ContinueStatement)
    def visit_continue_statement(self, node: QASMNode, context: Context):
        """
        Registers a continue statement.

        Args:
            node (QASMNode): the continue QASMNode.
            context (Context): the current context.
        """

        raise ContinueException(f"Continue statement encountered in {context.name}")

    @visit.register(BranchingStatement)
    def visit_branching_statement(self, node: QASMNode, context: Context):
        """
        Registers a branching statement. Like switches, uses qml.cond.

        Args:
            node (QASMNode): the branch QASMNode.
            context (Context): the current context.
        """
        context.init_branches_scope(node)

        # create the true body context
        context.scopes["branches"][f"branch_{node.span.start_line}"].update(
            {
                "true_body": context.init_clause_in_same_namespace(
                    context, f"{context.name}_branch_{node.span.start_line}_true_body"
                )
            }
        )

        if hasattr(node, "else_block"):

            # create the false body context
            context.scopes["branches"][f"branch_{node.span.start_line}"].update(
                {
                    "false_body": context.init_clause_in_same_namespace(
                        context, f"{context.name}_branch_{node.span.start_line}_false_body"
                    )
                }
            )

        ops.cond(
            self.visit(node.condition, context),
            partial(
                self.visit,
                node.if_block,
                context.scopes["branches"][f"branch_{node.span.start_line}"]["true_body"],
            ),
            (
                partial(
                    self.visit,
                    node.else_block,
                    context.scopes["branches"][f"branch_{node.span.start_line}"]["false_body"],
                )
                if hasattr(node, "else_block")
                else None
            ),
        )()

    @visit.register(SwitchStatement)
    def visit_switch_statement(self, node: QASMNode, context: Context):
        """
        Registers a switch statement.

        Args:
            node (QASMNode): the switch QASMNode.
            context (Context): the current context.
        """
        context.init_switches_scope(node)

        # switches need to have access to the outer context but not get called unless the condition is met

        i = 0
        # we need to keep track of each clause individually
        for i in range(len(node.cases)):
            context.scopes["switches"][f"switch_{node.span.start_line}"].update(
                {
                    f"cond_{i}": context.init_clause_in_same_namespace(
                        context, f"{context.name}_switch_{node.span.start_line}_cond_{i}"
                    )
                }
            )

        if hasattr(node, "default") and node.default is not None:
            context.scopes["switches"][f"switch_{node.span.start_line}"].update(
                {
                    f"cond_{i + 1}": context.init_clause_in_same_namespace(
                        context, f"{context.name}_switch_{node.span.start_line}_cond_{i + 1}"
                    )
                }
            )

        target = self.visit(node.target, context)
        ops.cond(
            target == self.visit(node.cases[0][0][0], context),
            partial(
                self.visit,
                node.cases[0][1].statements,
                context.scopes["switches"][f"switch_{node.span.start_line}"]["cond_0"],
            ),
            (
                partial(
                    self.visit,
                    node.default.statements,
                    context.scopes["switches"][f"switch_{node.span.start_line}"][f"cond_{i + 1}"],
                )
                if hasattr(node, "default") and node.default is not None
                else None
            ),
            [
                (
                    target == self.visit(node.cases[j + 1][0][0], context),
                    partial(
                        self.visit,
                        node.cases[j + 1][1].statements,
                        context.scopes["switches"][f"switch_{node.span.start_line}"][case],
                    ),
                )
                for j, case in enumerate(
                    list(context.scopes["switches"][f"switch_{node.span.start_line}"].keys())[1:-1]
                )
            ],
        )()

    @staticmethod
    def _handle_break(loop: Callable, execution_context: Context):
        """
        Handles when a break is encountered in the loop.

        Args:
            loop (Callable): the loop function.
            execution_context (Context): the context passed at execution time with current variable values, etc.
        """
        try:
            loop(execution_context)
        except BreakException:
            pass  # evaluation of the loop stops

    @visit.register(WhileLoop)
    def visit_while_loop(self, node: WhileLoop, context: Context):
        """
        Registers a while loop.

        Args:
            node (QASMNode): the loop node.
            context (Context): the current context.
        """
        context.init_loops_scope(node)

        @while_loop(partial(self.visit, node.while_condition))  # traces data dep through context
        def loop(context):
            """
            Executes a traceable while loop.

            Args:
                loop_context (Context): the context used to compile the while loop.
                execution_context (Context): the context passed at execution time with current variable values, etc.
            """
            try:
                # updates vars in context... need to propagate these to outer scope
                self.visit(node.block, context.scopes["loops"][f"while_{node.span.start_line}"])
            except ContinueException:
                pass  # evaluation of this iteration ends, and we continue to the next

            inner_context = context.scopes["loops"][f"while_{node.span.start_line}"]
            context.vars = inner_context.vars
            context.wires = inner_context.wires

            return context

        self._handle_break(loop, context)

    @visit.register(ForInLoop)
    def visit_for_in_loop(self, node: ForInLoop, context: Context):
        """
        Registers a for loop.

        Args:
            node (QASMNode): the loop node.
            context (Context): the current context.
        """
        context.init_loops_scope(node)

        # We need custom logic here for handling Identifiers in case they are of BitType.
        # If we introduce logic into retrieve_variable that returns BitType values as strings
        # this messes with unary and binary expressions' ability to handle BitType vars.
        # If we try to get a bit string directly from the integer representation natural to the parser here,
        # we can only guess at the size of the register since there might be leading zeroes.
        if isinstance(node.set_declaration, Identifier):
            loop_params = context.retrieve_variable(node.set_declaration.name)
            if isinstance(loop_params, Variable):
                if loop_params.ty == "BitType":
                    loop_params = _get_bit_type_val(loop_params)
                else:
                    loop_params = loop_params.val
        else:
            loop_params = self.visit(node.set_declaration, context)

        # TODO: support dynamic start, stop, step?
        if isinstance(loop_params, slice):
            start = loop_params.start
            stop = loop_params.stop
            step = loop_params.step
            if step is None:
                step = 1

            @for_loop(start, stop, step)
            def loop(i, execution_context):
                execution_context.scopes["loops"][f"for_{node.span.start_line}"].vars[
                    node.identifier.name
                ] = Variable(
                    ty=i.__class__.__name__,
                    val=i,
                    size=-1,
                    line=node.span.start_line,
                    constant=False,
                )
                try:
                    # we only want to execute the gates in the loop's scope
                    # updates vars in sub context... need to propagate these to outer context
                    self.visit(
                        node.block, context["scopes"]["loops"][f"for_{node.span.start_line}"]
                    )
                except ContinueException:
                    pass  # evaluation of the current iteration stops and we continue
                inner_context = execution_context.scopes["loops"][f"for_{node.span.start_line}"]
                context.vars = inner_context.vars
                context.wires = inner_context.wires

                return execution_context

            self._handle_break(loop, context)

        # we unroll the loop in the following case when we don't have a range since qml.for_loop() only
        # accepts (start, stop, step) and not a list of values.
        if isinstance(loop_params, Iterable):

            def unrolled(execution_context):
                for i in loop_params:
                    execution_context.scopes["loops"][f"for_{node.span.start_line}"].vars[
                        node.identifier.name
                    ] = Variable(
                        ty=i.__class__.__name__,
                        val=i,
                        size=-1,
                        line=node.span.start_line,
                        constant=False,
                    )
                    try:
                        # visit the nodes once per loop iteration
                        self.visit(
                            node.block, context.scopes["loops"][f"for_{node.span.start_line}"]
                        )  # updates vars in sub context if any measurements etc. occur
                    except ContinueException:
                        pass  # eval of current iteration stops and we continue

            self._handle_break(unrolled, context)

    @visit.register(QuantumMeasurementStatement)
    def visit_quantum_measurement_statement(self, node: QASMNode, context: Context):
        """
        Registers a quantum measurement.

        Args:
            node (QASMNode): the quantum measurement node.
            context (Context): the current context.
        """
        meas = partial(measure, self.visit(node.measure.qubit, context))
        name = _resolve_name(node.target)  # str or Identifier
        res = meas()
        context.vars[name].val = res
        context.vars[name].line = node.span.start_line
        return res

    @visit.register(QuantumReset)
    def visit_quantum_reset(self, node: QASMNode, context: dict):
        """
        Registers a reset of a quantum gate.

        Args:
            node (QASMNode): the quantum reset node.
            context (dict): the current context.
        """
        measure(self.visit(node.qubits, context), reset=True)

    def execute_custom_gate(self, node: QuantumGate, context: Context):
        """
        Executes a custom gate.

        Args:
            node (QuantumGate): the custom gate call.
            context (Context): the current context.
        """

        gate_context = context.scopes["custom_gates"][_resolve_name(node)]

        # bind subroutine arguments
        evald_args = [self.visit(raw_arg, context) for raw_arg in node.arguments] + [
            _resolve_name(qubit) for qubit in node.qubits
        ]
        for evald_arg, param in list(zip(evald_args, gate_context.params)):
            if not isinstance(evald_arg, str):  # this would indicate a quantum parameter
                gate_context.vars[param] = Variable(
                    evald_arg.__class__.__name__, evald_arg, None, node.span.start_line, False
                )
            else:
                if evald_arg in context.wire_map:
                    evald_arg = context.wire_map[evald_arg]
                if param != evald_arg:
                    gate_context.wire_map[param] = evald_arg

        # execute the subroutine
        self.visit(gate_context.body, gate_context)

        # reset context
        gate_context.vars = {k: v for k, v in gate_context.vars.items() if v.constant}

        # custom gates do not return a value

    @visit.register(FunctionCall)
    def visit_function_call(self, node: FunctionCall, context: Context):
        """
        Registers a function call. The node must refer to a subroutine that has been defined and
        is available in the current scope.

        Args:
            node (FunctionCall): The FunctionCall QASMNode.
            context (Context): The current context.

        Raises:
            NameError: When the subroutine is not defined.
        """
        name = _resolve_name(node)  # str or Identifier
        if name not in context.scopes["subroutines"]:
            raise NameError(
                f"Reference to subroutine {name} not available in calling namespace "
                f"on line {node.span.start_line}."
            )
        func_context = context.scopes["subroutines"][name]

        # reset return
        func_context.context["return"] = None

        # bind subroutine arguments
        evald_args = [self.visit(raw_arg, context) for raw_arg in node.arguments]
        for evald_arg, param in list(zip(evald_args, func_context.params)):
            if isinstance(evald_arg, str):  # this would indicate a quantum parameter
                if evald_arg in context.wire_map:
                    evald_arg = context.wire_map[evald_arg]
                if evald_arg != param:
                    func_context.wire_map[param] = evald_arg
            else:
                func_context.vars[param] = Variable(
                    evald_arg.__class__.__name__,
                    evald_arg,
                    None,
                    node.span.start_line,
                    False,
                    func_context.name,
                )

        # execute the subroutine
        self.visit(func_context.body, func_context)

        # reset context
        func_context.vars = {
            k: v for k, v in func_context.vars.items() if (v.scope == context.name) and v.constant
        }

        # the return value
        return getattr(func_context, "return")

    @visit.register(RangeDefinition)
    def visit_range(self, node: RangeDefinition, context: Context):
        """
        Processes a range definition.

        Args:
            node (RangeDefinition): The range to process.
            context (Context): the current context.

        Returns:
            slice: The slice that corresponds to the range.
        """
        start = self.visit(node.start, context) if node.start else None
        stop = self.visit(node.end, context) if node.end else None
        step = self.visit(node.step, context) if node.step else None
        return slice(start, stop, step)

    def _index_into_var(self, var: Iterable | Variable, node: IndexExpression, context: Context):
        """
        Index into a variable using an IndexExpression.

        Args:
            var (Variable): The data structure representing the variable to index.
            node (IndexExpression): The IndexExpression.
            context (Context): the current context.

        Returns:
            The indexed slice of the variable.
        """
        if not isinstance(var, Iterable):
            var = _get_bit_type_val(var) if var.ty == "BitType" else var.val
        index = self.visit(node.index[0], context)
        if not (isinstance(index, Iterable) and len(index) > 1):
            return var[index]
        raise NotImplementedError(
            f"Array index does not evaluate to a single RangeDefinition or Literal at line {node.span.start_line}."
        )

    @visit.register(IODeclaration)
    def visit_io_declaration(self, node: IODeclaration, context: Context):
        """
        When we encounter an input / output declaration we want to return a callable that will execute the
        OpenQASM 3.0 circuit at the end.

        Args:
            node (IODeclaration): The IODeclaration QASMNode.
            context (Context): the current context.
        """
        if node.io_identifier == IOKeyword.input:
            name = _resolve_name(node.identifier)
            if name in self.inputs:
                context.vars[name] = Variable(
                    node.type.__class__.__name__, self.inputs[name], -1, node.span.start_line, True
                )
            else:
                raise ValueError(
                    f"Missing input {name}. Please pass {name} as a keyword argument to from_qasm3."
                )
        elif node.io_identifier == IOKeyword.output:
            name = _resolve_name(node.identifier)
            context.vars[name] = Variable(
                node.type.__class__.__name__,
                None,
                -1,
                node.span.start_line,
                False,
            )
            self.outputs.append(_resolve_name(node.identifier))

    @visit.register(EndStatement)
    def visit_end_statement(self, node: QASMNode, context: Context):
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
    def visit_qubit_declaration(self, node: QubitDeclaration, context: Context):
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
        name = _resolve_name(node.lvalue)
        res = self.visit(node.rvalue, context)
        try:
            prev = context.retrieve_variable(name)
        except ValueError:
            # the var is not initialized
            context.vars[name] = Variable(
                res.__class__.__name__,
                res,
                len(_get_bit_type_val(res)) if isinstance(res, BitType) else -1,
                node.span.start_line,
                False,
            )
            prev = context.vars[name]
        if isinstance(prev.val, MeasurementValue):
            context.update_measurement(res, prev, node.op.name, node.span.start_line)
        else:
            context.update_var(res, name, node.op.name, node.span.start_line)

    @visit.register(AliasStatement)
    def visit_alias_statement(self, node: QASMNode, context: Context):
        """
        Registers an alias statement.
        Args:
            node (QASMNode): the alias QASMNode.
            context (Context): the current context.
        """
        context.aliases[node.target.name] = self.visit(node.value, context, aliasing=True)

    @visit.register(ReturnStatement)
    def visit_return_statement(self, node: QASMNode, context: Context):
        """
        Registers a return statement. Points to the var that needs to be set in an outer scope when this
        subroutine is called.
        """
        context.context["return"] = self.visit(node.expression, context)

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
                if getattr(node, "init_expression", None)
                else None
            ),
            (
                node.init_expression.width
                if hasattr(node, "init_expression") and hasattr(node.init_expression, "width")
                else None
            ),
            (
                node.init_expression.span.start_line
                if getattr(node, "init_expression", None)
                else node.span.start_line
            ),
            constant,
            context.name,
        )

    @visit.register(ImaginaryLiteral)
    def visit_imaginary_literal(self, node: ImaginaryLiteral, context: Context):
        """
        Registers an imaginary literal.

        Args:
            node (ImaginaryLiteral): The imaginary literal QASMNode.
            context (Context): the current context.

        Returns:
            complex: a complex number corresponding to the imaginary literal.
        """
        return 1j * node.value

    @visit.register(DiscreteSet)
    def visit_discrete_set(self, node: DiscreteSet, context: Context):
        """
        Evaluates a discrete set literal.

        Args:
            node (DiscreteSet): The set literal QASMNode.
            context (Context): The current context.

        Returns:
            list: The evaluated set.
        """
        return (self.visit(literal, context) for literal in node.values)

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

    @visit.register(QuantumGateDefinition)
    def visit_quantum_gate_definition(self, node: QuantumGateDefinition, context: Context):
        """
        Registers a quantum gate definition.

        Args:
            node (QuantumGateDefinition): The quantum gate definition QASMNode.
            context (Context): the current context.
        """
        context.init_custom_gate_scope(node)

        # register the params
        for param in node.arguments:
            context.scopes["custom_gates"][_resolve_name(node)].vars[_resolve_name(param)] = (
                Variable(
                    ty=param.__class__.__name__,
                    val=None,
                    size=-1,
                    line=param.span.start_line,
                    constant=False,
                )
            )
        for qubit in node.qubits:
            context.scopes["custom_gates"][_resolve_name(node)].wires.append(_resolve_name(qubit))

    @visit.register(SubroutineDefinition)
    def visit_subroutine_definition(self, node: QASMNode, context: Context):
        """
        Registers a subroutine definition. Maintains a namespace in the context, starts populating it with
        its parameters.
        Args:
            node (QASMNode): the subroutine node.
            context (Context): the current context.
        """
        context.init_subroutine_scope(node)

        # register the params
        for param in node.arguments:
            if isinstance(param, QuantumArgument):
                context.scopes["subroutines"][_resolve_name(node)].wires.append(
                    _resolve_name(param)
                )
            else:
                context.scopes["subroutines"][_resolve_name(node)].vars[_resolve_name(param)] = (
                    Variable(
                        ty=param.__class__.__name__,
                        val=None,
                        size=-1,
                        line=param.span.start_line,
                        constant=False,
                        scope=_resolve_name(node),
                    )
                )

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
        elif name in list(map(lambda n: n.upper(), context.scopes["custom_gates"].keys())):
            self.execute_custom_gate(node, context)
            return
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
        wires = [_resolve_name(node.qubits[q]) for q in range(len(node.qubits))]

        context.require_wires(wires)

        resolved_wires = list(
            map(lambda wire: context.wire_map[wire] if wire in context.wire_map else wire, wires)
        )

        return gate, args, resolved_wires

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

        Returns:
            Any: The argument cast to the appropriate type.

        Raises:
            TypeError: If the cast cannot be made.
        """
        arg = self.visit(node.argument, context)
        try:
            if isinstance(node.type, IntType):
                ret = int(arg)
            if isinstance(node.type, UintType):
                ret = uint(arg)
            if isinstance(node.type, FloatType):
                ret = float(arg)
            if isinstance(node.type, ComplexType):
                ret = complex(arg)
            if isinstance(node.type, BoolType):
                ret = bool(arg)
            # TODO: durations, angles, etc.
        except TypeError as e:
            raise TypeError(
                f"Unable to cast {arg.__class__.__name__} to {node.type.__class__.__name__}: {str(e)}"
            ) from e
        return ret

    @visit.register(BinaryExpression)
    def visit_binary_expression(
        self, node: BinaryExpression, context: Context
    ):  # pylint: disable=too-many-branches, too-many-return-statements
        """
        Registers a binary expression.

        Args:
            node (BinaryExpression): The binary expression.
            context (Context): The current context.

        Returns:
            The result of the evaluated expression.
        """
        lhs = preprocess_operands(self.visit(node.lhs, context))
        rhs = preprocess_operands(self.visit(node.rhs, context))
        if (isinstance(lhs, Variable) and isinstance(lhs.val, MeasurementValue)) or (
            isinstance(rhs, Variable) and isinstance(rhs.val, MeasurementValue)
        ):
            return context.process_measurement(node.op.name, node.span.start_line, lhs, rhs)
        else:
            return _eval_binary_op(lhs, node.op.name, rhs, node.span.start_line)

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
        operand = preprocess_operands(self.visit(node.expression, context))
        if isinstance(operand, Variable) and isinstance(operand.val, MeasurementValue):
            return context.process_measurement(node.op.name, node.span.start_line, operand, None)
        return _eval_unary_op(operand, node.op.name, node.span.start_line)

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
            return lambda cntxt: self._index_into_var(self._alias(node, cntxt), node, context)

        # else we are just evaluating an index
        var = context.retrieve_variable(node.collection.name)
        return self._index_into_var(var, node, context)

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
            return (
                context.retrieve_variable(node.collection.name)
                if getattr(node, "collection", None)
                else context.retrieve_variable(node.name)
            )
        except TypeError as e:
            raise TypeError(
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
            var = context.retrieve_variable(node.name)
            if isinstance(var, Variable):
                value = var.val
                var.line = node.span.start_line
            else:
                value = var
            return value
        except TypeError as e:
            raise TypeError(
                str(e) or f"Reference to an undeclared variable {node.name} in {context.name}."
            ) from e

    @visit.register(IntegerLiteral)
    @visit.register(FloatLiteral)
    @visit.register(BooleanLiteral)
    @visit.register(BitstringLiteral)
    @visit.register(DurationLiteral)
    def visit_literal(self, node: Expression, context: Context):
        """
        Visits a literal.

        Args:
            node (Literal): The literal.
            context (Context): The current context.

        Returns:
            The value of the literal.
        """
        return node.value
