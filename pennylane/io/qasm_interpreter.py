"""
This submodule contains the interpreter for OpenQASM 3.0.
"""

import functools
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Iterable

from numpy import uint
from openqasm3 import ast
from openqasm3.visitor import QASMNode

from pennylane import ops
from pennylane.control_flow import for_loop, while_loop
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
            context["scopes"] = {"subroutines": {}}
        if "wire_map" not in context or context["wire_map"] is None:
            context["wire_map"] = {}
        if "return" not in context:
            context["return"] = None
        self.context = context

    def init_subroutine_scope(self, node: ast.SubroutineDefinition):
        """
        Initializes a sub context with all the params, constants, subroutines and qubits it has access to.

        Args:
            node (SubroutineDefinition): the subroutine definition.
        """

        # outer scope variables are available to inner scopes... but not vice versa!
        self.scopes["subroutines"][node.name.name] = Context(
            {
                "vars": {k: v for k, v in self.vars.items() if v.constant},  # same namespace
                "wire_map": {},
                "wires": self.wires,
                "name": node.name.name,
                # we want subroutines declared in the global scope to be available
                "scopes": {"subroutines": self.scopes["subroutines"]},
                "body": node.body,
                "params": [param.name.name for param in node.arguments],
            }
        )

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
        match operator:
            case "=":
                self.vars[name].val = value
            case "+=":
                self.vars[name].val += value
            case "-=":
                self.vars[name].val -= value
            case "*=":
                self.vars[name].val = self.vars[name].val * value
            case "/=":
                self.vars[name].val = self.vars[name].val / value
            case "&=":
                self.vars[name].val = self.vars[name].val & value
            case "|=":
                self.vars[name].val = self.vars[name].val | value
            case "^=":
                self.vars[name].val = self.vars[name].val ^ value
            case "<<=":
                self.vars[name].val = self.vars[name].val << value
            case ">>=":
                self.vars[name].val = self.vars[name].val >> value
            case "%=":
                self.vars[name].val = self.vars[name].val % value
            case "**=":
                self.vars[name].val = self.vars[name].val ** value
            case _:  # pragma: no cover
                # we shouldn't ever get this error if the parser did its job right
                raise SyntaxError(  # pragma: no cover
                    f"Invalid operator {operator} encountered in assignment expression "
                    f"on line {line}."
                )  # pragma: no cover
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
    # parser will sometimes represent a name as a str and sometimes as an ast.Identifier
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

    def interpret(self, node: QASMNode, context: dict):
        """
        Entry point for visiting the QASMNodes of a parsed OpenQASM 3.0 program.

        Args:
            node (QASMNode): The top-most QASMNode.
            context (Context): The initial context populated with the name of the program (the outermost scope).

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

    @visit.register(ast.BreakStatement)
    def visit_break_statement(self, node: ast.BreakStatement, context: Context):
        """
        Registers a break statement.

        Args:
            node (BreakStatement): the break QASMNode.
            context (Context): the current context.
        """

        raise BreakException(f"Break statement encountered in {context.name}")

    @visit.register(ast.ContinueStatement)
    def visit_continue_statement(self, node: ast.ContinueStatement, context: Context):
        """
        Registers a continue statement.

        Args:
            node (ContinueStatement): the continue QASMNode.
            context (Context): the current context.
        """

        raise ContinueException(f"Continue statement encountered in {context.name}")

    @visit.register(ast.BranchingStatement)
    def visit_branching_statement(self, node: ast.BranchingStatement, context: Context):
        """
        Registers a branching statement. Like switches, uses qml.cond.

        Args:
            node (BranchingStatement): the branch QASMNode.
            context (Context): the current context.
        """
        ops.cond(
            self.visit(node.condition, context),
            partial(
                self.visit,
                node.if_block,
                context,
            ),
            (
                partial(
                    self.visit,
                    node.else_block,
                    context,
                )
                if hasattr(node, "else_block")
                else None
            ),
        )()

    @visit.register(ast.SwitchStatement)
    def visit_switch_statement(self, node: ast.SwitchStatement, context: Context):
        """
        Registers a switch statement.

        Args:
            node (SwitchStatement): the switch QASMNode.
            context (Context): the current context.
        """

        # switches need to have access to the outer context but not get called unless the condition is met

        target = self.visit(node.target, context)
        ops.cond(
            target == self.visit(node.cases[0][0][0], context),
            partial(
                self.visit,
                node.cases[0][1].statements,
                context,
            ),
            (
                partial(
                    self.visit,
                    node.default.statements,
                    context,
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
                        context,
                    ),
                )
                for j in range(len(node.cases) - 2)
            ],
        )()

    @staticmethod
    def execute_loop(loop: Callable, execution_context: Context):
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

    @visit.register(ast.WhileLoop)
    def visit_while_loop(self, node: ast.WhileLoop, context: Context):
        """
        Registers a while loop.

        Args:
            node (QASMNode): the loop node.
            context (Context): the current context.
        """

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
                self.visit(node.block, context)
            except ContinueException:
                pass  # evaluation of this iteration ends, and we continue to the next

            return context

        self.execute_loop(loop, context)

    @visit.register(ast.ForInLoop)
    def visit_for_in_loop(self, node: ast.ForInLoop, context: Context):
        """
        Registers a for loop.

        Args:
            node (QASMNode): the loop node.
            context (Context): the current context.
        """

        # We need custom logic here for handling ast.Identifiers in case they are of BitType.
        # If we introduce logic into retrieve_variable that returns BitType values as strings
        # this messes with unary and binary expressions' ability to handle BitType vars.
        # If we try to get a bit string directly from the integer representation natural to the parser here,
        # we can only guess at the size of the register since there might be leading zeroes.
        if isinstance(node.set_declaration, ast.Identifier):
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
            step = loop_params.step or 1
        elif isinstance(loop_params, Iterable):
            start = 0
            stop = len(loop_params)
            step = 1
        else:
            # we shouldn't be able to get here if the parser does its job
            raise TypeError(
                f"Expected iterable type or a range in loop, got {type(loop_params)}."
            )  # pragma: no cover

        @for_loop(start, stop, step)
        def loop(i, execution_context):
            if isinstance(loop_params, Iterable):
                execution_context.vars[node.identifier.name] = Variable(
                    ty=loop_params[i].__class__.__name__,
                    val=loop_params[i],
                    size=-1,
                    line=node.span.start_line,
                    constant=False,
                )
            else:
                execution_context.vars[node.identifier.name] = Variable(
                    ty=i.__class__.__name__,
                    val=i,
                    size=-1,
                    line=node.span.start_line,
                    constant=False,
                )
            try:
                # we only want to execute the gates in the loop's scope
                # updates vars in sub context... need to propagate these to outer context
                self.visit(node.block, execution_context)
            except ContinueException:
                pass  # evaluation of the current iteration stops and we continue

            return execution_context

        self.execute_loop(loop, context)

    @visit.register(ast.FunctionCall)
    def visit_function_call(self, node: ast.FunctionCall, context: Context):
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

    @visit.register(ast.RangeDefinition)
    def visit_range(self, node: ast.RangeDefinition, context: Context):
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

    def _index_into_var(
        self, var: Iterable | Variable, node: ast.IndexExpression, context: Context
    ):
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

    @visit.register(ast.EndStatement)
    def visit_end_statement(self, node: ast.EndStatement, context: Context):
        """
        Ends the program.
        Args:
            node (EndStatement): The end statement QASMNode.
            context (Context): the current context.
        """
        raise EndProgram(
            f"The QASM program was terminated om line {node.span.start_line}."
            f"There may be unprocessed QASM code."
        )

    # needs to have same signature as visit()
    @visit.register(ast.QubitDeclaration)
    def visit_qubit_declaration(self, node: ast.QubitDeclaration, context: Context):
        """
        Registers a qubit declaration. Named qubits are mapped to numbered wires by their indices
        in context.wires. Note: Qubit declarations must be global.

        Args:
            node (QASMNode): The QubitDeclaration QASMNode.
            context (Context): The current context.
        """
        context.wires.append(node.qubit.name)

    @visit.register(ast.ClassicalAssignment)
    def visit_classical_assignment(self, node: ast.ClassicalAssignment, context: Context):
        """
        Registers a classical assignment.
        Args:
            node (ClassicalAssignment): the assignment QASMNode.
            context (Context): the current context.
        """
        # references to an unresolved value see a func for now
        name = _resolve_name(node.lvalue)
        res = self.visit(node.rvalue, context)
        context.update_var(res, name, node.op.name, node.span.start_line)

    @visit.register(ast.AliasStatement)
    def visit_alias_statement(self, node: ast.AliasStatement, context: Context):
        """
        Registers an alias statement.
        Args:
            node (AliasStatement): the alias QASMNode.
            context (Context): the current context.
        """
        context.aliases[node.target.name] = self.visit(node.value, context, aliasing=True)

    @visit.register(ast.ReturnStatement)
    def visit_return_statement(self, node: ast.ReturnStatement, context: Context):
        """
        Registers a return statement. Points to the var that needs to be set in an outer scope when this
        subroutine is called.

        Args:
            node (ReturnStatement): The return statement QASMNode.
            context (Context): the current context.
        """
        context.context["return"] = self.visit(node.expression, context)

    @visit.register(ast.ConstantDeclaration)
    def visit_constant_declaration(self, node: ast.ConstantDeclaration, context: Context):
        """
        Registers a constant declaration. Traces data flow through the context, transforming QASMNodes into
        Python type variables that can be readily used in expression eval, etc.

        Args:
            node (ConstantDeclaration): The constant QASMNode.
            context (Context): The current context.
        """
        self.visit_classical_declaration(node, context, constant=True)

    @visit.register(ast.ClassicalDeclaration)
    def visit_classical_declaration(
        self, node: ast.ClassicalDeclaration, context: Context, constant: bool = False
    ):
        """
        Registers a classical declaration. Traces data flow through the context, transforming QASMNodes into Python
        type variables that can be readily used in expression evaluation, for example.
        Args:
            node (ClassicalDeclaration): The ClassicalDeclaration QASMNode.
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

    @visit.register(ast.ImaginaryLiteral)
    def visit_imaginary_literal(self, node: ast.ImaginaryLiteral, context: Context):
        """
        Registers an imaginary literal.

        Args:
            node (ImaginaryLiteral): The imaginary literal QASMNode.
            context (Context): the current context.

        Returns:
            complex: a complex number corresponding to the imaginary literal.
        """
        return 1j * node.value

    @visit.register(ast.DiscreteSet)
    def visit_discrete_set(self, node: ast.DiscreteSet, context: Context):
        """
        Evaluates a discrete set literal.

        Args:
            node (DiscreteSet): The set literal QASMNode.
            context (Context): The current context.

        Returns:
            list: The evaluated set.
        """
        return [self.visit(literal, context) for literal in node.values]

    @visit.register(ast.ArrayLiteral)
    def visit_array_literal(self, node: ast.ArrayLiteral, context: Context):
        """
        Evaluates an array literal.

        Args:
            node (ArrayLiteral): The array literal QASMNode.
            context (Context): The current context.

        Returns:
            list: The evaluated array.
        """
        return [self.visit(literal, context) for literal in node.values]

    @visit.register(ast.SubroutineDefinition)
    def visit_subroutine_definition(self, node: ast.SubroutineDefinition, context: Context):
        """
        Registers a subroutine definition. Maintains a namespace in the context, starts populating it with
        its parameters.
        Args:
            node (SubroutineDefinition): the subroutine node.
            context (Context): the current context.
        """
        context.init_subroutine_scope(node)

        # register the params
        for param in node.arguments:
            if isinstance(param, ast.QuantumArgument):
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

    @visit.register(ast.QuantumGate)
    def visit_quantum_gate(self, node: ast.QuantumGate, context: Context):
        """
        Registers a quantum gate application. Calls the appropriate handler based on the sort of gate
        (parameterized or non-parameterized).

        Args:
            node (QuantumGate): The QuantumGate QASMNode.
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

    def _gate_setup_helper(self, node: ast.QuantumGate, gates_dict: dict, context: Context):
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

    def apply_modifier(
        self, mod: ast.QuantumGate, previous: Operator, context: Context, wires: list
    ):
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

    @visit.register(ast.ExpressionStatement)
    def visit_expression_statement(self, node: ast.ExpressionStatement, context: Context):
        """
        Registers an expression statement.
        Args:
            node (ExpressionStatement): The expression statement.
            context (Context): The current context.
        """
        return self.visit(node.expression, context)

    @visit.register(ast.Cast)
    def visit_cast(self, node: ast.Cast, context: Context):
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
        ret = arg
        try:
            if isinstance(node.type, ast.IntType):
                ret = int(arg)
            if isinstance(node.type, ast.UintType):
                ret = uint(arg)
            if isinstance(node.type, ast.FloatType):
                ret = float(arg)
            if isinstance(node.type, ast.ComplexType):
                ret = complex(arg)
            if isinstance(node.type, ast.BoolType):
                ret = bool(arg)
            # TODO: durations, angles, etc.
        except TypeError as e:
            raise TypeError(
                f"Unable to cast {arg.__class__.__name__} to {node.type.__class__.__name__}: {str(e)}"
            ) from e
        return ret

    @visit.register(ast.BinaryExpression)
    def visit_binary_expression(
        self, node: ast.BinaryExpression, context: Context
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
        match node.op.name:
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
                    f"Invalid operator {node.op.name} encountered in binary expression "
                    f"on line {node.span.start_line}."
                )  # pragma: no cover

    @visit.register(ast.UnaryExpression)
    def visit_unary_expression(self, node: ast.UnaryExpression, context: Context):
        """
        Registers a unary expression.

        Args:
            node (UnaryExpression): The unary expression.
            context (Context): The current context.

        Returns:
            The result of the evaluated expression.
        """
        operand = preprocess_operands(self.visit(node.expression, context))
        if node.op.name == "!":
            return not operand
        if node.op.name == "-":
            return -operand  # pylint: disable=invalid-unary-operand-type
        if node.op.name == "~":
            return ~operand  # pylint: disable=invalid-unary-operand-type
        # we shouldn't ever get this error if the parser did its job right
        raise SyntaxError(  # pragma: no cover
            f"Invalid operator {node.op.name} encountered in unary expression "
            f"on line {node.span.start_line}."
        )  # pragma: no cover

    @visit.register(ast.IndexExpression)
    def visit_index_expression(
        self, node: ast.IndexExpression, context: Context, aliasing: bool = False
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

    def _alias(self, node: ast.Identifier | ast.IndexExpression, context: Context):
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

    @visit.register(ast.Identifier)
    def visit_identifier(self, node: ast.Identifier, context: Context, aliasing: bool = False):
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

    @visit.register(ast.IntegerLiteral)
    @visit.register(ast.FloatLiteral)
    @visit.register(ast.BooleanLiteral)
    @visit.register(ast.BitstringLiteral)
    @visit.register(ast.DurationLiteral)
    def visit_literal(self, node: ast.Expression, context: Context):
        """
        Visits a literal.

        Args:
            node (Literal): The literal.
            context (Context): The current context.

        Returns:
            The value of the literal.
        """
        return node.value
