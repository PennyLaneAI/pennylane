"""
This submodule contains the interpreter for OpenQASM 3.0.
"""

import functools
import re
from functools import partial
from typing import Callable, Iterable

from openqasm3.ast import (
    AliasStatement,
    ArrayLiteral,
    BinaryExpression,
    BitstringLiteral,
    BranchingStatement,
    BreakStatement,
    Cast,
    ClassicalAssignment,
    ClassicalDeclaration,
    ConstantDeclaration,
    ContinueStatement,
    Expression,
    ExpressionStatement,
    ForInLoop,
    FunctionCall,
    Identifier,
    IndexExpression,
    QuantumArgument,
    QuantumGate,
    QubitDeclaration,
    RangeDefinition,
    ReturnStatement,
    SubroutineDefinition,
    SwitchStatement,
    UnaryExpression,
    WhileLoop,
)
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


class BreakException(Exception):  # pragma: no cover
    """Exception raised when encountering a break statement."""


class ContinueException(Exception):  # pragma: no cover
    """Exception raised when encountering a continue statement."""


class QasmInterpreter:
    """
    Takes the top level node of the AST as a parameter and recursively descends the AST, calling the
    visitor function on each node.
    """

    @functools.singledispatchmethod
    def visit(self, node: QASMNode, context: dict, aliasing: bool = False):
        """
        Visitor function is called on each node in the AST, which is traversed using recursive descent.
        The purpose of this function is to pass each node to the appropriate handler.

        Args:
            node (QASMNode): the QASMNode to visit next.
            context (dict): the current context populated with any locally available variables, etc.

        Raises:
            NotImplementedError: when an unsupported QASMNode type is found.
            aliasing (bool): whether we are aliasing a variable in the context.
        """
        if re.search("Literal", node.__class__.__name__):  # There is no single "Literal" base class
            return self.visit_literal(node, context)
        if isinstance(node, Callable):
            return self.visit_callable(node, context)  # cannot register Callable
        if node is None:
            return None
        raise NotImplementedError(
            f"An unsupported QASM instruction {node.__class__.__name__} "
            f"was encountered on line {node.span.start_line}, in {context['name']}."
        )

    @visit.register(list)
    def visit_list(self, node_list: list, context: dict):
        """
        Visits a list of QASMNodes.

        Args:
            node_list (list): the list of QASMNodes to visit.
            context (dict): the current context.
        """
        for sub_node in node_list:
            self.visit(sub_node, context)

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

    @visit.register(BreakStatement)
    def visit_break_statement(self, node: QASMNode, context: dict):
        """
        Registers a break statement.

        Args:
            node (QASMNode): the break QASMNode.
            context (dict): the current context.
        """

        raise BreakException(f"Break statement encountered in {context['name']}")

    @visit.register(ContinueStatement)
    def visit_continue_statement(self, node: QASMNode, context: dict):
        """
        Registers a continue statement.

        Args:
            node (QASMNode): the continue QASMNode.
            context (dict): the current context.
        """

        raise ContinueException(f"Continue statement encountered in {context['name']}")

    @visit.register(BranchingStatement)
    def visit_branching_statement(self, node: QASMNode, context: dict):
        """
        Registers a branching statement. Like switches, uses qml.cond.

        Args:
            node (QASMNode): the branch QASMNode.
            context (dict): the current context.
        """
        self._init_branches_scope(node, context)

        # create the true body context
        context["scopes"]["branches"][f"branch_{node.span.start_line}"]["true_body"] = (
            self._init_clause_in_same_namespace(
                context, f'{context["name"]}_branch_{node.span.start_line}_true_body'
            )
        )

        if hasattr(node, "else_block"):

            # create the false body context
            context["scopes"]["branches"][f"branch_{node.span.start_line}"]["false_body"] = (
                self._init_clause_in_same_namespace(
                    context, f'{context["name"]}_branch_{node.span.start_line}_false_body'
                )
            )

        ops.cond(
            self.visit(node.condition, context),
            partial(
                self.visit,
                node.if_block,
                context["scopes"]["branches"][f"branch_{node.span.start_line}"]["true_body"],
            ),
            (
                partial(
                    self.visit,
                    node.else_block,
                    context["scopes"]["branches"][f"branch_{node.span.start_line}"]["false_body"],
                )
                if hasattr(node, "else_block")
                else None
            ),
        )()

    @visit.register(SwitchStatement)
    def visit_switch_statement(self, node: QASMNode, context: dict):
        """
        Registers a switch statement.

        Args:
            node (QASMNode): the switch QASMNode.
            context (dict): the current context.
        """
        self._init_switches_scope(node, context)

        # switches need to have access to the outer context but not get called unless the condition is met

        # we need to keep track of each clause individually
        for i, case in enumerate(node.cases):
            context["scopes"]["switches"][f"switch_{node.span.start_line}"][f"cond_{i}"] = (
                self._init_clause_in_same_namespace(
                    context, f'{context["name"]}_switch_{node.span.start_line}_cond_{i}'
                )
            )

        if hasattr(node, "default") and node.default is not None:
            context["scopes"]["switches"][f"switch_{node.span.start_line}"][f"cond_{i + 1}"] = (
                self._init_clause_in_same_namespace(
                    context, f'{context["name"]}_switch_{node.span.start_line}_cond_{i + 1}'
                )
            )

        target = self.visit(node.target, context)
        ops.cond(
            target == self.visit(node.cases[0][0][0], context),
            partial(
                self.visit,
                node.cases[0][1].statements,
                context["scopes"]["switches"][f"switch_{node.span.start_line}"]["cond_0"],
            ),
            (
                partial(
                    self.visit,
                    node.default.statements,
                    context["scopes"]["switches"][f"switch_{node.span.start_line}"][
                        f"cond_{i + 1}"
                    ],
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
                        context["scopes"]["switches"][f"switch_{node.span.start_line}"][case],
                    ),
                )
                for j, case in enumerate(
                    list(context["scopes"]["switches"][f"switch_{node.span.start_line}"].keys())[
                        1:-1
                    ]
                )
            ],
        )()

    def _init_switches_scope(self, node: QASMNode, context: dict):
        """
        Inits the switches scope on the current context.

        Args:
            node (QASMNode): the switch node.
            context (dict): the current context.
        """
        if not "scopes" in context:
            context["scopes"] = {"switches": dict()}
        elif "switches" not in context["scopes"]:
            context["scopes"]["switches"] = dict()

        context["scopes"]["switches"][f"switch_{node.span.start_line}"] = dict()

    def _init_branches_scope(self, node: QASMNode, context: dict):
        """
        Inits the branches scope on the current context.

        Args:
            node (QASMNode): the branch node.
            context (dict): the current context.
        """
        if not "scopes" in context:
            context["scopes"] = {"branches": dict()}
        elif "branches" not in context["scopes"]:
            context["scopes"]["branches"] = dict()

        context["scopes"]["branches"][f"branch_{node.span.start_line}"] = dict()

    def _init_outer_wires_list(self, context: dict):
        """
        Inits the outer wires list on a sub context.

        Args:
            context (dict): the current context.
        """
        if "outer_wires" not in context:
            context["outer_wires"] = []

    def _init_loops_scope(self, node: QASMNode, context: dict):
        """
        Inits the loops scope on the current context.

        Args:
            node (QASMNode): the loop node.
            context (dict): the current context.
        """
        if not "scopes" in context:
            context["scopes"] = {"loops": dict()}
        elif "loops" not in context["scopes"]:
            context["scopes"]["loops"] = dict()

        # the namespace is shared with the outer scope, but we need to keep track of the gates separately
        if isinstance(node, WhileLoop):
            context["scopes"]["loops"][f"while_{node.span.start_line}"] = (
                self._init_clause_in_same_namespace(
                    context, f'{context["name"]}_while_{node.span.start_line}'
                )
            )

        elif isinstance(node, ForInLoop):
            context["scopes"]["loops"][f"for_{node.span.start_line}"] = (
                self._init_clause_in_same_namespace(
                    context, f'{context["name"]}_for_{node.span.start_line}'
                )
            )

    @staticmethod
    def _handle_break(loop: Callable, execution_context: dict):
        """
        Handles when a break is encountered in the loop.

        Args:
            loop (Callable): the loop function.
            execution_context (dict): the context passed at execution time with current variable values, etc.
        """
        try:
            loop(execution_context)
        except BreakException as e:
            pass  # evaluation of the loop stops

    @visit.register(WhileLoop)
    def visit_while_loop(self, node: QASMNode, context: dict):
        """
        Registers a while loop.

        Args:
            node (QASMNode): the loop node.
            context (dict): the current context.
        """
        self._init_loops_scope(node, context)

        @while_loop(
            partial(self.visit, node.while_condition)
        )  # traces data dep through context
        def loop(context):
            """
            Executes a traceable while loop.

            Args:
                loop_context (dict): the context used to compile the while loop.
                execution_context (dict): the context passed at execution time with current variable values, etc.
            """
            try:
                # updates vars in context... need to propagate these to outer scope
                self.visit(node.block, context["scopes"]["loops"][f"while_{node.span.start_line}"])
            except ContinueException as e:
                pass  # evaluation of this iteration ends, and we continue to the next

            inner_context = context["scopes"]["loops"][f"while_{node.span.start_line}"]
            context["vars"] = inner_context["vars"] if "vars" in inner_context else None
            context["wires"] += inner_context["wires"] if "wires" in inner_context else None

            return context

        self._handle_break(loop, context)

    @visit.register(ForInLoop)
    def visit_for_in_loop(self, node: QASMNode, context: dict):
        """
        Registers a for loop.

        Args:
            node (QASMNode): the loop node.
            context (dict): the current context.
        """
        self._init_loops_scope(node, context)

        loop_params = node.set_declaration

        # de-referencing
        if isinstance(loop_params, Identifier):
            loop_params = self.retrieve_variable(loop_params.name, context)
            if isinstance(loop_params, dict) and "val" in loop_params and "ty" in loop_params:
                if loop_params["ty"] == "BitType":
                    loop_params = self._get_bit_type_val(loop_params)
                else:
                    loop_params = loop_params["val"]

        # TODO: support dynamic start, stop, step?
        if isinstance(loop_params, RangeDefinition):
            start = self.visit(loop_params.start, context)
            stop = self.visit(loop_params.end, context)
            step = self.visit(loop_params.step, context)
            if step is None:
                step = 1

            @for_loop(start, stop, step)
            def loop(i, execution_context):
                execution_context["scopes"]["loops"][f"for_{node.span.start_line}"]["vars"][
                    node.identifier.name
                ] = {
                    "ty": i.__class__.__name__,
                    "val": i,
                    "line": node.span.start_line,
                    "dirty": True,
                }
                try:
                    # we only want to execute the gates in the loop's scope
                    # updates vars in sub context... need to propagate these to outer context
                    self.visit(
                        node.block, context["scopes"]["loops"][f"for_{node.span.start_line}"]
                    )
                except ContinueException as e:
                    pass  # evaluation of the current iteration stops and we continue
                inner_context = execution_context["scopes"]["loops"][f"for_{node.span.start_line}"]
                context["vars"] = inner_context["vars"] if "vars" in inner_context else None
                context["wires"] += inner_context["wires"] if "wires" in inner_context else None

                return execution_context

            self._handle_break(loop, context)

        # we unroll the loop in the following case when we don't have a range since qml.for_loop() only
        # accepts (start, stop, step) and not a list of values.
        elif isinstance(loop_params, ArrayLiteral):
            iter = [self.visit(literal, context) for literal in loop_params.values]
        elif isinstance(
            loop_params, Iterable
        ):  # it's an array literal that's been eval'd before TODO: unify these reprs?
            iter = [val for val in loop_params]

            def unrolled(execution_context):
                for i in iter:
                    execution_context["scopes"]["loops"][f"for_{node.span.start_line}"]["vars"][
                        node.identifier.name
                    ] = {
                        "ty": i.__class__.__name__,
                        "val": i,
                        "line": node.span.start_line,
                        "dirty": True,
                    }
                    try:
                        # visit the nodes once per loop iteration
                        self.visit(
                            node.block, context["scopes"]["loops"][f"for_{node.span.start_line}"]
                        )  # updates vars in sub context if any measurements etc. occur
                    except ContinueException as e:
                        pass  # eval of current iteration stops and we continue

            self._handle_break(unrolled, context)
        elif loop_params is None:
            print(f"Uninitialized iterator in loop {f'for_{node.span.start_line}'}.")

    @visit.register(FunctionCall)
    def visit_function_call(self, node: FunctionCall, context: dict):
        """
        Registers a function call. The node must refer to a subroutine that has been defined and
        is available in the current scope.

        Args:
            node (FunctionCall): The FunctionCall QASMNode.
            context (dict): The current context.

        Raises:
            NameError: When the subroutine is not defined.
        """
        ret = None
        if (
            "scopes" in context
            and "subroutines" in context["scopes"]
            or "outer_scopes" in context
            and "subroutines" in context["outer_scopes"]
        ):
            name = node.name if isinstance(node.name, str) else node.name.name  # str or Identifier
            if ("scopes" in context and name not in context["scopes"]["subroutines"]) or (
                "outer_scopes" in context and name not in context["outer_scopes"]["subroutines"]
            ):
                raise NameError(
                    f"Reference to an undeclared subroutine {name} in {context['name']}."
                )
            else:
                # TODO: use ChainMap to get this from scopes or outer_scopes
                if (
                    "scopes" in context
                    and "subroutines" in context["scopes"]
                    and node.name.name in context["scopes"]["subroutines"]
                ):
                    func_context = context["scopes"]["subroutines"][node.name.name]
                elif (
                    "outer_scopes" in context
                    and "subroutines" in context["outer_scopes"]
                    and node.name.name in context["outer_scopes"]["subroutines"]
                ):
                    func_context = context["outer_scopes"]["subroutines"][node.name.name]
                else:
                    raise NameError(
                        f"Reference to subroutine {node.name.name} not available in calling namespace"
                        f"on line {node.span.start_line}."
                    )

                # bind subroutine arguments
                self._init_vars(func_context)
                evald_args = [self.visit(raw_arg, context) for raw_arg in node.arguments]
                for evald_arg, param in list(zip(evald_args, func_context["params"])):
                    if not isinstance(evald_arg, str):  # this would indicate a quantum parameter
                        func_context["vars"][param] = evald_arg
                    else:
                        self._init_wire_map(func_context)
                        if not param == evald_arg:
                            func_context["wire_map"][param] = evald_arg

                # execute the subroutine
                self.visit(func_context["body"], func_context)

                # the return value
                ret = func_context["return"] if "return" in func_context else None
        return ret

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

    @staticmethod
    def _update_var(value: any, name: str, node: QASMNode, context: dict):
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
        res = self.visit(node.rvalue, context)
        self._update_var(res, name, node, context)

    @visit.register(AliasStatement)
    def visit_alias_statement(self, node: QASMNode, context: dict):
        """
        Registers an alias statement.
        Args:
            node (QASMNode): the alias QASMNode.
            context (dict): the current context.
        """
        self._init_aliases(context)
        context["aliases"][node.target.name] = self.visit(node.value, context, aliasing=True)

    @staticmethod
    def retrieve_variable(name: str, context: dict):
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
            raise NameError(f"Attempt to reference uninitialized parameter {name}!")
        if (
            "wires" in context
            and context["wires"] is not None
            and name in context["wires"]
            or "outer_wires" in context
            and name in context["outer_wires"]
        ):
            return name
        if "aliases" in context and context["wires"] is not None and name in context["aliases"]:
            res = context["aliases"][name](context)  # evaluate the alias and de-reference
            if isinstance(res, str):
                return res
            if res["val"] is not None:
                return res
            raise NameError(f"Attempt to reference uninitialized parameter {name}!")
        raise TypeError(
            f"Attempt to use unevaluated variable {name} in {context['name']}, "
            f"last updated on line {context['vars'][name]['line'] if name in context['vars'] else 'unknown'}."
        )

    def _init_clause_in_same_namespace(self, outer_context: dict, name: str):
        """
        Initializes a clause that shares the namespace of the outer scope, but contains its own
        set of gates, operations, expressions, logic, etc.
        Args:
            outer_context (dict): the context of the outer scope.
            name (str): the name of the clause.
        Returns:
            dict: the inner context.
        """
        # we want wires declared in outer scopes to be available
        outer_wires = outer_context["wires"] if "wires" in outer_context else None
        if "outer_wires" in outer_context:
            outer_wires = outer_context["outer_wires"]
        context = {
            "vars": outer_context["vars"] if "vars" in outer_context else None,  # same namespace
            "outer_wires": outer_wires,
            "wire_map": outer_context["wire_map"],
            "wires": [],
            "name": name,
        }
        # we want subroutines declared in outer scopes to be available
        if "scopes" in outer_context and "subroutines" in outer_context["scopes"]:
            context["outer_scopes"] = {
                # no recursion here please! hence the filter
                "subroutines": {
                    k: v for k, v in outer_context["scopes"]["subroutines"].items() if k != name
                }
            }

        return context

    def _init_wire_map(self, context: dict):
        """
        Inits the wire map dict on the current context.
        Args:
            context (dict): the current context.
        """
        if "wire_map" not in context or context["wire_map"] is None:
            context["wire_map"] = dict()

    @staticmethod
    def _init_vars(context: dict):
        """
        Inits the vars dict on the current context.
        Args:
            context (dict): the current context.
        """
        if "vars" not in context:
            context["vars"] = dict()

    @staticmethod
    def _init_aliases(context: dict):
        """
        Inits the aliases dict on the current context.
        Args:
            context (dict): the current context.
        """
        if "aliases" not in context:
            context["aliases"] = dict()

    def _init_subroutine_scope(self, node: QASMNode, context: dict):
        """
        Inits the subroutine scope on the current context.
        Args:
            node (QASMNode): the subroutine node.
            context (dict): the current context.
        """
        if not "scopes" in context:
            context["scopes"] = {"subroutines": dict()}
        elif "subroutines" not in context["scopes"]:
            context["scopes"]["subroutines"] = dict()

        # outer scope variables are available to inner scopes... but not vice versa!
        # names prefixed with outer scope names for specificity
        context["scopes"]["subroutines"][node.name.name] = self._init_clause_in_same_namespace(
            context, f'{context["name"]}_{node.name.name}'
        )
        context["scopes"]["subroutines"][node.name.name]["sub"] = True
        context["scopes"]["subroutines"][node.name.name]["body"] = node.body
        context["scopes"]["subroutines"][node.name.name]["params"] = [
            param.name.name for param in node.arguments
        ]

    @staticmethod
    def _get_bit_type_val(var):
        return bin(var["val"])[2:].zfill(var["size"])

    @visit.register(ReturnStatement)
    def visit_return_statement(self, node: QASMNode, context: dict):
        """
        Registers a return statement. Points to the var that needs to be set in an outer scope when this
        subroutine is called.
        """
        context["return"] = self.visit(node.expression, context)

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
    def visit_classical_declaration(self, node: QASMNode, context: dict, constant: bool = False):
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
                    "val": self.visit(node.init_expression, context),
                    "size": node.init_expression.width,
                    "line": node.init_expression.span.start_line,
                    "constant": constant,
                }
            elif not isinstance(node.init_expression, ArrayLiteral):
                context["vars"][node.identifier.name] = {
                    "ty": node.type.__class__.__name__,
                    "val": self.visit(node.init_expression, context),
                    "line": node.init_expression.span.start_line,
                    "constant": constant,
                }
            else:
                context["vars"][node.identifier.name] = {
                    "ty": node.type.__class__.__name__,
                    "val": [
                        self.visit(literal, context) for literal in node.init_expression.values
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

    @visit.register(SubroutineDefinition)
    def visit_subroutine_definition(self, node: QASMNode, context: dict):
        """
        Registers a subroutine definition. Maintains a namespace in the context, starts populating it with
        its parameters.
        Args:
            node (QASMNode): the subroutine node.
            context (dict): the current context.
        """
        self._init_subroutine_scope(node, context)

        # register the params
        for param in node.arguments:
            if not isinstance(param, QuantumArgument):
                context["scopes"]["subroutines"][node.name.name]["vars"][param.name.name] = {
                    "ty": param.__class__.__name__,
                    "val": None,
                    "line": param.span.start_line,
                    "dirty": False,
                }
            else:
                context["scopes"]["subroutines"][node.name.name]["wires"].append(param.name.name)

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
        num_control = sum("ctrl" in mod.modifier.name for mod in node.modifiers)
        op_wires = wires[num_control:]
        control_wires = wires[:num_control]

        op = gate(*args, wires=op_wires)
        for mod in reversed(node.modifiers):
            op, control_wires = self.apply_modifier(mod, op, context, control_wires)

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

        self._require_wires(wires, context)

        resolved_wires = []
        if context["wire_map"] is not None:
            for wire in wires:
                resolving = wire
                while resolving in context["wire_map"]:
                    resolving = context["wire_map"][resolving]
                resolved_wires.append(resolving)
        else:
            resolved_wires = wires

        return gate, args, resolved_wires

    def apply_modifier(self, mod: QuantumGate, previous: Operator, context: dict, wires: list):
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
    def visit_expression_statement(self, node: ExpressionStatement, context: dict):
        """
        Registers an expression statement.

        Args:
            node (ExpressionStatement): The expression statement.
            context (dict): The current context.
        """
        return self.visit(node.expression, context)

    @visit.register(Cast)
    def visit_cast(self, node: Cast, context: dict):
        """
        Registers a Cast expression.

        Args:
            node (Cast): The Cast expression.
            context (dict): The current context.
        """
        return self.retrieve_variable(node.argument.name, context)["val"]

    @visit.register(BinaryExpression)
    def visit_binary_expression(self, node: BinaryExpression, context: dict):
        """
        Registers a binary expression.

        Args:
            node (BinaryExpression): The binary expression.
            context (dict): The current context.

        Returns:
            The result of the evaluated expression.
        """
        lhs = self.visit(node.lhs, context)
        rhs = self.visit(node.rhs, context)
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
    def visit_unary_expression(self, node: UnaryExpression, context: dict):
        """
        Registers a unary expression.

        Args:
            node (UnaryExpression): The unary expression.
            context (dict): The current context.

        Returns:
            The result of the evaluated expression.
        """
        if (
            node.op.name in NON_ASSIGNMENT_CLASSICAL_OPERATORS
        ):  # makes sure we are not executing anything malicious
            return eval(
                f"{node.op.name}{self.visit(node.expression, context)}"
            )  # pylint: disable=eval-used
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
    def visit_index_expression(self, node: IndexExpression, context: dict, aliasing: bool = False):
        """
        Registers an index expression.

        Args:
            node (IndexExpression): The index expression.
            context (dict): The current context.
            aliasing (bool): If ``True``, the expression will be treated as an alias.

        Returns:
            The slice of the indexed value.
        """

        def _index_into_var(var):
            if var["ty"] == "BitType":
                var = bin(var["val"])[2:].zfill(var["size"])
            else:
                var = var["val"]
            if isinstance(node.index[0], RangeDefinition):
                return var[node.index[0].start.value : node.index[0].end.value]
            if re.search("Literal", node.index[0].__class__.__name__):
                return var[node.index[0].value]
            raise TypeError(
                f"Array index is not a RangeDefinition or Literal at line {node.span.start_line}."
            )

        if aliasing:

            def alias(context):
                try:
                    var = self.retrieve_variable(node.collection.name, context)
                    return _index_into_var(var)
                except NameError as e:
                    raise NameError(
                        f"Attempt to alias an undeclared variable "
                        f"{node.collection.name} in {context['name']}."
                    ) from e

            return alias
        else:
            var = self.retrieve_variable(node.collection.name, context)
            return _index_into_var(var)

    @visit.register(Identifier)
    def visit_identifier(self, node: Identifier, context: dict, aliasing: bool = False):
        """
        Registers an identifier.

        Args:
            node (Identifier): The identifier.
            context (dict): The current context.
            aliasing (bool): If ``True``, the Identifier will be treated as an alias.

        Returns:
            The de-referenced identifier.
        """
        if aliasing:

            def alias(context):
                try:
                    return self.retrieve_variable(node.collection.name, context)
                except NameError as e:
                    raise NameError(
                        f"Attempt to alias an undeclared variable "
                        f"{node.name} in {context['name']}."
                    ) from e

            return alias
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
                    str(e)
                    or f"Reference to an undeclared variable {node.name} in {context['name']}."
                ) from e

    def visit_callable(self, func: Callable, context: dict):
        """
        Visits a Callable.

        Args:
            func (Callable): The callable.
            context (dict): The current context.

        Returns:
            The result of the called callable.
        """
        return func()

    def visit_literal(self, node: Expression, context: dict):
        """
        Visits a literal.

        Args:
            node (Literal): The literal.
            context (dict): The current context.

        Returns:
            The value of the literal.
        """
        return node.value

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
            if not (
                wire in context["wires"]
                or ("outer_wires" in context and wire in context["outer_wires"])
            ):
                missing_wires.append(wire)
        if len(missing_wires) > 0:
            raise NameError(
                f"Attempt to reference wire(s): {missing_wires} that have not been declared in {context['name']}"
            )
