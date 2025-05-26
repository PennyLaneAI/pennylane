"""
This submodule contains the interpreter for QASM 3.0.
"""

import copy
import inspect
import re
from functools import partial
from typing import Callable, Iterable

from pennylane import ops, wires
from pennylane.control_flow import for_loop, while_loop
from pennylane.control_flow.while_loop import WhileLoopCallable
from pennylane.measurements import measure

has_openqasm = True
try:
    from openqasm3.ast import (
        ArrayLiteral,
        BinaryExpression,
        BitstringLiteral,
        Cast,
        EndStatement,
        ForInLoop,
        FunctionCall,
        Identifier,
        IndexExpression,
        IntegerLiteral,
        QuantumArgument,
        RangeDefinition,
        UnaryExpression,
        WhileLoop,
    )
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


class DirtyError(Exception):  # pragma: no cover
    """Exception raised when attempt is made to use a dirty variable in a compilation context."""


class BreakException(Exception):  # pragma: no cover
    """Exception raised when encountering a break statement."""


class ContinueException(Exception):  # pragma: no cover
    """Exception raised when encountering a continue statement."""


class QasmInterpreter(QASMVisitor):
    """
    Overrides generic_visit(self, node: QASMNode, context: Optional[T]) which takes the
    top level node of the AST as a parameter and recursively descends the AST, calling the
    overriden visitor function on each node.

    There are two passes. The first queues Callables such as gate partials into a QNode,
    so that the QNode can be called and the program will be executed at that time. During
    this first pass, any available values provided by the program like literals are used to
    optimize the compilation with as much detail as possible. A simulation does not occur
    during the first pass which just creates a quantum function. The second pass occurs
    during the execution of the QNode and involves simulating everything. All remaining data
    flow and control flow is evaluated completely during this second pass. The control flow
    is handled using Pennylane provided qml.while_loop and qml.for_loop etc. to be compatible
    with qjit. The data flow is traced through the context, which is mutated during each pass.

    The first pass does optimization using static values only. We therefore need to track whether
    values are dirty.
    """

    def __init__(self, permissive=False):
        """
        Initializes a QASMInterpreter.

        Args:
            permissive (bool): whether to continue interpreting if an unsuppported node type is encountered.

        Raises:
            ImportError: if the openqasm3 package is not available.
        """
        if not has_openqasm:  # pragma: no cover
            raise ImportError(
                "QASM interpreter requires openqasm3 to be installed"
            )  # pragma: no cover
        else:
            super().__init__()

        self.permissive = permissive
        self.raise_if_dirty = False

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
            InterruptedError: When a QASM program is terminated by an end instruction.
        """
        handler_name = 'visit_' + node.__class__.__name__
        if node.__class__ == list:
            for sub_node in node:
                self.visit(sub_node, context)
        elif hasattr(self, handler_name):
            try:
                getattr(self, handler_name)(node, context)
            except NotImplementedError as e:
                if self.permissive:
                    pass
                else:
                    raise NotImplementedError(str(e)) from e
        elif self.permissive:
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

        init_context = copy.deepcopy(context)  # preserved for use in second (execution) pass
        try:
            super().generic_visit(node, context)
        except InterruptedError as e:
            print(str(e))
        execution_context = self.construct_qfunc(context, init_context)
        return context, execution_context

    @staticmethod
    def _execute_all(context: dict):
        """
        Executes all the gates in the context.

        Args:
            context (dict): the current context populated with the gates.
        """
        for func in context["gates"]:
            func()

    @staticmethod
    def _all_context_bound(quantum_function):
        """
        Checks whether a partial received all required context during compilation, or if
        it requires execution context because it has parameters that are based on, for example,
        the outcomes of mid-circuit measurements or subroutines.

        Args:
            partial_function (Callable): the partial with compilation context bound.
        """
        if hasattr(quantum_function, "func"):
            if isinstance(quantum_function.func, WhileLoopCallable):
                return False
            else:
                try:
                    inspect.signature(quantum_function.func).bind(
                        *quantum_function.args, **quantum_function.keywords
                    )
                    return True
                except TypeError:
                    return False
        elif len(inspect.signature(quantum_function).parameters) > 0:
            return False
        return True

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
        if not context["vars"][name]["constant"]:
            context["vars"][name]["dirty"] = True
        else:
            raise ValueError(
                f"Attempt to mutate a constant {name} on line {node.span.start_line} that was "
                f"defined on line {context['vars'][name]['line']}"
            )
        context["vars"][name]["line"] = node.span.start_line

    def visit_EndStatement(self, node: QASMNode, context: dict):
        """
        Ends the program.

        Args:
            node (QASMNode): The end statement QASMNode.
            context (dict): the current context.
        """
        raise InterruptedError(
            f"The QASM program was terminated om line {node.span.start_line}."
            f"There may be unprocessed QASM code."
        )

    def visit_BreakStatement(self, node: QASMNode, context: dict):
        """
        Registers a break statement.

        Args:
            node (QASMNode): the break QASMNode.
            context (dict): the current context.
        """

        def raiser():
            raise BreakException(f"Break statement encountered in {context['name']}")

        context["gates"].append(raiser)

    def visit_ContinueStatement(self, node: QASMNode, context: dict):
        """
        Registers a continue statement.

        Args:
            node (QASMNode): the continue QASMNode.
            context (dict): the current context.
        """

        def raiser():
            raise ContinueException(f"Continue statement encountered in {context['name']}")

        context["gates"].append(raiser)

    def visit_ClassicalAssignment(self, node: QASMNode, context: dict):
        """
        Registers a classical assignment.

        Args:
            node (QASMNode): the assignment QASMNode.
            context (dict): the current context.
        """

        def set_local_var(execution_context: dict):
            rhs = partial(self.eval_expr, node.rvalue, execution_context)
            name = (
                node.lvalue.name if isinstance(node.lvalue.name, str) else node.lvalue.name.name
            )  # str or Identifier
            res = rhs()
            self._update_var(res, name, node, execution_context)
            return res

        # references to an unresolved value see a func for now
        name = (
            node.lvalue.name if isinstance(node.lvalue.name, str) else node.lvalue.name.name
        )  # str or Identifier
        self._update_var(set_local_var, name, node, context)
        context["gates"].append(set_local_var)

    def _choose_context(self, func: Callable, context: dict, execution_context: dict):
        """
        Executes the callable with the right context depending on whether it was bound at compile time or
        needs execution context.

        Args:
            func (Callable): the callable to be executed.
            context (dict): the context to be passed to the function.
            execution_context (dict): the execution context to be passed to the function.

        Returns:
            dict: the context to use.
        """
        if not self._depends_on_dirty_vars(func):
            return context
        else:
            return execution_context

    def _depends_on_dirty_vars(self, func: Callable):
        """
        Checks if any state (variables) relevant to the node being processed is dirty.

        Args:
            func (Callable): the function which may use dirty state.
            context (dict): the current compilation context.

        Returns:
            bool: whether the func depends on dirty state. If it does, the state should not be bound during compilation.
        """
        self.raise_if_dirty = True
        try:
            func()
            self.raise_if_dirty = False
            return False
        except DirtyError:
            self.raise_if_dirty = False
            return True

    def visit_BranchingStatement(self, node: QASMNode, context: dict):
        """
        Registers a branching statement. Like switches, uses qml.cond.

        Args:
            node (QASMNode): the branch QASMNode.
            context (dict): the current context.
        """
        self._init_branches_scope(node, context)
        self._init_gates_list(context)

        # create the true body context
        context["scopes"]["branches"][f"branch_{node.span.start_line}"]["true_body"] = (
            self._init_clause_in_same_namespace(
                context, f'{context["name"]}_branch_{node.span.start_line}_true_body'
            )
        )

        # process the true body
        self.visit(
            node.if_block,
            context["scopes"]["branches"][f"branch_{node.span.start_line}"]["true_body"],
        )

        if hasattr(node, "else_block"):

            # create the false body context
            context["scopes"]["branches"][f"branch_{node.span.start_line}"]["false_body"] = (
                self._init_clause_in_same_namespace(
                    context, f'{context["name"]}_branch_{node.span.start_line}_false_body'
                )
            )

            # process the false body
            self.visit(
                node.else_block,
                context["scopes"]["branches"][f"branch_{node.span.start_line}"]["false_body"],
            )

        def branch(execution_context: dict):
            ops.cond(
                self.eval_expr(
                    node.condition,
                    self._choose_context(
                        partial(self.eval_expr, node.condition, context), context, execution_context
                    ),
                ),
                lambda: [
                    gate(execution_context) if not self._all_context_bound(gate) else gate()
                    for gate in context["scopes"]["branches"][f"branch_{node.span.start_line}"][
                        "true_body"
                    ]["gates"]
                ],
                lambda: (
                    [
                        gate(execution_context) if not self._all_context_bound(gate) else gate()
                        for gate in context["scopes"]["branches"][f"branch_{node.span.start_line}"][
                            "false_body"
                        ]["gates"]
                    ]
                    if "gates"
                    in context["scopes"]["branches"][f"branch_{node.span.start_line}"]["false_body"]
                    else None
                ),
            )()

        context["gates"].append(branch)

    def visit_SwitchStatement(self, node: QASMNode, context: dict):
        """
        Registers a switch statement.

        Args:
            node (QASMNode): the switch QASMNode.
            context (dict): the current context.
        """
        self._init_switches_scope(node, context)
        self._init_gates_list(context)

        # switches need to have access to the outer context but not get called unless the condition is met

        # we need to keep track of each clause individually
        for i, case in enumerate(node.cases):
            context["scopes"]["switches"][f"switch_{node.span.start_line}"][f"cond_{i}"] = (
                self._init_clause_in_same_namespace(
                    context, f'{context["name"]}_switch_{node.span.start_line}_cond_{i}'
                )
            )

            # process the individual clauses
            self.visit(
                case[1].statements,
                context["scopes"]["switches"][f"switch_{node.span.start_line}"][f"cond_{i}"],
            )

        if hasattr(node, "default") and node.default is not None:
            context["scopes"]["switches"][f"switch_{node.span.start_line}"][f"cond_{i + 1}"] = (
                self._init_clause_in_same_namespace(
                    context, f'{context["name"]}_switch_{node.span.start_line}_cond_{i + 1}'
                )
            )

            # process the default case
            self.visit(
                node.default.statements,
                context["scopes"]["switches"][f"switch_{node.span.start_line}"][f"cond_{i + 1}"],
            )

        def switch(execution_context: dict):
            target = self.eval_expr(
                node.target,
                self._choose_context(
                    partial(self.eval_expr, node.target, context), context, execution_context
                ),
            )
            ops.cond(
                target
                == self.eval_expr(
                    node.cases[0][0][0],
                    self._choose_context(
                        partial(self.eval_expr, node.cases[0][0][0], context),
                        context,
                        execution_context,
                    ),
                ),
                lambda: [
                    gate(execution_context) if not self._all_context_bound(gate) else gate()
                    for gate in context["scopes"]["switches"][f"switch_{node.span.start_line}"][
                        f"cond_0"
                    ]["gates"]
                ],
                lambda: (
                    [
                        gate(execution_context) if not self._all_context_bound(gate) else gate()
                        for gate in context["scopes"]["switches"][f"switch_{node.span.start_line}"][
                            f"cond_{i + 1}"
                        ]["gates"]
                    ]
                    if f"cond_{i + 1}"
                    in context["scopes"]["switches"][f"switch_{node.span.start_line}"]
                    else None
                ),
                [
                    (
                        target == self.eval_expr(node.cases[j + 1][0][0], execution_context),
                        lambda: [
                            gate(execution_context) if not self._all_context_bound(gate) else gate()
                            for gate in context["scopes"]["switches"][
                                f"switch_{node.span.start_line}"
                            ][case]["gates"]
                        ],
                    )
                    for j, case in enumerate(
                        list(
                            context["scopes"]["switches"][f"switch_{node.span.start_line}"].keys()
                        )[1:-1]
                    )
                ],
            )()

        context["gates"].append(switch)

    def visit_AliasStatement(self, node: QASMNode, context: dict):
        """
        Registers an alias statement.

        Args:
            node (QASMNode): the alias QASMNode.
            context (dict): the current context.
        """
        self._init_aliases(context)
        context["aliases"][node.target.name] = self.eval_expr(node.value, context, aliasing=True)

        # we append anything that needs to be computed at execution time to the gates list...
        # aliases can change throughout a program
        context["gates"].append(partial(self.visit_AliasStatement, node))

    def retrieve_variable(self, name: str, context: dict):
        """
        Attempts to retrieve a variable from the current context by name.

        Args:
            name (str): the name of the variable to retrieve.
            context (dict): the current context.
        """

        def _warning(context, name):
            raise TypeError(
                f"Attempt to use unevaluated variable {name} in {context['name']}, "
                f"last updated on line {context['vars'][name]['line'] if name in context['vars'] else 'unknown'}."
            )

        def _dirty(context, name):
            raise DirtyError(
                f"Attempt to use dirty variable {name} in compilation context {context['name']}."
            )

        if "vars" in context and context["vars"] is not None and name in context["vars"]:
            if isinstance(context["vars"][name], Callable):
                _warning(context, name)
            else:
                res = context["vars"][name]
                if res["dirty"] == True and self.raise_if_dirty:
                    _dirty(context, name)
                return res
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
            else:
                if res["dirty"] == True and self.raise_if_dirty:
                    _dirty(context, name)
            return res
        else:
            _warning(context, name)

    def eval_expr(self, node: QASMNode, context: dict, aliasing: bool = False):
        """
        Evaluates an expression.

        Args:
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
            res = eval(f"{lhs}{node.op.name}{rhs}")  # TODO: don't use eval
        elif isinstance(node, UnaryExpression):
            res = eval(f"{node.op.name}{self.eval_expr(node.expression, context)}")
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
                        var["val"] = (
                            value(context) if not self._all_context_bound(value) else value()
                        )
                        var["line"] = node.span.start_line
                        var["dirty"] = True
                        value = var["val"]
                    return value
                except NameError as e:
                    raise NameError(
                        f"Reference to an undeclared variable {node.name} in {context['name']}."
                    ) from e
        elif isinstance(node, FunctionCall):
            if (
                "scopes" in context
                and "subroutines" in context["scopes"]
                or "outer_scopes" in context
                and "subroutines" in context["outer_scopes"]
            ):
                name = (
                    node.name if isinstance(node.name, str) else node.name.name
                )  # str or Identifier
                if ("scopes" in context and name not in context["scopes"]["subroutines"]) or (
                    "outer_scopes" in context and name not in context["outer_scopes"]["subroutines"]
                ):
                    raise NameError(
                        f"Reference to an undeclared subroutine {name} in {context['name']}."
                    )
                else:
                    if "scopes" in context and name in context["scopes"]["subroutines"]:
                        func_context = context["scopes"]["subroutines"][name]
                    else:
                        func_context = context["outer_scopes"]["subroutines"][name]

                    # bind subroutine arguments
                    for arg in node.arguments:
                        self._init_vars(func_context)
                        evald_arg = self.eval_expr(arg, context)
                        # TODO: maybe we want to have a class for classical and a class for quantum parameters
                        if not isinstance(
                            evald_arg, str
                        ):  # this would indicate a quantum parameter
                            func_context["vars"][arg.name] = evald_arg

                    # execute the subroutine
                    [
                        gate(context) if not self._all_context_bound(gate) else gate()
                        for gate in func_context["gates"]
                    ]

                    # the return value
                    return self.retrieve_variable(func_context["return"], func_context)
        elif isinstance(node, Callable):
            res = node()
        elif re.search("Literal", node.__class__.__name__):
            res = node.value
        # TODO: include all other cases here
        return res

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

    def _init_vars(self, context: dict):
        """
        Inits the vars dict on the current context.

        Args:
            context (dict): the current context.
        """
        if "vars" not in context:
            context["vars"] = dict()

    def _init_wires(self, context: dict):
        """
        Inits the wires dict on the current context.

        Args:
            context (dict): the current context.
        """
        if "wires" not in context:
            context["wires"] = []

    def _init_aliases(self, context: dict):
        """
        Inits the aliases dict on the current context.

        Args:
            context (dict): the current context.
        """
        if "aliases" not in context:
            context["aliases"] = dict()

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

    def _init_gates_list(self, context: dict):
        """
        Inits the gates list on the current context.

        Args:
            context (dict): the current context.
        """
        if "gates" not in context:
            context["gates"] = []

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

    def visit_WhileLoop(self, node: QASMNode, context: dict):
        """
        Registers a while loop.

        Args:
            node (QASMNode): the loop node.
            context (dict): the current context.
        """
        self._init_gates_list(context)
        self._init_loops_scope(node, context)

        @while_loop(
            partial(self.eval_expr, node.while_condition)
        )  # traces data dep through context
        def loop(execution_context):
            """
            Executes a traceable while loop.

            Args:
                loop_context (dict): the context used to compile the while loop.
                execution_context (dict): the context passed at execution time with current variable values, etc.
            """
            # we don't want to populate the gates again with every call to visit
            context["scopes"]["loops"][f"while_{node.span.start_line}"]["gates"] = []
            # process loop body...
            inner_context = self.visit(
                node.block, context["scopes"]["loops"][f"while_{node.span.start_line}"]
            )
            try:
                for gate in inner_context["gates"]:
                    # updates vars in context... need to propagate these to outer scope
                    gate(execution_context) if not self._all_context_bound(gate) else gate()
            except ContinueException as e:
                pass  # evaluation of this iteration ends and we continue to the next
            context["vars"] = inner_context["vars"] if "vars" in inner_context else None
            context["wires"] += inner_context["wires"] if "wires" in inner_context else None

            return execution_context

        context["gates"].append(
            partial(self._handle_break, loop)
        )  # bind compilation context now, leave execution context

    @staticmethod
    def _get_bit_type_val(var):
        return bin(var["val"])[2:].zfill(var["size"])

    def visit_ForInLoop(self, node: QASMNode, context: dict):
        """
        Registers a for loop.

        Args:
            node (QASMNode): the loop node.
            context (dict): the current context.
        """
        self._init_gates_list(context)
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
            start = self.eval_expr(loop_params.start, context)
            stop = self.eval_expr(loop_params.end, context)
            step = self.eval_expr(loop_params.step, context)
            if step is None:
                step = 1

            @for_loop(start, stop, step)
            def loop(i, execution_context):
                context["scopes"]["loops"][f"for_{node.span.start_line}"]["vars"][
                    node.identifier.name
                ] = {
                    "ty": i.__class__.__name__,
                    "val": i,
                    "line": node.span.start_line,
                    "dirty": True,
                }
                # we don't want to populate the gates again with every call to visit
                context["scopes"]["loops"][f"for_{node.span.start_line}"]["gates"] = []
                # process loop body
                inner_context = self.visit(
                    node.block, context["scopes"]["loops"][f"for_{node.span.start_line}"]
                )
                try:
                    for gate in inner_context[
                        "gates"
                    ]:  # we only want to execute the gates in the loop's scope
                        # updates vars in sub context... need to propagate these to outer context
                        gate(execution_context) if not self._all_context_bound(gate) else gate()
                except ContinueException as e:
                    pass  # evaluation of the current iteration stops and we continue
                context["vars"] = inner_context["vars"] if "vars" in inner_context else None
                context["wires"] += inner_context["wires"] if "wires" in inner_context else None

                return execution_context

            context["gates"].append(partial(self._handle_break, loop))

        # we unroll the loop in the following case when we don't have a range since qml.for_loop() only
        # accepts (start, stop, step) and nto a list of values.
        elif isinstance(loop_params, ArrayLiteral):
            iter = [self.eval_expr(literal, context) for literal in loop_params.values]
        elif isinstance(
            loop_params, Iterable
        ):  # it's an array literal that's been eval'd before TODO: unify these reprs?
            iter = [val for val in loop_params]

            def unrolled(execution_context):
                for i in iter:
                    context["scopes"]["loops"][f"for_{node.span.start_line}"]["vars"][
                        node.identifier.name
                    ] = {
                        "ty": i.__class__.__name__,
                        "val": i,
                        "line": node.span.start_line,
                        "dirty": True,
                    }
                    context["scopes"]["loops"][f"for_{node.span.start_line}"]["gates"] = []
                    # visit the nodes once per loop iteration
                    self.visit(
                        node.block, context["scopes"]["loops"][f"for_{node.span.start_line}"]
                    )
                    try:
                        for gate in context["scopes"]["loops"][f"for_{node.span.start_line}"][
                            "gates"
                        ]:
                            (
                                gate(execution_context)
                                if not self._all_context_bound(gate)
                                else gate()
                            )  # updates vars in sub context if any measurements etc. occur
                    except ContinueException as e:
                        pass  # eval of current iteration stops and we continue

            context["gates"].append(partial(self._handle_break, unrolled))
        elif (
            loop_params is None
        ):  # could be func param... then it's a value that will be evaluated at "runtime" (when calling the QNode)
            print(f"Uninitialized iterator in loop {f'for_{node.span.start_line}'}.")

    def visit_ReturnStatement(self, node: QASMNode, context: dict):
        """
        Registers a return statement. Points to the var that needs to be set in an outer scope when this
        subroutine is called.
        """
        context["return"] = node.expression.name

    def visit_QuantumMeasurementStatement(self, node: QASMNode, context: dict):
        """
        Registers a quantum measurement.

        Args:
            node (QASMNode): the quantum measurement node.
            context (dict): the current context.
        """
        self._init_gates_list(context)
        if isinstance(node.measure.qubit, Identifier):
            meas = partial(measure, node.measure.qubit.name)

        elif isinstance(node.measure.qubit, IntegerLiteral):  # TODO: are all these cases necessary
            meas = partial(measure, node.measure.qubit.value)

        elif isinstance(node.measure.qubit, list):
            for qubit in node.measure.qubit:
                if isinstance(qubit, Identifier):
                    meas = partial(measure, qubit.name)

                elif isinstance(qubit, IntegerLiteral):
                    meas = partial(measure, qubit.value)

        # handle data flow. Note: data flow dependent on quantum operations deferred? Promises?
        def set_local_var(execution_context: dict):
            name = (
                node.target.name if isinstance(node.target.name, str) else node.target.name.name
            )  # str or Identifier
            res = meas()
            self._update_var(res, name, node, execution_context)
            return res

        # references to an unresolved value see a func for now
        name = (
            node.target.name if isinstance(node.target.name, str) else node.target.name.name
        )  # str or Identifier
        self._update_var(set_local_var, name, node, context)
        context["gates"].append(set_local_var)

    def visit_QuantumReset(self, node: QASMNode, context: dict):
        """
        Registers a reset of a quantum gate.

        Args:
            node (QASMNode): the quantum reset node.
            context (dict): the current context.
        """
        self._init_gates_list(context)
        if isinstance(node.qubits, Identifier):
            context["gates"].append(partial(measure, node.qubits.name, reset=True))
        elif isinstance(
            node.qubits, IntegerLiteral
        ):  # TODO: are all these cases necessary / supported
            context["gates"].append(partial(measure, node.qubits.value, reset=True))
        elif isinstance(node.qubits, list):
            for qubit in node.qubits:
                if isinstance(qubit, Identifier):
                    context["gates"].append(partial(measure, qubit.name, reset=True))
                elif isinstance(qubit, IntegerLiteral):
                    context["gates"].append(partial(measure, qubit.value, reset=True))

    def visit_SubroutineDefinition(self, node: QASMNode, context: dict):
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

        # process the subroutine body. Note we don't call the gates in the outer context until the subroutine is called.
        context["scopes"]["subroutines"][node.name.name] = self.visit(
            node.body, context["scopes"]["subroutines"][node.name.name]
        )

        # Should we visit now or when the function is called with arguments?
        # Now is fine b/c we evaluate vars at the end, and visit only constructs partials that
        # reference them during a visit.

    def _get_wires_helper(self, curr: dict, wires: list):
        """
        We need a device with enough wires to support all the qubit declarations in every sub-context.
        We need to instantiate a device with enough wires to support all qubit declarations, with names
        that give enough specificity to identify them when they are in different scopes but share the same
        name in the QASM file, for example.

        Args:
            curr (dict): the current context in our recursive descent.
            wires (list): the wires list we are building.

        Returns:
            list: the list of wires we have found.
        """
        if "scopes" in curr:  # TODO: raise warning when a variable is shadowed
            contexts = curr["scopes"]
            for context_type, typed_contexts in contexts.items():
                for typed_context_name, typed_context in typed_contexts.items():
                    if context_type != "switches" and context_type != "branches":
                        wires += [
                            f'{contexts[context_type][typed_context_name]["name"]}_{w}'
                            for w in contexts[context_type][typed_context_name]["wires"]
                        ]
                        wires = self._get_wires_helper(typed_context, wires)
                    else:
                        # TODO: account for: we don't need new wires for scopes that don't have their own namespaces
                        for cond in typed_context.keys():
                            wires += [
                                f'{typed_context[cond]["name"]}_{w}'
                                for w in typed_context[cond]["wires"]
                            ]
                            wires = self._get_wires_helper(typed_context[cond], wires)
        return wires

    def construct_qfunc(self, context: dict, init_context: dict):
        """
        Constructs a Callable quantum function that may be queued into a QNode.

        Args:
            context (dict): the final context resulting from the compilation pass.
            init_context (dict): the initial context.

        Returns:
            dict: the final executed context.
        """
        if "device" not in context:
            wires = [w for w in context["wires"]] if "wires" in context else []
            curr = context
            if "scopes" in curr:
                wires = self._get_wires_helper(curr, wires)
            context["wires"] = wires
        # passes and modifies init_context through gate calls during second pass
        # static variables are already bound in context["gates"]
        context["callable"] = lambda: [
            gate(init_context) if not self._all_context_bound(gate) else gate()
            for gate in context["gates"]
        ]
        return init_context

    @staticmethod
    def visit_QubitDeclaration(node: QASMNode, context: dict):
        """
        Registers a qubit declaration. Named qubits are mapped to numbered wires by their indices
        in context["wires"].

        Args:
            node (QASMNode): The QubitDeclaration QASMNode.
            context (dict): The current context.
        """
        context["wires"].append(node.qubit.name)

    def visit_ConstantDeclaration(self, node: QASMNode, context: dict):
        """
        Registers a constant declaration. Traces data flow through the context, transforming QASMNodes into
        Python type variables that can be readily used in expression eval, etc.

        Args:
            node (QASMNode): The constant QASMNode.
            context (dict): The current context.
        """
        self.visit_ClassicalDeclaration(node, context, constant=True)

    def visit_ClassicalDeclaration(self, node: QASMNode, context: dict, constant=False):
        """
        Registers a classical declaration. Traces data flow through the context, transforming QASMNodes into Python
        type variables that can be readily used in expression evaluation, for example.

        Args:
            node (QASMNode): The ClassicalDeclaration QASMNode.
            context (dict): The current context.
        """

        # compile time tracking of static variables
        self._init_vars(context)
        if node.init_expression is not None:
            # TODO: store AST objects in context instead of these dicts?

            # Note: vars which are clean may be bound at compile time, dirty ones
            # must be calculated during execution
            if isinstance(node.init_expression, BitstringLiteral):
                context["vars"][node.identifier.name] = {
                    "ty": node.type.__class__.__name__,
                    "val": self.eval_expr(node.init_expression, context),
                    "size": node.init_expression.width,
                    "line": node.init_expression.span.start_line,
                    "dirty": False,
                    "constant": constant,
                }
            elif not isinstance(node.init_expression, ArrayLiteral):
                context["vars"][node.identifier.name] = {
                    "ty": node.type.__class__.__name__,
                    "val": self.eval_expr(node.init_expression, context),
                    "line": node.init_expression.span.start_line,
                    "dirty": False,
                    "constant": constant,
                }
            else:
                context["vars"][node.identifier.name] = {
                    "ty": node.type.__class__.__name__,
                    "val": [
                        self.eval_expr(literal, context) for literal in node.init_expression.values
                    ],
                    "line": node.init_expression.span.start_line,
                    "dirty": False,
                    "constant": constant,
                }
        else:
            # the var is declared but uninitialized
            context["vars"][node.identifier.name] = {
                "ty": node.type.__class__.__name__,
                "val": None,
                "line": node.span.start_line,
                "dirty": False,
                "constant": constant,
            }

        # runtime Callable
        self._init_gates_list(context)
        context["gates"].append(partial(self.visit_ClassicalDeclaration, node))

    def visit_QuantumGate(self, node: QASMNode, context: dict):
        """
        Registers a quantum gate application. Calls the appropriate handler based on the sort of gate
        (parameterized or non-parameterized).

        Args:
            node (QASMNode): The QuantumGate QASMNode.
            context (dict): The current context.
        """
        self._init_gates_list(context)
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
                    wrapper = partial(
                        ops.pow, z=self.retrieve_variable(mod.argument.name, context)["val"]
                    )
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
        if len(context["wires"]) == 0 and (
            "outer_wires" not in context or len(context["outer_wires"]) == 0
        ):
            raise NameError(
                f"Attempt to reference wires that have not been declared in {context['name']}"
            )

    def gate(self, gates_dict: dict, node: QASMNode, context: dict):
        """
        Registers a gate application. Builds a Callable partial
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
        self._require_wires(context)
        args = []
        for arg in node.arguments:
            if hasattr(arg, "name") and "vars" in context and arg.name in context["vars"]:
                val = self.retrieve_variable(arg.name, context)["val"]
                if val is not None:
                    args.append(val)
                else:
                    raise NameError(f"Attempt to reference uninitialized parameter {arg.name}!")
            elif re.search("Literal", arg.__class__.__name__) is not None:
                args.append(arg.value)
            else:
                raise NameError(f"Undeclared variable {arg.name} encountered in QASM.")
        return partial(
            gates_dict[node.name.name.upper()],
            *args,
            wires=[self.eval_expr(node.qubits[q], context) for q in range(len(node.qubits))],
        )
